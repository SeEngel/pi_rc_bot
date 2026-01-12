from __future__ import annotations

import asyncio
import json
import os
import time
from glob import glob
from dataclasses import dataclass
from typing import Any

from agent.src.base_agent import BaseAgentConfig, BaseWorkbenchChatAgent
from agent.src.config import OpenAIClientConfig, load_yaml, resolve_repo_root

from .mcp_client import call_mcp_tool_json
from .sound_activity import detect_sound_activity
from .protocol import ProtocolLogger


@dataclass(frozen=True)
class AdvisorMemorizerConfig:
	"""Configuration for the optional MemorizerAgent integration."""

	enabled: bool
	# Path to agent/memorizer/config.yaml (absolute).
	config_path: str

	# Behavior toggles.
	ingest_user_utterances: bool
	recall_for_questions: bool

	# Retrieval settings.
	recall_top_n: int

	# Timeouts (seconds) so the advisor loop can't stall forever.
	ingest_timeout_seconds: float
	recall_timeout_seconds: float


@dataclass(frozen=True)
class AdvisorMemoryConfig:
	max_tokens: int
	avg_chars_per_token: int
	summary_dir: str
	summary_max_chars: int


@dataclass
class AdvisorLedger:
	"""A lightweight event log + rough token budgeting."""

	entries: list[str]
	chars: int

	def append(self, line: str) -> None:
		line = str(line)
		self.entries.append(line)
		self.chars += len(line)

	def text(self) -> str:
		return "\n".join(self.entries)


@dataclass(frozen=True)
class AdvisorSettings:
	name: str
	persona_instructions: str
	debug: bool
	debug_log_path: str | None
	response_language: str

	openai_model: str
	openai_base_url: str | None
	env_file_path: str | None

	listen_mcp_url: str
	speak_mcp_url: str
	observe_mcp_url: str

	# interaction
	min_transcript_chars: int
	max_listen_attempts: int
	reprompt_text: str
	listen_stream: bool
	interrupt_speech_on_sound: bool
	wait_for_speech_finish: bool
	speech_status_poll_interval_seconds: float
	speech_max_wait_seconds: float
	suppress_alone_mode_while_speaking: bool
	stop_speech_on_sound_while_waiting: bool
	post_speech_interaction_grace_seconds: float
	stop_words: list[str]

	# alone
	think_interval_seconds: float
	observation_question: str
	max_thought_chars: int

	# sound activity
	sound_enabled: bool
	sound_threshold_rms: int
	sound_sample_rate_hz: int
	sound_window_seconds: float
	sound_poll_interval_seconds: float
	sound_arecord_device: str | None
	sound_fallback_to_interaction_on_error: bool

	memory: AdvisorMemoryConfig
	memorizer: AdvisorMemorizerConfig


class AdvisorBrain(BaseWorkbenchChatAgent):
	"""LLM-only reasoning brain for the advisor (no MCP tools here)."""

	def __init__(self, *, name: str, instructions: str, model: str, base_url: str | None, env_file_path: str | None):
		cfg = BaseAgentConfig(
			name=name,
			instructions=instructions,
			openai=OpenAIClientConfig(model=model, base_url=base_url, env_file_path=env_file_path),
			mcp_servers=[],
		)
		super().__init__(cfg)


class AdvisorAgent:
	"""Long-running orchestrator for interaction + alone modes."""

	def __init__(self, settings: AdvisorSettings, *, dry_run: bool = False):
		self.settings = settings
		self._dry_run = bool(dry_run)
		self._ledger = AdvisorLedger(entries=[], chars=0)
		self._summary: str | None = None
		self._brain: AdvisorBrain | None = None
		self._memorizer: Any | None = None
		self._memorizer_task: asyncio.Task[None] | None = None
		self._last_alone_think_ts = 0.0
		self._force_interaction_until_ts = 0.0
		self._protocol = ProtocolLogger(enabled=bool(settings.debug), log_path=settings.debug_log_path)
		self._protocol.open()
		# Best-effort persistent memory: load newest summary file (if any) to seed this run.
		self._load_latest_persisted_summary()

	def _load_latest_persisted_summary(self) -> None:
		"""Load the newest persisted summary into `_summary`.

		Note: the advisor already writes summaries on rollover, but previously it did not
		reload them on startup. Without this, restarts lose all memory.
		"""
		try:
			mem = self.settings.memory
			repo_root = resolve_repo_root(os.path.dirname(__file__))
			summary_dir = os.path.join(repo_root, str(mem.summary_dir))
			pattern = os.path.join(summary_dir, "summary_*.txt")
			paths = sorted(glob(pattern))
			if not paths:
				return
			latest = paths[-1]
			with open(latest, "r", encoding="utf-8") as f:
				summary = (f.read() or "").strip()
			if not summary:
				return
			self._summary = summary
			# Seed ledger with summary so prompt-time recall can include it.
			self._ledger = AdvisorLedger(entries=[f"[summary]\n{summary}"], chars=len(summary))
			self._emit("memory_load", component="advisor.memory", path=latest, chars=len(summary))
		except Exception as exc:
			# Never fail startup due to memory IO.
			self._emit("memory_load_warning", component="advisor.memory", error=str(exc))

	def _conversation_context(self, *, max_chars: int) -> str:
		"""Return a tail snippet of the conversation (human/assistant) from the ledger."""
		max_chars_i = max(0, int(max_chars))
		if max_chars_i <= 0:
			return ""
		lines: list[str] = []
		for entry in self._ledger.entries:
			e = str(entry)
			if e.startswith("Human:") or e.startswith("Assistant:") or e.startswith("[summary]"):
				lines.append(e)
		text = "\n".join(lines).strip()
		if not text:
			return ""
		if len(text) <= max_chars_i:
			return text
		return text[-max_chars_i:]

	def _wants_conversation_summary(self, text: str) -> bool:
		low = (text or "").strip().lower()
		if not low:
			return False
		# German + English triggers.
		keywords = (
			"zusammenfassung",
			"fass zusammen",
			"fasse zusammen",
			"kurze zusammenfassung",
			"summary",
			"summarize",
			"recap",
			"was haben wir bisher",
			"unser bisheriges gespräch",
		)
		return any(k in low for k in keywords)

	async def _summarize_conversation_for_user(self) -> str:
		if self._dry_run:
			return "(dry_run) (Zusammenfassung)"
		brain = await self._ensure_brain()
		lang = self.settings.response_language
		# Use a bounded context window to avoid extremely long prompts.
		mem = self.settings.memory
		max_chars_budget = int(mem.max_tokens) * max(1, int(mem.avg_chars_per_token))
		ctx = self._conversation_context(max_chars=min(12000, max(1500, max_chars_budget // 6)))
		if not ctx:
			return "Ich habe in diesem Lauf noch kein Gesprächsprotokoll, das ich zusammenfassen kann."
		prompt = (
			"Gib eine kurze, hilfreiche Zusammenfassung unseres bisherigen Gesprächs.\n"
			"Behalte: wichtige Fakten, meine Präferenzen, offene Aufgaben, und was zuletzt passiert ist.\n"
			"Antworte NUR als Fließtext, keine JSON.\n"
			"Sprache der Ausgabe MUSS sein: "
			+ str(lang)
			+ ".\n\n"
			"Gesprächsverlauf (Ausschnitt, zuletzt am Ende):\n"
			+ ctx
		)
		start = time.perf_counter()
		self._emit("brain_call_start", component="advisor.brain", kind="user_summary")
		try:
			summary = str(await brain.run(prompt)).strip()
		finally:
			dur_ms = (time.perf_counter() - start) * 1000.0
			self._emit("brain_call_end", component="advisor.brain", kind="user_summary", duration_ms=round(dur_ms, 2))
		return summary or "Ich konnte gerade keine sinnvolle Zusammenfassung erzeugen."

	@classmethod
	def from_config_yaml(cls, path: str) -> "AdvisorAgent":
		cfg = load_yaml(path)
		here = os.path.dirname(os.path.abspath(path))
		repo_root = resolve_repo_root(here)
		env_file_path = os.path.join(repo_root, ".env") if os.path.isfile(os.path.join(repo_root, ".env")) else None

		advisor_cfg = cfg.get("advisor", {}) if isinstance(cfg, dict) else {}
		openai_cfg = cfg.get("openai", {}) if isinstance(cfg, dict) else {}
		mcp_cfg = cfg.get("mcp", {}) if isinstance(cfg, dict) else {}
		interaction_cfg = cfg.get("interaction", {}) if isinstance(cfg, dict) else {}
		alone_cfg = cfg.get("alone", {}) if isinstance(cfg, dict) else {}
		sound_cfg = cfg.get("sound_activity", {}) if isinstance(cfg, dict) else {}
		memory_cfg = cfg.get("memory", {}) if isinstance(cfg, dict) else {}
		memorizer_cfg = cfg.get("memorizer", {}) if isinstance(cfg, dict) else {}

		name = str((advisor_cfg or {}).get("name") or "AdvisorAgent")
		persona_instructions = str((advisor_cfg or {}).get("persona_instructions") or "").strip() or "You are an always-on robot advisor."
		debug = bool((advisor_cfg or {}).get("debug"))
		debug_log_path_raw = str((advisor_cfg or {}).get("debug_log_path") or "").strip()
		debug_log_path = debug_log_path_raw or None
		response_language = str((advisor_cfg or {}).get("response_language") or "de-DE").strip() or "de-DE"

		openai_model = str((openai_cfg or {}).get("model") or "gpt-4o-mini")
		base_url_raw = str((openai_cfg or {}).get("base_url") or "").strip()
		openai_base_url = base_url_raw or None

		listen_mcp_url = str((mcp_cfg or {}).get("listen_mcp_url") or "http://127.0.0.1:8602/mcp").strip()
		speak_mcp_url = str((mcp_cfg or {}).get("speak_mcp_url") or "http://127.0.0.1:8601/mcp").strip()
		observe_mcp_url = str((mcp_cfg or {}).get("observe_mcp_url") or "http://127.0.0.1:8603/mcp").strip()

		# Memorizer (optional, separate agent that uses services/memory MCP)
		memorizer_enabled = bool((memorizer_cfg or {}).get("enabled") if "enabled" in (memorizer_cfg or {}) else True)
		memorizer_ingest = bool(
			(memorizer_cfg or {}).get("ingest_user_utterances")
			if "ingest_user_utterances" in (memorizer_cfg or {})
			else True
		)
		memorizer_recall = bool(
			(memorizer_cfg or {}).get("recall_for_questions")
			if "recall_for_questions" in (memorizer_cfg or {})
			else True
		)
		recall_top_n = int((memorizer_cfg or {}).get("recall_top_n") or 3)
		recall_top_n = max(1, min(10, recall_top_n))
		ingest_timeout_seconds = float((memorizer_cfg or {}).get("ingest_timeout_seconds") or 12.0)
		recall_timeout_seconds = float((memorizer_cfg or {}).get("recall_timeout_seconds") or 12.0)

		config_path_raw = str((memorizer_cfg or {}).get("config_path") or "agent/memorizer/config.yaml").strip()
		# Resolve relative paths from repo root.
		memorizer_config_path = config_path_raw
		if not os.path.isabs(memorizer_config_path):
			memorizer_config_path = os.path.join(repo_root, memorizer_config_path)
		memorizer_config_path = os.path.abspath(memorizer_config_path)

		min_transcript_chars = int((interaction_cfg or {}).get("min_transcript_chars") or 3)
		max_listen_attempts = int((interaction_cfg or {}).get("max_listen_attempts") or 2)
		reprompt_text = str((interaction_cfg or {}).get("reprompt_text") or "Say it again please.")
		listen_stream = bool((interaction_cfg or {}).get("listen_stream"))
		interrupt_speech_on_sound = bool(
			(interaction_cfg or {}).get("interrupt_speech_on_sound")
			if "interrupt_speech_on_sound" in (interaction_cfg or {})
			else True
		)
		wait_for_speech_finish = bool(
			(interaction_cfg or {}).get("wait_for_speech_finish")
			if "wait_for_speech_finish" in (interaction_cfg or {})
			else True
		)
		speech_status_poll_interval_seconds = float(
			(interaction_cfg or {}).get("speech_status_poll_interval_seconds") or 0.2
		)
		speech_max_wait_seconds = float((interaction_cfg or {}).get("speech_max_wait_seconds") or 120.0)
		suppress_alone_mode_while_speaking = bool(
			(interaction_cfg or {}).get("suppress_alone_mode_while_speaking")
			if "suppress_alone_mode_while_speaking" in (interaction_cfg or {})
			else True
		)
		stop_speech_on_sound_while_waiting = bool(
			(interaction_cfg or {}).get("stop_speech_on_sound_while_waiting")
			if "stop_speech_on_sound_while_waiting" in (interaction_cfg or {})
			else False
		)
		post_speech_interaction_grace_seconds = float(
			(interaction_cfg or {}).get("post_speech_interaction_grace_seconds") or 1.5
		)
		stop_words_cfg = (interaction_cfg or {}).get("stop_words")
		stop_words: list[str] = []
		if isinstance(stop_words_cfg, list):
			stop_words = [str(x).strip().lower() for x in stop_words_cfg if str(x).strip()]
		elif isinstance(stop_words_cfg, str) and stop_words_cfg.strip():
			stop_words = [stop_words_cfg.strip().lower()]
		if not stop_words:
			stop_words = ["stop", "stopp", "halt", "genug"]

		think_interval_seconds = float((alone_cfg or {}).get("think_interval_seconds") or 20.0)
		observation_question = str((alone_cfg or {}).get("observation_question") or "Briefly describe what you see.")
		max_thought_chars = int((alone_cfg or {}).get("max_thought_chars") or 240)

		sound_enabled = bool((sound_cfg or {}).get("enabled"))
		sound_threshold_rms = int((sound_cfg or {}).get("threshold_rms") or 800)
		sound_sample_rate_hz = int((sound_cfg or {}).get("sample_rate_hz") or 16000)
		sound_window_seconds = float((sound_cfg or {}).get("window_seconds") or 0.15)
		sound_poll_interval_seconds = float((sound_cfg or {}).get("poll_interval_seconds") or 0.25)
		sound_arecord_device_raw = str((sound_cfg or {}).get("arecord_device") or "").strip()
		sound_arecord_device = sound_arecord_device_raw or None
		sound_fallback_to_interaction_on_error = bool(
			(sound_cfg or {}).get("fallback_to_interaction_on_error")
			if "fallback_to_interaction_on_error" in (sound_cfg or {})
			else True
		)

		mem = AdvisorMemoryConfig(
			max_tokens=int((memory_cfg or {}).get("max_tokens") or 30000),
			avg_chars_per_token=int((memory_cfg or {}).get("avg_chars_per_token") or 4),
			summary_dir=str((memory_cfg or {}).get("summary_dir") or "memory/advisor"),
			summary_max_chars=int((memory_cfg or {}).get("summary_max_chars") or 2500),
		)

		memorizer = AdvisorMemorizerConfig(
			enabled=bool(memorizer_enabled),
			config_path=memorizer_config_path,
			ingest_user_utterances=bool(memorizer_ingest),
			recall_for_questions=bool(memorizer_recall),
			recall_top_n=int(recall_top_n),
			ingest_timeout_seconds=float(ingest_timeout_seconds),
			recall_timeout_seconds=float(recall_timeout_seconds),
		)

		return cls(
			AdvisorSettings(
				name=name,
				persona_instructions=persona_instructions,
				debug=debug,
				debug_log_path=debug_log_path,
				response_language=response_language,
				openai_model=openai_model,
				openai_base_url=openai_base_url,
				env_file_path=env_file_path,
				listen_mcp_url=listen_mcp_url,
				speak_mcp_url=speak_mcp_url,
				observe_mcp_url=observe_mcp_url,
				min_transcript_chars=min_transcript_chars,
				max_listen_attempts=max_listen_attempts,
				reprompt_text=reprompt_text,
				listen_stream=listen_stream,
				interrupt_speech_on_sound=interrupt_speech_on_sound,
				wait_for_speech_finish=wait_for_speech_finish,
				speech_status_poll_interval_seconds=speech_status_poll_interval_seconds,
				speech_max_wait_seconds=speech_max_wait_seconds,
				suppress_alone_mode_while_speaking=suppress_alone_mode_while_speaking,
				stop_speech_on_sound_while_waiting=stop_speech_on_sound_while_waiting,
				post_speech_interaction_grace_seconds=post_speech_interaction_grace_seconds,
				stop_words=stop_words,
				think_interval_seconds=think_interval_seconds,
				observation_question=observation_question,
				max_thought_chars=max_thought_chars,
				sound_enabled=sound_enabled,
				sound_threshold_rms=sound_threshold_rms,
				sound_sample_rate_hz=sound_sample_rate_hz,
				sound_window_seconds=sound_window_seconds,
				sound_poll_interval_seconds=sound_poll_interval_seconds,
				sound_arecord_device=sound_arecord_device,
				sound_fallback_to_interaction_on_error=sound_fallback_to_interaction_on_error,
				memory=mem,
				memorizer=memorizer,
			)
		)

	async def _ensure_memorizer(self) -> Any:
		"""Create and enter the MemorizerAgent if enabled."""
		if self._dry_run:
			raise RuntimeError("Memorizer disabled in dry-run mode")
		if not bool(self.settings.memorizer.enabled):
			raise RuntimeError("Memorizer is disabled by config")
		if self._memorizer is not None:
			return self._memorizer

		# Lazy import so advisor can run even if memorizer module isn't present.
		from agent.memorizer.src.memorizer_agent import MemorizerAgent

		cfg_path = str(self.settings.memorizer.config_path)
		memorizer = MemorizerAgent.from_config_yaml(cfg_path)
		await memorizer.__aenter__()
		self._memorizer = memorizer
		self._emit(
			"memorizer_start",
			component="advisor.memorizer",
			config_path=cfg_path,
		)
		return memorizer

	async def _wait_for_speech_completion(self, *, context: str, max_wait_seconds: float | None = None) -> None:
		"""Optionally wait until the speak service reports not speaking.

		This keeps the main behavior closer to the old (blocking) speak behavior,
		while still allowing interruption on sound by stopping playback.
		"""
		if self._dry_run:
			return
		poll_s = max(0.05, float(self.settings.speech_status_poll_interval_seconds or 0.2))
		max_wait = float(max_wait_seconds) if max_wait_seconds is not None else float(self.settings.speech_max_wait_seconds or 120.0)
		start = time.perf_counter()
		self._emit(
			"speech_wait_start",
			component="advisor",
			context=context,
			poll_interval_s=round(poll_s, 3),
			max_wait_s=round(max_wait, 3),
		)
		saw_sound = False
		max_grace = max(0.0, float(self.settings.post_speech_interaction_grace_seconds or 0.0))
		while True:
			st = await self._speaker_status()
			speaking = bool(isinstance(st, dict) and st.get("speaking"))
			if not speaking:
				if saw_sound and max_grace > 0:
					self._force_interaction_until_ts = max(self._force_interaction_until_ts, time.time() + max_grace)
					self._emit(
						"speech_wait_post_grace",
						component="advisor",
						context=context,
						grace_seconds=round(max_grace, 3),
					)
				self._emit("speech_wait_end", component="advisor", context=context, reason="not_speaking")
				return
			elapsed = time.perf_counter() - start
			if elapsed >= max_wait:
				self._emit(
					"speech_wait_end",
					component="advisor",
					context=context,
					reason="timeout",
					elapsed_s=round(elapsed, 3),
				)
				return

			# If sound is detected while we wait, it's often just the robot's own voice.
			# We therefore DO NOT stop speech by default.
			# Instead, we remember that sound happened and (after speech ends) we force
			# one interaction attempt shortly, so the human can be heard right after.
			if self.settings.sound_enabled:
				try:
					res = detect_sound_activity(
						threshold_rms=self.settings.sound_threshold_rms,
						sample_rate_hz=self.settings.sound_sample_rate_hz,
						window_seconds=self.settings.sound_window_seconds,
						arecord_device=self.settings.sound_arecord_device,
					)
					if bool(res.active):
						saw_sound = True
						self._emit(
							"speech_wait_sound",
							component="advisor",
							context=context,
							backend=str(res.backend),
							rms=int(res.rms),
						)
						if self.settings.stop_speech_on_sound_while_waiting and self.settings.interrupt_speech_on_sound:
							self._emit(
								"interrupt",
								component="advisor",
								kind="sound_while_waiting_speech",
								backend=str(res.backend),
								rms=int(res.rms),
							)
							await self._speaker_stop()
							self._emit("speech_wait_end", component="advisor", context=context, reason="stopped_on_sound")
							return
				except Exception as exc:
					self._emit(
						"speech_wait_warning",
						component="advisor",
						context=context,
						error=str(exc),
					)

			await asyncio.sleep(poll_s)

	async def _ensure_brain(self) -> AdvisorBrain:
		if self._dry_run:
			raise RuntimeError("Advisor brain disabled in dry-run mode")
		if self._brain is not None:
			return self._brain

		seed = ""
		if self._summary:
			seed = f"\n\nConversation summary so far:\n{self._summary.strip()}\n"

		lang = self.settings.response_language
		instructions = (
			f"{self.settings.persona_instructions.strip()}\n\n"
			"You will be given transcripts and observations from sensors/services.\n"
			"When asked to respond to a human, reply as the robot speaking out loud.\n"
			"You may reason internally in English, but ALL user-facing text you output MUST be in: "
			+ str(lang)
			+ ".\n"
			"Keep responses short unless the user asks for detail.\n"
			+ seed
		)

		brain = AdvisorBrain(
			name=self.settings.name,
			instructions=instructions,
			model=self.settings.openai_model,
			base_url=self.settings.openai_base_url,
			env_file_path=self.settings.env_file_path,
		)
		# Enter context now (so subsequent calls are fast)
		await brain.__aenter__()
		self._brain = brain
		return brain

	async def close(self) -> None:
		brain = self._brain
		self._brain = None
		if brain is not None:
			await brain.__aexit__(None, None, None)
		mem_task = self._memorizer_task
		self._memorizer_task = None
		if mem_task is not None and not mem_task.done():
			mem_task.cancel()
			try:
				await mem_task
			except Exception:
				pass
		memorizer = self._memorizer
		self._memorizer = None
		if memorizer is not None:
			try:
				await memorizer.__aexit__(None, None, None)
			except Exception:
				pass
		self._protocol.close()

	def _should_query_memorizer(self, text: str) -> bool:
		"""Heuristic: only ask the memorizer for recall when it seems useful."""
		low = (text or "").strip().lower()
		if not low:
			return False

		# Direct memory cues.
		cues = (
			"remember",
			"do you remember",
			"what did i",
			"what was my",
			"what's my",
			"what is my",
			"who am i",
			"my name",
			"where did i",
			"where is my",
			"last time",
			"earlier",
			"before",
			"we talked",
			"i told you",
			"did i tell you",
			# German
			"erinnerst du",
			"erinnern",
			"weißt du noch",
			"wie heiße ich",
			"mein name",
			"wo habe ich",
			"vorhin",
			"letztes mal",
			"habe ich dir gesagt",
		)
		if any(c in low for c in cues):
			return True

		# If the user asks something personal/ongoing (often preference or past fact).
		if low.startswith("my ") or low.startswith("mein ") or "my " in low or "mein " in low:
			# Avoid spamming recall for every "my"; keep it to question forms.
			if "?" in low or any(x in low for x in ("what", "where", "which", "wer", "was", "wo", "welche")):
				return True

		return False

	async def _memorizer_recall(self, query: str) -> str | None:
		if self._dry_run:
			return None
		if not self.settings.memorizer.enabled or not self.settings.memorizer.recall_for_questions:
			return None
		if not self._should_query_memorizer(query):
			return None

		try:
			memorizer = await self._ensure_memorizer()
		except Exception as exc:
			self._emit("memorizer_warning", component="advisor.memorizer", kind="ensure_failed", error=str(exc))
			return None

		start = time.perf_counter()
		self._emit(
			"memorizer_call_start",
			component="advisor.memorizer",
			kind="recall",
			query=query,
			url=getattr(memorizer.settings, "memory_mcp_url", None),
		)
		try:
			out = await asyncio.wait_for(
				memorizer.recall(query, top_n=int(self.settings.memorizer.recall_top_n)),
				timeout=float(self.settings.memorizer.recall_timeout_seconds),
			)
		except asyncio.TimeoutError:
			dur_ms = (time.perf_counter() - start) * 1000.0
			self._emit(
				"memorizer_call_error",
				component="advisor.memorizer",
				kind="recall",
				duration_ms=round(dur_ms, 2),
				error="timeout",
			)
			return None
		except Exception as exc:
			dur_ms = (time.perf_counter() - start) * 1000.0
			self._emit(
				"memorizer_call_error",
				component="advisor.memorizer",
				kind="recall",
				duration_ms=round(dur_ms, 2),
				error=str(exc),
			)
			return None
		else:
			dur_ms = (time.perf_counter() - start) * 1000.0
			out_s = str(out or "").strip()
			self._emit(
				"memorizer_call_end",
				component="advisor.memorizer",
				kind="recall",
				duration_ms=round(dur_ms, 2),
				preview=out_s[:400],
			)
			if not out_s:
				return None
			# If memorizer returns an explicit error, don't inject into prompt.
			if out_s.lower().startswith("error:"):
				return None
			return out_s

	async def _memorizer_ingest_background(self, text: str) -> None:
		"""Ask memorizer to decide whether to store the user utterance.

		Runs as a background task (best-effort) so it doesn't block speaking.
		"""
		if self._dry_run:
			return
		if not self.settings.memorizer.enabled or not self.settings.memorizer.ingest_user_utterances:
			return
		if self._is_bad_transcript(text):
			return
		if self._matches_stop_word(text):
			return

		try:
			memorizer = await self._ensure_memorizer()
		except Exception as exc:
			self._emit("memorizer_warning", component="advisor.memorizer", kind="ensure_failed", error=str(exc))
			return

		start = time.perf_counter()
		self._emit(
			"memorizer_call_start",
			component="advisor.memorizer",
			kind="ingest",
			chars=len(text or ""),
			preview=text,
			url=getattr(memorizer.settings, "memory_mcp_url", None),
		)
		try:
			out = await asyncio.wait_for(
				memorizer.ingest(text, force_store=False),
				timeout=float(self.settings.memorizer.ingest_timeout_seconds),
			)
		except asyncio.TimeoutError:
			dur_ms = (time.perf_counter() - start) * 1000.0
			self._emit(
				"memorizer_call_error",
				component="advisor.memorizer",
				kind="ingest",
				duration_ms=round(dur_ms, 2),
				error="timeout",
			)
			return
		except Exception as exc:
			dur_ms = (time.perf_counter() - start) * 1000.0
			self._emit(
				"memorizer_call_error",
				component="advisor.memorizer",
				kind="ingest",
				duration_ms=round(dur_ms, 2),
				error=str(exc),
			)
			return
		else:
			dur_ms = (time.perf_counter() - start) * 1000.0
			out_s = str(out or "").strip()
			self._emit(
				"memorizer_call_end",
				component="advisor.memorizer",
				kind="ingest",
				duration_ms=round(dur_ms, 2),
				preview=out_s[:400],
			)
			if out_s:
				self._ledger.append(f"[memorizer] {out_s}")

	def _emit(self, event: str, **fields: Any) -> None:
		# Streaming JSONL protocol for live debugging.
		self._protocol.emit(event, **fields)

	def _log(self, msg: str) -> None:
		# Back-compat helper: a simple textual message inside the protocol.
		self._emit("log", message=str(msg))

	def _is_bad_transcript(self, text: str) -> bool:
		t = (text or "").strip()
		if len(t) < int(self.settings.min_transcript_chars or 0):
			return True
		# Common "no speech" markers.
		low = t.lower()
		if low in {"(dry_run)", "[unk]", "unk", "", "..."}:
			return True
		return False

	def _matches_stop_word(self, text: str) -> bool:
		low = (text or "").strip().lower()
		if not low:
			return False
		# Exact match or prefix ("stop!", "stopp bitte")
		for w in self.settings.stop_words:
			w = (w or "").strip().lower()
			if not w:
				continue
			if low == w or low.startswith(w + " ") or low.startswith(w + "!") or low.startswith(w + "."):
				return True
		return False

	def _wants_vision(self, text: str) -> bool:
		"""Heuristic: does the human ask about what the robot can/could see?"""
		low = (text or "").strip().lower()
		if not low:
			return False
		keywords = (
			# English
			"what do you see",
			"what can you see",
			"can you see",
			"look",
			"in front",
			"camera",
			"show me",
			"what is this",
			"what's this",
			# German
			"was siehst",
			"was kannst du sehen",
			"kannst du sehen",
			"siehst du",
			"schau",
			"sieh",
			"vor dir",
			"vor dir ist",
			"kamera",
			"was ist das",
			"was ist vor",
			"observier",
			"beobacht",
		)
		return any(k in low for k in keywords)

	def _parse_decision_json(self, raw: str) -> tuple[str | None, bool | None]:
		"""Parse a brain JSON response. Returns (response_text, need_observe)."""
		def _strip_code_fence(s: str) -> str:
			s = (s or "").strip()
			if not s.startswith("```"):
				return s
			lines = s.splitlines()
			if not lines:
				return ""
			# Drop opening fence line (``` or ```json)
			if lines[0].lstrip().startswith("```"):
				lines = lines[1:]
			# Drop closing fence line
			if lines and lines[-1].strip().startswith("```"):
				lines = lines[:-1]
			return "\n".join(lines).strip()

		def _try_load_json(s: str) -> Any | None:
			s = (s or "").strip()
			if not s:
				return None
			try:
				return json.loads(s)
			except Exception:
				pass
			# Fallback: extract the largest {...} span (handles stray text around the JSON).
			start = s.find("{")
			end = s.rfind("}")
			if start != -1 and end != -1 and end > start:
				candidate = s[start : end + 1].strip()
				try:
					return json.loads(candidate)
				except Exception:
					return None
			return None

		normalized = _strip_code_fence(raw or "")
		obj = _try_load_json(normalized)
		if obj is None and normalized != (raw or ""):
			# One more try on the original (in case the JSON wasn't inside the code fence).
			obj = _try_load_json(raw or "")
		if obj is None:
			return (None, None)
		if not isinstance(obj, dict):
			return (None, None)
		resp = obj.get("response_text")
		need = obj.get("need_observe")
		resp_s = str(resp).strip() if resp is not None else None
		need_b: bool | None
		if isinstance(need, bool):
			need_b = need
		elif isinstance(need, (int, float)):
			need_b = bool(need)
		elif isinstance(need, str):
			need_b = need.strip().lower() in {"true", "1", "yes", "y"}
		else:
			need_b = None
		return (resp_s if resp_s else None, need_b)

	async def _listen_once(self) -> str:
		if self._dry_run:
			self._emit("tool_call", tool="listen", component="mcp.listen", dry_run=True)
			return "(dry_run transcript)"
		start = time.perf_counter()
		self._emit("tool_call_start", tool="listen", component="mcp.listen", url=self.settings.listen_mcp_url)
		payload: dict[str, Any] = {}
		if self.settings.listen_stream:
			payload["stream"] = True
		try:
			res = await call_mcp_tool_json(url=self.settings.listen_mcp_url, tool_name="listen", timeout_seconds=180.0, **payload)
		except Exception as exc:
			dur_ms = (time.perf_counter() - start) * 1000.0
			self._emit(
				"tool_call_error",
				tool="listen",
				component="mcp.listen",
				url=self.settings.listen_mcp_url,
				duration_ms=round(dur_ms, 2),
				error=str(exc),
			)
			raise
		dur_ms = (time.perf_counter() - start) * 1000.0
		self._emit(
			"tool_call_end",
			tool="listen",
			component="mcp.listen",
			url=self.settings.listen_mcp_url,
			duration_ms=round(dur_ms, 2),
			result_keys=sorted([str(k) for k in res.keys()]) if isinstance(res, dict) else None,
		)
		# listening service returns {ok, text, raw}
		text = str(res.get("text") or "")
		return text.strip()

	async def _speak(self, text: str) -> None:
		t = (text or "").strip()
		if not t:
			return
		if self._dry_run:
			self._emit("tool_call", tool="speak", component="mcp.speak", dry_run=True, text=t)
			return
		start = time.perf_counter()
		self._emit("tool_call_start", tool="speak", component="mcp.speak", url=self.settings.speak_mcp_url, text=t)
		try:
			res = await call_mcp_tool_json(url=self.settings.speak_mcp_url, tool_name="speak", timeout_seconds=120.0, text=t)
		except Exception as exc:
			dur_ms = (time.perf_counter() - start) * 1000.0
			self._emit(
				"tool_call_error",
				tool="speak",
				component="mcp.speak",
				url=self.settings.speak_mcp_url,
				duration_ms=round(dur_ms, 2),
				error=str(exc),
			)
			raise
		dur_ms = (time.perf_counter() - start) * 1000.0
		self._emit(
			"tool_call_end",
			tool="speak",
			component="mcp.speak",
			url=self.settings.speak_mcp_url,
			duration_ms=round(dur_ms, 2),
			result_keys=sorted([str(k) for k in res.keys()]) if isinstance(res, dict) else None,
		)
		if self.settings.wait_for_speech_finish:
			await self._wait_for_speech_completion(context="after_speak")

	async def _speaker_status(self) -> dict[str, Any]:
		if self._dry_run:
			return {"ok": True, "speaking": False}
		start = time.perf_counter()
		self._emit("tool_call_start", tool="status", component="mcp.speak", url=self.settings.speak_mcp_url)
		try:
			res = await call_mcp_tool_json(url=self.settings.speak_mcp_url, tool_name="status", timeout_seconds=5.0)
		except Exception as exc:
			dur_ms = (time.perf_counter() - start) * 1000.0
			self._emit(
				"tool_call_error",
				tool="status",
				component="mcp.speak",
				url=self.settings.speak_mcp_url,
				duration_ms=round(dur_ms, 2),
				error=str(exc),
			)
			return {"ok": False, "speaking": False, "error": str(exc)}
		dur_ms = (time.perf_counter() - start) * 1000.0
		self._emit(
			"tool_call_end",
			tool="status",
			component="mcp.speak",
			url=self.settings.speak_mcp_url,
			duration_ms=round(dur_ms, 2),
			result_keys=sorted([str(k) for k in res.keys()]) if isinstance(res, dict) else None,
		)
		return res if isinstance(res, dict) else {"ok": True, "value": res}

	async def _speaker_stop(self) -> bool:
		if self._dry_run:
			self._emit("tool_call", tool="stop", component="mcp.speak", dry_run=True)
			return True
		start = time.perf_counter()
		self._emit("tool_call_start", tool="stop", component="mcp.speak", url=self.settings.speak_mcp_url)
		try:
			res = await call_mcp_tool_json(url=self.settings.speak_mcp_url, tool_name="stop", timeout_seconds=5.0)
		except Exception as exc:
			dur_ms = (time.perf_counter() - start) * 1000.0
			self._emit(
				"tool_call_error",
				tool="stop",
				component="mcp.speak",
				url=self.settings.speak_mcp_url,
				duration_ms=round(dur_ms, 2),
				error=str(exc),
			)
			return False
		dur_ms = (time.perf_counter() - start) * 1000.0
		self._emit(
			"tool_call_end",
			tool="stop",
			component="mcp.speak",
			url=self.settings.speak_mcp_url,
			duration_ms=round(dur_ms, 2),
			result_keys=sorted([str(k) for k in res.keys()]) if isinstance(res, dict) else None,
		)
		if isinstance(res, dict):
			return bool(res.get("stopped"))
		return True

	async def _observe(self, question: str) -> str:
		if self._dry_run:
			q = str(question or "").strip() or "Describe the scene."
			self._emit("tool_call", tool="observe", component="mcp.observe", dry_run=True, question=q)
			return f"(dry_run observe) {q}"
		q = str(question or "").strip() or "Describe the scene."
		start = time.perf_counter()
		self._emit("tool_call_start", tool="observe", component="mcp.observe", url=self.settings.observe_mcp_url, question=q)
		try:
			res = await call_mcp_tool_json(
				url=self.settings.observe_mcp_url,
				tool_name="observe",
				timeout_seconds=120.0,
				question=q,
			)
		except Exception as exc:
			dur_ms = (time.perf_counter() - start) * 1000.0
			self._emit(
				"tool_call_error",
				tool="observe",
				component="mcp.observe",
				url=self.settings.observe_mcp_url,
				duration_ms=round(dur_ms, 2),
				error=str(exc),
			)
			raise
		dur_ms = (time.perf_counter() - start) * 1000.0
		self._emit(
			"tool_call_end",
			tool="observe",
			component="mcp.observe",
			url=self.settings.observe_mcp_url,
			duration_ms=round(dur_ms, 2),
			result_keys=sorted([str(k) for k in res.keys()]) if isinstance(res, dict) else None,
		)
		return str(res.get("text") or res.get("value") or res.get("raw") or "").strip()

	async def _maybe_summarize_and_reset(self) -> None:
		if self._dry_run:
			return
		mem = self.settings.memory
		max_chars = int(mem.max_tokens) * max(1, int(mem.avg_chars_per_token))
		if self._ledger.chars < max_chars:
			return

		self._emit(
			"memory_summarize_start",
			component="advisor.memory",
			ledger_chars=self._ledger.chars,
			max_chars=max_chars,
		)

		brain = await self._ensure_brain()
		log_text = self._ledger.text()
		prompt = (
			"Summarize the following event log for a long-running robot assistant.\n"
			"Keep: key facts, user preferences, any ongoing tasks, and what happened most recently.\n"
			"Output plain text only. Target <= "
			+ str(int(mem.summary_max_chars))
			+ " characters.\n\n"
			"Event log:\n"
			+ log_text
		)
		try:
			start = time.perf_counter()
			self._emit("brain_call_start", component="advisor.brain", kind="summarize")
			summary = str(await brain.run(prompt)).strip()
			dur_ms = (time.perf_counter() - start) * 1000.0
			self._emit("brain_call_end", component="advisor.brain", kind="summarize", duration_ms=round(dur_ms, 2))
		except Exception as exc:
			summary = f"(summary failed: {exc})"
			self._emit("brain_call_error", component="advisor.brain", kind="summarize", error=str(exc))

		# Persist summary
		repo_root = resolve_repo_root(os.path.dirname(__file__))
		summary_dir = os.path.join(repo_root, str(mem.summary_dir))
		os.makedirs(summary_dir, exist_ok=True)
		ts = time.strftime("%Y%m%d_%H%M%S")
		path = os.path.join(summary_dir, f"summary_{ts}.txt")
		try:
			with open(path, "w", encoding="utf-8") as f:
				f.write(summary)
		except Exception:
			pass

		self._summary = summary
		self._ledger = AdvisorLedger(entries=[f"[summary]\n{summary}"], chars=len(summary))

		# Reset the brain (new LLM context seeded with summary).
		await self.close()
		# Re-open protocol after closing brain
		self._protocol.open()
		self._emit("memory_summarize_end", component="advisor.memory", path=path)

	async def _decide_and_respond(
		self,
		*,
		human_text: str,
		observation: str | None,
		memory_hint: str | None,
	) -> tuple[str, bool | None]:
		if self._dry_run:
			# Minimal, deterministic behavior for test runs.
			if observation and observation.strip():
				return (f"(dry_run) Based on what I see: {observation.strip()}", None)
			return (f"(dry_run) I heard: {human_text.strip()}", None)

		brain = await self._ensure_brain()

		obs_block = ""
		if observation and observation.strip():
			obs_block = f"\n\nObservation (camera):\n{observation.strip()}\n"

		mem_block = ""
		if memory_hint and memory_hint.strip():
			# Keep it clearly separated so the model treats it as tool-provided context.
			mem_block = "\n\nMemory report (from memorizer agent):\n" + memory_hint.strip() + "\n"

		# Include a small recall window so the model can answer questions about what was
		# said earlier (otherwise each prompt is effectively standalone).
		mem = self.settings.memory
		max_chars_budget = int(mem.max_tokens) * max(1, int(mem.avg_chars_per_token))
		ctx = self._conversation_context(max_chars=min(8000, max(1200, max_chars_budget // 10)))
		ctx_block = ""
		if ctx:
			ctx_block = "\n\nConversation so far (most recent last):\n" + ctx + "\n"

		# Ask for a JSON decision so we can optionally trigger observation earlier later.
		lang = self.settings.response_language
		prompt = (
			"You are responding to a human speaking to the robot.\n"
			"The robot MUST speak in language: "
			+ str(lang)
			+ ".\n"
			"You may think in English, but the returned response_text MUST be in that language.\n"
			"Return ONLY a JSON object with keys:\n"
			"- response_text: string (what the robot should say out loud)\n"
			"- need_observe: boolean (true if you need camera info to answer well)\n\n"
			+ ctx_block
			+ f"Human said: {human_text.strip()}"
			+ obs_block
			+ mem_block
		)
		start = time.perf_counter()
		self._emit(
			"brain_call_start",
			component="advisor.brain",
			kind="decide_and_respond",
			has_observation=bool(observation and observation.strip()),
		)
		raw = str(await brain.run(prompt)).strip()
		dur_ms = (time.perf_counter() - start) * 1000.0
		self._emit(
			"brain_call_end",
			component="advisor.brain",
			kind="decide_and_respond",
			duration_ms=round(dur_ms, 2),
			raw_preview=raw,
		)
		resp, need_observe = self._parse_decision_json(raw)
		self._emit(
			"decision",
			component="advisor.brain",
			need_observe=need_observe,
			has_observation=bool(observation and observation.strip()),
			response_preview=resp or raw,
		)
		if resp is not None:
			return (resp, need_observe)

		# Fallback: treat model output as plain speech.
		return (raw, None)

	async def _interaction_step(self) -> None:
		self._emit("state", component="advisor", state="interaction_start")
		# Listen (maybe retry once with reprompt)
		text = ""
		for attempt in range(int(self.settings.max_listen_attempts or 1)):
			self._emit("state", component="advisor", state="interaction_listen", attempt=attempt + 1)
			text = await self._listen_once()
			self._emit(
				"listen_result",
				component="mcp.listen",
				attempt=attempt + 1,
				chars=len(text or ""),
				is_bad=self._is_bad_transcript(text),
				preview=text,
			)
			if not self._is_bad_transcript(text):
				break
			if attempt < int(self.settings.max_listen_attempts or 1) - 1:
				self._emit("state", component="advisor", state="interaction_reprompt")
				await self._speak(self.settings.reprompt_text)

		if self._is_bad_transcript(text):
			self._emit("state", component="advisor", state="interaction_no_transcript")
			return

		# If user says a stop word, stop any ongoing speech and keep quiet.
		if self._matches_stop_word(text):
			self._emit("interrupt", component="advisor", kind="stop_word", transcript=text)
			await self._speaker_stop()
			return

		self._ledger.append(f"Human: {text}")
		# If user explicitly asks for a conversation summary, do it directly.
		if self._wants_conversation_summary(text):
			summary = await self._summarize_conversation_for_user()
			self._emit("state", component="advisor", state="interaction_speak_summary")
			await self._speak(summary)
			self._ledger.append(f"Assistant: {summary}")
			await self._maybe_summarize_and_reset()
			self._emit("state", component="advisor", state="interaction_end")
			return
		await self._maybe_summarize_and_reset()

		wants_vision = self._wants_vision(text)
		self._emit("interaction_intent", component="advisor", wants_vision=wants_vision)

		# If the human clearly asks about seeing, observe first.
		obs: str | None = None
		if wants_vision:
			self._emit("state", component="advisor", state="interaction_observe")
			obs = await self._observe(f"Answer the human by describing what the robot sees now. Human said: {text}")

		# Decide response.
		self._emit("state", component="advisor", state="interaction_think")
		mem_hint = await self._memorizer_recall(text)
		if mem_hint:
			self._ledger.append(f"[recall] {mem_hint}")
		response, need_observe = await self._decide_and_respond(human_text=text, observation=obs, memory_hint=mem_hint)

		# If the brain asked for an observation and we haven't taken one yet, do it once.
		if need_observe is True and not (obs and obs.strip()):
			self._emit("state", component="advisor", state="interaction_observe_assist")
			obs = await self._observe(f"Help answer the human request: {text}")
			response, _ = await self._decide_and_respond(human_text=text, observation=obs, memory_hint=mem_hint)

		response = response.strip()
		if not response:
			response = "I heard you, but I didn't get enough to answer. Could you say that again?"

		self._emit("state", component="advisor", state="interaction_speak")
		await self._speak(response)
		self._ledger.append(f"Assistant: {response}")

		# Ask memorizer to decide whether to store this user utterance (best-effort, non-blocking).
		if self.settings.memorizer.enabled and self.settings.memorizer.ingest_user_utterances and not self._dry_run:
			# Cancel previous background ingest if still running (avoid backlog).
			if self._memorizer_task is not None and not self._memorizer_task.done():
				self._memorizer_task.cancel()
			self._memorizer_task = asyncio.create_task(self._memorizer_ingest_background(text))

		await self._maybe_summarize_and_reset()
		self._emit("state", component="advisor", state="interaction_end")

	async def _alone_step(self) -> None:
		now = time.time()
		if (now - self._last_alone_think_ts) < float(self.settings.think_interval_seconds or 0.0):
			return
		if self.settings.suppress_alone_mode_while_speaking:
			st = await self._speaker_status()
			if bool(isinstance(st, dict) and st.get("speaking")):
				self._emit("state", component="advisor", state="alone_skip_speaking")
				return
		self._last_alone_think_ts = now
		self._emit("state", component="advisor", state="alone_start")

		obs = await self._observe(self.settings.observation_question)
		self._ledger.append(f"[alone] observation: {obs}")
		await self._maybe_summarize_and_reset()

		lang = self.settings.response_language
		if self._dry_run:
			# In dry-run, still show the intended spoken language.
			thought = (
				f"(dry_run {lang}) Ich sehe: {obs}"[: int(self.settings.max_thought_chars)]
			)
		else:
			brain = await self._ensure_brain()
			prompt = (
				"You are alone (no human speaking right now). Think out loud briefly based on the observation.\n"
				"Output MUST be in language: "
				+ str(lang)
				+ ".\n"
				"Keep it under "
				+ str(int(self.settings.max_thought_chars))
				+ " characters.\n\n"
				"Observation:\n"
				+ (obs or "")
			)
			thought = str(await brain.run(prompt)).strip()
		if thought:
			self._emit("state", component="advisor", state="alone_speak")
			await self._speak(thought)
			self._ledger.append(f"[alone] thought: {thought}")
			await self._maybe_summarize_and_reset()
		self._emit("state", component="advisor", state="alone_end")

	async def run_forever(self, *, max_iterations: int | None = None) -> None:
		"""Run the advisor loop.

		If `max_iterations` is set, stops after that many polling iterations (useful for testing).
		"""
		iters = 0
		self._emit(
			"loop_start",
			component="advisor",
			dry_run=self._dry_run,
			sound_enabled=self.settings.sound_enabled,
			response_language=self.settings.response_language,
		)
		try:
			while True:
				iters += 1
				if max_iterations is not None and iters > int(max_iterations):
					self._emit("loop_stop", component="advisor", reason="max_iterations", iterations=iters - 1)
					return

				try:
					self._emit("iteration", component="advisor", iteration=iters)
					active = False
					rms = 0
					backend = None
					if self._dry_run:
						active = False
					elif time.time() < float(self._force_interaction_until_ts or 0.0):
						active = True
						self._emit(
							"mode_hint",
							component="advisor",
							hint="force_interaction",
							until_ts=round(float(self._force_interaction_until_ts), 3),
						)
					elif self.settings.sound_enabled:
						res = detect_sound_activity(
							threshold_rms=self.settings.sound_threshold_rms,
							sample_rate_hz=self.settings.sound_sample_rate_hz,
							window_seconds=self.settings.sound_window_seconds,
							arecord_device=self.settings.sound_arecord_device,
						)
						active = bool(res.active)
						rms = int(res.rms)
						backend = str(res.backend)
						reason = getattr(res, "reason", None)
						self._emit("sound", component="advisor.sound", backend=backend, rms=rms, active=active, reason=reason)
						if (reason or backend == "none") and self.settings.sound_fallback_to_interaction_on_error:
							self._emit(
								"sound_fallback",
								component="advisor.sound",
								fallback="interaction",
								backend=backend,
								reason=reason,
							)
							active = True
					else:
						# If sound detection is disabled, default to interaction mode.
						active = True
						self._emit("sound", component="advisor.sound", backend=None, rms=None, active=True, disabled=True)

					if active:
						# If someone interrupts while the robot is speaking, stop playback before listening.
						if self.settings.interrupt_speech_on_sound:
							st = await self._speaker_status()
							speaking = bool(isinstance(st, dict) and st.get("speaking"))
							if speaking:
								self._emit("interrupt", component="advisor", kind="sound_while_speaking")
								await self._speaker_stop()
								await asyncio.sleep(0.15)
							self._emit("mode", component="advisor", mode="interaction")
							await self._interaction_step()
					else:
							if self.settings.suppress_alone_mode_while_speaking:
								st = await self._speaker_status()
								if bool(isinstance(st, dict) and st.get("speaking")):
									self._emit("mode", component="advisor", mode="speaking")
								else:
									self._emit("mode", component="advisor", mode="alone")
									await self._alone_step()
							else:
								self._emit("mode", component="advisor", mode="alone")
								await self._alone_step()
				except asyncio.CancelledError:
					raise
				except Exception as exc:
					# Keep the advisor alive; emit error and continue after a short delay.
					self._emit("iteration_error", component="advisor", iteration=iters, error=str(exc))
					self._ledger.append(f"[error] iteration {iters}: {exc}")
					await asyncio.sleep(1.0)

				await asyncio.sleep(float(self.settings.sound_poll_interval_seconds or 0.25))
		except KeyboardInterrupt:
			self._emit("loop_stop", component="advisor", reason="keyboard_interrupt", iterations=iters)
			return
		finally:
			self._emit("loop_end", component="advisor", iterations=iters)
			await self.close()
