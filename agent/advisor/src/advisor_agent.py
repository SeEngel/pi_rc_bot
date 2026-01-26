from __future__ import annotations

import asyncio
import json
import os
import re
import time
from glob import glob
from dataclasses import dataclass
from typing import Any

from agent.src.base_agent import BaseAgentConfig, BaseWorkbenchChatAgent
from agent.src.config import OpenAIClientConfig, load_yaml, resolve_repo_root
from agent.todo.src import TodoAgent, TaskPlanner, TaskPlannerSettings, LLMTodoAgent, LLMTodoAgentSettings

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
class AdvisorTodoConfig:
	"""Configuration for the local TodoAgent integration with optional LLM task planning."""

	enabled: bool
	# Path to agent/todo/config.yaml (absolute).
	config_path: str

	# If true, always provide todo status to the brain prompt.
	include_in_prompt: bool

	# If true, instruct the brain to always mention the next open task.
	# (Keeps the human aware of what happens next.)
	mention_next_in_response: bool

	# If true, use LLM-based TaskPlanner to decompose multi-step requests.
	use_task_planner: bool = True

	# Minimum words to trigger task planning (avoid planning for simple commands).
	task_planner_min_words: int = 6

	# If true, review the plan after creation and ask if changes are needed.
	review_after_planning: bool = True

	# If true, review after all tasks are done to check if mission is complete.
	review_after_completion: bool = True

	# If true, enable autonomous todo generation in alone mode.
	autonomous_mode_enabled: bool = True


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
class AdvisorDecision:
	response_text: str | None
	need_observe: bool | None
	actions: list[dict[str, Any]]


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
	move_mcp_url: str
	head_mcp_url: str
	proximity_mcp_url: str
	perception_mcp_url: str
	safety_mcp_url: str

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
	# Brief listen after completing tasks to allow human interruption
	brief_listen_timeout_seconds: float

	# alone
	think_interval_seconds: float
	observation_question: str
	max_thought_chars: int

	# sound activity
	sound_enabled: bool
	sound_threshold_rms: int
	# Require this many consecutive "active" sound windows before entering interaction mode.
	# Helps avoid false triggers from brief ambient noise spikes.
	sound_active_windows_required: int
	sound_sample_rate_hz: int
	sound_window_seconds: float
	sound_poll_interval_seconds: float
	sound_arecord_device: str | None
	sound_fallback_to_interaction_on_error: bool

	memory: AdvisorMemoryConfig
	memorizer: AdvisorMemorizerConfig
	todo: AdvisorTodoConfig

	# Fields with default values must come last in dataclasses
	# Enable streaming TTS (sends text chunks to speaker as LLM generates them)
	streaming_tts_enabled: bool = True
	# alone (optional exploration/motion)
	alone_explore_enabled: bool = False
	alone_explore_speed: int = 20
	alone_explore_threshold_cm: float = 35.0
	alone_explore_duration_s: float = 0.6
	alone_explore_far_duration_s: float = 1.2
	alone_explore_speak: bool = False


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
		self._todo: TodoAgent | None = None
		self._task_planner: TaskPlanner | None = None
		self._llm_todo: LLMTodoAgent | None = None  # New LLM-based todo agent
		self._memorizer_task: asyncio.Task[None] | None = None
		self._last_alone_think_ts = 0.0
		self._force_interaction_until_ts = 0.0
		self._sound_active_streak = 0
		self._protocol = ProtocolLogger(enabled=bool(settings.debug), log_path=settings.debug_log_path)
		self._protocol.open()
		self._init_todo_agent()
		self._init_llm_todo_agent()  # Initialize LLM todo agent
		self._init_task_planner()
		# Best-effort persistent memory: load newest summary file (if any) to seed this run.
		self._load_latest_persisted_summary()
    
	def _init_todo_agent(self) -> None:
		"""Initialize local TodoAgent (best-effort; must never crash startup)."""
		try:
			cfg = self.settings.todo
			if not bool(cfg.enabled):
				return
			# The TodoAgent manages its own persistence.
			agent = TodoAgent.from_config_yaml(cfg.config_path)
			agent.load()
			self._todo = agent
			self._emit("todo_init", component="advisor.todo", enabled=True, state_path=agent.settings.state_path)
		except Exception as exc:
			self._emit("todo_init_warning", component="advisor.todo", error=str(exc))
			self._todo = None

	def _init_llm_todo_agent(self) -> None:
		"""Initialize the LLM-based TodoAgent (best-effort; must never crash startup)."""
		try:
			cfg = self.settings.todo
			if not bool(cfg.enabled) or not bool(cfg.use_task_planner):
				self._emit("llm_todo_init_skip", component="advisor.llm_todo", reason="disabled_in_config")
				return
			if self._dry_run:
				self._emit("llm_todo_init_skip", component="advisor.llm_todo", reason="dry_run")
				return

			# LLMTodoAgent uses the same OpenAI settings as the advisor brain.
			llm_todo_settings = LLMTodoAgentSettings(
				enabled=True,
				model=self.settings.openai_model,
				base_url=self.settings.openai_base_url,
				env_file_path=self.settings.env_file_path,
				language=self.settings.response_language,
				review_after_planning=bool(cfg.review_after_planning),
				review_after_completion=bool(cfg.review_after_completion),
				autonomous_mode_enabled=bool(cfg.autonomous_mode_enabled),
			)
			self._llm_todo = LLMTodoAgent(llm_todo_settings)
			is_enabled = self._llm_todo.is_enabled()
			self._emit(
				"llm_todo_init",
				component="advisor.llm_todo",
				enabled=True,
				is_enabled=is_enabled,
				model=self.settings.openai_model,
				review_after_planning=bool(cfg.review_after_planning),
				review_after_completion=bool(cfg.review_after_completion),
				autonomous_mode_enabled=bool(cfg.autonomous_mode_enabled),
			)
		except Exception as exc:
			self._emit("llm_todo_init_warning", component="advisor.llm_todo", error=str(exc))
			self._llm_todo = None

	def _init_task_planner(self) -> None:
		"""Initialize the LLM-based TaskPlanner (best-effort; must never crash startup)."""
		try:
			cfg = self.settings.todo
			if not bool(cfg.enabled) or not bool(cfg.use_task_planner):
				return
			if self._dry_run:
				return

			# TaskPlanner uses the same OpenAI settings as the advisor brain.
			planner_settings = TaskPlannerSettings(
				enabled=True,
				model=self.settings.openai_model,
				base_url=self.settings.openai_base_url,
				env_file_path=self.settings.env_file_path,
				min_words_for_planning=cfg.task_planner_min_words,
			)
			self._task_planner = TaskPlanner(planner_settings)
			self._emit("task_planner_init", component="advisor.task_planner", enabled=True)
		except Exception as exc:
			self._emit("task_planner_init_warning", component="advisor.task_planner", error=str(exc))
			self._task_planner = None

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
	def settings_from_config_yaml(cls, path: str) -> AdvisorSettings:
		"""Load AdvisorSettings from YAML without constructing an AdvisorAgent.

		This keeps config parsing side-effect free (no protocol/todo init) so callers
		can decide how/when to instantiate the agent (e.g. dry-run).
		"""
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
		todo_cfg = cfg.get("todo", {}) if isinstance(cfg, dict) else {}

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
		move_mcp_url = str((mcp_cfg or {}).get("move_mcp_url") or "http://127.0.0.1:8605/mcp").strip()
		head_mcp_url = str((mcp_cfg or {}).get("head_mcp_url") or "http://127.0.0.1:8606/mcp").strip()
		proximity_mcp_url = str((mcp_cfg or {}).get("proximity_mcp_url") or "http://127.0.0.1:8607/mcp").strip()
		perception_mcp_url = str((mcp_cfg or {}).get("perception_mcp_url") or "http://127.0.0.1:8608/mcp").strip()
		safety_mcp_url = str((mcp_cfg or {}).get("safety_mcp_url") or "http://127.0.0.1:8609/mcp").strip()

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
        
		# TodoAgent (local-only)
		todo_enabled = bool((todo_cfg or {}).get("enabled") if "enabled" in (todo_cfg or {}) else True)
		todo_config_path_raw = str((todo_cfg or {}).get("config_path") or "agent/todo/config.yaml").strip()
		todo_config_path = todo_config_path_raw
		if not os.path.isabs(todo_config_path):
			todo_config_path = os.path.join(repo_root, todo_config_path)
		todo_config_path = os.path.abspath(todo_config_path)
		include_in_prompt = bool((todo_cfg or {}).get("include_in_prompt") if "include_in_prompt" in (todo_cfg or {}) else True)
		mention_next_in_response = bool(
			(todo_cfg or {}).get("mention_next_in_response")
			if "mention_next_in_response" in (todo_cfg or {})
			else True
		)
		use_task_planner = bool(
			(todo_cfg or {}).get("use_task_planner")
			if "use_task_planner" in (todo_cfg or {})
			else True
		)
		task_planner_min_words = int((todo_cfg or {}).get("task_planner_min_words") or 6)
		review_after_planning = bool(
			(todo_cfg or {}).get("review_after_planning")
			if "review_after_planning" in (todo_cfg or {})
			else True
		)
		review_after_completion = bool(
			(todo_cfg or {}).get("review_after_completion")
			if "review_after_completion" in (todo_cfg or {})
			else True
		)
		autonomous_mode_enabled = bool(
			(todo_cfg or {}).get("autonomous_mode_enabled")
			if "autonomous_mode_enabled" in (todo_cfg or {})
			else True
		)

		todo = AdvisorTodoConfig(
			enabled=bool(todo_enabled),
			config_path=todo_config_path,
			include_in_prompt=bool(include_in_prompt),
			mention_next_in_response=bool(mention_next_in_response),
			use_task_planner=bool(use_task_planner),
			task_planner_min_words=int(task_planner_min_words),
			review_after_planning=bool(review_after_planning),
			review_after_completion=bool(review_after_completion),
			autonomous_mode_enabled=bool(autonomous_mode_enabled),
		)

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
		brief_listen_timeout_seconds = float(
			(interaction_cfg or {}).get("brief_listen_timeout_seconds") or 3.0
		)
		brief_listen_timeout_seconds = max(0.5, min(10.0, brief_listen_timeout_seconds))
		streaming_tts_enabled = bool(
			(interaction_cfg or {}).get("streaming_tts_enabled")
			if "streaming_tts_enabled" in (interaction_cfg or {})
			else True  # Default to enabled
		)

		think_interval_seconds = float((alone_cfg or {}).get("think_interval_seconds") or 20.0)
		observation_question = str((alone_cfg or {}).get("observation_question") or "Briefly describe what you see.")
		max_thought_chars = int((alone_cfg or {}).get("max_thought_chars") or 240)

		alone_explore_enabled = bool((alone_cfg or {}).get("explore_enabled") if "explore_enabled" in (alone_cfg or {}) else False)
		alone_explore_speed = int((alone_cfg or {}).get("explore_speed") or 20)
		alone_explore_speed = max(-100, min(100, alone_explore_speed))
		alone_explore_threshold_cm = float((alone_cfg or {}).get("explore_threshold_cm") or 35.0)
		alone_explore_threshold_cm = max(5.0, min(300.0, alone_explore_threshold_cm))
		alone_explore_duration_s = float((alone_cfg or {}).get("explore_duration_s") or 0.6)
		alone_explore_duration_s = max(0.1, min(5.0, alone_explore_duration_s))
		alone_explore_far_duration_s = float((alone_cfg or {}).get("explore_far_duration_s") or 1.2)
		alone_explore_far_duration_s = max(0.1, min(8.0, alone_explore_far_duration_s))
		alone_explore_speak = bool((alone_cfg or {}).get("explore_speak") if "explore_speak" in (alone_cfg or {}) else False)

		sound_enabled = bool((sound_cfg or {}).get("enabled"))
		sound_threshold_rms = int((sound_cfg or {}).get("threshold_rms") or 800)
		sound_active_windows_required = int((sound_cfg or {}).get("active_windows_required") or 1)
		sound_active_windows_required = max(1, min(10, sound_active_windows_required))
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

		return AdvisorSettings(
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
			move_mcp_url=move_mcp_url,
			head_mcp_url=head_mcp_url,
			proximity_mcp_url=proximity_mcp_url,
			perception_mcp_url=perception_mcp_url,
			safety_mcp_url=safety_mcp_url,
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
			brief_listen_timeout_seconds=brief_listen_timeout_seconds,
			streaming_tts_enabled=streaming_tts_enabled,
			think_interval_seconds=think_interval_seconds,
			observation_question=observation_question,
			max_thought_chars=max_thought_chars,
			alone_explore_enabled=alone_explore_enabled,
			alone_explore_speed=alone_explore_speed,
			alone_explore_threshold_cm=alone_explore_threshold_cm,
			alone_explore_duration_s=alone_explore_duration_s,
			alone_explore_far_duration_s=alone_explore_far_duration_s,
			alone_explore_speak=alone_explore_speak,
			sound_enabled=sound_enabled,
			sound_threshold_rms=sound_threshold_rms,
			sound_active_windows_required=sound_active_windows_required,
			sound_sample_rate_hz=sound_sample_rate_hz,
			sound_window_seconds=sound_window_seconds,
			sound_poll_interval_seconds=sound_poll_interval_seconds,
			sound_arecord_device=sound_arecord_device,
			sound_fallback_to_interaction_on_error=sound_fallback_to_interaction_on_error,
			memory=mem,
			memorizer=memorizer,
			todo=todo,
		)

	@classmethod
	def from_config_yaml(cls, path: str) -> "AdvisorAgent":
		settings = cls.settings_from_config_yaml(path)
		return cls(settings)

	def _todo_status_block(self) -> str:
		cfg = self.settings.todo
		if not bool(cfg.enabled) or not bool(cfg.include_in_prompt):
			return ""
		agent = self._todo
		if agent is None:
			return ""
		try:
			status = agent.status_text().strip()
		except Exception:
			return ""
		if not status:
			return ""
		return "\n\nTodo status (local):\n" + status + "\n"

	def _todo_policy_block(self) -> str:
		cfg = self.settings.todo
		if not bool(cfg.enabled) or not bool(cfg.include_in_prompt):
			return ""
		if not bool(cfg.mention_next_in_response):
			return ""
		return (
			"\n\nTodo policy:\n"
			"- Always keep the human aware of what happens next.\n"
			"- After answering, include the next OPEN todo item in your spoken response.\n"
			"- If there is no open task, explicitly say that there are no open tasks.\n"
			"- If the human changes the plan, acknowledge and ask for a new short step list if needed.\n"
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
			+ ".\n\n"
			"SPEECH STYLE - VERY IMPORTANT:\n"
			"- Keep responses SHORT and to the point (1-2 sentences max).\n"
			"- Do NOT ramble or add unnecessary commentary.\n"
			"- Do NOT ask follow-up questions unless absolutely necessary.\n"
			"- Be direct and action-oriented.\n"
			"- ONLY give long, detailed responses when the user EXPLICITLY asks for it.\n"
			"  Examples of when to give longer responses:\n"
			"  * 'tell me a story/tale' or 'erzähl mir eine Geschichte'\n"
			"  * 'explain in detail' or 'erkläre mir das genau'\n"
			"  * 'tell me more' or 'erzähl mir mehr'\n"
			"  * 'I want to know everything about...' or 'ich will alles wissen über...'\n"
			"  * explicit questions asking for detailed explanations\n"
			"- For simple commands, just acknowledge briefly: 'OK', 'Mach ich', 'Erledigt', etc.\n"
			"- Don't repeat back what the user just said unless clarifying ambiguity.\n"
			"\n"
			"IMPORTANT about audio errors:\n"
			"If you receive error messages about audio recording (e.g. 'recording too short', "
			"'no speech detected', 'audio duration less than X seconds'), these are NORMAL technical "
			"occurrences that happen when the user didn't speak long enough or there was background noise.\n"
			"DO NOT try to 'fix' or 'solve' these errors. DO NOT discuss the technical details.\n"
			"Simply ask the user briefly if they said something, e.g. 'Did you say something?' or "
			"'I didn't catch that, could you repeat?'\n"
			"Never mention specific error messages or technical audio parameters to the user.\n"
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
		# Clean up TaskPlanner
		task_planner = self._task_planner
		self._task_planner = None
		if task_planner is not None:
			try:
				await task_planner.close()
			except Exception:
				pass
		# Clean up LLMTodoAgent
		llm_todo = self._llm_todo
		self._llm_todo = None
		if llm_todo is not None:
			try:
				await llm_todo.close()
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
			"can you remember",
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
			"kannst du dich erinnern",
			"merkst du dir",
			"merk dir",
			"hast du dir gemerkt",
			"wie war nochmal",
			"wie heiße ich",
			"mein name",
			"wie heißt mein",
			"was war mein",
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

	async def _get_memory_mcp_url(self) -> str | None:
		"""Get the memory MCP URL from the memorizer agent's settings."""
		try:
			memorizer = await self._ensure_memorizer()
			return getattr(memorizer.settings, "memory_mcp_url", None)
		except Exception:
			return None

	async def _memorizer_recall_direct(self, query: str) -> str | None:
		"""Direct memory recall via MCP (LLM-triggered, bypasses heuristics).

		This is called when the Brain decides to use memory_recall action.
		Unlike _memorizer_recall, this doesn't use the MemorizerAgent LLM,
		it calls the memory MCP directly.
		"""
		if self._dry_run:
			return "(dry_run) Memory recall not available"
		if not self.settings.memorizer.enabled:
			return None

		memory_url = await self._get_memory_mcp_url()
		if not memory_url:
			self._emit("memory_direct_error", component="advisor.memory", action="recall", error="no_memory_url")
			return None

		start = time.perf_counter()
		self._emit(
			"memory_direct_call_start",
			component="advisor.memory",
			action="recall",
			query=query,
			url=memory_url,
		)
		try:
			result = await asyncio.wait_for(
				call_mcp_tool_json(
					url=memory_url,
					tool_name="get_top_n_memory_by_tags",
					timeout_seconds=15.0,
					content=query,
					top_n=self.settings.memorizer.recall_top_n,
					top_k_tags=5,  # Pre-filter by top 5 matching tags
				),
				timeout=float(self.settings.memorizer.recall_timeout_seconds),
			)
			dur_ms = (time.perf_counter() - start) * 1000.0

			# Extract memories from MCP response - format for natural speech
			if isinstance(result, dict):
				# Collect memory contents only (no technical details)
				memory_contents = []
				for mem in result.get("short_term_memory", []):
					content = mem.get("content", "").strip()
					if content:
						memory_contents.append(content)
				for mem in result.get("long_term_memory", []):
					content = mem.get("content", "").strip()
					if content:
						memory_contents.append(content)
				
				if memory_contents:
					# Join memories naturally for speech
					if len(memory_contents) == 1:
						text = memory_contents[0]
					else:
						# Multiple memories - join with natural connectors
						text = " Außerdem: ".join(memory_contents)
				else:
					text = None  # No memories found
			else:
				text = str(result) if result else None

			self._emit(
				"memory_direct_call_end",
				component="advisor.memory",
				action="recall",
				duration_ms=round(dur_ms, 2),
				result_preview=str(text)[:300] if text else None,
			)
			return str(text).strip() if text else None

		except asyncio.TimeoutError:
			dur_ms = (time.perf_counter() - start) * 1000.0
			self._emit("memory_direct_error", component="advisor.memory", action="recall", duration_ms=round(dur_ms, 2), error="timeout")
			return None
		except Exception as exc:
			dur_ms = (time.perf_counter() - start) * 1000.0
			self._emit("memory_direct_error", component="advisor.memory", action="recall", duration_ms=round(dur_ms, 2), error=str(exc))
			return None

	async def _memorizer_store_direct(self, content: str, tags: list[str] | None = None) -> bool:
		"""Direct memory store via MCP (LLM-triggered).

		This is called when the Brain decides to use memory_store action.
		"""
		if self._dry_run:
			return True
		if not self.settings.memorizer.enabled:
			return False

		memory_url = await self._get_memory_mcp_url()
		if not memory_url:
			self._emit("memory_direct_error", component="advisor.memory", action="store", error="no_memory_url")
			return False

		start = time.perf_counter()
		self._emit(
			"memory_direct_call_start",
			component="advisor.memory",
			action="store",
			content_preview=content[:100],
			tags=tags,
			url=memory_url,
		)
		try:
			await asyncio.wait_for(
				call_mcp_tool_json(
					url=memory_url,
					tool_name="store_memory",
					timeout_seconds=15.0,
					content=content,
					tags=tags or [],
				),
				timeout=float(self.settings.memorizer.ingest_timeout_seconds),
			)
			dur_ms = (time.perf_counter() - start) * 1000.0
			self._emit(
				"memory_direct_call_end",
				component="advisor.memory",
				action="store",
				duration_ms=round(dur_ms, 2),
				ok=True,
			)
			return True

		except asyncio.TimeoutError:
			dur_ms = (time.perf_counter() - start) * 1000.0
			self._emit("memory_direct_error", component="advisor.memory", action="store", duration_ms=round(dur_ms, 2), error="timeout")
			return False
		except Exception as exc:
			dur_ms = (time.perf_counter() - start) * 1000.0
			self._emit("memory_direct_error", component="advisor.memory", action="store", duration_ms=round(dur_ms, 2), error=str(exc))
			return False

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
		# Audio error messages from listen service - these are not real transcripts
		audio_error_markers = (
			"recording too short",
			"audio duration",
			"less than",
			"no speech detected",
			"speech not detected",
			"audio error",
			"microphone error",
			"aufnahme zu kurz",
			"keine sprache erkannt",
		)
		for marker in audio_error_markers:
			if marker in low:
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
			"look at",
			"look at me",
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
			"schau mich an",
			"schau mal",
			"schau",
			"guck",
			"guck mal",
			"guck mich an",
			"schau",
			"sieh",
			"vor dir",
			"vor dir ist",
			"kamera",
			"was ist das",
			"was ist vor",
			"observier",
			"beobacht",
			"was ist vor dir",
			"was ist da vorne",
			"was ist da",
			"was steht da",
		)
		return any(k in low for k in keywords)

	@staticmethod
	def _parse_number_with_unit(text: str, *, unit_words: tuple[str, ...]) -> float | None:
		"""Extract a floating point number with a unit (e.g. "0.7 sek")."""
		low = (text or "").lower()
		if not low:
			return None
		# Require an explicit unit word to reduce accidental matches (e.g., a speed value).
		pat = r"(\d+(?:[\.,]\d+)?)\s*(?:" + "|".join([re.escape(u) for u in unit_words]) + r")\b"
		m = re.search(pat, low)
		if not m:
			return None
		try:
			return float(m.group(1).replace(",", "."))
		except Exception:
			return None

	@staticmethod
	def _parse_speed(text: str) -> int | None:
		low = (text or "").lower()
		m = re.search(r"\b(?:speed|tempo|geschwindigkeit)\s*(?:ist\s*)?(\-?\d{1,3})\b", low)
		if not m:
			return None
		try:
			v = int(m.group(1))
			return max(-100, min(100, v))
		except Exception:
			return None

	async def _proximity_distance_cm(self) -> float | None:
		if self._dry_run:
			return 100.0
		start = time.perf_counter()
		self._emit("tool_call_start", tool="distance_cm", component="mcp.proximity", url=self.settings.proximity_mcp_url)
		try:
			res = await call_mcp_tool_json(url=self.settings.proximity_mcp_url, tool_name="distance_cm", timeout_seconds=5.0)
		except Exception as exc:
			dur_ms = (time.perf_counter() - start) * 1000.0
			self._emit(
				"tool_call_error",
				tool="distance_cm",
				component="mcp.proximity",
				url=self.settings.proximity_mcp_url,
				duration_ms=round(dur_ms, 2),
				error=str(exc),
			)
			return None
		dur_ms = (time.perf_counter() - start) * 1000.0
		self._emit(
			"tool_call_end",
			tool="distance_cm",
			component="mcp.proximity",
			url=self.settings.proximity_mcp_url,
			duration_ms=round(dur_ms, 2),
			result_keys=sorted([str(k) for k in res.keys()]) if isinstance(res, dict) else None,
		)
		if isinstance(res, dict):
			try:
				v = res.get("distance_cm")
				return float(v) if v is not None else None
			except Exception:
				return None
		return None

	async def _safety_stop(self) -> None:
		if self._dry_run:
			self._emit("tool_call", tool="stop", component="mcp.safety", dry_run=True)
			return
		start = time.perf_counter()
		self._emit("tool_call_start", tool="stop", component="mcp.safety", url=self.settings.safety_mcp_url)
		try:
			await call_mcp_tool_json(url=self.settings.safety_mcp_url, tool_name="stop", timeout_seconds=10.0)
		except Exception as exc:
			dur_ms = (time.perf_counter() - start) * 1000.0
			self._emit(
				"tool_call_error",
				tool="stop",
				component="mcp.safety",
				url=self.settings.safety_mcp_url,
				duration_ms=round(dur_ms, 2),
				error=str(exc),
			)
			return
		dur_ms = (time.perf_counter() - start) * 1000.0
		self._emit("tool_call_end", tool="stop", component="mcp.safety", url=self.settings.safety_mcp_url, duration_ms=round(dur_ms, 2))

	async def _safety_estop_on(self) -> None:
		if self._dry_run:
			self._emit("tool_call", tool="estop_on", component="mcp.safety", dry_run=True)
			return
		start = time.perf_counter()
		self._emit("tool_call_start", tool="estop_on", component="mcp.safety", url=self.settings.safety_mcp_url)
		try:
			await call_mcp_tool_json(url=self.settings.safety_mcp_url, tool_name="estop_on", timeout_seconds=5.0)
		except Exception as exc:
			dur_ms = (time.perf_counter() - start) * 1000.0
			self._emit(
				"tool_call_error",
				tool="estop_on",
				component="mcp.safety",
				url=self.settings.safety_mcp_url,
				duration_ms=round(dur_ms, 2),
				error=str(exc),
			)
			return
		dur_ms = (time.perf_counter() - start) * 1000.0
		self._emit("tool_call_end", tool="estop_on", component="mcp.safety", url=self.settings.safety_mcp_url, duration_ms=round(dur_ms, 2))

	async def _safety_estop_off(self) -> None:
		if self._dry_run:
			self._emit("tool_call", tool="estop_off", component="mcp.safety", dry_run=True)
			return
		start = time.perf_counter()
		self._emit("tool_call_start", tool="estop_off", component="mcp.safety", url=self.settings.safety_mcp_url)
		try:
			await call_mcp_tool_json(url=self.settings.safety_mcp_url, tool_name="estop_off", timeout_seconds=5.0)
		except Exception as exc:
			dur_ms = (time.perf_counter() - start) * 1000.0
			self._emit(
				"tool_call_error",
				tool="estop_off",
				component="mcp.safety",
				url=self.settings.safety_mcp_url,
				duration_ms=round(dur_ms, 2),
				error=str(exc),
			)
			return
		dur_ms = (time.perf_counter() - start) * 1000.0
		self._emit("tool_call_end", tool="estop_off", component="mcp.safety", url=self.settings.safety_mcp_url, duration_ms=round(dur_ms, 2))

	async def _safety_guarded_drive(
		self,
		*,
		speed: int,
		steer_deg: int = 0,
		duration_s: float | None = None,
		threshold_cm: float | None = None,
		await_completion: bool = False,
	) -> dict[str, Any] | None:
		"""Execute a guarded drive command.

		Args:
			speed: Signed speed percentage (-100 to 100).
			steer_deg: Steering angle in degrees (-35 to 35).
			duration_s: Duration in seconds. If provided, the motion runs for this long.
			threshold_cm: Obstacle detection threshold in cm.
			await_completion: If True, wait for the motion duration to complete before returning.
				This is useful for sequential task execution where the next task should
				only start after this motion is done.
		"""
		if self._dry_run:
			self._emit(
				"tool_call",
				tool="guarded_drive",
				component="mcp.safety",
				dry_run=True,
				speed=int(speed),
				steer_deg=int(steer_deg),
				duration_s=duration_s,
				threshold_cm=threshold_cm,
				await_completion=await_completion,
			)
			# In dry-run mode, still simulate the wait if requested
			if await_completion and duration_s is not None and duration_s > 0:
				await asyncio.sleep(float(duration_s))
			return {"ok": True, "dry_run": True}
		payload: dict[str, Any] = {"speed": int(speed), "steer_deg": int(steer_deg)}
		if duration_s is not None:
			payload["duration_s"] = float(duration_s)
		if threshold_cm is not None:
			payload["threshold_cm"] = float(threshold_cm)
		start = time.perf_counter()
		self._emit("tool_call_start", tool="guarded_drive", component="mcp.safety", url=self.settings.safety_mcp_url, await_completion=await_completion, **payload)
		try:
			res = await call_mcp_tool_json(url=self.settings.safety_mcp_url, tool_name="guarded_drive", timeout_seconds=10.0, **payload)
		except Exception as exc:
			dur_ms = (time.perf_counter() - start) * 1000.0
			self._emit(
				"tool_call_error",
				tool="guarded_drive",
				component="mcp.safety",
				url=self.settings.safety_mcp_url,
				duration_ms=round(dur_ms, 2),
				error=str(exc),
			)
			return None

		# If await_completion is requested and the motion was started successfully,
		# wait for the duration to elapse. The move service spawns a subprocess that
		# runs for duration_s, but returns immediately. We need to wait here.
		if await_completion and duration_s is not None and duration_s > 0:
			if isinstance(res, dict) and res.get("ok") and not res.get("blocked"):
				self._emit(
					"tool_call_wait",
					tool="guarded_drive",
					component="mcp.safety",
					wait_seconds=duration_s,
				)
				# Add a small buffer (0.1s) to ensure the motion subprocess completes
				await asyncio.sleep(float(duration_s) + 0.1)

		dur_ms = (time.perf_counter() - start) * 1000.0
		self._emit(
			"tool_call_end",
			tool="guarded_drive",
			component="mcp.safety",
			url=self.settings.safety_mcp_url,
			duration_ms=round(dur_ms, 2),
			result_keys=sorted([str(k) for k in res.keys()]) if isinstance(res, dict) else None,
		)
		return res if isinstance(res, dict) else {"ok": True, "value": res}

	async def _head_set_angles(self, *, pan_deg: int | None = None, tilt_deg: int | None = None) -> None:
		if self._dry_run:
			self._emit("tool_call", tool="set_angles", component="mcp.head", dry_run=True, pan_deg=pan_deg, tilt_deg=tilt_deg)
			return
		payload: dict[str, Any] = {}
		if pan_deg is not None:
			payload["pan_deg"] = int(pan_deg)
		if tilt_deg is not None:
			payload["tilt_deg"] = int(tilt_deg)
		start = time.perf_counter()
		self._emit("tool_call_start", tool="set_angles", component="mcp.head", url=self.settings.head_mcp_url, **payload)
		try:
			await call_mcp_tool_json(url=self.settings.head_mcp_url, tool_name="set_angles", timeout_seconds=10.0, **payload)
		except Exception as exc:
			dur_ms = (time.perf_counter() - start) * 1000.0
			self._emit(
				"tool_call_error",
				tool="set_angles",
				component="mcp.head",
				url=self.settings.head_mcp_url,
				duration_ms=round(dur_ms, 2),
				error=str(exc),
			)
			return
		dur_ms = (time.perf_counter() - start) * 1000.0
		self._emit("tool_call_end", tool="set_angles", component="mcp.head", url=self.settings.head_mcp_url, duration_ms=round(dur_ms, 2))

	async def _head_center(self) -> None:
		if self._dry_run:
			self._emit("tool_call", tool="center", component="mcp.head", dry_run=True)
			return
		start = time.perf_counter()
		self._emit("tool_call_start", tool="center", component="mcp.head", url=self.settings.head_mcp_url)
		try:
			await call_mcp_tool_json(url=self.settings.head_mcp_url, tool_name="center", timeout_seconds=10.0)
		except Exception as exc:
			dur_ms = (time.perf_counter() - start) * 1000.0
			self._emit(
				"tool_call_error",
				tool="center",
				component="mcp.head",
				url=self.settings.head_mcp_url,
				duration_ms=round(dur_ms, 2),
				error=str(exc),
			)
			return
		dur_ms = (time.perf_counter() - start) * 1000.0
		self._emit("tool_call_end", tool="center", component="mcp.head", url=self.settings.head_mcp_url, duration_ms=round(dur_ms, 2))

	async def _head_stop(self) -> None:
		if self._dry_run:
			self._emit("tool_call", tool="stop", component="mcp.head", dry_run=True)
			return
		try:
			await call_mcp_tool_json(url=self.settings.head_mcp_url, tool_name="stop", timeout_seconds=5.0)
		except Exception:
			return

	async def _try_handle_robot_command(self, text: str) -> str | None:
		low = (text or "").strip().lower()
		if not low:
			return None

		# Emergency stop.
		if any(k in low for k in ("notaus", "not-aus", "emergency stop", "e-stop", "estop")):
			await self._speaker_stop()
			await self._safety_estop_on()
			await self._safety_stop()
			return "Not-Aus aktiviert. Ich stoppe sofort."

		# Release estop.
		if any(k in low for k in ("notaus aus", "not-aus aus", "estop aus", "e-stop aus", "release estop")):
			await self._safety_estop_off()
			return "Not-Aus deaktiviert."

		# Distance query.
		if any(k in low for k in ("abstand", "wie weit", "distance", "how far")):
			d = await self._proximity_distance_cm()
			if d is None:
				return "Ich kann den Abstand gerade nicht messen."
			return f"Der Abstand nach vorne ist ungefähr {int(round(d))} Zentimeter."

		# Head/look commands.
		# Head and motion commands are handled by the LLM to properly parse natural language.
		# The LLM has access to head MCP tools (set_angles, scan, center) and will use them
		# based on understanding the user's intent.

		# Perception quick checks.
		if any(k in low for k in ("gesicht", "faces", "people", "person", "personen", "menschen")) and any(
			k in low for k in ("siehst", "siehst du", "do you see", "see", "erkennst", "detect")
		):
			res = await self._perception_detect()
			if not isinstance(res, dict) or not bool(res.get("ok")):
				return "Ich kann gerade keine Erkennung durchführen."
			faces = res.get("faces") if isinstance(res.get("faces"), list) else []
			people = res.get("people") if isinstance(res.get("people"), list) else []
			if faces:
				return f"Ich sehe {len(faces)} Gesicht(er)."
			if people:
				return f"Ich sehe {len(people)} Person(en), aber keine klaren Gesichter."
			return "Ich sehe keine Personen oder Gesichter."

		return None

	def _parse_decision_json(self, raw: str) -> AdvisorDecision | None:
		"""Parse a brain JSON response.

		Supported shapes:
		- v1: {"response_text": str, "need_observe": bool}
		- v2: {"response_text": str, "need_observe": bool, "actions": [ {"type": ..., ...}, ... ]}
		"""
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
			return None
		if not isinstance(obj, dict):
			return None
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

		actions_raw = obj.get("actions")
		actions: list[dict[str, Any]] = []
		if isinstance(actions_raw, list):
			for item in actions_raw:
				if isinstance(item, dict):
					actions.append({str(k): v for k, v in item.items()})
		# Cap to avoid runaway behavior (increased from 3 to 10 for multi-step tasks).
		actions = actions[:10]
		return AdvisorDecision(response_text=(resp_s if resp_s else None), need_observe=need_b, actions=actions)

	async def _execute_planned_actions(
		self,
		actions: list[dict[str, Any]],
		*,
		await_completion: bool = False,
		allow_interruption: bool = True,
	) -> list[dict[str, Any]]:
		"""Execute a small, whitelisted set of actions suggested by the brain.

		This is intentionally conservative: unknown actions are ignored.

		Args:
			actions: List of action dicts with 'type' and parameters.
			await_completion: If True, wait for time-based actions (like drive) to complete
				before returning. This is essential for sequential task execution.
			allow_interruption: If True and there are multiple actions, allow human to
				interrupt between actions with brief pauses.
		"""
		results: list[dict[str, Any]] = []
		if not actions:
			return results

		lang = self.settings.response_language
		is_german = str(lang).lower().startswith("de")
		total_actions = min(len(actions), 10)  # Max 10 actions

		for idx, a in enumerate(actions[:10]):
			atype = str(a.get("type") or a.get("action") or "").strip().lower()
			if not atype:
				continue

			res: dict[str, Any] = {"type": atype, "ok": True}
			try:
				# --- Head ---
				if atype in {"head_center", "center_head", "look_center"}:
					await self._head_center()
				elif atype in {"head_set_angles", "set_head_angles", "look"}:
					pan = a.get("pan_deg")
					tilt = a.get("tilt_deg")
					pan_i = int(pan) if pan is not None else None
					tilt_i = int(tilt) if tilt is not None else None
					if pan_i is not None:
						pan_i = max(-90, min(90, pan_i))
					if tilt_i is not None:
						tilt_i = max(-35, min(35, tilt_i))
					await self._head_set_angles(pan_deg=pan_i, tilt_deg=tilt_i)
				elif atype in {"head_scan", "scan", "look_around"}:
					pattern = str(a.get("pattern") or "sweep")
					duration_s = float(a.get("duration_s") or 3.0)
					duration_s = max(0.5, min(10.0, duration_s))
					if self._dry_run:
						self._emit(
							"tool_call",
							tool="scan",
							component="mcp.head",
							dry_run=True,
							pattern=pattern,
							duration_s=duration_s,
						)
						# Simulate wait in dry-run mode if await_completion
						if await_completion:
							await asyncio.sleep(duration_s)
					else:
						await call_mcp_tool_json(
							url=self.settings.head_mcp_url,
							tool_name="scan",
							timeout_seconds=5.0,
							pattern=pattern,
							duration_s=duration_s,
						)
						# Head scan is typically blocking on the service side,
						# but wait anyway if requested
						if await_completion:
							await asyncio.sleep(duration_s)

				# --- Safety / Motion ---
				elif atype in {"stop", "safety_stop", "halt"}:
					await self._safety_stop()
				elif atype in {"estop_on", "e_stop", "emergency_stop"}:
					await self._speaker_stop()
					await self._safety_estop_on()
					await self._safety_stop()
				elif atype in {"estop_off", "release_estop"}:
					await self._safety_estop_off()
				elif atype in {"guarded_drive", "drive", "move"}:
					speed = int(a.get("speed") if a.get("speed") is not None else a.get("speed_pct") or 25)
					steer = int(a.get("steer_deg") if a.get("steer_deg") is not None else 0)
					duration_s = float(a.get("duration_s") or 0.7)
					threshold_cm = float(a.get("threshold_cm") or 35.0)
					speed = max(-100, min(100, speed))
					steer = max(-45, min(45, steer))
					duration_s = max(0.1, min(10.0, duration_s))
					threshold_cm = max(5.0, min(150.0, threshold_cm))
					drive_res = await self._safety_guarded_drive(
						speed=speed,
						steer_deg=steer,
						duration_s=duration_s,
						threshold_cm=threshold_cm,
						await_completion=True,  # Wait for motion to complete before next action
					)
					res["drive_result"] = drive_res

				# --- Memory Actions (LLM-triggered) ---
				elif atype == "memory_recall":
					query = str(a.get("query") or "").strip()
					if query:
						self._emit("memory_action_start", component="advisor.memory", action="recall", query=query)
						recall_result = await self._memorizer_recall_direct(query)
						res["recall_result"] = recall_result
						self._emit("memory_action_end", component="advisor.memory", action="recall", result_preview=str(recall_result)[:200] if recall_result else None)
					else:
						res["ok"] = False
						res["reason"] = "empty_query"

				elif atype == "memory_store":
					content = str(a.get("content") or "").strip()
					tags = a.get("tags") or []
					if isinstance(tags, str):
						tags = [t.strip() for t in tags.split(",") if t.strip()]
					if content:
						self._emit("memory_action_start", component="advisor.memory", action="store", content_preview=content[:100])
						store_result = await self._memorizer_store_direct(content, tags)
						res["store_result"] = store_result
						self._emit("memory_action_end", component="advisor.memory", action="store", ok=store_result)
					else:
						res["ok"] = False
						res["reason"] = "empty_content"

				# Unknown action type -> ignore (do not fail the interaction).
				else:
					res["ok"] = False
					res["ignored"] = True
					res["reason"] = "unknown_action"
			except Exception as exc:
				res["ok"] = False
				res["error"] = str(exc)
			results.append(res)
			
			# Check for human interruption between actions (if multiple actions and allowed)
			if allow_interruption and (idx + 1) < total_actions and res.get("ok", False):
				# Brief pause to allow human to interrupt
				interrupt_prompt = (
					"Soll ich weitermachen?"
					if is_german
					else "Should I continue?"
				)
				human_interrupt = await self._interruptible_pause(
					interrupt_prompt,
					initial_timeout_seconds=1.0,
					context="between_actions",
				)
				if human_interrupt:
					# Human interrupted - stop execution and record the interrupt
					self._emit(
						"action_execution_interrupted",
						component="advisor.actions",
						completed_actions=idx + 1,
						total_actions=total_actions,
						human_input=human_interrupt[:100],
					)
					self._ledger.append(f"Human (interrupt): {human_interrupt}")
					# Add marker to results so caller knows we were interrupted
					results.append({
						"type": "_interrupted",
						"human_input": human_interrupt,
						"completed": idx + 1,
						"total": total_actions,
					})
					break
					
		return results

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

	async def _listen_briefly(self, timeout_seconds: float | None = None) -> str:
		"""Listen for a short time to allow human interruption.

		Uses a shorter silence timeout to quickly return if no speech is detected.
		Returns the transcript if speech was detected, or empty string otherwise.
		"""
		if timeout_seconds is None:
			timeout_seconds = float(self.settings.brief_listen_timeout_seconds or 3.0)
		timeout_seconds = max(0.5, min(10.0, timeout_seconds))

		if self._dry_run:
			self._emit("tool_call", tool="listen_briefly", component="mcp.listen", dry_run=True, timeout_seconds=timeout_seconds)
			return ""
		start = time.perf_counter()
		self._emit(
			"tool_call_start",
			tool="listen_briefly",
			component="mcp.listen",
			url=self.settings.listen_mcp_url,
			timeout_seconds=timeout_seconds,
		)
		payload: dict[str, Any] = {"speech_pause_seconds": timeout_seconds}
		if self.settings.listen_stream:
			payload["stream"] = True
		try:
			res = await call_mcp_tool_json(
				url=self.settings.listen_mcp_url,
				tool_name="listen",
				timeout_seconds=timeout_seconds + 10.0,  # Extra margin for network
				**payload,
			)
		except Exception as exc:
			dur_ms = (time.perf_counter() - start) * 1000.0
			self._emit(
				"tool_call_error",
				tool="listen_briefly",
				component="mcp.listen",
				url=self.settings.listen_mcp_url,
				duration_ms=round(dur_ms, 2),
				error=str(exc),
			)
			# Don't raise - brief listen failures should not crash the flow
			return ""
		dur_ms = (time.perf_counter() - start) * 1000.0
		self._emit(
			"tool_call_end",
			tool="listen_briefly",
			component="mcp.listen",
			url=self.settings.listen_mcp_url,
			duration_ms=round(dur_ms, 2),
			result_keys=sorted([str(k) for k in res.keys()]) if isinstance(res, dict) else None,
		)
		text = str(res.get("text") or "")
		return text.strip()

	async def _interruptible_pause(
		self,
		prompt: str,
		*,
		initial_timeout_seconds: float = 2.0,
		context: str = "pause",
	) -> str | None:
		"""Speak a prompt and listen briefly for human input.

		This allows the human to interrupt or provide additional input at key moments.
		If the human starts speaking during the initial wait, we acknowledge them
		and switch to full listening mode (like a normal conversation).

		Args:
			prompt: What to say (e.g., "Say something or I move on")
			initial_timeout_seconds: How long to wait initially for sound (default 2s)
			context: Label for logging

		Returns:
			The transcript if meaningful speech was detected, None otherwise.
		"""
		lang = self.settings.response_language
		is_german = str(lang).lower().startswith("de")

		self._emit("interruptible_pause_start", component="advisor", context=context, prompt=prompt[:50])

		# Speak the prompt
		await self._speak(prompt)
		self._ledger.append(f"Assistant: {prompt}")

		# Wait and check for sound activity during the pause window
		# Poll multiple times since detect_sound_activity has max 1s window
		sound_detected = False
		if self.settings.sound_enabled and not self._dry_run:
			poll_window = min(self.settings.sound_window_seconds, 0.5)  # Short poll windows
			elapsed = 0.0
			
			while elapsed < initial_timeout_seconds and not sound_detected:
				try:
					# Run sound detection in thread pool since it's blocking
					loop = asyncio.get_running_loop()
					result = await loop.run_in_executor(
						None,
						lambda: detect_sound_activity(
							threshold_rms=self.settings.sound_threshold_rms,
							sample_rate_hz=self.settings.sound_sample_rate_hz,
							window_seconds=poll_window,
							arecord_device=self.settings.sound_arecord_device,
						),
					)
					if result.active:
						sound_detected = True
						self._emit(
							"interruptible_pause_sound_detected",
							component="advisor",
							context=context,
							rms=result.rms,
							backend=result.backend,
							elapsed_seconds=round(elapsed, 2),
						)
						break
				except Exception as exc:
					self._emit(
						"interruptible_pause_sound_error",
						component="advisor",
						context=context,
						error=str(exc),
					)
					break
				
				elapsed += poll_window
				# Small gap between polls
				if elapsed < initial_timeout_seconds:
					await asyncio.sleep(0.05)

		self._emit(
			"interruptible_pause_sound_check",
			component="advisor",
			context=context,
			sound_detected=sound_detected,
		)

		if not sound_detected:
			# No sound - user didn't want to interrupt, continue with task
			self._emit("interruptible_pause_timeout", component="advisor", context=context)
			return None

		# Sound detected! User wants to say something.
		# Acknowledge and switch to full listening mode
		ack_prompt = "Ok, was möchtest du?" if is_german else "Ok, what's your request?"
		self._emit(
			"interruptible_pause_acknowledged",
			component="advisor",
			context=context,
			ack_prompt=ack_prompt,
		)
		await self._speak(ack_prompt)
		self._ledger.append(f"Assistant: {ack_prompt}")

		# Now listen fully like a normal conversation (up to 3 minutes)
		self._emit("interruptible_pause_full_listen", component="advisor", context=context)
		try:
			transcript = await self._listen_once()
		except Exception as exc:
			self._emit(
				"interruptible_pause_listen_error",
				component="advisor",
				context=context,
				error=str(exc),
			)
			return None

		# Check if we got meaningful input
		if transcript and not self._is_bad_transcript(transcript):
			self._emit(
				"interruptible_pause_input",
				component="advisor",
				context=context,
				chars=len(transcript),
				preview=transcript[:100],
			)
			return transcript

		self._emit("interruptible_pause_no_input", component="advisor", context=context)
		return None

	async def _silent_interrupt_check(
		self,
		timeout_seconds: float = 1.5,
		context: str = "silent_check",
	) -> str | None:
		"""Silently check for sound and listen if detected, WITHOUT speaking any prompt.

		This allows the human to interrupt at key moments without the robot asking
		"Do you want to say something?" every time - which is annoying.

		Args:
			timeout_seconds: How long to check for sound activity
			context: Label for logging

		Returns:
			The transcript if meaningful speech was detected, None otherwise.
		"""
		self._emit("silent_interrupt_check_start", component="advisor", context=context)

		# Check for sound activity without speaking
		sound_detected = False
		if self.settings.sound_enabled and not self._dry_run:
			poll_window = min(self.settings.sound_window_seconds, 0.5)
			elapsed = 0.0
			
			while elapsed < timeout_seconds and not sound_detected:
				try:
					loop = asyncio.get_running_loop()
					result = await loop.run_in_executor(
						None,
						lambda: detect_sound_activity(
							threshold_rms=self.settings.sound_threshold_rms,
							sample_rate_hz=self.settings.sound_sample_rate_hz,
							window_seconds=poll_window,
							arecord_device=self.settings.sound_arecord_device,
						),
					)
					if result.active:
						sound_detected = True
						self._emit(
							"silent_interrupt_sound_detected",
							component="advisor",
							context=context,
							rms=result.rms,
							backend=result.backend,
							elapsed_seconds=round(elapsed, 2),
						)
						break
				except Exception as exc:
					self._emit(
						"silent_interrupt_sound_error",
						component="advisor",
						context=context,
						error=str(exc),
					)
					break
				
				elapsed += poll_window
				if elapsed < timeout_seconds:
					await asyncio.sleep(0.05)

		if not sound_detected:
			self._emit("silent_interrupt_no_sound", component="advisor", context=context)
			return None

		# Sound detected - listen without asking "what do you want?"
		# Just switch to listening mode silently
		self._emit("silent_interrupt_listening", component="advisor", context=context)
		try:
			transcript = await self._listen_once()
		except Exception as exc:
			self._emit(
				"silent_interrupt_listen_error",
				component="advisor",
				context=context,
				error=str(exc),
			)
			return None

		if transcript and not self._is_bad_transcript(transcript):
			self._emit(
				"silent_interrupt_input",
				component="advisor",
				context=context,
				chars=len(transcript),
				preview=transcript[:100],
			)
			return transcript

		self._emit("silent_interrupt_no_input", component="advisor", context=context)
		return None

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

	# ========== STREAMING TTS METHODS ==========
	# These methods enable low-latency streaming TTS by sending text chunks
	# as they are generated by the LLM, rather than waiting for the full response.

	async def _speak_stream_start(self) -> str | None:
		"""Start a streaming TTS session. Returns session_id or None on failure."""
		if self._dry_run:
			self._emit("tool_call", tool="stream_start", component="mcp.speak", dry_run=True)
			return "dry_run_session"
		
		# Use HTTP directly since MCP may not expose the streaming endpoints
		speak_url = self.settings.speak_mcp_url
		# Convert MCP URL to HTTP API URL (e.g., http://127.0.0.1:8601/mcp -> http://127.0.0.1:8001)
		import urllib.parse
		parsed = urllib.parse.urlparse(speak_url)
		# MCP port is typically API port + 600
		api_port = parsed.port - 600 if parsed.port else 8001
		api_url = f"{parsed.scheme}://{parsed.hostname}:{api_port}/stream/start"
		
		start = time.perf_counter()
		self._emit("tool_call_start", tool="stream_start", component="mcp.speak", url=api_url)
		
		try:
			import aiohttp
			async with aiohttp.ClientSession() as session:
				async with session.post(api_url, timeout=aiohttp.ClientTimeout(total=10.0)) as resp:
					data = await resp.json()
					if not data.get("ok"):
						raise RuntimeError(data.get("detail", "stream_start failed"))
					session_id = data.get("session_id")
		except Exception as exc:
			dur_ms = (time.perf_counter() - start) * 1000.0
			self._emit(
				"tool_call_error",
				tool="stream_start",
				component="mcp.speak",
				url=api_url,
				duration_ms=round(dur_ms, 2),
				error=str(exc),
			)
			return None
		
		dur_ms = (time.perf_counter() - start) * 1000.0
		self._emit(
			"tool_call_end",
			tool="stream_start",
			component="mcp.speak",
			url=api_url,
			duration_ms=round(dur_ms, 2),
			session_id=session_id,
		)
		return session_id

	async def _speak_stream_chunk(self, session_id: str, text: str, http_session: Any = None) -> bool:
		"""Send a text chunk to a streaming TTS session. Returns True on success.
		
		Args:
			session_id: The TTS streaming session ID
			text: Text chunk to send
			http_session: Optional aiohttp.ClientSession to reuse (faster)
		"""
		if not session_id or not text:
			return False
		
		if self._dry_run:
			self._emit("tool_call", tool="stream_chunk", component="mcp.speak", dry_run=True, text=text[:50])
			return True
		
		speak_url = self.settings.speak_mcp_url
		import urllib.parse
		parsed = urllib.parse.urlparse(speak_url)
		api_port = parsed.port - 600 if parsed.port else 8001
		api_url = f"{parsed.scheme}://{parsed.hostname}:{api_port}/stream/chunk"
		
		try:
			import aiohttp
			
			async def do_request(session: aiohttp.ClientSession) -> bool:
				async with session.post(
					api_url,
					json={"session_id": session_id, "text": text},
					timeout=aiohttp.ClientTimeout(total=5.0),
				) as resp:
					data = await resp.json()
					return bool(data.get("ok"))
			
			if http_session is not None:
				return await do_request(http_session)
			else:
				async with aiohttp.ClientSession() as session:
					return await do_request(session)
		except Exception as exc:
			self._emit(
				"tool_call_error",
				tool="stream_chunk",
				component="mcp.speak",
				error=str(exc),
			)
			return False

	async def _speak_stream_end(self, session_id: str) -> bool:
		"""End a streaming TTS session. Returns True on success."""
		if not session_id:
			return False
		
		if self._dry_run:
			self._emit("tool_call", tool="stream_end", component="mcp.speak", dry_run=True)
			return True
		
		speak_url = self.settings.speak_mcp_url
		import urllib.parse
		parsed = urllib.parse.urlparse(speak_url)
		api_port = parsed.port - 600 if parsed.port else 8001
		api_url = f"{parsed.scheme}://{parsed.hostname}:{api_port}/stream/end"
		
		start = time.perf_counter()
		self._emit("tool_call_start", tool="stream_end", component="mcp.speak", url=api_url, session_id=session_id)
		
		try:
			import aiohttp
			async with aiohttp.ClientSession() as session:
				async with session.post(
					api_url,
					json={"session_id": session_id},
					timeout=aiohttp.ClientTimeout(total=10.0),
				) as resp:
					data = await resp.json()
					success = bool(data.get("ok"))
		except Exception as exc:
			dur_ms = (time.perf_counter() - start) * 1000.0
			self._emit(
				"tool_call_error",
				tool="stream_end",
				component="mcp.speak",
				url=api_url,
				duration_ms=round(dur_ms, 2),
				error=str(exc),
			)
			return False
		
		dur_ms = (time.perf_counter() - start) * 1000.0
		self._emit(
			"tool_call_end",
			tool="stream_end",
			component="mcp.speak",
			url=api_url,
			duration_ms=round(dur_ms, 2),
			success=success,
		)
		return success

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

	async def _observe_direction(self, question: str | None = None) -> dict[str, Any]:
		"""Ask the observe service to suggest a movement direction."""
		if self._dry_run:
			q = (question or "").strip() or "Where should the robot move next?"
			self._emit("tool_call", tool="observe_direction", component="mcp.observe", dry_run=True, question=q)
			return {"cell": {"row": 1, "col": 1}, "action": "forward", "why": "dry_run", "fit": "dry_run"}

		q = (question or "").strip() or "Where should the robot move next to approach the most interesting object?"
		start = time.perf_counter()
		self._emit(
			"tool_call_start",
			tool="observe_direction",
			component="mcp.observe",
			url=self.settings.observe_mcp_url,
			question=q,
		)
		try:
			res = await call_mcp_tool_json(
				url=self.settings.observe_mcp_url,
				tool_name="observe_direction",
				timeout_seconds=120.0,
				question=q,
			)
		except Exception as exc:
			dur_ms = (time.perf_counter() - start) * 1000.0
			self._emit(
				"tool_call_error",
				tool="observe_direction",
				component="mcp.observe",
				url=self.settings.observe_mcp_url,
				duration_ms=round(dur_ms, 2),
				error=str(exc),
			)
			return {"ok": False, "error": str(exc)}

		dur_ms = (time.perf_counter() - start) * 1000.0
		self._emit(
			"tool_call_end",
			tool="observe_direction",
			component="mcp.observe",
			url=self.settings.observe_mcp_url,
			duration_ms=round(dur_ms, 2),
			result_keys=sorted([str(k) for k in res.keys()]) if isinstance(res, dict) else None,
		)
		return res if isinstance(res, dict) else {"ok": True, "value": res}

	async def _perception_detect(self) -> dict[str, Any] | None:
		"""Best-effort perception detection call."""
		if self._dry_run:
			return {"ok": True, "available": True, "dry_run": True, "faces": [], "people": []}
		start = time.perf_counter()
		self._emit("tool_call_start", tool="detect", component="mcp.perception", url=self.settings.perception_mcp_url)
		try:
			res = await call_mcp_tool_json(url=self.settings.perception_mcp_url, tool_name="detect", timeout_seconds=8.0)
		except Exception as exc:
			dur_ms = (time.perf_counter() - start) * 1000.0
			self._emit(
				"tool_call_error",
				tool="detect",
				component="mcp.perception",
				url=self.settings.perception_mcp_url,
				duration_ms=round(dur_ms, 2),
				error=str(exc),
			)
			return None
		dur_ms = (time.perf_counter() - start) * 1000.0
		self._emit(
			"tool_call_end",
			tool="detect",
			component="mcp.perception",
			url=self.settings.perception_mcp_url,
			duration_ms=round(dur_ms, 2),
			result_keys=sorted([str(k) for k in res.keys()]) if isinstance(res, dict) else None,
		)
		return res if isinstance(res, dict) else {"ok": True, "value": res}

	@staticmethod
	def _drive_plan_from_observe_action(action: str | None, *, near_s: float, far_s: float) -> tuple[int, float]:
		"""Map observe_direction 'action' -> (steer_deg, duration_s)."""
		a = (action or "").strip().lower()
		dur = float(far_s) if "far" in a else float(near_s)
		steer = 0
		if "left" in a:
			steer = -25
		elif "right" in a:
			steer = 25
		return (int(steer), max(0.1, float(dur)))

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

	async def _ask_brain_simple(self, prompt: str) -> str | None:
		"""Ask the brain LLM a simple question and get a plain text response.
		
		This is used for things like formatting memory results into natural speech.
		No JSON parsing, no action handling - just text in, text out.
		"""
		if self._dry_run:
			return f"(dry_run) {prompt[:50]}"
		
		try:
			brain = await self._ensure_brain()
			thread = brain.get_new_thread()
			response = await brain._agent.run(prompt, thread=thread)
			
			# Extract text from response
			if hasattr(response, "text"):
				return str(response.text).strip()
			elif hasattr(response, "content"):
				return str(response.content).strip()
			else:
				return str(response).strip()
		except Exception as exc:
			self._emit("brain_simple_error", component="advisor.brain", error=str(exc))
			return None

	async def _decide_and_respond(
		self,
		*,
		human_text: str,
		observation: str | None,
		memory_hint: str | None,
	) -> tuple[str, bool | None, list[dict[str, Any]]]:
		if self._dry_run:
			# Minimal, deterministic behavior for test runs.
			if observation and observation.strip():
				return (f"(dry_run) Based on what I see: {observation.strip()}", None, [])
			return (f"(dry_run) I heard: {human_text.strip()}", None, [])

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

		todo_block = self._todo_status_block()
		todo_policy = self._todo_policy_block()

		# Ask for a JSON decision so we can optionally trigger observation and/or
		# execute a small set of safe robot actions.
		lang = self.settings.response_language
		few_shots = (
			"Examples (decision JSON only):\n"
			"Human: Was siehst du?\n"
			"Assistant: {\"response_text\": \"Einen Moment, ich schaue nach.\", \"need_observe\": true, \"actions\": []}\n\n"
			"Human: Schau mich an.\n"
			"Assistant: {\"response_text\": \"Okay, ich schaue dich an.\", \"need_observe\": true, \"actions\": [{\"type\":\"head_center\"}]}\n\n"
			"Human: Schau nach oben.\n"
			"Assistant: {\"response_text\": \"Okay, ich schaue nach oben.\", \"need_observe\": false, \"actions\": [{\"type\":\"head_set_angles\", \"tilt_deg\": 15}]}\n\n"
			"Human: Guck mal hoch bitte.\n"
			"Assistant: {\"response_text\": \"Okay.\", \"need_observe\": false, \"actions\": [{\"type\":\"head_set_angles\", \"tilt_deg\": 15}]}\n\n"
			"Human: Look down.\n"
			"Assistant: {\"response_text\": \"Okay, looking down.\", \"need_observe\": false, \"actions\": [{\"type\":\"head_set_angles\", \"tilt_deg\": -15}]}\n\n"
			"Human: Schau nach links.\n"
			"Assistant: {\"response_text\": \"Okay, ich schaue nach links.\", \"need_observe\": false, \"actions\": [{\"type\":\"head_set_angles\", \"pan_deg\": -30}]}\n\n"
			"Human: Turn your head to the right.\n"
			"Assistant: {\"response_text\": \"Okay.\", \"need_observe\": false, \"actions\": [{\"type\":\"head_set_angles\", \"pan_deg\": 30}]}\n\n"
			"Human: Schau dich mal um.\n"
			"Assistant: {\"response_text\": \"Okay, ich schaue mich um.\", \"need_observe\": false, \"actions\": [{\"type\":\"head_scan\", \"pattern\": \"sweep\", \"duration_s\": 3.0}]}\n\n"
			"Human: Fahr drei Sekunden nach links.\n"
			"Assistant: {\"response_text\": \"Okay, ich fahre drei Sekunden nach links.\", \"need_observe\": false, \"actions\": [{\"type\":\"guarded_drive\", \"speed\": 25, \"steer_deg\": -25, \"duration_s\": 3.0}]}\n\n"
			"Human: Drive forward for 5 seconds.\n"
			"Assistant: {\"response_text\": \"Okay, driving forward for 5 seconds.\", \"need_observe\": false, \"actions\": [{\"type\":\"guarded_drive\", \"speed\": 30, \"steer_deg\": 0, \"duration_s\": 5.0}]}\n\n"
			"Human: Fahr mal kurz vorwärts.\n"
			"Assistant: {\"response_text\": \"Okay.\", \"need_observe\": false, \"actions\": [{\"type\":\"guarded_drive\", \"speed\": 25, \"steer_deg\": 0, \"duration_s\": 0.7}]}\n\n"
			"Human: Was ist vor dir?\n"
			"Assistant: {\"response_text\": \"Einen Moment, ich beschreibe, was vor mir ist.\", \"need_observe\": true, \"actions\": []}\n\n"
			"Human: Wie spät ist es?\n"
			"Assistant: {\"response_text\": \"Ich kann dir helfen, aber ich habe keine Uhrzeit-Sensorik.\", \"need_observe\": false, \"actions\": []}\n\n"
			# Memory examples - LLM decides when to use memory
			"Human: Wer ist Sebastian?\n"
			"Assistant: {\"response_text\": \"Moment, ich schaue in meinem Gedächtnis nach.\", \"need_observe\": false, \"actions\": [{\"type\":\"memory_recall\", \"query\": \"Sebastian\"}]}\n\n"
			"Human: Erinnerst du dich an Katharina?\n"
			"Assistant: {\"response_text\": \"Lass mich nachdenken...\", \"need_observe\": false, \"actions\": [{\"type\":\"memory_recall\", \"query\": \"Katharina\"}]}\n\n"
			"Human: Suche in deiner Erinnerung nach meiner Familie.\n"
			"Assistant: {\"response_text\": \"Ich schaue mal, was ich über deine Familie weiß.\", \"need_observe\": false, \"actions\": [{\"type\":\"memory_recall\", \"query\": \"Familie\"}]}\n\n"
			"Human: Merk dir, dass ich Pizza mag.\n"
			"Assistant: {\"response_text\": \"Okay, ich merke mir, dass du Pizza magst.\", \"need_observe\": false, \"actions\": [{\"type\":\"memory_store\", \"content\": \"Der Benutzer mag Pizza.\", \"tags\": [\"vorlieben\", \"essen\"]}]}\n\n"
			"Human: Was weißt du über mich?\n"
			"Assistant: {\"response_text\": \"Lass mich schauen, was ich über dich weiß.\", \"need_observe\": false, \"actions\": [{\"type\":\"memory_recall\", \"query\": \"Benutzer Informationen Vorlieben\"}]}\n"
		)
		prompt = (
			"You are responding to a human speaking to the robot.\n"
			"The robot MUST speak in language: "
			+ str(lang)
			+ ".\n"
			"You may think in English, but the returned response_text MUST be in that language.\n"
			"Return ONLY a JSON object with keys:\n"
			"- response_text: string (what the robot should say out loud)\n"
			"- need_observe: boolean (true if you need camera info to answer well)\n"
			"- actions: array of small robot actions (optional; may be empty)\n\n"
			"Allowed action types (use EXACT type strings):\n"
			"- head_center\n"
			"- head_set_angles (pan_deg?: int, tilt_deg?: int)\n"
			"- head_scan (pattern?: string, duration_s?: number)\n"
			"- guarded_drive (speed: int -100..100, steer_deg?: int, duration_s: number 0.1-10.0, threshold_cm?: number)\n"
			"  IMPORTANT: When the user specifies a time duration (e.g. 'drive 3 seconds left'), you MUST include duration_s in the action!\n"
			"- safety_stop\n"
			"- estop_on\n"
			"- estop_off\n"
			"- memory_recall (query: string) - Search robot's memory for information about a person, topic, or fact\n"
			"- memory_store (content: string, tags?: string[]) - Store important information in robot's memory\n"
			"\nIMPORTANT: Use memory_recall whenever the user asks about:\n"
			"- People (names, family, friends)\n"
			"- Past conversations or things they told you\n"
			"- User preferences or personal information\n"
			"- Anything that requires remembering previous interactions\n\n"
			+ few_shots
			+ "\n"
			+ ctx_block
			+ f"Human said: {human_text.strip()}"
			+ obs_block
			+ mem_block
			+ todo_block
			+ todo_policy
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
		decision = self._parse_decision_json(raw)
		self._emit(
			"decision",
			component="advisor.brain",
			need_observe=(decision.need_observe if decision else None),
			has_observation=bool(observation and observation.strip()),
			response_preview=(decision.response_text if decision and decision.response_text else raw),
		)
		if decision is not None and decision.response_text is not None:
			return (decision.response_text, decision.need_observe, decision.actions)

		# Fallback: treat model output as plain speech.
		return (raw, None, [])

	async def _decide_and_respond_streaming(
		self,
		*,
		human_text: str,
		observation: str | None,
		memory_hint: str | None,
	) -> tuple[str, bool | None, list[dict[str, Any]]]:
		"""Streaming version of _decide_and_respond that speaks text as it arrives.
		
		This method streams the LLM response and sends text chunks to TTS as they arrive,
		significantly reducing the time to first audio. The full response is still accumulated
		for parsing actions and need_observe.
		
		Returns the same tuple as _decide_and_respond: (response_text, need_observe, actions)
		"""
		if self._dry_run:
			# Minimal, deterministic behavior for test runs.
			if observation and observation.strip():
				return (f"(dry_run) Based on what I see: {observation.strip()}", None, [])
			return (f"(dry_run) I heard: {human_text.strip()}", None, [])

		brain = await self._ensure_brain()

		obs_block = ""
		if observation and observation.strip():
			obs_block = f"\n\nObservation (camera):\n{observation.strip()}\n"

		mem_block = ""
		if memory_hint and memory_hint.strip():
			mem_block = "\n\nMemory report (from memorizer agent):\n" + memory_hint.strip() + "\n"

		mem = self.settings.memory
		max_chars_budget = int(mem.max_tokens) * max(1, int(mem.avg_chars_per_token))
		ctx = self._conversation_context(max_chars=min(8000, max(1200, max_chars_budget // 10)))
		ctx_block = ""
		if ctx:
			ctx_block = "\n\nConversation so far (most recent last):\n" + ctx + "\n"

		todo_block = self._todo_status_block()
		todo_policy = self._todo_policy_block()

		# Modified prompt that asks for speech text FIRST, then JSON metadata
		# This allows us to stream the speech text while the model is still generating
		lang = self.settings.response_language
		prompt = (
			"You are responding to a human speaking to the robot.\n"
			"The robot MUST speak in language: "
			+ str(lang)
			+ ".\n\n"
			"IMPORTANT OUTPUT FORMAT FOR STREAMING:\n"
			"1. First, output the text the robot should say (response_text)\n"
			"2. Then output a separator line: ---JSON---\n"
			"3. Then output a JSON object with: {\"need_observe\": boolean, \"actions\": array}\n\n"
			"Example output:\n"
			"Hallo! Ich schaue mich mal um.\n"
			"---JSON---\n"
			"{\"need_observe\": false, \"actions\": [{\"type\": \"head_scan\", \"pattern\": \"sweep\"}]}\n\n"
			"Allowed action types:\n"
			"- head_center, head_set_angles (pan_deg, tilt_deg), head_scan (pattern, duration_s)\n"
			"- guarded_drive (speed: -100..100, steer_deg, duration_s: 0.1-10.0)\n"
			"- safety_stop, estop_on, estop_off\n"
			"- memory_recall (query: string), memory_store (content: string, tags?: string[])\n\n"
			+ ctx_block
			+ f"Human said: {human_text.strip()}"
			+ obs_block
			+ mem_block
			+ todo_block
			+ todo_policy
		)

		start = time.perf_counter()
		self._emit(
			"brain_call_start",
			component="advisor.brain",
			kind="decide_and_respond_streaming",
			has_observation=bool(observation and observation.strip()),
		)

		# Start streaming TTS session
		session_id = await self._speak_stream_start()
		if not session_id:
			# Fallback to non-streaming if TTS streaming unavailable
			self._emit("streaming_fallback", component="advisor", reason="tts_session_failed")
			return await self._decide_and_respond(
				human_text=human_text,
				observation=observation,
				memory_hint=memory_hint,
			)

		full_response = ""
		speech_text = ""
		json_part = ""
		separator_seen = False
		pending_speech_chunk = ""  # Buffer to reduce HTTP calls
		chunk_send_threshold = 40  # Send to TTS every ~40 chars (faster first response)
		
		# Create a persistent HTTP session for all chunk requests (much faster)
		import aiohttp
		async with aiohttp.ClientSession() as http_session:
			try:
				async for update in brain.run_stream(prompt):
					# Extract text from the update
					chunk_text = ""
					if hasattr(update, "contents"):
						for content in update.contents:
							if hasattr(content, "text") and content.text:
								chunk_text += str(content.text)
					elif hasattr(update, "text"):
						chunk_text = str(update.text)
					elif hasattr(update, "content"):
						chunk_text = str(update.content)
					
					if not chunk_text:
						continue
					
					full_response += chunk_text
					
					# Check for JSON separator
					if not separator_seen:
						# Look for the separator in accumulated text
						if "---JSON---" in full_response:
							separator_seen = True
							parts = full_response.split("---JSON---", 1)
							speech_text = parts[0].strip()
							json_part = parts[1] if len(parts) > 1 else ""
							
							# Send any remaining buffered speech text before separator
							if pending_speech_chunk.strip():
								await self._speak_stream_chunk(session_id, pending_speech_chunk, http_session)
								pending_speech_chunk = ""
						else:
							# Still in speech section - buffer chunks to reduce HTTP calls
							pending_speech_chunk += chunk_text
							speech_text = full_response.strip()
							
							# Send when buffer is big enough
							if len(pending_speech_chunk) >= chunk_send_threshold:
								await self._speak_stream_chunk(session_id, pending_speech_chunk, http_session)
								pending_speech_chunk = ""
					else:
						# In JSON section, accumulate for parsing
						json_part += chunk_text
				
				# Send any remaining buffered speech
				if pending_speech_chunk.strip():
					await self._speak_stream_chunk(session_id, pending_speech_chunk, http_session)
			
			except Exception as exc:
				self._emit(
					"brain_stream_error",
					component="advisor.brain",
					error=str(exc),
				)
				# End TTS session on error
				await self._speak_stream_end(session_id)
				raise

		# End TTS streaming session
		await self._speak_stream_end(session_id)

		dur_ms = (time.perf_counter() - start) * 1000.0
		self._emit(
			"brain_call_end",
			component="advisor.brain",
			kind="decide_and_respond_streaming",
			duration_ms=round(dur_ms, 2),
			speech_chars=len(speech_text),
			json_chars=len(json_part),
		)

		# Parse the JSON metadata
		need_observe: bool | None = None
		actions: list[dict[str, Any]] = []
		
		if json_part.strip():
			try:
				# Clean up JSON (remove markdown code blocks if present)
				json_clean = json_part.strip()
				if json_clean.startswith("```"):
					json_clean = re.sub(r"^```(?:json)?\s*", "", json_clean)
					json_clean = re.sub(r"\s*```$", "", json_clean)
				
				obj = json.loads(json_clean)
				if isinstance(obj, dict):
					need_observe = bool(obj.get("need_observe")) if "need_observe" in obj else None
					actions = list(obj.get("actions") or []) if "actions" in obj else []
			except Exception as parse_exc:
				self._emit(
					"json_parse_error",
					component="advisor.brain",
					error=str(parse_exc),
					json_preview=json_part[:200],
				)

		# If no separator was found, try to parse as original JSON format
		if not separator_seen:
			decision = self._parse_decision_json(full_response)
			if decision is not None and decision.response_text is not None:
				return (decision.response_text, decision.need_observe, decision.actions)
			# Fallback: treat entire output as speech
			speech_text = full_response.strip()

		self._emit(
			"decision",
			component="advisor.brain",
			need_observe=need_observe,
			has_observation=bool(observation and observation.strip()),
			response_preview=speech_text[:100] if speech_text else "(empty)",
			streaming=True,
		)

		return (speech_text or full_response.strip(), need_observe, actions)

	def _is_multi_step_request(self, text: str) -> bool:
		"""Check if a user request might need multi-step task planning."""
		# Debug: log entry point
		llm_todo_available = self._llm_todo is not None
		llm_todo_enabled = self._llm_todo.is_enabled() if llm_todo_available else False
		self._emit("multi_step_check_entry", component="advisor", text_preview=text[:100] if text else "", llm_todo_available=llm_todo_available, llm_todo_enabled=llm_todo_enabled)
		
		# Check if LLMTodoAgent is available first
		if self._llm_todo is not None and self._llm_todo.is_enabled():
			# Use heuristics to detect multi-step requests
			lower = (text or "").lower()
			words = lower.split()
			if len(words) < 6:
				self._emit("multi_step_check", component="advisor", result=False, reason="too_few_words", word_count=len(words))
				return False
			# Look for German/English sequence indicators
			multi_step_indicators = [
				" und dann ", " dann ", " danach ", " zuerst ", " erst ",
				" anschließend ", " nachdem ", " bevor ", " schließlich ",
				" and then ", " then ", " after ", " first ", " next ",
				" finally ", " before ", " sekunden ", " seconds ",
			]
			for indicator in multi_step_indicators:
				if indicator in lower:
					self._emit("multi_step_check", component="advisor", result=True, reason="indicator_found", indicator=indicator.strip())
					return True
			# Check for numbered lists or multiple action verbs
			action_verbs = ["schau", "look", "fahr", "drive", "dreh", "turn", "sag", "say", "beschreib", "describe", "erzähl", "tell"]
			verb_count = sum(1 for v in action_verbs if v in lower)
			if verb_count >= 2:
				self._emit("multi_step_check", component="advisor", result=True, reason="multiple_verbs", verb_count=verb_count)
				return True
			self._emit("multi_step_check", component="advisor", result=False, reason="no_indicators", verb_count=verb_count)
			return False

		# Fallback to TaskPlanner if available
		if self._task_planner is not None:
			result = self._task_planner.is_multi_step_request(text)
			self._emit("multi_step_check", component="advisor", result=result, reason="task_planner_fallback")
			return result

		self._emit("multi_step_check", component="advisor", result=False, reason="no_planner_available")
		return False

	async def _plan_and_execute_tasks(self, user_request: str) -> str | None:
		"""Plan tasks using LLMTodoAgent and execute them step-by-step.

		The flow is:
		1. Ask LLMTodoAgent to plan tasks from user request
		2. Loop: ask LLMTodoAgent "what's next?" -> execute ONE task -> wait -> mark done
		3. Repeat until no more tasks

		Returns the final response to speak, or None if planning failed.
		"""
		# Prefer LLMTodoAgent if available, fall back to old TaskPlanner
		if self._llm_todo is not None and self._llm_todo.is_enabled():
			return await self._plan_and_execute_tasks_llm(user_request)

		# Fallback to old TaskPlanner-based approach
		if self._task_planner is None or self._todo is None:
			return None

		return await self._plan_and_execute_tasks_legacy(user_request)

	async def _plan_and_execute_tasks_llm(self, user_request: str) -> str | None:
		"""Execute tasks using the LLM-based TodoAgent - step by step with waiting."""
		if self._llm_todo is None:
			return None

		self._emit("state", component="advisor", state="llm_task_planning_start")

		lang = self.settings.response_language
		is_german = str(lang).lower().startswith("de")

		# Step 1: Ask LLMTodoAgent to plan tasks
		try:
			tasks = await self._llm_todo.plan_tasks(user_request)
		except Exception as exc:
			self._emit("llm_task_planning_error", component="advisor.llm_todo", error=str(exc))
			return None

		if not tasks:
			self._emit("llm_task_planning_empty", component="advisor.llm_todo")
			return None

		self._emit(
			"llm_task_planning_done",
			component="advisor.llm_todo",
			task_count=len(tasks),
			tasks=[t.get("title") for t in tasks],
		)

		# Step 1b: Review the plan (optional) - LLM checks if plan is sufficient
		# NOTE: If not approved, we auto-modify WITHOUT asking the user to avoid annoying re-asks.
		# The LLM decides autonomously if the plan needs adjustments.
		if self.settings.todo.review_after_planning:
			review_result = await self._llm_todo.review_plan()
			self._emit(
				"llm_task_plan_review",
				component="advisor.llm_todo",
				approved=review_result.get("approved", True),
				suggestion=review_result.get("suggestion"),
				reason=review_result.get("reason"),
			)

			if not review_result.get("approved", True):
				suggestion = review_result.get("suggestion")
				if suggestion:
					# Auto-modify the plan based on the suggestion - NO user confirmation needed
					self._emit("llm_task_plan_modify", component="advisor.llm_todo", suggestion=suggestion)
					await self._llm_todo.modify_plan(suggestion)
					tasks = await self._llm_todo.get_next_task()  # Refresh tasks
					if tasks is None:
						tasks = []
					else:
						tasks = [tasks]  # Just get the list from the agent

		# Brief announcement - just say "OK" and get going (no detailed plan explanation)
		plan_announcement = (
			"OK, los geht's."
			if is_german
			else "OK, let's go."
		)
		await self._speak(plan_announcement)
		self._ledger.append(f"Assistant: {plan_announcement}")

		# Silent brief listen - allow human to interrupt before starting, but don't ask/prompt
		# This preserves the ability to interrupt without annoying "Say something or I start" prompts.
		self._emit("silent_interrupt_check_start", component="advisor", context="before_task_execution")
		human_input = await self._silent_interrupt_check(timeout_seconds=1.0)
		if human_input:
			# Human said something - check if they want to modify the plan
			low = human_input.lower()
			modify_cues = ("änder", "change", "add", "hinzufüg", "mehr", "more", "anders", "different", "warte", "wait", "stop", "nein", "no")
			if any(cue in low for cue in modify_cues):
				# Human wants to modify - let them add to the plan
				self._emit("plan_modification_requested", component="advisor", input=human_input[:100])
				modify_response = (
					"Okay, was soll ich ändern oder hinzufügen?"
					if is_german
					else "Okay, what should I change or add?"
				)
				await self._speak(modify_response)
				self._ledger.append(f"Assistant: {modify_response}")

				# Listen for their modification
				modification = await self._listen_briefly(timeout_seconds=5.0)
				if modification and not self._is_bad_transcript(modification):
					self._ledger.append(f"Human: {modification}")
					# Re-plan with the modification
					combined_request = f"{user_request}. Außerdem: {modification}" if is_german else f"{user_request}. Also: {modification}"
					await self._llm_todo.modify_plan(modification)
					self._emit("plan_modified", component="advisor", modification=modification[:100])

		self._emit("task_execution_starting", component="advisor", task_count=len(self._llm_todo._tasks))

		# Step 2: Execute tasks ONE BY ONE with waiting
		completed_count = 0

		while self._llm_todo.has_pending_tasks():
			# Ask LLMTodoAgent: "What's the next task?"
			next_task = await self._llm_todo.get_next_task()
			if next_task is None:
				break

			task_id = next_task.get("id")
			task_title = str(next_task.get("title") or "")
			action = next_task.get("action", {})

			self._emit(
				"llm_task_execution_start",
				component="advisor.llm_todo",
				task_id=task_id,
				task_title=task_title,
				action=action,
			)

			task_result: str | None = None

			try:
				# Execute the task based on its action type
				action_type = str(action.get("type") or "").lower()

				if action_type == "observe":
					# Just observe and describe
					question = str(action.get("question") or "Beschreibe was du siehst.")
					obs = await self._observe(question)
					if obs:
						await self._speak(obs)
						self._ledger.append(f"Assistant: {obs}")
					task_result = obs

				elif action_type in {"head_set_angles", "set_head_angles", "look"}:
					# Head movement with optional observe_after
					pan = action.get("pan_deg")
					tilt = action.get("tilt_deg")
					observe_after = bool(action.get("observe_after", False))

					# Execute head movement
					pan_i = int(pan) if pan is not None else None
					tilt_i = int(tilt) if tilt is not None else None
					if pan_i is not None:
						pan_i = max(-90, min(90, pan_i))
					if tilt_i is not None:
						tilt_i = max(-35, min(35, tilt_i))

					self._emit(
						"llm_task_head_move",
						component="advisor.llm_todo",
						task_id=task_id,
						pan_deg=pan_i,
						tilt_deg=tilt_i,
					)

					await self._head_set_angles(pan_deg=pan_i, tilt_deg=tilt_i)

					# WAIT for head to settle (important!)
					await asyncio.sleep(1.0)

					# If observe_after, take picture and describe
					if observe_after:
						question = str(action.get("observe_question") or action.get("question") or "Beschreibe kurz was du siehst.")
						self._emit(
							"llm_task_observe_after",
							component="advisor.llm_todo",
							task_id=task_id,
							question=question,
						)
						obs = await self._observe(question)
						if obs:
							await self._speak(obs)
							self._ledger.append(f"Assistant: {obs}")
						task_result = obs

				elif action_type in {"guarded_drive", "drive", "move"}:
					# Driving action - WAIT for completion
					speed = int(action.get("speed") or 25)
					steer_deg = int(action.get("steer_deg") or 0)
					duration_s = float(action.get("duration_s") or 1.0)

					speed = max(-100, min(100, speed))
					steer_deg = max(-45, min(45, steer_deg))
					duration_s = max(0.1, min(10.0, duration_s))

					self._emit(
						"llm_task_drive",
						component="advisor.llm_todo",
						task_id=task_id,
						speed=speed,
						steer_deg=steer_deg,
						duration_s=duration_s,
					)

					await self._safety_guarded_drive(
						speed=speed,
						steer_deg=steer_deg,
						duration_s=duration_s,
						threshold_cm=35.0,
						await_completion=True,  # WAIT for drive to complete
					)
					task_result = f"Gefahren: speed={speed}, steer={steer_deg}, {duration_s}s"

				elif action_type == "speak":
					# Just speak text
					text = str(action.get("text") or "")
					if text:
						await self._speak(text)
						self._ledger.append(f"Assistant: {text}")
					task_result = text

				elif action_type == "memory_recall":
					# Search memory database and let LLM formulate response
					query = str(action.get("query") or "")
					if query:
						self._emit(
							"llm_task_memory_recall",
							component="advisor.llm_todo",
							task_id=task_id,
							query=query,
						)
						recall_result = await self._memorizer_recall_direct(query)
						if recall_result:
							# Let the Brain LLM formulate a natural response based on memory
							llm_prompt = (
								f"Der Benutzer hat nach '{query}' gefragt. "
								f"Hier sind relevante Informationen aus deinem Gedächtnis:\n\n{recall_result}\n\n"
								f"Formuliere eine kurze, natürliche Antwort auf Deutsch basierend auf diesen Informationen. "
								f"Antworte direkt ohne Einleitung wie 'Basierend auf meinem Gedächtnis'."
							)
							natural_response = await self._ask_brain_simple(llm_prompt)
							if natural_response:
								await self._speak(natural_response)
								self._ledger.append(f"Assistant: {natural_response}")
								task_result = natural_response
							else:
								# Fallback: speak raw result
								await self._speak(recall_result)
								self._ledger.append(f"Assistant: {recall_result}")
								task_result = recall_result
						else:
							no_memory_msg = "Dazu habe ich leider nichts in meinem Gedächtnis gefunden."
							await self._speak(no_memory_msg)
							self._ledger.append(f"Assistant: {no_memory_msg}")
							task_result = no_memory_msg
					else:
						task_result = "Keine Query angegeben"

				elif action_type == "memory_store":
					# Store information in memory
					content = str(action.get("content") or "")
					tags = action.get("tags", [])
					if isinstance(tags, str):
						tags = [tags]
					if content:
						self._emit(
							"llm_task_memory_store",
							component="advisor.llm_todo",
							task_id=task_id,
							content=content[:100],
							tags=tags,
						)
						success = await self._memorizer_store_direct(content, tags)
						if success:
							confirm_msg = "Ich habe mir das gemerkt."
							await self._speak(confirm_msg)
							self._ledger.append(f"Assistant: {confirm_msg}")
							task_result = f"Gespeichert: {content[:50]}"
						else:
							fail_msg = "Speichern fehlgeschlagen."
							await self._speak(fail_msg)
							task_result = fail_msg
					else:
						task_result = "Kein Inhalt zum Speichern"

				else:
					# Unknown action type - try to execute via brain
					self._emit(
						"llm_task_unknown_action",
						component="advisor.llm_todo",
						task_id=task_id,
						action_type=action_type,
					)
					task_result = f"Unbekannte Aktion: {action_type}"

				# Mark task as done
				await self._llm_todo.mark_task_done(task_id, task_result)
				completed_count += 1

				self._emit(
					"llm_task_execution_done",
					component="advisor.llm_todo",
					task_id=task_id,
					task_title=task_title,
					result=task_result[:100] if task_result else None,
				)

				# Check if there are more tasks - allow silent interrupt check between tasks
				# NOTE: We do NOT speak "Soll ich weitermachen?" - that's annoying.
				# Instead, we silently check for sound and only listen if user starts talking.
				if self._llm_todo.has_pending_tasks():
					# Silent brief check for human interruption between tasks
					human_input_between = await self._silent_interrupt_check(
						timeout_seconds=1.0,
						context="between_tasks",
					)
					if human_input_between:
						# Human wants to adjust - handle the input
						self._emit(
							"llm_task_interrupted_between",
							component="advisor.llm_todo",
							completed_so_far=completed_count,
							human_input=human_input_between[:50],
						)
						# Add the human input to the ledger and let the brain decide
						self._ledger.append(f"User: {human_input_between}")
						# Break out of the loop to handle the new input
						# The human's input will be processed in the next interaction cycle
						break
				else:
					# Last task - just a small pause
					await asyncio.sleep(0.3)

			except Exception as exc:
				self._emit(
					"llm_task_execution_error",
					component="advisor.llm_todo",
					task_id=task_id,
					task_title=task_title,
					error=str(exc),
				)
				# Mark as done anyway to avoid infinite loop
				await self._llm_todo.mark_task_done(task_id, f"Fehler: {exc}")

		# Step 3: Final status
		self._emit(
			"llm_task_execution_complete",
			component="advisor.llm_todo",
			completed=completed_count,
		)

		# Step 4: Review if mission is complete (optional)
		# Step 4: Review if mission is complete (optional)
		# NOTE: We do this review silently and only act if LLM suggests important missing tasks.
		# We do NOT ask the user "should I do more?" for every trivial thing.
		if self.settings.todo.review_after_completion:
			review_result = await self._llm_todo.review_completion()
			self._emit(
				"llm_task_completion_review",
				component="advisor.llm_todo",
				complete=review_result.get("complete", True),
				reason=review_result.get("reason"),
			)

			if not review_result.get("complete", True):
				additional_tasks = review_result.get("additional_tasks")
				if additional_tasks:
					# Don't ask - just do it if there are obvious follow-up tasks
					# This reduces unnecessary "should I continue?" questions
					self._emit(
						"llm_task_additional_auto_execute",
						component="advisor.llm_todo",
						additional_count=len(additional_tasks),
					)
					# Add the tasks and continue execution
					await self._llm_todo.add_tasks(additional_tasks)
					# Don't speak anything - just continue executing the additional tasks
					# The loop will continue in the next iteration

		# Simple completion message - just "Fertig!" (short and sweet)
		final_response = (
			"Fertig!"
			if is_german
			else "Done!"
		)

		# Speak the completion message
		await self._speak(final_response)
		self._ledger.append(f"Assistant: {final_response}")

		# Silent interrupt check after completion - allow user to add more without asking
		# NOTE: We do NOT ask "Zufrieden?" - that's annoying. User will speak if needed.
		self._emit("state", component="advisor", state="completion_silent_check")
		human_feedback = await self._silent_interrupt_check(
			timeout_seconds=1.5,
			context="after_completion",
		)
		
		if human_feedback and not self._is_bad_transcript(human_feedback):
			self._emit(
				"completion_feedback",
				component="advisor",
				chars=len(human_feedback),
				preview=human_feedback[:100],
			)
			# Human said something - handle it as new request
			self._ledger.append(f"Human: {human_feedback}")

			# Check for stop words first
			if self._matches_stop_word(human_feedback):
				self._emit("interrupt", component="advisor", kind="stop_word", transcript=human_feedback)
				self._llm_todo.clear()
				return None  # Caller will not speak anything

			# Check if it's a positive response (happy with result) - just acknowledge briefly
			low = human_feedback.lower().strip()
			positive_responses = {
				"ja", "yes", "gut", "good", "super", "toll", "great", "perfekt", "perfect",
				"danke", "thanks", "thank you", "ok", "okay", "passt", "alles klar",
				"zufrieden", "happy", "fine", "cool", "nice", "wunderbar", "excellent"
			}
			if low in positive_responses or any(p in low for p in positive_responses):
				# User is happy - just acknowledge very briefly
				self._llm_todo.clear()
				return None  # No need to say "Super, freut mich!" - "Fertig!" was enough

			# User said something else - treat it as new task/request
			self._llm_todo.clear()
			self._emit("state", component="advisor", state="adjustment_request")
			return await self._plan_and_execute_tasks_llm(human_feedback)
		else:
			self._emit(
				"completion_feedback",
				component="advisor",
				chars=0,
				preview=None,
				reason="no_speech_or_timeout",
			)

		# Clear the todo list
		self._llm_todo.clear()

		return None  # Already spoke the final response

	async def _plan_and_execute_tasks_legacy(self, user_request: str) -> str | None:
		"""Legacy task execution using TaskPlanner (fallback)."""
		if self._task_planner is None or self._todo is None:
			return None

		self._emit("state", component="advisor", state="task_planning_start")

		# Step 1: Plan tasks using LLM
		try:
			planned_tasks = await self._task_planner.plan_tasks(
				user_request,
				language=self.settings.response_language,
			)
		except Exception as exc:
			self._emit("task_planning_error", component="advisor.task_planner", error=str(exc))
			return None

		if not planned_tasks:
			self._emit("task_planning_empty", component="advisor.task_planner")
			return None

		self._emit(
			"task_planning_done",
			component="advisor.task_planner",
			task_count=len(planned_tasks),
			tasks=[t.get("title") for t in planned_tasks],
		)

		# Step 2: Set up todo list with the planned tasks
		task_titles = [t.get("title", "") for t in planned_tasks if t.get("title")]
		mission = user_request[:100] + ("..." if len(user_request) > 100 else "")
		self._todo.set_mission(mission, tasks=task_titles)

		# Store action hints for each task (by title)
		action_hints: dict[str, dict[str, Any]] = {}
		for t in planned_tasks:
			title = t.get("title", "")
			hint = t.get("action_hint")
			if title and hint:
				action_hints[title] = hint

		# Announce the plan
		lang = self.settings.response_language
		is_german = str(lang).lower().startswith("de")
		plan_announcement = (
			f"Okay, ich habe {len(task_titles)} Schritte geplant. Los geht's!"
			if is_german
			else f"Okay, I've planned {len(task_titles)} steps. Let's go!"
		)
		await self._speak(plan_announcement)
		self._ledger.append(f"Assistant: {plan_announcement}")

		# Step 3: Execute tasks one by one
		completed_count = 0
		task_results: list[dict[str, Any]] = []

		while self._todo.has_open_tasks():
			current_task = self._todo.next_task()
			if current_task is None:
				break

			task_id = current_task.get("id")
			task_title = str(current_task.get("title") or "")

			self._emit(
				"task_execution_start",
				component="advisor",
				task_id=task_id,
				task_title=task_title,
			)

			# Get the action hint for this task
			action_hint = action_hints.get(task_title)
			task_result: dict[str, Any] = {"task_id": task_id, "title": task_title, "ok": True}

			try:
				if action_hint:
					# Execute the pre-planned action
					action_type = str(action_hint.get("type") or "").lower()

					if action_type == "observe":
						# Special case: observe and describe
						question = str(action_hint.get("question") or "Beschreibe was du siehst.")
						obs = await self._observe(question)
						if obs:
							await self._speak(obs)
							self._ledger.append(f"Assistant: {obs}")
						task_result["observation"] = obs

					elif action_type in {"head_set_angles", "set_head_angles", "look"}:
						# Head movement - possibly with observe_after
						pan = action_hint.get("pan_deg")
						tilt = action_hint.get("tilt_deg")
						observe_after = bool(action_hint.get("observe_after", False))

						# Execute head movement
						pan_i = int(pan) if pan is not None else None
						tilt_i = int(tilt) if tilt is not None else None
						if pan_i is not None:
							pan_i = max(-90, min(90, pan_i))
						if tilt_i is not None:
							tilt_i = max(-35, min(35, tilt_i))
						await self._head_set_angles(pan_deg=pan_i, tilt_deg=tilt_i)

						# Wait for head to settle
						await asyncio.sleep(1.0)

						# If observe_after, take a picture and describe
						if observe_after:
							question = str(action_hint.get("observe_question") or action_hint.get("question") or "Beschreibe was du siehst.")
							obs = await self._observe(question)
							if obs:
								await self._speak(obs)
								self._ledger.append(f"Assistant: {obs}")
							task_result["observation"] = obs

					elif action_type in {"guarded_drive", "drive", "move"}:
						# Execute drive with await_completion
						action_results = await self._execute_planned_actions([action_hint])
						task_result["action_results"] = action_results

					elif action_type == "speak":
						# Just speak
						text = str(action_hint.get("text") or "")
						if text:
							await self._speak(text)
							self._ledger.append(f"Assistant: {text}")
						task_result["spoken"] = text

					else:
						# Regular action - execute via _execute_planned_actions
						action_results = await self._execute_planned_actions([action_hint])
						task_result["action_results"] = action_results
				else:
					# No action hint - ask the brain what to do for this task
					response, need_observe, actions = await self._decide_and_respond(
						human_text=f"Führe diese Aufgabe aus: {task_title}",
						observation=None,
						memory_hint=None,
					)
					if actions:
						action_results = await self._execute_planned_actions(actions)
						task_result["action_results"] = action_results
					if need_observe:
						obs = await self._observe(f"Für die Aufgabe: {task_title}")
						if obs:
							await self._speak(obs)
							self._ledger.append(f"Assistant: {obs}")
						task_result["observation"] = obs
					elif response:
						await self._speak(response)
						self._ledger.append(f"Assistant: {response}")
						task_result["response"] = response

				# Mark task as done
				self._todo.complete_task(task_id)
				completed_count += 1
				self._emit(
					"task_execution_done",
					component="advisor",
					task_id=task_id,
					task_title=task_title,
				)

				# Pause between tasks
				await asyncio.sleep(0.3)

			except Exception as exc:
				task_result["ok"] = False
				task_result["error"] = str(exc)
				self._emit(
					"task_execution_error",
					component="advisor",
					task_id=task_id,
					task_title=task_title,
					error=str(exc),
				)
				# Mark as done anyway to avoid infinite loop
				self._todo.complete_task(task_id)

			task_results.append(task_result)
			self._ledger.append(f"[task] {task_result}")

		# Step 4: Final summary
		self._emit(
			"task_execution_complete",
			component="advisor",
			total_tasks=len(task_titles),
			completed=completed_count,
		)

		final_response = (
			f"Fertig! Ich habe alle {completed_count} Schritte abgeschlossen."
			if is_german
			else f"Done! I completed all {completed_count} steps."
		)
		return final_response

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
			# Best-effort: also stop robot motion and head scan jobs.
			try:
				await self._safety_stop()
			except Exception:
				pass
			try:
				await self._head_stop()
			except Exception:
				pass
			return

		self._ledger.append(f"Human: {text}")

		# Fast path: local todo commands.
		todo_response = await self._try_handle_todo_command(text)
		if todo_response is not None:
			self._emit("state", component="advisor", state="interaction_todo")
			await self._speak(todo_response)
			self._ledger.append(f"Assistant: {todo_response}")
			await self._maybe_summarize_and_reset()
			self._emit("state", component="advisor", state="interaction_end")
			return

		# Fast path: handle simple robot commands without involving the LLM.
		cmd_response = await self._try_handle_robot_command(text)
		if cmd_response is not None:
			self._emit("state", component="advisor", state="interaction_command")
			await self._speak(cmd_response)
			self._ledger.append(f"Assistant: {cmd_response}")
			await self._maybe_summarize_and_reset()
			self._emit("state", component="advisor", state="interaction_end")
			return
		# If user explicitly asks for a conversation summary, do it directly.
		if self._wants_conversation_summary(text):
			summary = await self._summarize_conversation_for_user()
			self._emit("state", component="advisor", state="interaction_speak_summary")
			await self._speak(summary)
			self._ledger.append(f"Assistant: {summary}")
			await self._maybe_summarize_and_reset()
			self._emit("state", component="advisor", state="interaction_end")
			return

		# NEW: Check if this is a multi-step request that needs task planning
		if self._is_multi_step_request(text):
			self._emit("state", component="advisor", state="interaction_multi_step")
			final_response = await self._plan_and_execute_tasks(text)
			if final_response is not None:
				await self._speak(final_response)
				self._ledger.append(f"Assistant: {final_response}")
				await self._maybe_summarize_and_reset()
				self._emit("state", component="advisor", state="interaction_end")
				return
			# If planning failed, fall through to normal flow

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
		
		# Use streaming TTS if enabled (reduces latency by speaking as LLM generates)
		# Note: With streaming, we speak first, then handle actions. This is a tradeoff.
		use_streaming = bool(self.settings.streaming_tts_enabled) and not self._dry_run
		
		if use_streaming:
			self._emit("state", component="advisor", state="interaction_think_streaming")
			response, need_observe, actions = await self._decide_and_respond_streaming(
				human_text=text, observation=obs, memory_hint=mem_hint
			)
			# Note: With streaming, speech has already been sent during generation
			speech_already_streamed = True
		else:
			response, need_observe, actions = await self._decide_and_respond(
				human_text=text, observation=obs, memory_hint=mem_hint
			)
			speech_already_streamed = False

		# If the brain proposed safe actions, execute them before speaking.
		# Special handling for memory_recall: if present, get memory and re-decide with results
		memory_recall_result: str | None = None
		if actions:
			self._emit("planned_actions", component="advisor", count=len(actions), actions=actions)
			
			# Check for memory_recall actions first - execute them and get results
			for action in actions:
				atype = str(action.get("type") or "").strip().lower()
				if atype == "memory_recall":
					query = str(action.get("query") or "").strip()
					if query:
						self._emit("memory_recall_triggered", component="advisor", query=query)
						memory_recall_result = await self._memorizer_recall_direct(query)
						if memory_recall_result:
							self._ledger.append(f"[memory_recall] {memory_recall_result[:200]}...")
			
			# If we got memory results, re-decide with that information
			if memory_recall_result:
				combined_mem_hint = mem_hint or ""
				if combined_mem_hint:
					combined_mem_hint += "\n\n"
				combined_mem_hint += f"Memory search results:\n{memory_recall_result}"
				
				self._emit("state", component="advisor", state="interaction_think_with_memory")
				# Re-deciding after memory recall uses non-streaming (original response was likely incomplete)
				response, need_observe, remaining_actions = await self._decide_and_respond(
					human_text=text, 
					observation=obs, 
					memory_hint=combined_mem_hint
				)
				speech_already_streamed = False  # Need to speak this new response
				# Execute remaining non-memory actions
				non_memory_actions = [a for a in actions if str(a.get("type") or "").strip().lower() not in ("memory_recall",)]
				if non_memory_actions:
					action_results = await self._execute_planned_actions(non_memory_actions)
					self._ledger.append(f"[actions] {action_results}")
					# Check if we were interrupted
					if action_results and action_results[-1].get("type") == "_interrupted":
						human_input = action_results[-1].get("human_input", "")
						self._emit("interaction_interrupted", component="advisor", human_input=human_input[:100])
						# Handle the interruption - recurse with new input
						await self._maybe_summarize_and_reset()
						await self._interaction_step(human_input)
						return
			else:
				# No memory results, execute all actions normally
				action_results = await self._execute_planned_actions(actions)
				self._ledger.append(f"[actions] {action_results}")
				# Check if we were interrupted
				if action_results and action_results[-1].get("type") == "_interrupted":
					human_input = action_results[-1].get("human_input", "")
					self._emit("interaction_interrupted", component="advisor", human_input=human_input[:100])
					# Handle the interruption - recurse with new input
					await self._maybe_summarize_and_reset()
					await self._interaction_step(human_input)
					return

		# If the brain asked for an observation and we haven't taken one yet, do it once.
		if need_observe is True and not (obs and obs.strip()):
			self._emit("state", component="advisor", state="interaction_observe_assist")
			obs = await self._observe(f"Help answer the human request: {text}")
			# If we need to re-decide after observation, use non-streaming (need full response for actions)
			response, _, _ = await self._decide_and_respond(human_text=text, observation=obs, memory_hint=mem_hint)
			speech_already_streamed = False  # Need to speak this new response

		response = response.strip()
		if not response:
			response = "I heard you, but I didn't get enough to answer. Could you say that again?"

		# Optional: always speak the next todo task (keeps the human aligned).
		if self.settings.todo.enabled and self.settings.todo.mention_next_in_response and self._todo is not None:
			try:
				nxt = self._todo.next_task()
			except Exception:
				nxt = None
			if nxt is not None and str(nxt.get("title") or "").strip():
				prefix = "Nächster Schritt:" if str(self.settings.response_language).lower().startswith("de") else "Next:"
				hint = f"{prefix} {str(nxt.get('title')).strip()}"
				if hint.lower() not in response.lower():
					response = response.rstrip() + "\n\n" + hint
					# If we're adding a hint, need to speak it (even if main response was streamed)
					if speech_already_streamed:
						# Just speak the hint part
						await self._speak(hint)
						self._ledger.append(f"Assistant: {response}")
						speech_already_streamed = True  # Mark as handled

		# Only speak if not already streamed
		if not speech_already_streamed:
			self._emit("state", component="advisor", state="interaction_speak")
			await self._speak(response)
		else:
			self._emit("state", component="advisor", state="interaction_speak_streamed")
		self._ledger.append(f"Assistant: {response}")

		# Ask memorizer to decide whether to store this user utterance (best-effort, non-blocking).
		if self.settings.memorizer.enabled and self.settings.memorizer.ingest_user_utterances and not self._dry_run:
			# Cancel previous background ingest if still running (avoid backlog).
			if self._memorizer_task is not None and not self._memorizer_task.done():
				self._memorizer_task.cancel()
			self._memorizer_task = asyncio.create_task(self._memorizer_ingest_background(text))

		await self._maybe_summarize_and_reset()
		self._emit("state", component="advisor", state="interaction_end")

	async def _try_handle_todo_command(self, text: str) -> str | None:
		cfg = self.settings.todo
		agent = self._todo
		if not bool(cfg.enabled) or agent is None:
			return None
		low = (text or "").strip().lower()
		if not low:
			return None

		# Status / next.
		if low in {"todo", "todos", "todo status", "status", "tasks", "aufgaben", "aufgaben status", "liste"}:
			return agent.status_text()
		if low in {"next", "next task", "nächste", "nächste aufgabe", "was als nächstes", "was kommt als nächstes"}:
			nxt = agent.next_task()
			if nxt is None:
				return "Keine offenen Aufgaben." if str(self.settings.response_language).lower().startswith("de") else "No open tasks."
			return f"Nächster Schritt: {nxt.get('title')}" if str(self.settings.response_language).lower().startswith("de") else f"Next: {nxt.get('title')}"

		# Mark done (optionally with an id).
		done_m = re.search(r"\b(done|erledigt|fertig|abgehakt)\b(?:\s+#?(\d+))?", low)
		if done_m:
			tid_s = done_m.group(2)
			updated = None
			if tid_s:
				updated = agent.complete_task(int(tid_s), note=None)
			else:
				updated = agent.complete_current_or_next(note=None)
			if updated is None:
				return "Ich habe gerade keine offene Aufgabe zum Abhaken." if str(self.settings.response_language).lower().startswith("de") else "I don't have an open task to mark done."
			nxt = agent.next_task()
			if nxt is None:
				return "Erledigt. Keine offenen Aufgaben mehr." if str(self.settings.response_language).lower().startswith("de") else "Done. No open tasks left."
			return (
				f"Erledigt. Nächster Schritt: {nxt.get('title')}"
				if str(self.settings.response_language).lower().startswith("de")
				else f"Done. Next: {nxt.get('title')}"
			)

		# Clear.
		if any(k in low for k in ("clear todo", "clear todos", "todos löschen", "liste leeren", "aufgaben löschen", "clear tasks")):
			agent.clear()
			return "Okay, ich habe die Todo-Liste geleert." if str(self.settings.response_language).lower().startswith("de") else "Okay, I cleared the todo list."

		# Add task.
		add_m = re.match(r"^(?:todo\s+)?(?:add|hinzufügen|füge hinzu|aufgabe hinzufügen)\s*[:\-]?\s*(.+)$", low)
		if add_m:
			title = (add_m.group(1) or "").strip()
			if not title:
				return None
			agent.add_task(title)
			nxt = agent.next_task()
			if nxt is not None:
				return (
					f"Okay. Nächster Schritt: {nxt.get('title')}"
					if str(self.settings.response_language).lower().startswith("de")
					else f"Okay. Next: {nxt.get('title')}"
				)
			return "Okay." 

		# Replan from a freeform list (if the user speaks a list).
		if any(x in text for x in ("\n", "- ", "* ")) or re.search(r"\b\d+\s*[\.)]", text):
			if any(k in low for k in ("plan", "replan", "neuer plan", "todo")):
				agent.set_from_freeform_text(text)
				nxt = agent.next_task()
				if nxt is None:
					return "Plan aktualisiert. Keine offenen Aufgaben." if str(self.settings.response_language).lower().startswith("de") else "Plan updated. No open tasks."
				return (
					f"Plan aktualisiert. Nächster Schritt: {nxt.get('title')}"
					if str(self.settings.response_language).lower().startswith("de")
					else f"Plan updated. Next: {nxt.get('title')}"
				)

		return None

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

		# Check if we should use autonomous todo generation
		use_autonomous_todo = (
			self.settings.todo.autonomous_mode_enabled
			and self._llm_todo is not None
			and self._llm_todo.is_enabled()
		)

		if use_autonomous_todo:
			# Use autonomous todo generation instead of simple observe + think
			await self._alone_step_autonomous()
		else:
			# Original behavior: observe and think out loud
			await self._alone_step_simple()

		self._emit("state", component="advisor", state="alone_end")

	async def _alone_step_simple(self) -> None:
		"""Original alone mode: observe and think out loud."""
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

		# Optional: a small, safety-guarded exploration step.
		if bool(self.settings.alone_explore_enabled):
			try:
				st = await self._speaker_status()
				if bool(isinstance(st, dict) and st.get("speaking")):
					self._emit("state", component="advisor", state="alone_explore_skip_speaking")
				else:
					self._emit("state", component="advisor", state="alone_explore")
					dir_res = await self._observe_direction("Pick a safe direction to move a little. Avoid obstacles.")
					action = str((dir_res or {}).get("action") or "") if isinstance(dir_res, dict) else ""
					steer_deg, duration_s = self._drive_plan_from_observe_action(
						action,
						near_s=float(self.settings.alone_explore_duration_s),
						far_s=float(self.settings.alone_explore_far_duration_s),
					)
					res = await self._safety_guarded_drive(
						speed=int(self.settings.alone_explore_speed),
						steer_deg=int(steer_deg),
						duration_s=float(duration_s),
						threshold_cm=float(self.settings.alone_explore_threshold_cm),
					)
					self._ledger.append(f"[alone] explore action={action!r} steer={steer_deg} dur={duration_s} res={res}")
					await self._maybe_summarize_and_reset()
					if self.settings.alone_explore_speak:
						if res is None:
							await self._speak("Ich kann mich gerade nicht bewegen.")
						elif bool(isinstance(res, dict) and res.get("blocked")):
							await self._speak("Ich bleibe stehen, weil es nicht sicher ist.")
			except Exception as exc:
				self._emit("alone_explore_error", component="advisor", error=str(exc))

	async def _alone_step_autonomous(self) -> None:
		"""Autonomous alone mode: generate and execute random todos."""
		self._emit("state", component="advisor", state="alone_autonomous_start")

		# First, observe the environment to provide context
		obs = await self._observe(self.settings.observation_question)
		self._ledger.append(f"[alone] observation: {obs}")

		# Generate autonomous tasks based on the observation
		try:
			context = f"Aktuelle Beobachtung: {obs}" if obs else None
			tasks = await self._llm_todo.generate_autonomous_tasks(context)

			if not tasks:
				self._emit("state", component="advisor", state="alone_autonomous_no_tasks")
				# Fallback to simple mode
				await self._alone_step_simple()
				return

			self._emit(
				"alone_autonomous_tasks_generated",
				component="advisor.llm_todo",
				task_count=len(tasks),
				tasks=[t.get("title") for t in tasks],
			)

			# Execute the autonomous tasks (usually just 1-3)
			lang = self.settings.response_language
			is_german = str(lang).lower().startswith("de")

			for task in tasks[:3]:  # Limit to 3 tasks per alone step
				if not self._llm_todo.has_pending_tasks():
					break

				next_task = await self._llm_todo.get_next_task()
				if next_task is None:
					break

				task_id = next_task.get("id")
				task_title = str(next_task.get("title") or "")
				action = next_task.get("action", {})

				self._emit(
					"alone_autonomous_task_start",
					component="advisor.llm_todo",
					task_id=task_id,
					task_title=task_title,
				)

				# Execute the action
				task_result = await self._execute_autonomous_action(action)

				# Mark task as done
				await self._llm_todo.mark_task_done(task_id, task_result)

				self._emit(
					"alone_autonomous_task_done",
					component="advisor.llm_todo",
					task_id=task_id,
					task_title=task_title,
					result=task_result[:100] if task_result else None,
				)

				# Small pause between tasks
				await asyncio.sleep(0.5)

			# Clear the autonomous todo list
			self._llm_todo.clear()

			self._emit("state", component="advisor", state="alone_autonomous_complete")

		except Exception as exc:
			self._emit("alone_autonomous_error", component="advisor", error=str(exc))
			# Fallback to simple mode on error
			await self._alone_step_simple()

	async def _execute_autonomous_action(self, action: dict[str, Any]) -> str:
		"""Execute a single autonomous action and return the result."""
		action_type = str(action.get("type") or "").lower()
		task_result = ""

		try:
			if action_type == "observe":
				question = str(action.get("question") or "Beschreibe was du siehst.")
				obs = await self._observe(question)
				if obs:
					await self._speak(obs)
					self._ledger.append(f"[alone] autonomous observe: {obs}")
				task_result = obs or "Keine Beobachtung"

			elif action_type in {"head_set_angles", "set_head_angles", "look"}:
				pan = action.get("pan_deg")
				tilt = action.get("tilt_deg")
				observe_after = bool(action.get("observe_after", False))

				pan_i = int(pan) if pan is not None else None
				tilt_i = int(tilt) if tilt is not None else None
				if pan_i is not None:
					pan_i = max(-90, min(90, pan_i))
				if tilt_i is not None:
					tilt_i = max(-35, min(35, tilt_i))

				await self._head_set_angles(pan_deg=pan_i, tilt_deg=tilt_i)
				await asyncio.sleep(1.0)

				if observe_after:
					question = str(action.get("observe_question") or action.get("question") or "Was siehst du?")
					obs = await self._observe(question)
					if obs:
						await self._speak(obs)
						self._ledger.append(f"[alone] autonomous observe: {obs}")
					task_result = obs or "Keine Beobachtung"
				else:
					task_result = f"Kopf bewegt: pan={pan_i}, tilt={tilt_i}"

			elif action_type in {"guarded_drive", "drive", "move"}:
				speed = int(action.get("speed") or 20)
				steer_deg = int(action.get("steer_deg") or 0)
				duration_s = float(action.get("duration_s") or 0.5)

				speed = max(-100, min(100, speed))
				steer_deg = max(-45, min(45, steer_deg))
				duration_s = max(0.1, min(3.0, duration_s))  # Shorter limit for autonomous

				res = await self._safety_guarded_drive(
					speed=speed,
					steer_deg=steer_deg,
					duration_s=duration_s,
					threshold_cm=float(self.settings.alone_explore_threshold_cm),
					await_completion=True,
				)

				if res is None:
					task_result = "Konnte nicht fahren"
				elif isinstance(res, dict) and res.get("blocked"):
					task_result = "Fahrt blockiert (Hindernis)"
				else:
					task_result = f"Gefahren: speed={speed}, steer={steer_deg}, {duration_s}s"

				self._ledger.append(f"[alone] autonomous drive: {task_result}")

			elif action_type == "speak":
				text = str(action.get("text") or "")
				if text:
					await self._speak(text)
					self._ledger.append(f"[alone] autonomous speak: {text}")
				task_result = text

			else:
				task_result = f"Unbekannte autonome Aktion: {action_type}"

		except Exception as exc:
			task_result = f"Fehler: {exc}"

		return task_result

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
					raw_active = False
					rms = 0
					backend = None
					if self._dry_run:
						raw_active = False
					elif time.time() < float(self._force_interaction_until_ts or 0.0):
						raw_active = True
						self._sound_active_streak = int(self.settings.sound_active_windows_required or 1)
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
						raw_active = bool(res.active)
						rms = int(res.rms)
						backend = str(res.backend)
						reason = getattr(res, "reason", None)
						self._emit("sound", component="advisor.sound", backend=backend, rms=rms, active=raw_active, reason=reason)
						if (reason or backend == "none") and self.settings.sound_fallback_to_interaction_on_error:
							self._emit(
								"sound_fallback",
								component="advisor.sound",
								fallback="interaction",
								backend=backend,
								reason=reason,
							)
							raw_active = True
							self._sound_active_streak = int(self.settings.sound_active_windows_required or 1)
					else:
						# If sound detection is disabled, default to interaction mode.
						raw_active = True
						self._sound_active_streak = int(self.settings.sound_active_windows_required or 1)
						self._emit("sound", component="advisor.sound", backend=None, rms=None, active=True, disabled=True)

					# Require N consecutive active windows before interacting.
					req = int(self.settings.sound_active_windows_required or 1)
					req = max(1, req)
					if raw_active:
						self._sound_active_streak = min(self._sound_active_streak + 1, req)
					else:
						self._sound_active_streak = 0
					active = bool(raw_active and (self._sound_active_streak >= req))
					if raw_active and not active and req > 1:
						self._emit(
							"sound_gate",
							component="advisor.sound",
							gate="streak",
							required=req,
							streak=self._sound_active_streak,
						)

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
