from __future__ import annotations

import os
import collections
import inspect
import tempfile
import wave
from dataclasses import dataclass
from typing import Any, Callable


class ListenerError(RuntimeError):
	pass


@dataclass(frozen=True)
class ListenerSettings:
	engine: str = "vosk"  # vosk | openai
	language: str = "de"

	print_text: bool = True
	dry_run: bool = False

	# Text post-processing
	normalize_lowercase: bool = True
	normalize_strip: bool = True

	# --- OpenAI STT settings (engine=openai) ---
	openai_model: str = "gpt-4o-mini-transcribe"
	openai_language: str | None = "de"  # ISO-639-1 recommended by OpenAI (e.g. "de", "en")
	openai_prompt: str | None = None
	# Maximum capture duration (seconds). Recording may stop earlier if silence-stop is enabled.
	openai_record_seconds: float = 6.0
	# Some ALSA input devices reject 16000 Hz (PortAudio: paInvalidSampleRate).
	# Set to null in config to auto-use the device default sample rate.
	openai_sample_rate_hz: int | None = 16000
	openai_device: int | None = None

	# --- OpenAI mic capture behavior ---
	# Stop the recording after this much *continuous* silence (seconds) once speech has started.
	# Increase this to allow short pauses/breathing between sentences.
	openai_stop_silence_seconds: float = 2.0
	# Simple energy threshold for speech detection (0..~30000 for int16).
	# If you get early stop, lower it; if it records lots of noise, raise it.
	openai_energy_threshold: float = 300.0
	# Chunk size for streaming capture.
	openai_chunk_ms: int = 50
	# Keep a bit of audio from before speech starts.
	openai_pre_roll_seconds: float = 0.25


def _call_with_accepted_kwargs(fn: Callable[..., Any], /, **kwargs: Any) -> Any:
	"""Call `fn` with only those kwargs it accepts.

	This keeps the wrapper compatible with different `robot_hat.stt` versions.
	"""
	try:
		sig = inspect.signature(fn)
	except Exception:
		return fn(**kwargs)

	accepted: dict[str, Any] = {}
	for k, v in kwargs.items():
		if k in sig.parameters:
			accepted[k] = v
	return fn(**accepted)


class Listener:
	"""Simple Speech-to-Text wrapper (Vosk).

	Keeps initialization/config similar to the speak service.
	"""

	def __init__(self, settings: ListenerSettings):
		self.settings = settings
		self._stt_obj: Any | None = None
		self._openai_client: Any | None = None
		self._available_reason: str | None = None

		# In dry-run mode we never access hardware or external services.
		# Keep the listener "available" so the HTTP service can respond.
		if self.settings.dry_run:
			return

		engine = (settings.engine or "").strip().lower()
		if engine == "vosk":
			self._init_vosk()
			return
		if engine == "openai":
			self._init_openai()
			return
		raise ListenerError(f"Unsupported STT engine: {settings.engine!r}")

	@property
	def is_available(self) -> bool:
		return self._available_reason is None

	@property
	def unavailable_reason(self) -> str | None:
		return self._available_reason

	def listen_once(
		self,
		*,
		stream: bool = False,
		speech_pause_seconds: float | None = None,
	) -> dict[str, Any]:
		"""Listen once and return the raw result dict.

		On success, the underlying implementation typically returns:
		  {"text": "..."}
		"""
		if self.settings.dry_run:
			return {"text": "(dry_run)"}
		if not self.is_available:
			reason = self._available_reason or "STT unavailable"
			raise ListenerError(reason)

		engine = (self.settings.engine or "").strip().lower()
		if engine == "openai":
			return self._listen_openai_once(speech_pause_seconds=speech_pause_seconds)
		if engine != "vosk":
			raise ListenerError(f"Unsupported STT engine: {self.settings.engine!r}")

		if self._stt_obj is None:
			raise ListenerError(self._available_reason or "STT unavailable")

		listen_fn = getattr(self._stt_obj, "listen", None)
		if not callable(listen_fn):
			raise ListenerError("STT object has no listen()")

		res = _call_with_accepted_kwargs(listen_fn, stream=bool(stream))
		if isinstance(res, dict):
			return res
		# Normalize unknown return types.
		return {"text": str(res)}

	def _listen_openai_once(self, *, speech_pause_seconds: float | None = None) -> dict[str, Any]:
		"""Record a short WAV and transcribe it via OpenAI."""
		client = self._openai_client
		if client is None:
			raise ListenerError(self._available_reason or "OpenAI client not initialized")

		try:
			import sounddevice as sd  # type: ignore
			import numpy as np  # type: ignore
		except Exception as exc:  # pragma: no cover
			raise ListenerError(f"Missing dependency for OpenAI recording: {exc}") from exc

		max_record_s = float(self.settings.openai_record_seconds or 0.0)
		if max_record_s <= 0:
			raise ListenerError("openai.record_seconds must be > 0")
		device = self.settings.openai_device

		configured_sr = self.settings.openai_sample_rate_hz
		candidates: list[int] = []
		if configured_sr is not None:
			try:
				candidates.append(int(configured_sr))
			except Exception:
				pass

		device_default_sr: int | None = None
		try:
			info = sd.query_devices(device, "input")
			val = info.get("default_samplerate") if isinstance(info, dict) else None
			if val is not None:
				device_default_sr = int(round(float(val)))
		except Exception:
			device_default_sr = None
		if device_default_sr is not None:
			candidates.append(device_default_sr)

		# Common ALSA input rates.
		candidates.extend([48000, 44100, 32000, 16000, 8000])

		seen: set[int] = set()
		unique_rates: list[int] = []
		for sr in candidates:
			try:
				sr_i = int(sr)
			except Exception:
				continue
			if sr_i < 8000 or sr_i > 192000:
				continue
			if sr_i in seen:
				continue
			seen.add(sr_i)
			unique_rates.append(sr_i)

		if speech_pause_seconds is None:
			stop_silence_s = float(self.settings.openai_stop_silence_seconds or 0.0)
		else:
			try:
				stop_silence_s = float(speech_pause_seconds)
			except Exception:
				stop_silence_s = float(self.settings.openai_stop_silence_seconds or 0.0)
		stop_silence_s = max(0.0, stop_silence_s)
		energy_th = float(self.settings.openai_energy_threshold or 0.0)
		energy_th = max(0.0, energy_th)
		chunk_ms = int(self.settings.openai_chunk_ms or 50)
		chunk_ms = max(10, min(500, chunk_ms))
		pre_roll_s = float(self.settings.openai_pre_roll_seconds or 0.0)
		pre_roll_s = max(0.0, min(2.0, pre_roll_s))

		def _rms_int16(x: "np.ndarray") -> float:
			# x: shape (frames, 1) or (frames,)
			y = x.astype(np.float32)
			return float(np.sqrt(np.mean(y * y))) if y.size else 0.0

		last_exc: Exception | None = None
		audio_arr = None
		sample_rate: int | None = None
		for sr in unique_rates:
			frames_per_chunk = max(1, int(sr * (chunk_ms / 1000.0)))
			max_frames = int(max_record_s * sr)
			if max_frames <= 0:
				continue

			pre_roll_frames = int(pre_roll_s * sr)
			pre_buf: "collections.deque[np.ndarray]" = collections.deque()
			pre_buf_frames = 0

			chunks: list["np.ndarray"] = []
			started = False
			silence_frames = 0
			total_frames = 0

			try:
				with sd.InputStream(
					samplerate=sr,
					channels=1,
					dtype="int16",
					blocksize=frames_per_chunk,
					device=device,
				) as stream:
					while total_frames < max_frames:
						data, _overflowed = stream.read(frames_per_chunk)
						arr = np.asarray(data, dtype=np.int16)
						frames = int(arr.shape[0])
						total_frames += frames

						level = _rms_int16(arr)
						is_speech = level >= energy_th

						if not started:
							# Accumulate pre-roll while waiting for speech.
							if pre_roll_frames > 0:
								pre_buf.append(arr)
								pre_buf_frames += frames
								while pre_buf and pre_buf_frames > pre_roll_frames:
									p = pre_buf.popleft()
									pre_buf_frames -= int(p.shape[0])

							if is_speech:
								started = True
								if pre_buf:
									chunks.extend(list(pre_buf))
									pre_buf.clear()
									pre_buf_frames = 0
								chunks.append(arr)
								silence_frames = 0
							continue

						# After speech has started, keep recording and stop after sustained silence.
						chunks.append(arr)
						if stop_silence_s > 0.0:
							if is_speech:
								silence_frames = 0
							else:
								silence_frames += frames
								if (silence_frames / sr) >= stop_silence_s:
									break

				# If we never detected speech, still return what we captured (up to max_record_s).
				audio_arr = np.concatenate(chunks, axis=0) if chunks else np.zeros((0, 1), dtype=np.int16)
				sample_rate = sr
				break
			except Exception as exc:
				last_exc = exc
				audio_arr = None
				sample_rate = None
				continue

		if audio_arr is None or sample_rate is None:
			msg = f"Microphone recording failed for all sample rates tried: {unique_rates}"
			if device_default_sr is not None:
				msg += f" (device default: {device_default_sr})"
			if last_exc is not None:
				msg += f"; last error: {last_exc}"
			msg += (
				". Try setting stt.openai.sample_rate_hz to 44100 or 48000, "
				"or set it to null to auto-use the device default."
			)
			raise ListenerError(msg)

		try:
			pcm16 = np.asarray(audio_arr, dtype=np.int16)
			data = pcm16.tobytes()
		except Exception as exc:
			raise ListenerError(f"Audio conversion failed: {exc}") from exc

		fd, wav_path = tempfile.mkstemp(prefix="pi_rc_bot_listening_", suffix=".wav")
		os.close(fd)
		try:
			with wave.open(wav_path, "wb") as wf:
				wf.setnchannels(1)
				wf.setsampwidth(2)  # int16
				wf.setframerate(int(sample_rate))
				wf.writeframes(data)

			model = str(self.settings.openai_model or "gpt-4o-mini-transcribe")
			kwargs: dict[str, Any] = {"model": model}
			lang = (self.settings.openai_language or "").strip()
			if lang:
				kwargs["language"] = lang
			prompt = self.settings.openai_prompt
			if prompt is not None and str(prompt).strip() != "":
				kwargs["prompt"] = str(prompt)

			with open(wav_path, "rb") as audio_file:
				transcription = client.audio.transcriptions.create(file=audio_file, **kwargs)

			text = getattr(transcription, "text", None)
			if text is None:
				# Some SDK responses may be dict-like.
				try:
					text = transcription.get("text")  # type: ignore[attr-defined]
				except Exception:
					text = None
			text = "" if text is None else str(text)
			return {"text": text}
		except ListenerError:
			raise
		except Exception as exc:
			raise ListenerError(f"OpenAI transcription failed: {exc}") from exc
		finally:
			try:
				os.remove(wav_path)
			except Exception:
				pass

	def extract_text(self, res: dict[str, Any]) -> str:
		text = ""
		try:
			text = str(res.get("text") or "")
		except Exception:
			text = ""

		if self.settings.normalize_lowercase:
			text = text.lower()
		if self.settings.normalize_strip:
			text = text.strip()

		if self.settings.print_text and text:
			print(f"[listening] {text}")
		return text

	@staticmethod
	def from_config_dict(cfg: dict[str, Any]) -> "Listener":
		stt_cfg = (cfg or {}).get("stt", {}) if isinstance(cfg, dict) else {}
		if not isinstance(stt_cfg, dict):
			stt_cfg = {}

		def _get_str(key: str, default: str) -> str:
			val = stt_cfg.get(key, default)
			return default if val is None else str(val)

		def _get_bool(key: str, default: bool) -> bool:
			val = stt_cfg.get(key, default)
			return bool(val) if val is not None else default

		def _get_float(key: str, default: float) -> float:
			val = stt_cfg.get(key, default)
			try:
				return float(val)
			except Exception:
				return default

		def _get_int(key: str, default: int) -> int:
			val = stt_cfg.get(key, default)
			try:
				return int(val)
			except Exception:
				return default

		openai_cfg = stt_cfg.get("openai", {}) if isinstance(stt_cfg, dict) else {}
		if not isinstance(openai_cfg, dict):
			openai_cfg = {}

		openai_model = str(openai_cfg.get("model") or "gpt-4o-mini-transcribe")
		openai_lang_raw = openai_cfg.get("language")
		openai_language = None
		if openai_lang_raw is not None and str(openai_lang_raw).strip() != "":
			openai_language = str(openai_lang_raw).strip()
		openai_prompt = None
		if openai_cfg.get("prompt") is not None and str(openai_cfg.get("prompt")).strip() != "":
			openai_prompt = str(openai_cfg.get("prompt")).strip()

		record_seconds = 6.0
		if openai_cfg.get("record_seconds") is not None and openai_cfg.get("record_seconds") != "":
			try:
				record_seconds = float(openai_cfg.get("record_seconds"))
			except Exception:
				record_seconds = 6.0

		stop_silence_seconds = 1.2
		if openai_cfg.get("stop_silence_seconds") is not None and openai_cfg.get("stop_silence_seconds") != "":
			try:
				stop_silence_seconds = float(openai_cfg.get("stop_silence_seconds"))
			except Exception:
				stop_silence_seconds = 1.2

		energy_threshold = 300.0
		if openai_cfg.get("energy_threshold") is not None and openai_cfg.get("energy_threshold") != "":
			try:
				energy_threshold = float(openai_cfg.get("energy_threshold"))
			except Exception:
				energy_threshold = 300.0

		chunk_ms = 50
		if openai_cfg.get("chunk_ms") is not None and openai_cfg.get("chunk_ms") != "":
			try:
				chunk_ms = int(openai_cfg.get("chunk_ms"))
			except Exception:
				chunk_ms = 50

		pre_roll_seconds = 0.25
		if openai_cfg.get("pre_roll_seconds") is not None and openai_cfg.get("pre_roll_seconds") != "":
			try:
				pre_roll_seconds = float(openai_cfg.get("pre_roll_seconds"))
			except Exception:
				pre_roll_seconds = 0.25

		sample_rate_hz: int | None = 16000
		if "sample_rate_hz" in openai_cfg:
			val = openai_cfg.get("sample_rate_hz")
			if val is None or val == "":
				sample_rate_hz = None
			else:
				try:
					sample_rate_hz = int(val)
				except Exception:
					sample_rate_hz = 16000

		device = None
		if "device" in openai_cfg:
			val = openai_cfg.get("device")
			if val is None or val == "":
				device = None
			else:
				try:
					device = int(val)
				except Exception:
					device = None

		norm_cfg = stt_cfg.get("normalize", {}) if isinstance(stt_cfg, dict) else {}
		lower = True
		strip = True
		if isinstance(norm_cfg, dict):
			lower = bool(norm_cfg.get("lowercase")) if norm_cfg.get("lowercase") is not None else True
			strip = bool(norm_cfg.get("strip")) if norm_cfg.get("strip") is not None else True

		settings = ListenerSettings(
			engine=_get_str("engine", "vosk"),
			language=_get_str("language", "de"),
			print_text=_get_bool("print_text", True),
			dry_run=_get_bool("dry_run", False),
			normalize_lowercase=bool(lower),
			normalize_strip=bool(strip),
			openai_model=openai_model,
			openai_language=openai_language,
			openai_prompt=openai_prompt,
			openai_record_seconds=float(record_seconds),
			openai_sample_rate_hz=sample_rate_hz,
			openai_device=device,
			openai_stop_silence_seconds=float(stop_silence_seconds),
			openai_energy_threshold=float(energy_threshold),
			openai_chunk_ms=int(chunk_ms),
			openai_pre_roll_seconds=float(pre_roll_seconds),
		)
		return Listener(settings)

	# --- init helpers ---

	def _init_vosk(self) -> None:
		Vosk = None
		# Prefer robot_hat directly.
		try:
			from robot_hat.stt import Vosk as _Vosk  # type: ignore

			Vosk = _Vosk
		except Exception:
			# Fallback: picarx re-exports robot_hat.stt
			try:
				from picarx.stt import Vosk as _Vosk  # type: ignore

				Vosk = _Vosk
			except Exception:
				Vosk = None

		if Vosk is None:
			self._available_reason = (
				"robot_hat.stt / picarx.stt not importable. "
				"Install the PiCar-X stack (picar-x + robot-hat) to enable STT."
			)
			self._stt_obj = None
			return

		# Try to pass language if supported.
		try:
			obj = _call_with_accepted_kwargs(Vosk, language=self.settings.language)
		except TypeError:
			try:
				obj = Vosk(self.settings.language)
			except Exception as exc:
				self._available_reason = f"Vosk init error: {exc}"
				self._stt_obj = None
				return
		except Exception as exc:
			self._available_reason = f"Vosk init error: {exc}"
			self._stt_obj = None
			return

		self._stt_obj = obj
		self._available_reason = None

	def _init_openai(self) -> None:
		# Allow dry_run without an API key.
		api_key = os.getenv("OPENAI_API_KEY")
		if not api_key and not self.settings.dry_run:
			self._available_reason = "Missing OpenAI API key (set OPENAI_API_KEY)"
			self._openai_client = None
			return

		try:
			from openai import OpenAI  # type: ignore
		except Exception as exc:
			self._available_reason = f"OpenAI SDK not installed/importable: {exc}"
			self._openai_client = None
			return

		# Recording deps
		try:
			import sounddevice  # type: ignore  # noqa: F401
			import numpy  # type: ignore  # noqa: F401
		except Exception as exc:
			self._available_reason = f"Missing audio recording dependency: {exc}"
			self._openai_client = None
			return

		try:
			self._openai_client = OpenAI()
			self._available_reason = None
		except Exception as exc:
			self._available_reason = f"OpenAI client init failed: {exc}"
			self._openai_client = None
