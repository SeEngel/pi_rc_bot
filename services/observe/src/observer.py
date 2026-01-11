from __future__ import annotations

import base64
import os
import tempfile
import time
from dataclasses import dataclass
from typing import Any


class ObserverError(RuntimeError):
	pass


@dataclass(frozen=True)
class ObserverSettings:
	engine: str = "openai"  # openai
	camera_backend: str = "picamera2"  # picamera2
	camera_width: int = 1280
	camera_height: int = 720
	camera_warmup_seconds: float = 2.0
	camera_save_last_path: str | None = None

	dry_run: bool = False
	default_question: str = "Describe what you see in this image."
	instructions: str | None = "You are a helpful assistant that describes what is visible in the image."

	openai_model: str = "gpt-4o"
	openai_base_url: str | None = None
	openai_temperature: float = 0.2
	openai_max_tokens: int = 200


class Observer:
	def __init__(self, settings: ObserverSettings):
		self.settings = settings
		self._available_reason: str | None = None
		self._camera: Any | None = None
		self._openai_client: Any | None = None

		if self.settings.dry_run:
			return

		engine = (self.settings.engine or "").strip().lower()
		if engine != "openai":
			raise ObserverError(f"Unsupported vision engine: {self.settings.engine!r}")

		self._init_camera()
		self._init_openai()

	@property
	def is_available(self) -> bool:
		return self._available_reason is None

	@property
	def unavailable_reason(self) -> str | None:
		return self._available_reason

	def close(self) -> None:
		"""Best-effort resource cleanup (camera)."""
		cam = self._camera
		self._camera = None
		try:
			if cam is not None and hasattr(cam, "stop"):
				cam.stop()
		except Exception:
			pass

	def observe_once(self, *, question: str | None = None) -> dict[str, Any]:
		"""Capture one image and return a dict with text + metadata."""

		if self.settings.dry_run:
			q = (question or self.settings.default_question or "").strip() or "(no question)"
			return {"text": f"(dry_run) {q}", "raw": {}, "model": None}

		if not self.is_available:
			raise ObserverError(self._available_reason or "Vision unavailable")

		jpeg_bytes, saved_path = self._capture_jpeg_bytes()
		res = self._ask_openai_vision(jpeg_bytes=jpeg_bytes, question=question)
		# Attach capture info for debugging.
		res["capture"] = {
			"backend": self.settings.camera_backend,
			"width": int(self.settings.camera_width),
			"height": int(self.settings.camera_height),
			"saved_path": saved_path,
		}
		return res

	@staticmethod
	def from_config_dict(cfg: dict[str, Any]) -> "Observer":
		camera_cfg = (cfg or {}).get("camera", {}) if isinstance(cfg, dict) else {}
		vision_cfg = (cfg or {}).get("vision", {}) if isinstance(cfg, dict) else {}
		openai_cfg = (vision_cfg or {}).get("openai", {}) if isinstance(vision_cfg, dict) else {}

		def _get_str(d: dict[str, Any], key: str, default: str) -> str:
			val = d.get(key, default)
			return default if val is None else str(val)

		def _get_bool(d: dict[str, Any], key: str, default: bool) -> bool:
			val = d.get(key, default)
			return bool(val) if val is not None else default

		def _get_int(d: dict[str, Any], key: str, default: int) -> int:
			val = d.get(key, default)
			try:
				return int(val)
			except Exception:
				return default

		def _get_float(d: dict[str, Any], key: str, default: float) -> float:
			val = d.get(key, default)
			try:
				return float(val)
			except Exception:
				return default

		base_url = _get_str(openai_cfg if isinstance(openai_cfg, dict) else {}, "base_url", "").strip()
		if base_url == "":
			base_url_val: str | None = None
		else:
			base_url_val = base_url

		save_last_path = _get_str(camera_cfg if isinstance(camera_cfg, dict) else {}, "save_last_path", "").strip()
		if save_last_path == "":
			save_last_val: str | None = None
		else:
			save_last_val = save_last_path

		dry_run_cfg = _get_bool(vision_cfg if isinstance(vision_cfg, dict) else {}, "dry_run", False)
		# Convenience override for development/testing.
		# If set, we will not access camera or external APIs.
		if str(os.getenv("OBSERVE_DRY_RUN") or "").strip().lower() in {"1", "true", "yes", "y", "on"}:
			dry_run_cfg = True

		settings = ObserverSettings(
			engine=_get_str(vision_cfg if isinstance(vision_cfg, dict) else {}, "engine", "openai"),
			camera_backend=_get_str(camera_cfg if isinstance(camera_cfg, dict) else {}, "backend", "picamera2"),
			camera_width=_get_int(camera_cfg if isinstance(camera_cfg, dict) else {}, "width", 1280),
			camera_height=_get_int(camera_cfg if isinstance(camera_cfg, dict) else {}, "height", 720),
			camera_warmup_seconds=_get_float(camera_cfg if isinstance(camera_cfg, dict) else {}, "warmup_seconds", 2.0),
			camera_save_last_path=save_last_val,
			dry_run=dry_run_cfg,
			default_question=_get_str(vision_cfg if isinstance(vision_cfg, dict) else {}, "default_question", "Describe what you see in this image."),
			instructions=_get_str(vision_cfg if isinstance(vision_cfg, dict) else {}, "instructions", "").strip() or None,
			openai_model=_get_str(openai_cfg if isinstance(openai_cfg, dict) else {}, "model", "gpt-4o"),
			openai_base_url=base_url_val,
			openai_temperature=_get_float(openai_cfg if isinstance(openai_cfg, dict) else {}, "temperature", 0.2),
			openai_max_tokens=_get_int(openai_cfg if isinstance(openai_cfg, dict) else {}, "max_tokens", 200),
		)
		return Observer(settings)

	def _init_camera(self) -> None:
		backend = (self.settings.camera_backend or "").strip().lower()
		if backend != "picamera2":
			self._available_reason = f"Unsupported camera backend: {self.settings.camera_backend!r}"
			return

		try:
			from picamera2 import Picamera2  # type: ignore
		except Exception as exc:
			self._available_reason = f"picamera2 not available: {exc}"
			return

		try:
			cam = Picamera2()
			config = cam.create_still_configuration(main={"size": (int(self.settings.camera_width), int(self.settings.camera_height))})
			cam.configure(config)
			cam.start()
			warm = float(self.settings.camera_warmup_seconds or 0.0)
			if warm > 0:
				time.sleep(warm)
			self._camera = cam
		except Exception as exc:
			self._available_reason = f"Camera init failed: {exc}"
			self._camera = None

	def _init_openai(self) -> None:
		api_key = os.getenv("OPENAI_API_KEY")
		if not api_key:
			self._available_reason = "Missing OPENAI_API_KEY"
			return

		try:
			from openai import OpenAI  # type: ignore
		except Exception as exc:
			self._available_reason = f"openai package not available: {exc}"
			return

		kwargs: dict[str, Any] = {"api_key": api_key}
		base_url = (self.settings.openai_base_url or "").strip()
		# If base_url is empty, use the OpenAI default base URL.
		# If set, it must point to an OpenAI-compatible API (often ending with /v1).
		if base_url:
			kwargs["base_url"] = base_url

		try:
			self._openai_client = OpenAI(**kwargs)
		except Exception as exc:
			self._available_reason = f"OpenAI client init failed: {exc}"
			self._openai_client = None

	def _capture_jpeg_bytes(self) -> tuple[bytes, str | None]:
		cam = self._camera
		if cam is None:
			raise ObserverError(self._available_reason or "Camera not initialized")

		# Capture to a temporary file (Picamera2 writes files efficiently).
		fd, tmp_path = tempfile.mkstemp(prefix="pi_rc_bot_observe_", suffix=".jpg")
		os.close(fd)
		saved_path: str | None = None
		try:
			cam.capture_file(tmp_path)
			with open(tmp_path, "rb") as f:
				jpeg_bytes = f.read()

			if self.settings.camera_save_last_path:
				try:
					# Best-effort copy for debugging.
					import shutil

					shutil.copyfile(tmp_path, self.settings.camera_save_last_path)
					saved_path = str(self.settings.camera_save_last_path)
				except Exception:
					saved_path = None
			return jpeg_bytes, saved_path
		finally:
			try:
				os.remove(tmp_path)
			except Exception:
				pass

	def _ask_openai_vision(self, *, jpeg_bytes: bytes, question: str | None) -> dict[str, Any]:
		client = self._openai_client
		if client is None:
			raise ObserverError(self._available_reason or "OpenAI client not initialized")

		q = (question or self.settings.default_question or "").strip()
		if not q:
			q = "Describe what you see in this image."

		b64 = base64.b64encode(jpeg_bytes).decode("ascii")
		data_url = f"data:image/jpeg;base64,{b64}"

		messages: list[dict[str, Any]] = []
		if self.settings.instructions:
			messages.append({"role": "system", "content": str(self.settings.instructions)})
		messages.append(
			{
				"role": "user",
				"content": [
					{"type": "text", "text": q},
					{"type": "image_url", "image_url": {"url": data_url}},
				],
			}
		)

		model = (self.settings.openai_model or "").strip() or "gpt-4o"
		try:
			resp = client.chat.completions.create(
				model=model,
				messages=messages,
				temperature=float(self.settings.openai_temperature),
				max_tokens=int(self.settings.openai_max_tokens),
			)
		except Exception as exc:
			raise ObserverError(f"Vision request failed: {exc}") from exc

		text = ""
		try:
			choice0 = resp.choices[0]
			msg = getattr(choice0, "message", None)
			text = str(getattr(msg, "content", "") or "")
		except Exception:
			text = ""

		raw: Any = {}
		try:
			raw = resp.model_dump()  # type: ignore[attr-defined]
		except Exception:
			try:
				raw = dict(resp)  # type: ignore[arg-type]
			except Exception:
				raw = {}

		return {"text": text, "model": model, "raw": raw}
