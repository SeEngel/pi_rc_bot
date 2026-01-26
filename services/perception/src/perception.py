from __future__ import annotations

import os
import time
from dataclasses import dataclass
from typing import Any


class PerceptionError(RuntimeError):
	pass


@dataclass(frozen=True)
class PerceptionSettings:
	dry_run: bool = False
	camera_backend: str = "picamera2"
	camera_width: int = 640
	camera_height: int = 480
	camera_warmup_seconds: float = 1.0
	enable_faces: bool = True
	enable_people: bool = True


class Perception:
	def __init__(self, settings: PerceptionSettings):
		self.settings = settings
		self._available_reason: str | None = None
		self._camera_error: str | None = None
		self._last_camera_attempt_ts: float | None = None
		self._camera: Any | None = None
		self._cv2: Any | None = None
		self._last_ts: float | None = None
		self._last_result: dict[str, Any] | None = None

		if self.settings.dry_run:
			return

		# Optional deps
		try:
			import cv2  # type: ignore

			self._cv2 = cv2
		except Exception as exc:
			self._available_reason = f"opencv (cv2) not available: {exc}"
			return

		# Camera is initialized lazily on first `detect_once()` call to avoid
		# grabbing the camera at startup (which can conflict with other services).
		backend = (self.settings.camera_backend or "").strip().lower()
		if backend not in {"picamera2"}:
			self._available_reason = f"Unsupported camera backend: {self.settings.camera_backend!r}"
			return

		self._available_reason = None

	@property
	def is_available(self) -> bool:
		return self._available_reason is None

	@property
	def unavailable_reason(self) -> str | None:
		return self._available_reason

	def close(self) -> None:
		cam = self._camera
		self._camera = None
		try:
			if cam is not None and hasattr(cam, "stop"):
				cam.stop()
		except Exception:
			pass

	def _ensure_camera(self) -> bool:
		if self.settings.dry_run:
			return True
		if not self.is_available:
			return False
		if self._camera is not None:
			return True

		now = time.time()
		# Avoid hot-looping camera init attempts if something else holds the device.
		if self._last_camera_attempt_ts is not None and (now - self._last_camera_attempt_ts) < 2.0:
			return False
		self._last_camera_attempt_ts = now

		backend = (self.settings.camera_backend or "").strip().lower()
		if backend != "picamera2":
			self._camera_error = f"Unsupported camera backend: {self.settings.camera_backend!r}"
			return False

		try:
			from picamera2 import Picamera2  # type: ignore

			cam = Picamera2()
			cam.configure(
				cam.create_still_configuration(
					main={"size": (int(self.settings.camera_width), int(self.settings.camera_height))}
				)
			)
			cam.start()
			time.sleep(max(0.0, float(self.settings.camera_warmup_seconds)))
			self._camera = cam
			self._camera_error = None
			return True
		except Exception as exc:
			self._camera = None
			self._camera_error = str(exc)
			return False

	def _capture_bgr(self):
		if self.settings.dry_run:
			return None
		if not self.is_available or self._camera is None:
			return None
		cv2 = self._cv2
		if cv2 is None:
			return None
		try:
			# Picamera2 returns RGB array; convert to BGR for OpenCV
			rgb = self._camera.capture_array()
			bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
			return bgr
		except Exception:
			return None

	def detect_once(self) -> dict[str, Any]:
		"""Run enabled detectors on one frame."""
		if self.settings.dry_run:
			out = {
				"ok": True,
				"available": True,
				"dry_run": True,
				"faces": [{"bbox": [10, 10, 100, 100], "score": 0.5}],
				"people": [],
				"ts": time.time(),
			}
			self._last_result = out
			self._last_ts = out["ts"]
			return out

		if not self.is_available:
			return {
				"ok": False,
				"available": False,
				"reason": self._available_reason,
				"faces": [],
				"people": [],
				"ts": time.time(),
			}

		if not self._ensure_camera() or self._camera is None:
			reason = self._camera_error or "camera not available"
			return {
				"ok": False,
				"available": False,
				"reason": reason,
				"faces": [],
				"people": [],
				"ts": time.time(),
			}

		cv2 = self._cv2
		frame = self._capture_bgr()
		if cv2 is None or frame is None:
			return {
				"ok": False,
				"available": False,
				"reason": "camera capture failed",
				"faces": [],
				"people": [],
				"ts": time.time(),
			}

		faces: list[dict[str, Any]] = []
		people: list[dict[str, Any]] = []

		try:
			gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
		except Exception:
			gray = None

		if bool(self.settings.enable_faces) and gray is not None:
			try:
				cascade_path = None
				try:
					# Works for many OpenCV installs.
					cascade_path = os.path.join(cv2.data.haarcascades, "haarcascade_frontalface_default.xml")
				except Exception:
					cascade_path = None

				if cascade_path:
					face_cascade = cv2.CascadeClassifier(cascade_path)
					rects = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
					for (x, y, w, h) in rects:
						faces.append({"bbox": [int(x), int(y), int(w), int(h)], "score": None})
			except Exception:
				pass

		if bool(self.settings.enable_people):
			try:
				hog = cv2.HOGDescriptor()
				hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())
				rects, weights = hog.detectMultiScale(frame, winStride=(8, 8), padding=(16, 16), scale=1.05)
				for i, (x, y, w, h) in enumerate(rects):
					score = None
					try:
						if weights is not None and len(weights) > i:
							score = float(weights[i])
					except Exception:
						score = None
					people.append({"bbox": [int(x), int(y), int(w), int(h)], "score": score})
			except Exception:
				pass

		out = {
			"ok": True,
			"available": True,
			"dry_run": False,
			"faces": faces,
			"people": people,
			"ts": time.time(),
		}
		self._last_result = out
		self._last_ts = out["ts"]
		return out

	def status(self) -> dict[str, Any]:
		return {
			"ok": True,
			"available": bool(self.settings.dry_run) or (self.is_available and self._camera is not None),
			"unavailable_reason": self._available_reason,
			"camera_started": self._camera is not None,
			"camera_error": self._camera_error,
			"dry_run": bool(self.settings.dry_run),
			"camera_backend": self.settings.camera_backend,
			"camera_width": int(self.settings.camera_width),
			"camera_height": int(self.settings.camera_height),
			"enable_faces": bool(self.settings.enable_faces),
			"enable_people": bool(self.settings.enable_people),
			"last_ts": self._last_ts,
		}

	@staticmethod
	def from_config_dict(cfg: dict[str, Any]) -> "Perception":
		p_cfg = (cfg or {}).get("perception", {}) if isinstance(cfg, dict) else {}

		def _get_bool(key: str, default: bool) -> bool:
			if not isinstance(p_cfg, dict) or key not in p_cfg:
				return default
			val = p_cfg.get(key)
			if isinstance(val, str):
				v = val.strip().lower()
				if v in {"1", "true", "yes", "y", "on"}:
					return True
				if v in {"0", "false", "no", "n", "off"}:
					return False
			return bool(val)

		def _get_int(key: str, default: int) -> int:
			if not isinstance(p_cfg, dict) or key not in p_cfg:
				return default
			try:
				return int(p_cfg.get(key))
			except Exception:
				return default

		def _get_float(key: str, default: float) -> float:
			if not isinstance(p_cfg, dict) or key not in p_cfg:
				return default
			try:
				return float(p_cfg.get(key))
			except Exception:
				return default

		def _get_str(key: str, default: str) -> str:
			if not isinstance(p_cfg, dict) or key not in p_cfg:
				return default
			val = p_cfg.get(key)
			return default if val is None else str(val)

		dry_run_cfg = _get_bool("dry_run", False)
		if str(os.getenv("PERCEPTION_DRY_RUN") or "").strip().lower() in {"1", "true", "yes", "y", "on"}:
			dry_run_cfg = True

		settings = PerceptionSettings(
			dry_run=dry_run_cfg,
			camera_backend=_get_str("camera_backend", "picamera2"),
			camera_width=max(160, min(1920, _get_int("camera_width", 640))),
			camera_height=max(120, min(1080, _get_int("camera_height", 480))),
			camera_warmup_seconds=max(0.0, min(10.0, _get_float("camera_warmup_seconds", 1.0))),
			enable_faces=_get_bool("enable_faces", True),
			enable_people=_get_bool("enable_people", True),
		)
		return Perception(settings)
