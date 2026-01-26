from __future__ import annotations

import json
import os
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from typing import Any


class HeadError(RuntimeError):
	pass


@dataclass(frozen=True)
class HeadSettings:
	dry_run: bool = False
	robot_base_url: str = "http://127.0.0.1:8010"
	max_pan_deg: int = 35
	max_tilt_deg: int = 35
	scan_step_deg: int = 5
	scan_interval_s: float = 0.08


class HeadController:
	def __init__(self, settings: HeadSettings):
		self.settings = settings
		self._available_reason: str | None = None
		self._pan: int = 0
		self._tilt: int = 0
		self._last_cmd: dict[str, Any] | None = None
		self._last_cmd_ts: float | None = None
		self._last_robot_check_ts: float | None = None
		self._last_robot_available: bool | None = None
		self._last_robot_reason: str | None = None

		if self.settings.dry_run:
			return

		# We don't touch GPIO here. Hardware is owned by services/robot.
		self._available_reason = None

	@property
	def is_available(self) -> bool:
		if self.settings.dry_run:
			return True
		self._refresh_robot_status(max_age_s=1.0)
		return bool(self._last_robot_available)

	@property
	def unavailable_reason(self) -> str | None:
		if self.settings.dry_run:
			return None
		self._refresh_robot_status(max_age_s=1.0)
		return self._last_robot_reason or self._available_reason

	def _http_json(self, method: str, url: str, payload: dict[str, Any] | None = None, timeout_s: float = 3.0) -> dict[str, Any]:
		data = None
		headers = {"Accept": "application/json"}
		if payload is not None:
			body = json.dumps(payload).encode("utf-8")
			data = body
			headers["Content-Type"] = "application/json"
		req = urllib.request.Request(url, data=data, headers=headers, method=method.upper())
		try:
			with urllib.request.urlopen(req, timeout=float(timeout_s)) as resp:
				raw = resp.read().decode("utf-8", errors="replace")
				try:
					out = json.loads(raw) if raw else {}
				except Exception:
					out = {"raw": raw}
				return out if isinstance(out, dict) else {"value": out}
		except urllib.error.HTTPError as exc:
			try:
				raw = exc.read().decode("utf-8", errors="replace")
			except Exception:
				raw = str(exc)
			return {"ok": False, "error": f"HTTP {exc.code}", "detail": raw}
		except Exception as exc:
			return {"ok": False, "error": str(exc)}

	def _refresh_robot_status(self, *, max_age_s: float = 1.0) -> None:
		now = time.time()
		if self._last_robot_check_ts is not None and (now - self._last_robot_check_ts) <= float(max_age_s):
			return
		self._last_robot_check_ts = now
		base = str(self.settings.robot_base_url or "").rstrip("/")
		if not base:
			self._last_robot_available = False
			self._last_robot_reason = "robot_base_url not configured"
			return
		res = self._http_json("GET", base + "/healthz", payload=None, timeout_s=2.0)
		avail = bool(res.get("available")) if isinstance(res, dict) else False
		reason = None
		if not avail:
			reason = str(res.get("unavailable_reason") or res.get("reason") or res.get("error") or "robot unavailable")
		self._last_robot_available = avail
		self._last_robot_reason = reason

	def status(self) -> dict[str, Any]:
		if not self.settings.dry_run:
			self._refresh_robot_status(max_age_s=0.0)
		return {
			"ok": True,
			"available": self.is_available,
			"unavailable_reason": self.unavailable_reason,
			"dry_run": bool(self.settings.dry_run),
			"robot_base_url": str(self.settings.robot_base_url),
			"robot_available": self._last_robot_available,
			"robot_reason": self._last_robot_reason,
			"pan_deg": int(self._pan),
			"tilt_deg": int(self._tilt),
			"last_cmd": self._last_cmd,
			"last_cmd_ts": self._last_cmd_ts,
		}

	def _clamp(self, v: int, *, max_abs: int) -> int:
		v = int(v)
		m = max(1, int(max_abs))
		return max(-m, min(m, v))

	def set_angles(self, *, pan_deg: int | None = None, tilt_deg: int | None = None) -> None:
		pan = self._pan if pan_deg is None else self._clamp(int(pan_deg), max_abs=self.settings.max_pan_deg)
		tilt = self._tilt if tilt_deg is None else self._clamp(int(tilt_deg), max_abs=self.settings.max_tilt_deg)
		self._pan, self._tilt = int(pan), int(tilt)
		self._last_cmd = {"cmd": "set_angles", "pan_deg": self._pan, "tilt_deg": self._tilt}
		self._last_cmd_ts = time.time()

		if self.settings.dry_run:
			return
		base = str(self.settings.robot_base_url or "").rstrip("/")
		res = self._http_json(
			"POST",
			base + "/head/set_angles",
			payload={"pan_deg": int(self._pan), "tilt_deg": int(self._tilt)},
			timeout_s=3.0,
		)
		if not bool(res.get("ok", False)):
			raise HeadError(str(res.get("detail") or res.get("error") or "Head move failed"))

	def center(self) -> None:
		self.set_angles(pan_deg=0, tilt_deg=0)

	@staticmethod
	def from_config_dict(cfg: dict[str, Any]) -> "HeadController":
		head_cfg = (cfg or {}).get("head", {}) if isinstance(cfg, dict) else {}
		scan_cfg = head_cfg.get("scan", {}) if isinstance(head_cfg, dict) else {}

		def _get_bool(obj: Any, key: str, default: bool) -> bool:
			if not isinstance(obj, dict) or key not in obj:
				return default
			val = obj.get(key)
			if isinstance(val, str):
				v = val.strip().lower()
				if v in {"1", "true", "yes", "y", "on"}:
					return True
				if v in {"0", "false", "no", "n", "off"}:
					return False
			return bool(val)

		def _get_int(obj: Any, key: str, default: int) -> int:
			if not isinstance(obj, dict) or key not in obj:
				return default
			try:
				return int(obj.get(key))
			except Exception:
				return default

		def _get_float(obj: Any, key: str, default: float) -> float:
			if not isinstance(obj, dict) or key not in obj:
				return default
			try:
				return float(obj.get(key))
			except Exception:
				return default

		def _get_str(obj: Any, key: str, default: str) -> str:
			if not isinstance(obj, dict) or key not in obj:
				return default
			val = obj.get(key)
			return default if val is None else str(val)

		dry_run_cfg = _get_bool(head_cfg, "dry_run", False)
		if str(os.getenv("HEAD_DRY_RUN") or "").strip().lower() in {"1", "true", "yes", "y", "on"}:
			dry_run_cfg = True

		settings = HeadSettings(
			dry_run=dry_run_cfg,
			robot_base_url=_get_str(head_cfg, "robot_base_url", str(os.getenv("ROBOT_BASE_URL") or "http://127.0.0.1:8010")),
			max_pan_deg=max(5, min(45, _get_int(head_cfg, "max_pan_deg", 35))),
			max_tilt_deg=max(5, min(45, _get_int(head_cfg, "max_tilt_deg", 35))),
			scan_step_deg=max(1, min(20, _get_int(scan_cfg, "step_deg", 5))),
			scan_interval_s=max(0.02, min(1.0, _get_float(scan_cfg, "interval_s", 0.08))),
		)
		return HeadController(settings)
