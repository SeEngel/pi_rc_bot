from __future__ import annotations

import json
import os
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from typing import Any


class MoveError(RuntimeError):
	pass


@dataclass(frozen=True)
class MoveSettings:
	dry_run: bool = False
	robot_base_url: str = "http://127.0.0.1:8010"
	max_steer_deg: int = 35
	max_speed: int = 100
	default_duration_s: float = 0.6


class MoveController:
	"""Thin wrapper around PiCar-X motion primitives.

	This controller is intentionally defensive:
	- if hardware libs aren't importable, it reports unavailable
	- in dry_run, it never touches hardware
	"""

	def __init__(self, settings: MoveSettings):
		self.settings = settings
		self._available_reason: str | None = None
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
			"last_cmd": self._last_cmd,
			"last_cmd_ts": self._last_cmd_ts,
		}

	def stop(self) -> None:
		self._last_cmd = {"cmd": "stop"}
		self._last_cmd_ts = time.time()
		if self.settings.dry_run:
			return
		base = str(self.settings.robot_base_url or "").rstrip("/")
		res = self._http_json("POST", base + "/stop", payload={}, timeout_s=2.0)
		# Never raise on stop.
		_ = res

	def drive(self, *, speed: int, steer_deg: int = 0) -> None:
		spd = int(speed)
		max_spd = max(1, int(self.settings.max_speed))
		if spd > max_spd:
			spd = max_spd
		if spd < -max_spd:
			spd = -max_spd

		steer = int(steer_deg)
		max_steer = max(1, int(self.settings.max_steer_deg))
		if steer > max_steer:
			steer = max_steer
		if steer < -max_steer:
			steer = -max_steer

		self._last_cmd = {"cmd": "drive", "speed": spd, "steer_deg": steer}
		self._last_cmd_ts = time.time()

		if self.settings.dry_run:
			return
		base = str(self.settings.robot_base_url or "").rstrip("/")
		res = self._http_json("POST", base + "/drive", payload={"speed": spd, "steer_deg": steer}, timeout_s=3.0)
		if not bool(res.get("ok", False)):
			raise MoveError(str(res.get("detail") or res.get("error") or "Drive failed"))

	@staticmethod
	def from_config_dict(cfg: dict[str, Any]) -> "MoveController":
		move_cfg = (cfg or {}).get("move", {}) if isinstance(cfg, dict) else {}

		def _get_bool(key: str, default: bool) -> bool:
			if not isinstance(move_cfg, dict) or key not in move_cfg:
				return default
			val = move_cfg.get(key)
			if isinstance(val, str):
				v = val.strip().lower()
				if v in {"1", "true", "yes", "y", "on"}:
					return True
				if v in {"0", "false", "no", "n", "off"}:
					return False
			return bool(val)

		def _get_int(key: str, default: int) -> int:
			if not isinstance(move_cfg, dict) or key not in move_cfg:
				return default
			try:
				return int(move_cfg.get(key))
			except Exception:
				return default

		def _get_float(key: str, default: float) -> float:
			if not isinstance(move_cfg, dict) or key not in move_cfg:
				return default
			try:
				return float(move_cfg.get(key))
			except Exception:
				return default

		def _get_str(key: str, default: str) -> str:
			if not isinstance(move_cfg, dict) or key not in move_cfg:
				return default
			val = move_cfg.get(key)
			return default if val is None else str(val)

		dry_run_cfg = _get_bool("dry_run", False)
		if str(os.getenv("MOVE_DRY_RUN") or "").strip().lower() in {"1", "true", "yes", "y", "on"}:
			dry_run_cfg = True

		settings = MoveSettings(
			dry_run=dry_run_cfg,
			robot_base_url=_get_str("robot_base_url", str(os.getenv("ROBOT_BASE_URL") or "http://127.0.0.1:8010")),
			max_steer_deg=max(5, min(45, _get_int("max_steer_deg", 35))),
			max_speed=max(10, min(100, _get_int("max_speed", 100))),
			default_duration_s=max(0.05, min(10.0, _get_float("default_duration_s", 0.6))),
		)
		return MoveController(settings)
