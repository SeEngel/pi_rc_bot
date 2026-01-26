from __future__ import annotations

import json
import os
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from typing import Any


class ProximityError(RuntimeError):
	pass


@dataclass(frozen=True)
class ProximitySettings:
	dry_run: bool = False
	robot_base_url: str = "http://127.0.0.1:8010"
	obstacle_threshold_cm: float = 35.0


class ProximitySensor:
	def __init__(self, settings: ProximitySettings):
		self.settings = settings
		self._available_reason: str | None = None
		self._last_read_cm: float | None = None
		self._last_read_ts: float | None = None
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

	def read_distance_cm(self) -> float | None:
		if self.settings.dry_run:
			self._last_read_cm = 100.0
			self._last_read_ts = time.time()
			return float(self._last_read_cm)
		if not self.is_available:
			return None
		base = str(self.settings.robot_base_url or "").rstrip("/")
		res = self._http_json("GET", base + "/ultrasonic/distance", payload=None, timeout_s=2.0)
		if not bool(res.get("ok", False)):
			return None
		d = res.get("distance_cm")
		try:
			val = float(d) if d is not None else None
		except Exception:
			val = None
		if val is None or val <= 0:
			return None
		self._last_read_cm = val
		self._last_read_ts = time.time()
		return val

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
			"last_distance_cm": self._last_read_cm,
			"last_read_ts": self._last_read_ts,
			"obstacle_threshold_cm": float(self.settings.obstacle_threshold_cm),
		}

	@staticmethod
	def from_config_dict(cfg: dict[str, Any]) -> "ProximitySensor":
		prox_cfg = (cfg or {}).get("proximity", {}) if isinstance(cfg, dict) else {}

		def _get_bool(key: str, default: bool) -> bool:
			if not isinstance(prox_cfg, dict) or key not in prox_cfg:
				return default
			val = prox_cfg.get(key)
			if isinstance(val, str):
				v = val.strip().lower()
				if v in {"1", "true", "yes", "y", "on"}:
					return True
				if v in {"0", "false", "no", "n", "off"}:
					return False
			return bool(val)

		def _get_float(key: str, default: float) -> float:
			if not isinstance(prox_cfg, dict) or key not in prox_cfg:
				return default
			try:
				return float(prox_cfg.get(key))
			except Exception:
				return default

		def _get_str(key: str, default: str) -> str:
			if not isinstance(prox_cfg, dict) or key not in prox_cfg:
				return default
			val = prox_cfg.get(key)
			return default if val is None else str(val)

		dry_run_cfg = _get_bool("dry_run", False)
		if str(os.getenv("PROXIMITY_DRY_RUN") or "").strip().lower() in {"1", "true", "yes", "y", "on"}:
			dry_run_cfg = True

		settings = ProximitySettings(
			dry_run=dry_run_cfg,
			robot_base_url=_get_str("robot_base_url", str(os.getenv("ROBOT_BASE_URL") or "http://127.0.0.1:8010")),
			obstacle_threshold_cm=max(1.0, min(300.0, _get_float("obstacle_threshold_cm", 35.0))),
		)
		return ProximitySensor(settings)
