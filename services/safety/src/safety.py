from __future__ import annotations

import json
import os
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from typing import Any


@dataclass
class SafetySettings:
	dry_run: bool = False
	estop_default_on: bool = False
	move_base_url: str = "http://127.0.0.1:8005"
	proximity_base_url: str = "http://127.0.0.1:8007"
	obstacle_threshold_cm: float = 35.0


class SafetyController:
	def __init__(self, settings: SafetySettings):
		self.settings = settings
		self._estop_on: bool = bool(settings.estop_default_on)
		self._last_check: dict[str, Any] | None = None

	def estop_on(self) -> dict[str, Any]:
		self._estop_on = True
		return {"ok": True, "estop": True}

	def estop_off(self) -> dict[str, Any]:
		self._estop_on = False
		return {"ok": True, "estop": False}

	def _http_json(self, method: str, url: str, payload: dict[str, Any] | None = None, timeout_s: float = 5.0) -> dict[str, Any]:
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

	def check(self, threshold_cm: float | None = None) -> dict[str, Any]:
		thr = float(threshold_cm) if threshold_cm is not None else float(self.settings.obstacle_threshold_cm)

		if self.settings.dry_run:
			res = {
				"ok": True,
				"estop": bool(self._estop_on),
				"distance_cm": 100.0,
				"threshold_cm": thr,
				"obstacle": False,
				"safe_to_drive": (not self._estop_on),
				"ts": time.time(),
			}
			self._last_check = res
			return res

		prox_url = self.settings.proximity_base_url.rstrip("/") + "/obstacle"
		# We pass threshold as query param.
		prox_url = prox_url + f"?threshold_cm={thr:.3f}"
		p = self._http_json("GET", prox_url, payload=None, timeout_s=3.0)
		distance_cm = p.get("distance_cm")
		obstacle = bool(p.get("obstacle")) if isinstance(p, dict) else False

		safe = (not self._estop_on) and (not obstacle)
		res = {
			"ok": True,
			"estop": bool(self._estop_on),
			"distance_cm": distance_cm,
			"threshold_cm": thr,
			"obstacle": obstacle,
			"safe_to_drive": safe,
			"proximity": p,
			"ts": time.time(),
		}
		self._last_check = res
		return res

	def guarded_drive(self, *, speed: int, steer_deg: int = 0, duration_s: float | None = None, threshold_cm: float | None = None) -> dict[str, Any]:
		chk = self.check(threshold_cm=threshold_cm)
		if not bool(chk.get("safe_to_drive")):
			return {"ok": False, "blocked": True, "reason": "unsafe_to_drive", "check": chk}

		if self.settings.dry_run:
			return {
				"ok": True,
				"blocked": False,
				"dry_run": True,
				"delegated": {"speed": int(speed), "steer_deg": int(steer_deg), "duration_s": duration_s},
				"check": chk,
			}

		move_url = self.settings.move_base_url.rstrip("/") + "/drive"
		payload: dict[str, Any] = {"speed": int(speed), "steer_deg": int(steer_deg)}
		if duration_s is not None:
			payload["duration_s"] = float(duration_s)
		res = self._http_json("POST", move_url, payload=payload, timeout_s=5.0)
		return {"ok": bool(res.get("ok", True)), "blocked": False, "move": res, "check": chk}

	def stop(self) -> dict[str, Any]:
		if self.settings.dry_run:
			return {"ok": True, "stopped": True, "dry_run": True}
		move_url = self.settings.move_base_url.rstrip("/") + "/stop"
		res = self._http_json("POST", move_url, payload={}, timeout_s=5.0)
		return {"ok": True, "stopped": True, "move": res}

	def status(self) -> dict[str, Any]:
		return {
			"ok": True,
			"dry_run": bool(self.settings.dry_run),
			"estop": bool(self._estop_on),
			"move_base_url": self.settings.move_base_url,
			"proximity_base_url": self.settings.proximity_base_url,
			"obstacle_threshold_cm": float(self.settings.obstacle_threshold_cm),
			"last_check": self._last_check,
		}

	@staticmethod
	def from_config_dict(cfg: dict[str, Any]) -> "SafetyController":
		s_cfg = (cfg or {}).get("safety", {}) if isinstance(cfg, dict) else {}

		def _get_bool(key: str, default: bool) -> bool:
			if not isinstance(s_cfg, dict) or key not in s_cfg:
				return default
			val = s_cfg.get(key)
			if isinstance(val, str):
				v = val.strip().lower()
				if v in {"1", "true", "yes", "y", "on"}:
					return True
				if v in {"0", "false", "no", "n", "off"}:
					return False
			return bool(val)

		def _get_float(key: str, default: float) -> float:
			if not isinstance(s_cfg, dict) or key not in s_cfg:
				return default
			try:
				return float(s_cfg.get(key))
			except Exception:
				return default

		def _get_str(key: str, default: str) -> str:
			if not isinstance(s_cfg, dict) or key not in s_cfg:
				return default
			val = s_cfg.get(key)
			return default if val is None else str(val)

		dry_run_cfg = _get_bool("dry_run", False)
		if str(os.getenv("SAFETY_DRY_RUN") or "").strip().lower() in {"1", "true", "yes", "y", "on"}:
			dry_run_cfg = True

		settings = SafetySettings(
			dry_run=dry_run_cfg,
			estop_default_on=_get_bool("estop_default_on", False),
			move_base_url=_get_str("move_base_url", "http://127.0.0.1:8005"),
			proximity_base_url=_get_str("proximity_base_url", "http://127.0.0.1:8007"),
			obstacle_threshold_cm=max(1.0, min(300.0, _get_float("obstacle_threshold_cm", 35.0))),
		)
		return SafetyController(settings)
