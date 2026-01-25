from __future__ import annotations

import os
import time
from dataclasses import dataclass
from typing import Any


class RobotError(RuntimeError):
	pass


@dataclass(frozen=True)
class RobotSettings:
	dry_run: bool = False
	max_steer_deg: int = 35
	max_speed: int = 100
	# Per-motor trim (software offset) to compensate mismatched motors.
	# Applied to the *magnitude* of each motor command (preserves sign).
	left_speed_offset: int = 0
	right_speed_offset: int = 0
	max_pan_deg: int = 35
	max_tilt_deg: int = 35


class RobotController:
	"""Single-process owner of PiCar-X hardware (GPIO).

	Other services should *not* instantiate `picarx.Picarx()` directly; they should
	proxy to this controller over HTTP.
	"""

	def __init__(self, settings: RobotSettings):
		self.settings = settings
		self._available_reason: str | None = None
		self._px: Any | None = None
		self._last_cmd: dict[str, Any] | None = None
		self._last_cmd_ts: float | None = None
		self._pan: int = 0
		self._tilt: int = 0
		self._last_distance_cm: float | None = None
		self._last_distance_ts: float | None = None

		if self.settings.dry_run:
			return

		try:
			from picarx import Picarx  # type: ignore
		except Exception as exc:
			self._available_reason = f"picarx not available: {exc}"
			return

		try:
			self._px = Picarx()
			self._available_reason = None
		except Exception as exc:
			self._available_reason = f"Picarx init failed: {exc}"
			self._px = None

	@property
	def is_available(self) -> bool:
		return self._available_reason is None

	@property
	def unavailable_reason(self) -> str | None:
		return self._available_reason

	@property
	def pan_deg(self) -> int:
		return int(self._pan)

	@property
	def tilt_deg(self) -> int:
		return int(self._tilt)

	def _clamp(self, v: int, *, max_abs: int) -> int:
		v = int(v)
		m = max(1, int(max_abs))
		return max(-m, min(m, v))

	def _apply_offset(self, cmd: int, *, offset: int) -> int:
		"""Apply per-motor trim while preserving direction.

		Examples:
		- cmd=10, offset=5  -> 15
		- cmd=-10, offset=5 -> -15
		- cmd=0, offset=5   -> 0
		"""
		c = int(cmd)
		o = int(offset)
		if c == 0 or o == 0:
			return c
		return c + (o if c > 0 else -o)

	def _compute_motor_commands(self, *, speed: int, steer_deg: int) -> tuple[int, int, int]:
		"""Return (steer_used_deg, left_cmd, right_cmd).

		Motor sign conventions follow the upstream `picarx.Picarx` implementation:
		- forward(): left motor positive, right motor negative
		- backward(): left motor negative, right motor positive
		"""
		# Picar-X constrains steering to +/-30 internally.
		steer_used = self._clamp(int(steer_deg), max_abs=min(30, int(self.settings.max_steer_deg)))
		max_spd = max(1, int(self.settings.max_speed))
		spd = max(-max_spd, min(max_spd, int(speed)))
		base = abs(spd)

		if base == 0:
			return steer_used, 0, 0

		# Replicates `picarx.Picarx.forward/backward` power scaling based on steering.
		abs_angle = abs(steer_used)
		if abs_angle > 30:
			abs_angle = 30
		power_scale = (100 - abs_angle) / 100.0
		scaled = int(base * power_scale)

		if spd >= 0:
			# forward
			if steer_used != 0:
				if steer_used > 0:
					left_cmd, right_cmd = scaled, -base
				else:
					left_cmd, right_cmd = base, -scaled
			else:
				left_cmd, right_cmd = base, -base
		else:
			# backward
			if steer_used != 0:
				if steer_used > 0:
					left_cmd, right_cmd = -base, scaled
				else:
					left_cmd, right_cmd = -scaled, base
			else:
				left_cmd, right_cmd = -base, base

		# Apply trims and clamp to configured speed limits.
		left_cmd = self._apply_offset(left_cmd, offset=self.settings.left_speed_offset)
		right_cmd = self._apply_offset(right_cmd, offset=self.settings.right_speed_offset)
		left_cmd = max(-max_spd, min(max_spd, int(left_cmd)))
		right_cmd = max(-max_spd, min(max_spd, int(right_cmd)))

		return steer_used, int(left_cmd), int(right_cmd)

	def status(self) -> dict[str, Any]:
		return {
			"ok": True,
			"available": bool(self.settings.dry_run) or self.is_available,
			"unavailable_reason": self._available_reason,
			"dry_run": bool(self.settings.dry_run),
			"pan_deg": int(self._pan),
			"tilt_deg": int(self._tilt),
			"last_cmd": self._last_cmd,
			"last_cmd_ts": self._last_cmd_ts,
			"last_distance_cm": self._last_distance_cm,
			"last_distance_ts": self._last_distance_ts,
		}

	def drive(self, *, speed: int, steer_deg: int = 0) -> None:
		steer_used, left_cmd, right_cmd = self._compute_motor_commands(speed=int(speed), steer_deg=int(steer_deg))
		# Preserve the user-facing speed for observability.
		max_spd = max(1, int(self.settings.max_speed))
		spd = max(-max_spd, min(max_spd, int(speed)))
		self._last_cmd = {
			"cmd": "drive",
			"speed": int(spd),
			"steer_deg": int(steer_used),
			"motor_left": int(left_cmd),
			"motor_right": int(right_cmd),
			"motor_trim_left": int(self.settings.left_speed_offset),
			"motor_trim_right": int(self.settings.right_speed_offset),
		}
		self._last_cmd_ts = time.time()

		if self.settings.dry_run:
			return
		if not self.is_available or self._px is None:
			raise RobotError(self._available_reason or "Robot unavailable")

		try:
			self._px.set_dir_servo_angle(int(steer_used))
			# Set individual motor speeds so we can apply per-motor trim.
			self._px.set_motor_speed(1, int(left_cmd))
			self._px.set_motor_speed(2, int(right_cmd))
		except Exception as exc:
			raise RobotError(f"Drive failed: {exc}") from exc

	def stop(self) -> None:
		self._last_cmd = {"cmd": "stop"}
		self._last_cmd_ts = time.time()
		if self.settings.dry_run:
			return
		if not self.is_available or self._px is None:
			return
		try:
			self._px.stop()
			try:
				self._px.set_dir_servo_angle(0)
			except Exception:
				pass
		except Exception:
			return

	def set_head_angles(self, *, pan_deg: int | None = None, tilt_deg: int | None = None) -> None:
		pan = self._pan if pan_deg is None else self._clamp(int(pan_deg), max_abs=self.settings.max_pan_deg)
		tilt = self._tilt if tilt_deg is None else self._clamp(int(tilt_deg), max_abs=self.settings.max_tilt_deg)
		self._pan, self._tilt = int(pan), int(tilt)
		self._last_cmd = {"cmd": "head_set_angles", "pan_deg": self._pan, "tilt_deg": self._tilt}
		self._last_cmd_ts = time.time()

		if self.settings.dry_run:
			return
		if not self.is_available or self._px is None:
			raise RobotError(self._available_reason or "Robot unavailable")

		try:
			self._px.set_cam_pan_angle(self._pan)
			self._px.set_cam_tilt_angle(self._tilt)
		except Exception as exc:
			raise RobotError(f"Head move failed: {exc}") from exc

	def center_head(self) -> None:
		self.set_head_angles(pan_deg=0, tilt_deg=0)

	def read_distance_cm(self) -> float | None:
		if self.settings.dry_run:
			self._last_distance_cm = 100.0
			self._last_distance_ts = time.time()
			return float(self._last_distance_cm)

		if not self.is_available or self._px is None:
			return None

		try:
			if not hasattr(self._px, "ultrasonic"):
				return None
			d = float(self._px.ultrasonic.read())
			if d <= 0:
				return None
			self._last_distance_cm = d
			self._last_distance_ts = time.time()
			return d
		except Exception:
			return None

	@staticmethod
	def from_config_dict(cfg: dict[str, Any]) -> "RobotController":
		r_cfg = (cfg or {}).get("robot", {}) if isinstance(cfg, dict) else {}

		def _get_bool(key: str, default: bool) -> bool:
			if not isinstance(r_cfg, dict) or key not in r_cfg:
				return default
			val = r_cfg.get(key)
			if isinstance(val, str):
				v = val.strip().lower()
				if v in {"1", "true", "yes", "y", "on"}:
					return True
				if v in {"0", "false", "no", "n", "off"}:
					return False
			return bool(val)

		def _get_int(key: str, default: int) -> int:
			if not isinstance(r_cfg, dict) or key not in r_cfg:
				return default
			try:
				return int(r_cfg.get(key))
			except Exception:
				return default

		dry_run_cfg = _get_bool("dry_run", False)
		if str(os.getenv("ROBOT_DRY_RUN") or "").strip().lower() in {"1", "true", "yes", "y", "on"}:
			dry_run_cfg = True

		settings = RobotSettings(
			dry_run=dry_run_cfg,
			max_steer_deg=max(5, min(45, _get_int("max_steer_deg", 35))),
			max_speed=max(10, min(100, _get_int("max_speed", 100))),
			left_speed_offset=max(-100, min(100, _get_int("left_speed_offset", 0))),
			right_speed_offset=max(-100, min(100, _get_int("right_speed_offset", 0))),
			max_pan_deg=max(5, min(45, _get_int("max_pan_deg", 35))),
			max_tilt_deg=max(5, min(45, _get_int("max_tilt_deg", 35))),
		)
		return RobotController(settings)
