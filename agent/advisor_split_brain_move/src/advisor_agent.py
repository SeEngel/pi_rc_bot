from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass
from typing import Any
from urllib.parse import urlparse

from agent.advisor.src.advisor_agent import AdvisorAgent as LegacyAdvisorAgent
from agent.advisor.src.mcp_client import call_mcp_tool_json
from agent.src.config import load_yaml


@dataclass(frozen=True)
class SplitBrainAdvisorSettings:
	"""Proxy settings that extends legacy AdvisorSettings with move_advisor endpoint."""

	base: Any
	move_advisor_mcp_url: str
	move_advisor_preflight: bool = True

	def __getattr__(self, name: str) -> Any:  # pragma: no cover
		return getattr(self.base, name)


class SplitBrainAdvisorAgent(LegacyAdvisorAgent):
	"""AdvisorAgent variant that delegates motion/safety/proximity/perception via move_advisor."""

	@staticmethod
	async def _tcp_preflight(url: str, *, timeout_seconds: float = 0.5) -> tuple[bool, str | None]:
		"""Best-effort connectivity check.

		This intentionally avoids calling the MCP client stack when the target is down,
		because some MCP/anyio versions can produce noisy shutdown errors on failed connects.
		"""
		try:
			p = urlparse(str(url))
			host = p.hostname
			if not host:
				return (False, "invalid_url")
			if p.port is not None:
				port = int(p.port)
			else:
				port = 443 if (p.scheme or "").lower() == "https" else 80
			reader, writer = await asyncio.wait_for(asyncio.open_connection(host, port), timeout=float(timeout_seconds))
			try:
				_ = reader  # silence lint
				return (True, None)
			finally:
				writer.close()
				try:
					await writer.wait_closed()
				except Exception:
					pass
		except Exception as exc:
			return (False, str(exc))

	@classmethod
	def settings_from_config_yaml(cls, path: str) -> SplitBrainAdvisorSettings:
		base = LegacyAdvisorAgent.settings_from_config_yaml(path)
		cfg = load_yaml(path)
		mcp_cfg = cfg.get("mcp", {}) if isinstance(cfg, dict) else {}
		move_adv = str(mcp_cfg.get("move_advisor_mcp_url") or "http://127.0.0.1:8611/mcp").strip()
		if not move_adv:
			move_adv = "http://127.0.0.1:8611/mcp"
		preflight = bool(mcp_cfg.get("move_advisor_preflight") if "move_advisor_preflight" in mcp_cfg else True)
		return SplitBrainAdvisorSettings(base=base, move_advisor_mcp_url=move_adv, move_advisor_preflight=preflight)

	@classmethod
	def from_config_yaml(cls, path: str) -> "SplitBrainAdvisorAgent":
		settings = cls.settings_from_config_yaml(path)
		return cls(settings)

	async def _move_advisor_execute_action(
		self,
		action: dict[str, Any],
		*,
		background: bool = False,
		timeout_seconds: float = 30.0,
		request_id: str | None = None,
	) -> dict[str, Any]:
		if self._dry_run:
			self._emit(
				"tool_call",
				tool="execute_action",
				component="mcp.move_advisor",
				dry_run=True,
				action=action,
				background=bool(background),
				request_id=request_id,
			)
			return {"ok": True, "dry_run": True, "result": {"ok": True, "dry_run": True, **(action or {})}}

		start = time.perf_counter()
		self._emit(
			"tool_call_start",
			tool="execute_action",
			component="mcp.move_advisor",
			url=self.settings.move_advisor_mcp_url,
			background=bool(background),
			request_id=request_id,
			action=action,
		)

		if bool(getattr(self.settings, "move_advisor_preflight", True)):
			ok, err = await self._tcp_preflight(self.settings.move_advisor_mcp_url, timeout_seconds=0.6)
			if not ok:
				dur_ms = (time.perf_counter() - start) * 1000.0
				self._emit(
					"tool_call_error",
					tool="execute_action",
					component="mcp.move_advisor",
					url=self.settings.move_advisor_mcp_url,
					duration_ms=round(dur_ms, 2),
					error=f"connect_preflight_failed: {err}",
				)
				return {
					"ok": False,
					"request_id": request_id,
					"error": f"connect_preflight_failed: {err}",
					"result": None,
				}
		try:
			res = await call_mcp_tool_json(
				url=self.settings.move_advisor_mcp_url,
				tool_name="execute_action",
				timeout_seconds=float(timeout_seconds),
				action=dict(action or {}),
				background=bool(background),
				request_id=request_id,
			)
		except Exception as exc:
			dur_ms = (time.perf_counter() - start) * 1000.0
			self._emit(
				"tool_call_error",
				tool="execute_action",
				component="mcp.move_advisor",
				url=self.settings.move_advisor_mcp_url,
				duration_ms=round(dur_ms, 2),
				error=str(exc),
			)
			# IMPORTANT: never crash the advisor loop due to a missing move cluster.
			# Return a structured error so callers can handle it gracefully.
			return {
				"ok": False,
				"request_id": request_id,
				"error": str(exc),
				"result": None,
			}
		dur_ms = (time.perf_counter() - start) * 1000.0
		self._emit(
			"tool_call_end",
			tool="execute_action",
			component="mcp.move_advisor",
			url=self.settings.move_advisor_mcp_url,
			duration_ms=round(dur_ms, 2),
			result_keys=sorted([str(k) for k in res.keys()]) if isinstance(res, dict) else None,
		)
		return res if isinstance(res, dict) else {"ok": True, "value": res}

	@staticmethod
	def _unwrap_action_result(res: dict[str, Any] | None) -> dict[str, Any] | None:
		if not isinstance(res, dict):
			return None
		inner = res.get("result")
		if isinstance(inner, dict):
			return inner
		# Some callers may return already-unwrapped payloads.
		return res

	async def _proximity_distance_cm(self) -> float | None:
		if self._dry_run:
			return 100.0
		res = await self._move_advisor_execute_action({"type": "proximity_distance"}, timeout_seconds=10.0)
		inner = self._unwrap_action_result(res)
		if not isinstance(inner, dict):
			return None
		try:
			v = inner.get("distance_cm")
			return float(v) if v is not None else None
		except Exception:
			return None

	async def _perception_detect(self) -> dict[str, Any] | None:
		if self._dry_run:
			return {"ok": True, "available": True, "dry_run": True, "faces": [], "people": []}
		res = await self._move_advisor_execute_action({"type": "perception_detect"}, timeout_seconds=30.0)
		inner = self._unwrap_action_result(res)
		return inner if isinstance(inner, dict) else None

	async def _safety_stop(self) -> None:
		if self._dry_run:
			self._emit("tool_call", tool="stop", component="mcp.move_advisor", dry_run=True)
			return
		try:
			await self._move_advisor_execute_action({"type": "stop"}, timeout_seconds=10.0)
		except Exception as exc:  # pragma: no cover
			# Best-effort only.
			self._emit("tool_call_error", tool="stop", component="mcp.move_advisor", error=str(exc))

	async def _safety_estop_on(self) -> None:
		if self._dry_run:
			self._emit("tool_call", tool="estop_on", component="mcp.move_advisor", dry_run=True)
			return
		try:
			await self._move_advisor_execute_action({"type": "estop_on"}, timeout_seconds=10.0)
		except Exception as exc:  # pragma: no cover
			self._emit("tool_call_error", tool="estop_on", component="mcp.move_advisor", error=str(exc))

	async def _safety_estop_off(self) -> None:
		if self._dry_run:
			self._emit("tool_call", tool="estop_off", component="mcp.move_advisor", dry_run=True)
			return
		try:
			await self._move_advisor_execute_action({"type": "estop_off"}, timeout_seconds=10.0)
		except Exception as exc:  # pragma: no cover
			self._emit("tool_call_error", tool="estop_off", component="mcp.move_advisor", error=str(exc))

	async def _safety_guarded_drive(
		self,
		*,
		speed: int,
		steer_deg: int = 0,
		duration_s: float | None = None,
		threshold_cm: float | None = None,
		await_completion: bool = False,
	) -> dict[str, Any] | None:
		if self._dry_run:
			self._emit(
				"tool_call",
				tool="guarded_drive",
				component="mcp.move_advisor",
				dry_run=True,
				speed=int(speed),
				steer_deg=int(steer_deg),
				duration_s=duration_s,
				threshold_cm=threshold_cm,
			)
			return {"ok": True, "dry_run": True}

		action: dict[str, Any] = {"type": "guarded_drive", "speed": int(speed), "steer_deg": int(steer_deg)}
		if duration_s is not None:
			action["duration_s"] = float(duration_s)
		if threshold_cm is not None:
			action["threshold_cm"] = float(threshold_cm)

		res = await self._move_advisor_execute_action(action, timeout_seconds=30.0)
		if not isinstance(res, dict) or not bool(res.get("ok")):
			return None
		inner = self._unwrap_action_result(res)
		if not isinstance(inner, dict):
			return None
		# If the downstream action itself says ok=false, return it so callers may inspect,
		# but keep legacy command handling in mind (it treats None as service error).
		if "ok" in inner and not bool(inner.get("ok")):
			return inner

		# Wait for motion to actually complete if requested
		# The move service spawns a subprocess and returns immediately,
		# so we need to explicitly wait for the duration here.
		if await_completion and duration_s is not None and duration_s > 0:
			await asyncio.sleep(duration_s + 0.1)

		return inner
