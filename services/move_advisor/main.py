from __future__ import annotations

import asyncio
import json
import time
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

import yaml

FASTAPI_AVAILABLE = False

try:
	from fastapi import Body, FastAPI, HTTPException
	from pydantic import BaseModel, Field
	from fastmcp import FastMCP

	FASTAPI_AVAILABLE = True
except Exception:  # pragma: no cover
	# Minimal stubs so the module can be imported even if service deps aren't installed.
	class FastAPI:  # type: ignore[no-redef]
		def __init__(self, *args: Any, **kwargs: Any):
			pass

		def get(self, *args: Any, **kwargs: Any):
			def _decorator(fn):
				return fn
			return _decorator

		def post(self, *args: Any, **kwargs: Any):
			def _decorator(fn):
				return fn
			return _decorator

	class HTTPException(Exception):  # type: ignore[no-redef]
		def __init__(self, status_code: int, detail: str):
			super().__init__(f"{status_code}: {detail}")

	def Body(*args: Any, **kwargs: Any) -> Any:  # type: ignore[no-redef]
		return None

	class BaseModel:  # type: ignore[no-redef]
		pass

	def Field(*args: Any, **kwargs: Any) -> Any:  # type: ignore[no-redef]
		return None

	FastMCP = None  # type: ignore[assignment]


def _now_ts() -> float:
	return time.time()


def load_config() -> dict[str, Any]:
	cfg_path = Path(__file__).with_name("config.yaml")
	if not cfg_path.exists():
		return {}
	try:
		with cfg_path.open("r", encoding="utf-8") as f:
			data = yaml.safe_load(f) or {}
			return data if isinstance(data, dict) else {}
	except Exception:
		return {}


_cfg = load_config()
_service_cfg = _cfg.get("service", {}) if isinstance(_cfg, dict) else {}
HOST = str(_service_cfg.get("host", "0.0.0.0"))
PORT = int(_service_cfg.get("port", 8011))
MCP_PORT = PORT + 600

_mcp_cfg = _cfg.get("mcp", {}) if isinstance(_cfg, dict) else {}
SAFETY_MCP_URL = str(_mcp_cfg.get("safety_mcp_url", "http://127.0.0.1:8609/mcp"))
PROXIMITY_MCP_URL = str(_mcp_cfg.get("proximity_mcp_url", "http://127.0.0.1:8607/mcp"))
PERCEPTION_MCP_URL = str(_mcp_cfg.get("perception_mcp_url", "http://127.0.0.1:8608/mcp"))
MOVE_MCP_URL = str(_mcp_cfg.get("move_mcp_url", "http://127.0.0.1:8605/mcp"))

_ma_cfg = _cfg.get("move_advisor", {}) if isinstance(_cfg, dict) else {}
DRY_RUN = bool(_ma_cfg.get("dry_run", False))
MAX_SPEED = int(_ma_cfg.get("max_speed", 100))
MAX_STEER_DEG = int(_ma_cfg.get("max_steer_deg", 45))
MAX_DURATION_S = float(_ma_cfg.get("max_duration_s", 10.0))
DEFAULT_THRESHOLD_CM = float(_ma_cfg.get("default_threshold_cm", 35.0))
MAX_JOBS = int(_ma_cfg.get("max_jobs", 50))
JOB_TTL_SECONDS = float(_ma_cfg.get("job_ttl_seconds", 600.0))


async def _call_mcp_tool_json(*, url: str, tool_name: str, timeout_seconds: float = 30.0, **kwargs: Any) -> dict[str, Any]:
	"""Call a downstream tool.

	This service is launched via `services/main.sh` using global `/usr/bin/python3`.
	On many robots, `agent_framework` is only present in a dev/uv environment.
	So for *service-to-service* calls we avoid an MCP client dependency and call the
	downstream FastAPI HTTP endpoints directly.

	The configuration provides MCP URLs like `http://127.0.0.1:8609/mcp`.
	Across this repo, MCP is served on `PORT + 600`, so we can derive the HTTP port
	by subtracting 600.
	"""

	def _derive_http_base(mcp_url: str) -> str:
		p = urlparse(str(mcp_url))
		scheme = p.scheme or "http"
		host = p.hostname or "127.0.0.1"
		port = p.port
		# Convention: MCP_PORT = PORT + 600.
		if port is not None and port >= 600:
			port = port - 600
		if ":" in host and not host.startswith("["):
			host = f"[{host}]"  # IPv6
		return f"{scheme}://{host}:{port}" if port is not None else f"{scheme}://{host}"

	async def _http_json(
		*,
		method: str,
		base: str,
		path: str,
		params: dict[str, Any] | None = None,
		body: dict[str, Any] | None = None,
	) -> dict[str, Any]:
		# Prefer httpx if available; fall back to urllib (sync in a thread).
		try:
			import httpx  # type: ignore
		except Exception:
			httpx = None  # type: ignore

		url_full = f"{base}{path}"
		if httpx is not None:
			try:
				async with httpx.AsyncClient(timeout=float(timeout_seconds)) as client:
					resp = await client.request(method.upper(), url_full, params=params, json=body)
					resp.raise_for_status()
					data = resp.json()
					return data if isinstance(data, dict) else {"ok": True, "value": data}
			except Exception as exc:
				return {"ok": False, "error": str(exc), "where": url_full}

		# stdlib fallback (best-effort)
		import urllib.parse
		import urllib.request

		def _do() -> dict[str, Any]:
			try:
				u = url_full
				if params:
					u = f"{u}?{urllib.parse.urlencode(params)}"
				data_bytes = None
				headers = {"Accept": "application/json"}
				if body is not None:
					data_bytes = json.dumps(body).encode("utf-8")
					headers["Content-Type"] = "application/json"
				req = urllib.request.Request(u, data=data_bytes, headers=headers, method=method.upper())
				with urllib.request.urlopen(req, timeout=float(timeout_seconds)) as resp:  # nosec - local HTTP only
					payload = resp.read().decode("utf-8", errors="replace")
					try:
						obj = json.loads(payload)
						return obj if isinstance(obj, dict) else {"ok": True, "value": obj}
					except Exception:
						return {"ok": True, "text": payload}
			except Exception as exc:
				return {"ok": False, "error": str(exc), "where": url_full}

		return await asyncio.to_thread(_do)

	base = _derive_http_base(url)
	name = str(tool_name).strip()

	# Map operation_id (FastMCP tool name) -> HTTP endpoint.
	if name == "healthz_healthz_get":
		return await _http_json(method="GET", base=base, path="/healthz")
	if name == "status":
		return await _http_json(method="GET", base=base, path="/status")
	if name == "distance_cm":
		return await _http_json(method="GET", base=base, path="/distance")
	if name == "detect":
		return await _http_json(method="GET", base=base, path="/detect")
	if name == "check":
		return await _http_json(method="GET", base=base, path="/check", params=dict(kwargs))
	if name == "guarded_drive":
		return await _http_json(method="POST", base=base, path="/guarded_drive", body=dict(kwargs))
	if name == "drive":
		return await _http_json(method="POST", base=base, path="/drive", body=dict(kwargs))
	if name == "stop":
		return await _http_json(method="POST", base=base, path="/stop")
	if name == "estop_on":
		return await _http_json(method="POST", base=base, path="/estop/on")
	if name == "estop_off":
		return await _http_json(method="POST", base=base, path="/estop/off")

	return {"ok": False, "error": f"unknown downstream tool_name: {name}", "base": base}


def _clamp_int(v: Any, lo: int, hi: int, *, default: int) -> int:
	try:
		i = int(v)
	except Exception:
		i = int(default)
	return max(int(lo), min(int(hi), i))


def _clamp_float(v: Any, lo: float, hi: float, *, default: float) -> float:
	try:
		f = float(v)
	except Exception:
		f = float(default)
	return max(float(lo), min(float(hi), f))


class ServiceInfo(BaseModel):
	host: str
	port: int
	mcp_port: int


class HealthzResponse(BaseModel):
	ok: bool = True
	service: ServiceInfo
	downstream: dict[str, Any] = Field(default_factory=dict)
	dry_run: bool = False


class ActionRequest(BaseModel):
	"""Request to execute an action through the move advisor.

	The move advisor acts as a high-level coordinator that dispatches actions
	to downstream services (safety, proximity, perception, move).

	Supported action types:
	- stop, safety_stop, halt: Stop all motion
	- estop_on, e_stop, emergency_stop: Engage emergency stop
	- estop_off, release_estop: Release emergency stop
	- check, safety_check: Check if it's safe to drive
	- perception_detect, detect: Run perception/object detection
	- proximity_distance, distance_cm: Read ultrasonic distance
	- guarded_drive, drive, move: Drive with safety checks
	- move_drive_direct: Drive bypassing safety (use with caution)
	"""
	action: dict[str, Any] = Field(
		description="Action payload. Must include a 'type' key specifying the action. Additional keys depend on action type.",
		examples=[
			{"type": "guarded_drive", "speed": 30, "steer_deg": 0, "duration_s": 1.0},
			{"type": "stop"},
			{"type": "check", "threshold_cm": 35.0},
			{"type": "estop_on"},
		],
	)
	background: bool = Field(
		default=False,
		description="If True, execute as a background job and return job_id immediately. Use /job_status to poll for completion.",
	)
	request_id: str | None = Field(
		default=None,
		description="Optional caller-supplied ID for correlation/tracking.",
		examples=["req-123", "move-001"],
	)


class ActionResponse(BaseModel):
	"""Response from /execute_action."""
	ok: bool = Field(default=True, description="Whether the operation completed without errors.")
	request_id: str | None = Field(default=None, description="Echo of the request_id if provided.")
	job_id: str | None = Field(default=None, description="Job ID if background=True. Use with /job_status.", examples=["abc123def456"])
	result: dict[str, Any] | None = Field(default=None, description="Action result if executed synchronously.")
	error: str | None = Field(default=None, description="Error message if the action failed.")


class JobStatusResponse(BaseModel):
	"""Response from /job_status with background job information."""
	ok: bool = Field(default=True, description="Whether the query completed without errors.")
	job_id: str = Field(description="The job ID being queried.")
	state: str = Field(
		description="Current job state: queued, running, done, error, or canceled.",
		examples=["queued", "running", "done", "error", "canceled"],
	)
	created_ts: float = Field(description="Unix timestamp when the job was created.")
	started_ts: float | None = Field(default=None, description="Unix timestamp when execution started.")
	finished_ts: float | None = Field(default=None, description="Unix timestamp when execution finished.")
	result: dict[str, Any] | None = Field(default=None, description="Job result if completed.")
	error: str | None = Field(default=None, description="Error message if the job failed.")


@dataclass
class _Job:
	job_id: str
	created_ts: float
	state: str
	request_id: str | None
	action: dict[str, Any]
	started_ts: float | None = None
	finished_ts: float | None = None
	result: dict[str, Any] | None = None
	error: str | None = None
	cancel_event: asyncio.Event | None = None


_jobs_lock = asyncio.Lock()
_jobs: dict[str, _Job] = {}


async def _prune_jobs() -> None:
	"""Keep the job table bounded."""
	now = _now_ts()
	async with _jobs_lock:
		# Drop expired
		expired = [jid for jid, j in _jobs.items() if (now - j.created_ts) > JOB_TTL_SECONDS]
		for jid in expired:
			_jobs.pop(jid, None)
		# Hard cap (drop oldest)
		if len(_jobs) <= MAX_JOBS:
			return
		ordered = sorted(_jobs.values(), key=lambda j: j.created_ts)
		for j in ordered[: max(0, len(_jobs) - MAX_JOBS)]:
			_jobs.pop(j.job_id, None)


async def _execute_action(action: dict[str, Any], *, cancel_event: asyncio.Event | None = None) -> dict[str, Any]:
	"""Dispatch an action to downstream services.

	This is deliberately conservative: unknown actions return ok=false.
	"""
	atype = str(action.get("type") or action.get("action") or "").strip().lower()
	if not atype:
		return {"ok": False, "error": "missing action.type"}

	# Allow cancel checks in longer loops (future expansion).
	def _canceled() -> bool:
		return bool(cancel_event and cancel_event.is_set())

	if atype in {"stop", "safety_stop", "halt"}:
		if DRY_RUN:
			return {"ok": True, "dry_run": True, "type": atype}
		return await _call_mcp_tool_json(url=SAFETY_MCP_URL, tool_name="stop", timeout_seconds=10.0)

	if atype in {"estop_on", "e_stop", "emergency_stop"}:
		if DRY_RUN:
			return {"ok": True, "dry_run": True, "type": atype, "estop": True}
		# Engage estop and stop motion best-effort.
		res1 = await _call_mcp_tool_json(url=SAFETY_MCP_URL, tool_name="estop_on", timeout_seconds=10.0)
		res2 = await _call_mcp_tool_json(url=SAFETY_MCP_URL, tool_name="stop", timeout_seconds=10.0)
		return {"ok": True, "estop_on": res1, "stop": res2}

	if atype in {"estop_off", "release_estop"}:
		if DRY_RUN:
			return {"ok": True, "dry_run": True, "type": atype, "estop": False}
		return await _call_mcp_tool_json(url=SAFETY_MCP_URL, tool_name="estop_off", timeout_seconds=10.0)

	if atype in {"check", "safety_check"}:
		thr = _clamp_float(action.get("threshold_cm"), 5.0, 150.0, default=DEFAULT_THRESHOLD_CM)
		if DRY_RUN:
			return {"ok": True, "dry_run": True, "safe": True, "blocked": False, "threshold_cm": thr}
		return await _call_mcp_tool_json(url=SAFETY_MCP_URL, tool_name="check", timeout_seconds=10.0, threshold_cm=thr)

	if atype in {"perception_detect", "detect"}:
		if DRY_RUN:
			return {"ok": True, "dry_run": True, "faces": [], "people": []}
		return await _call_mcp_tool_json(url=PERCEPTION_MCP_URL, tool_name="detect", timeout_seconds=30.0)

	if atype in {"proximity_distance", "distance_cm"}:
		if DRY_RUN:
			return {"ok": True, "dry_run": True, "distance_cm": None}
		return await _call_mcp_tool_json(url=PROXIMITY_MCP_URL, tool_name="distance_cm", timeout_seconds=10.0)

	if atype in {"guarded_drive", "drive", "move"}:
		speed = _clamp_int(action.get("speed") if action.get("speed") is not None else action.get("speed_pct"), -MAX_SPEED, MAX_SPEED, default=25)
		steer = _clamp_int(action.get("steer_deg"), -MAX_STEER_DEG, MAX_STEER_DEG, default=0)
		duration_s = _clamp_float(action.get("duration_s"), 0.1, MAX_DURATION_S, default=0.7)
		thr = _clamp_float(action.get("threshold_cm"), 5.0, 150.0, default=DEFAULT_THRESHOLD_CM)
		if _canceled():
			return {"ok": False, "canceled": True}
		if DRY_RUN:
			return {
				"ok": True,
				"dry_run": True,
				"speed": speed,
				"steer_deg": steer,
				"duration_s": duration_s,
				"threshold_cm": thr,
			}
		# Prefer safety controller.
		return await _call_mcp_tool_json(
			url=SAFETY_MCP_URL,
			tool_name="guarded_drive",
			timeout_seconds=30.0,
			speed=speed,
			steer_deg=steer,
			duration_s=duration_s,
			threshold_cm=thr,
		)

	# Backdoor: allow direct move service calls if explicitly asked.
	if atype in {"move_drive_direct"}:
		speed = _clamp_int(action.get("speed"), -MAX_SPEED, MAX_SPEED, default=25)
		steer = _clamp_int(action.get("steer_deg"), -MAX_STEER_DEG, MAX_STEER_DEG, default=0)
		duration_s = action.get("duration_s")
		payload: dict[str, Any] = {"speed": speed, "steer_deg": steer}
		if duration_s is not None:
			payload["duration_s"] = _clamp_float(duration_s, 0.0, MAX_DURATION_S, default=0.7)
		if DRY_RUN:
			return {"ok": True, "dry_run": True, **payload}
		return await _call_mcp_tool_json(url=MOVE_MCP_URL, tool_name="drive", timeout_seconds=30.0, **payload)

	return {"ok": False, "error": f"unknown action type: {atype}"}


app = FastAPI(title="pi_rc_bot move_advisor", version="0.1.0")


@app.get(
	"/healthz",
	operation_id="healthz_healthz_get",
	summary="Health check",
	description="Returns service health and connectivity status to downstream services (safety, proximity, perception, move).",
	response_model=HealthzResponse,
)
async def healthz() -> dict[str, Any]:
	# Best-effort downstream pings (fast; failure is non-fatal).
	down: dict[str, Any] = {}
	if DRY_RUN:
		down = {"dry_run": True}
	else:
		for name, url in (
			("safety", SAFETY_MCP_URL),
			("proximity", PROXIMITY_MCP_URL),
			("perception", PERCEPTION_MCP_URL),
			("move", MOVE_MCP_URL),
		):
			try:
				down[name] = await _call_mcp_tool_json(url=url, tool_name="healthz_healthz_get", timeout_seconds=5.0)
			except Exception as exc:
				down[name] = {"ok": False, "error": str(exc)}
	return {
		"ok": True,
		"dry_run": DRY_RUN,
		"service": {"host": HOST, "port": PORT, "mcp_port": MCP_PORT},
		"downstream": down,
	}


@app.post(
	"/execute_action",
	operation_id="execute_action",
	summary="Execute an action (sync or background)",
	description=(
		"Execute a high-level action through the move advisor. Actions are dispatched to appropriate downstream services.\n\n"
		"**Supported action types:**\n"
		"- `stop`, `safety_stop`, `halt`: Stop all motion immediately\n"
		"- `estop_on`, `e_stop`, `emergency_stop`: Engage emergency stop (blocks all motion)\n"
		"- `estop_off`, `release_estop`: Release emergency stop\n"
		"- `check`, `safety_check`: Check if forward motion is safe\n"
		"- `perception_detect`, `detect`: Run object/face detection\n"
		"- `proximity_distance`, `distance_cm`: Read ultrasonic distance sensor\n"
		"- `guarded_drive`, `drive`, `move`: Drive with obstacle safety checks (RECOMMENDED)\n"
		"- `move_drive_direct`: Drive bypassing safety (use with caution)\n\n"
		"Set `background=true` to run asynchronously and poll `/job_status` for results."
	),
	response_model=ActionResponse,
)
async def execute_action(payload: ActionRequest = Body(...)) -> dict[str, Any]:
	action = dict(payload.action or {})
	request_id = payload.request_id

	await _prune_jobs()

	if not payload.background:
		try:
			res = await _execute_action(action)
			ok = bool(res.get("ok")) if isinstance(res, dict) else True
			return {"ok": ok, "request_id": request_id, "result": res}
		except Exception as exc:
			raise HTTPException(status_code=500, detail=str(exc)) from exc

	job_id = uuid.uuid4().hex
	cancel_event = asyncio.Event()
	job = _Job(job_id=job_id, created_ts=_now_ts(), state="queued", request_id=request_id, action=action, cancel_event=cancel_event)

	async with _jobs_lock:
		_jobs[job_id] = job

	async def _runner() -> None:
		async with _jobs_lock:
			j = _jobs.get(job_id)
			if j is None:
				return
			j.state = "running"
			j.started_ts = _now_ts()
		try:
			res = await _execute_action(action, cancel_event=cancel_event)
			async with _jobs_lock:
				j = _jobs.get(job_id)
				if j is None:
					return
				if cancel_event.is_set():
					j.state = "canceled"
					j.finished_ts = _now_ts()
					j.result = {"ok": False, "canceled": True, "result": res}
					return
				ok = True
				if isinstance(res, dict) and "ok" in res:
					ok = bool(res.get("ok"))
				j.state = "done" if ok else "error"
				j.finished_ts = _now_ts()
				j.result = res if isinstance(res, dict) else {"ok": True, "value": res}
		except Exception as exc:
			async with _jobs_lock:
				j = _jobs.get(job_id)
				if j is None:
					return
				j.state = "error"
				j.finished_ts = _now_ts()
				j.error = str(exc)

	asyncio.create_task(_runner())
	return {"ok": True, "request_id": request_id, "job_id": job_id}


@app.get(
	"/job_status",
	operation_id="job_status",
	summary="Get background job status",
	description="Query the status of a background job started with /execute_action (background=true). Returns current state, timestamps, and result when completed.",
	response_model=JobStatusResponse,
)
async def job_status(job_id: str) -> dict[str, Any]:
	await _prune_jobs()
	async with _jobs_lock:
		j = _jobs.get(str(job_id))
		if j is None:
			raise HTTPException(status_code=404, detail="unknown job_id")
		return {
			"ok": True,
			"job_id": j.job_id,
			"state": j.state,
			"created_ts": j.created_ts,
			"started_ts": j.started_ts,
			"finished_ts": j.finished_ts,
			"result": j.result,
			"error": j.error,
		}


@app.post(
	"/job_cancel",
	operation_id="job_cancel",
	summary="Cancel a background job (best-effort)",
	description="Attempt to cancel a running background job. Also stops any motion in progress as a safety measure. The job may not be immediately cancellable depending on its state.",
)
async def job_cancel(job_id: str) -> dict[str, Any]:
	await _prune_jobs()
	async with _jobs_lock:
		j = _jobs.get(str(job_id))
		if j is None:
			raise HTTPException(status_code=404, detail="unknown job_id")
		if j.cancel_event is not None:
			j.cancel_event.set()

	# Always attempt to stop motion.
	try:
		stop_res = await _execute_action({"type": "stop"})
	except Exception as exc:
		stop_res = {"ok": False, "error": str(exc)}

	return {"ok": True, "job_id": str(job_id), "stop": stop_res}


mcp = FastMCP.from_fastapi(app, name="move_advisor") if (FASTAPI_AVAILABLE and FastMCP is not None) else None


if __name__ == "__main__":
	if not FASTAPI_AVAILABLE or mcp is None:
		print("[move_advisor] FastAPI/fastmcp not available")
		raise SystemExit(1)

	import uvicorn
	import threading

	api_thread = threading.Thread(
		target=lambda: uvicorn.run(app, host=HOST, port=PORT, log_level="info"),
		daemon=True,
	)
	api_thread.start()

	# Serve MCP on a separate port (pattern used across this repo).
	uvicorn.run(mcp.http_app(path="/mcp"), host=HOST, port=MCP_PORT, log_level="info")
