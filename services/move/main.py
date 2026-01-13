from __future__ import annotations

import os
import signal
import shutil
import socket
import subprocess
import sys
import threading
import time
from typing import Any

FASTAPI_AVAILABLE = False

try:
	from fastapi import Body, FastAPI, HTTPException
	from pydantic import BaseModel, Field
	import uvicorn

	FASTAPI_AVAILABLE = True
except Exception:  # pragma: no cover
	class Request:  # type: ignore[no-redef]
		pass

	class FastAPI:  # type: ignore[no-redef]
		pass

	class HTTPException(Exception):  # type: ignore[no-redef]
		def __init__(self, status_code: int, detail: str):
			super().__init__(f"{status_code}: {detail}")

	uvicorn = None  # type: ignore[assignment]


if FASTAPI_AVAILABLE:
	class DriveRequest(BaseModel):
		speed: int = Field(description="Signed speed (-100..100). Negative = backward.")
		steer_deg: int = Field(default=0, description="Steering angle in degrees (-35..35).")
		duration_s: float | None = Field(
			default=None,
			description="If provided: run for this duration then stop. If omitted: uses service default.",
		)

	class DriveResponse(BaseModel):
		"""Drive call result.

		If duration_s > 0: a short-lived subprocess job is started (interruptible via /stop).
		If duration_s <= 0: the controller enters continuous drive mode.
		"""
		ok: bool = True
		started: bool = Field(description="True if a subprocess job was started.")
		pid: int | None = Field(default=None, description="PID of the subprocess job (if started).")
		stopped_previous: bool = Field(default=False, description="True if a previous job was stopped first.")
		continuous: bool | None = Field(default=None, description="True if this call entered continuous drive mode.")

	class StopResponse(BaseModel):
		ok: bool = True
		stopped: bool = True
		stopped_job: bool = False

	class StatusResponse(BaseModel):
		model_config = {"extra": "allow"}


def _load_dotenv(path: str) -> None:
	if not os.path.exists(path):
		return
	try:
		with open(path, "r", encoding="utf-8") as f:
			for raw in f:
				line = raw.strip()
				if not line or line.startswith("#") or "=" not in line:
					continue
				k, v = line.split("=", 1)
				k = k.strip()
				if k.startswith("export "):
					k = k[len("export ") :].strip()
				v = v.strip().strip('"').strip("'")
				if not k:
					continue
				existing = os.environ.get(k)
				if existing is None or existing == "":
					os.environ[k] = v
	except Exception:
		return


def _load_yaml(path: str) -> dict[str, Any]:
	try:
		import yaml  # type: ignore
	except Exception as exc:
		raise RuntimeError(
			"PyYAML is required to load config.yaml. Install it with: pip install -r services/move/requirements.txt"
		) from exc

	with open(path, "r", encoding="utf-8") as f:
		data = yaml.safe_load(f) or {}
	if not isinstance(data, dict):
		raise ValueError("config.yaml must contain a YAML mapping at the root")
	return data


def _is_port_in_use(port: int, host: str = "0.0.0.0") -> bool:
	with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
		s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
		try:
			s.bind((host, int(port)))
			return False
		except OSError as exc:
			return getattr(exc, "errno", None) == 98


def _kill_listeners_on_port(port: int) -> bool:
	port = int(port)
	if port <= 0:
		return False

	if shutil.which("fuser") is not None:
		try:
			res = subprocess.run(
				["fuser", "-k", f"{port}/tcp"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL
			)
			return res.returncode == 0
		except Exception:
			return False
	return False


def _ensure_port_free(port: int) -> None:
	if not _is_port_in_use(port):
		return
	print(f"[move] Port {port} is already in use; attempting to stop the existing service...")
	_kill_listeners_on_port(port)
	for _ in range(20):
		if not _is_port_in_use(port):
			return
		time.sleep(0.1)
	raise OSError(98, f"Port {port} still in use")


def main() -> int:
	here = os.path.dirname(os.path.abspath(__file__))
	if not FASTAPI_AVAILABLE or uvicorn is None:
		print("[move] FastAPI/uvicorn not available")
		return 1

	# Load .env from repo root and service folder (optional)
	repo_root = os.path.dirname(os.path.dirname(here))
	_load_dotenv(os.path.join(repo_root, ".env"))
	_load_dotenv(os.path.join(here, ".env"))

	if here not in sys.path:
		sys.path.insert(0, here)

	from contextlib import asynccontextmanager

	from src import MoveController

	cfg_path = os.path.join(here, "config.yaml")
	cfg = _load_yaml(cfg_path)
	service_cfg = (cfg or {}).get("service", {}) if isinstance(cfg, dict) else {}
	host = str((service_cfg or {}).get("host") or "0.0.0.0")
	port = int((service_cfg or {}).get("port") or 8005)

	controller_lock = threading.Lock()
	controller: MoveController | None = None

	current_job: subprocess.Popen[bytes] | None = None
	current_job_started_ts: float | None = None
	current_job_preview: str | None = None

	def _stop_job_only() -> bool:
		nonlocal current_job, current_job_started_ts, current_job_preview
		p = current_job
		if p is None:
			return False
		if p.poll() is not None:
			current_job = None
			current_job_started_ts = None
			current_job_preview = None
			return False
		try:
			p.terminate()
		except Exception:
			pass
		try:
			p.wait(timeout=1.0)
		except Exception:
			try:
				p.kill()
			except Exception:
				pass
			try:
				p.wait(timeout=1.0)
			except Exception:
				pass
		current_job = None
		current_job_started_ts = None
		current_job_preview = None
		return True

	def _init_controller() -> MoveController:
		mc = MoveController.from_config_dict(cfg)
		if not mc.is_available:
			print(f"[move] hardware unavailable: {mc.unavailable_reason}")
			print("[move] Tip: install picar-x system-wide, or set move.dry_run=true")
		return mc

	@asynccontextmanager
	async def lifespan(_):
		nonlocal controller
		controller = _init_controller()
		yield
		try:
			if controller is not None:
				controller.stop()
		except Exception:
			pass

	app = FastAPI(title="pi_rc_bot Move Service", version="1.0.0", lifespan=lifespan)

	@app.get("/healthz", operation_id="healthz_healthz_get")
	async def healthz() -> dict[str, Any]:
		mc = controller
		available = bool(mc and mc.is_available)
		return {
			"ok": True,
			"move_available": available,
			"move_unavailable_reason": (mc.unavailable_reason if mc else "Controller not initialized"),
		}

	@app.post(
		"/drive",
		operation_id="drive",
		summary="Drive with steering",
		description=(
			"Drive with signed speed (-100..100) and steering degrees (-35..35). "
			"If duration_s > 0, runs as an interruptible subprocess job (stop via /stop)."
		),
		response_model=DriveResponse,
	)
	async def drive(payload: DriveRequest = Body(...)) -> dict[str, Any]:
		mc = controller
		if mc is None:
			raise HTTPException(status_code=503, detail="Controller not initialized")
		if not mc.is_available:
			raise HTTPException(status_code=503, detail=mc.unavailable_reason or "Move unavailable")

		speed = int(payload.speed)
		steer = int(payload.steer_deg or 0)
		duration_s = payload.duration_s
		try:
			dur = float(duration_s) if duration_s is not None else float(mc.settings.default_duration_s)
		except Exception:
			dur = float(mc.settings.default_duration_s)
		# If duration <= 0, treat as "continuous" drive (no job process)

		with controller_lock:
			stopped_prev = _stop_job_only()
			if dur and dur > 0:
				job_path = os.path.join(here, "src", "move_job.py")
				try:
					p = subprocess.Popen(
						[
							sys.executable,
							os.path.abspath(job_path),
							"--config",
							os.path.abspath(cfg_path),
							"--speed",
							str(speed),
							"--steer-deg",
							str(steer),
							"--duration-s",
							str(dur),
						],
						stdout=subprocess.DEVNULL,
						stderr=subprocess.DEVNULL,
						cwd=here,
					)
				except Exception as exc:
					raise HTTPException(status_code=500, detail=f"Drive failed (spawn): {exc}") from exc

				nonlocal current_job, current_job_started_ts, current_job_preview
				current_job = p
				current_job_started_ts = time.time()
				current_job_preview = f"speed={speed}, steer={steer}, dur={dur:.2f}s"
				return {
					"ok": True,
					"started": True,
					"pid": int(p.pid) if p.pid is not None else None,
					"stopped_previous": bool(stopped_prev),
				}

			# Continuous drive
			try:
				mc.drive(speed=speed, steer_deg=steer)
			except Exception as exc:
				raise HTTPException(status_code=500, detail=str(exc)) from exc
			return {"ok": True, "started": False, "continuous": True, "stopped_previous": bool(stopped_prev)}

	@app.post(
		"/stop",
		operation_id="stop",
		summary="Stop motion",
		description="Stops any current move job and stops the motion controller (best-effort).",
		response_model=StopResponse,
	)
	async def stop() -> dict[str, Any]:
		mc = controller
		if mc is None:
			raise HTTPException(status_code=503, detail="Controller not initialized")
		with controller_lock:
			stopped_job = _stop_job_only()
			try:
				mc.stop()
			except Exception:
				pass
		return {"ok": True, "stopped": True, "stopped_job": bool(stopped_job)}

	@app.get(
		"/status",
		operation_id="status",
		summary="Motion status",
		description="Reports controller availability and whether a move job is currently running.",
		response_model=StatusResponse,
	)
	async def status() -> dict[str, Any]:
		mc = controller
		with controller_lock:
			p = current_job
			speaking = bool(p is not None and p.poll() is None)
			pid = int(p.pid) if (p is not None and p.pid is not None) else None
			started_ts = current_job_started_ts
			age_ms = int((time.time() - started_ts) * 1000) if (speaking and started_ts) else None
			preview = current_job_preview
		base = mc.status() if mc is not None else {"ok": True, "available": False, "unavailable_reason": "not initialized"}
		base.update({"job_running": speaking, "job_pid": pid, "job_age_ms": age_ms, "job_preview": preview})
		return base

	# Start the HTTP API server + MCP server (separate port)
	import asyncio

	try:
		from fastmcp import FastMCP
	except Exception as exc:
		print(f"[move] fastmcp not installed: {exc}")
		print("[move] Install it with: pip3 install fastmcp")
		return 1

	mcp_host = host
	mcp_port = int(port) + 600

	mcp = FastMCP.from_fastapi(app=app, name="pi_rc_bot Move MCP")
	mcp_app = mcp.http_app(path="/mcp")

	_ensure_port_free(port)
	_ensure_port_free(mcp_port)

	async def _serve_both() -> None:
		api_server = uvicorn.Server(uvicorn.Config(app, host=host, port=port, log_level="info", loop="asyncio"))
		mcp_server = uvicorn.Server(uvicorn.Config(mcp_app, host=mcp_host, port=mcp_port, log_level="info", loop="asyncio"))

		api_server.install_signal_handlers = lambda: None  # type: ignore[method-assign]
		mcp_server.install_signal_handlers = lambda: None  # type: ignore[method-assign]

		def _request_shutdown() -> None:
			api_server.should_exit = True
			mcp_server.should_exit = True

		try:
			loop = asyncio.get_running_loop()
			for sig in (signal.SIGINT, signal.SIGTERM):
				try:
					loop.add_signal_handler(sig, _request_shutdown)
				except NotImplementedError:
					signal.signal(sig, lambda *_: _request_shutdown())
		except Exception:
			pass

		t1 = asyncio.create_task(api_server.serve())
		t2 = asyncio.create_task(mcp_server.serve())
		done, pending = await asyncio.wait({t1, t2}, return_when=asyncio.FIRST_EXCEPTION)

		exc: BaseException | None = None
		for d in done:
			try:
				e = d.exception()
			except asyncio.CancelledError:
				e = None
			if e is not None:
				exc = e
				break

		_request_shutdown()
		for p in pending:
			p.cancel()
		await asyncio.gather(*pending, return_exceptions=True)
		if exc is not None:
			raise exc

	asyncio.run(_serve_both())
	return 0


if __name__ == "__main__":
	raise SystemExit(main())
