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
	class SetAnglesRequest(BaseModel):
		pan_deg: int | None = Field(default=None, description="Pan angle in degrees.")
		tilt_deg: int | None = Field(default=None, description="Tilt angle in degrees.")

	class ScanRequest(BaseModel):
		pattern: str = Field(default="sweep", description="Scan pattern.")
		duration_s: float = Field(default=3.0, description="Total scan duration in seconds.")
		step_deg: int | None = Field(default=None, description="Optional step size override (degrees).")
		interval_s: float | None = Field(default=None, description="Optional interval override between steps (seconds).")

	class ScanResponse(BaseModel):
		ok: bool = True
		started: bool = True
		pid: int | None = None
		stopped_previous: bool = False


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
			"PyYAML is required to load config.yaml. Install it with: pip install -r services/head/requirements.txt"
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
	if shutil.which("fuser") is None:
		return False
	try:
		res = subprocess.run(["fuser", "-k", f"{int(port)}/tcp"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
		return res.returncode == 0
	except Exception:
		return False


def _ensure_port_free(port: int) -> None:
	if not _is_port_in_use(port):
		return
	print(f"[head] Port {port} is already in use; attempting to stop the existing service...")
	_kill_listeners_on_port(port)
	for _ in range(20):
		if not _is_port_in_use(port):
			return
		time.sleep(0.1)
	raise OSError(98, f"Port {port} still in use")


def main() -> int:
	here = os.path.dirname(os.path.abspath(__file__))
	if not FASTAPI_AVAILABLE or uvicorn is None:
		print("[head] FastAPI/uvicorn not available")
		return 1

	repo_root = os.path.dirname(os.path.dirname(here))
	_load_dotenv(os.path.join(repo_root, ".env"))
	_load_dotenv(os.path.join(here, ".env"))

	if here not in sys.path:
		sys.path.insert(0, here)

	from contextlib import asynccontextmanager

	from src import HeadController

	cfg_path = os.path.join(here, "config.yaml")
	cfg = _load_yaml(cfg_path)
	service_cfg = (cfg or {}).get("service", {}) if isinstance(cfg, dict) else {}
	host = str((service_cfg or {}).get("host") or "0.0.0.0")
	port = int((service_cfg or {}).get("port") or 8006)

	controller_lock = threading.Lock()
	controller: HeadController | None = None

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

	def _init_controller() -> HeadController:
		hc = HeadController.from_config_dict(cfg)
		if not hc.is_available:
			print(f"[head] hardware unavailable: {hc.unavailable_reason}")
			print("[head] Tip: install picar-x system-wide, or set head.dry_run=true")
		return hc

	@asynccontextmanager
	async def lifespan(_):
		nonlocal controller
		controller = _init_controller()
		yield
		try:
			if controller is not None:
				controller.center()
		except Exception:
			pass

	app = FastAPI(title="pi_rc_bot Head Service", version="1.0.0", lifespan=lifespan)

	@app.get("/healthz", operation_id="healthz_healthz_get")
	async def healthz() -> dict[str, Any]:
		hc = controller
		available = bool(hc and hc.is_available)
		return {
			"ok": True,
			"head_available": available,
			"head_unavailable_reason": (hc.unavailable_reason if hc else "Controller not initialized"),
		}

	@app.post(
		"/set_angles",
		operation_id="set_angles",
		summary="Set head angles",
		description="Sets head pan/tilt angles (degrees).",
	)
	async def set_angles(payload: SetAnglesRequest = Body(...)) -> dict[str, Any]:
		hc = controller
		if hc is None:
			raise HTTPException(status_code=503, detail="Controller not initialized")
		if not hc.is_available:
			raise HTTPException(status_code=503, detail=hc.unavailable_reason or "Head unavailable")

		pan = payload.pan_deg
		tilt = payload.tilt_deg

		with controller_lock:
			_stop_job_only()
			try:
				hc.set_angles(pan_deg=pan, tilt_deg=tilt)
			except Exception as exc:
				raise HTTPException(status_code=500, detail=str(exc)) from exc
			return {"ok": True, **hc.status()}

	@app.post("/center", operation_id="center")
	async def center() -> dict[str, Any]:
		hc = controller
		if hc is None:
			raise HTTPException(status_code=503, detail="Controller not initialized")
		with controller_lock:
			_stop_job_only()
			try:
				hc.center()
			except Exception:
				pass
		return {"ok": True}

	@app.post(
		"/scan",
		operation_id="scan",
		summary="Start a scan pattern",
		description="Runs a scan job in a subprocess so it can be interrupted via /stop.",
	)
	async def scan(payload: ScanRequest | None = Body(default=None)) -> dict[str, Any]:
		hc = controller
		if hc is None:
			raise HTTPException(status_code=503, detail="Controller not initialized")
		if not hc.is_available:
			raise HTTPException(status_code=503, detail=hc.unavailable_reason or "Head unavailable")

		payload_obj = payload or ScanRequest()
		pattern = str(payload_obj.pattern or "sweep")
		if pattern not in {"sweep", "nod"}:
			pattern = "sweep"
		duration_s = float(payload_obj.duration_s)
		step_deg = payload_obj.step_deg
		interval_s = payload_obj.interval_s

		with controller_lock:
			stopped_prev = _stop_job_only()
			job_path = os.path.join(here, "src", "head_job.py")
			try:
				p = subprocess.Popen(
					[
						sys.executable,
						os.path.abspath(job_path),
						"--config",
						os.path.abspath(cfg_path),
						"--pattern",
						pattern,
						"--duration-s",
						str(duration_s),
						*(["--step-deg", str(int(step_deg))] if step_deg is not None else []),
						*(["--interval-s", str(float(interval_s))] if interval_s is not None else []),
					],
					stdout=subprocess.DEVNULL,
					stderr=subprocess.DEVNULL,
					cwd=here,
				)
			except Exception as exc:
				raise HTTPException(status_code=500, detail=f"Scan failed (spawn): {exc}") from exc

			nonlocal current_job, current_job_started_ts, current_job_preview
			current_job = p
			current_job_started_ts = time.time()
			current_job_preview = f"pattern={pattern}, dur={duration_s:.2f}s"

		return {
			"ok": True,
			"started": True,
			"pid": int(p.pid) if p.pid is not None else None,
			"stopped_previous": bool(stopped_prev),
		}

	@app.post("/stop", operation_id="stop", summary="Stop scan job")
	async def stop() -> dict[str, Any]:
		with controller_lock:
			stopped = _stop_job_only()
		return {"ok": True, "stopped": bool(stopped)}

	@app.get("/status", operation_id="status")
	async def status() -> dict[str, Any]:
		hc = controller
		with controller_lock:
			p = current_job
			running = bool(p is not None and p.poll() is None)
			pid = int(p.pid) if (p is not None and p.pid is not None) else None
			started_ts = current_job_started_ts
			age_ms = int((time.time() - started_ts) * 1000) if (running and started_ts) else None
			preview = current_job_preview
		base = hc.status() if hc is not None else {"ok": True, "available": False, "unavailable_reason": "not initialized"}
		base.update({"job_running": running, "job_pid": pid, "job_age_ms": age_ms, "job_preview": preview})
		return base

	# Serve HTTP + MCP
	import asyncio

	try:
		from fastmcp import FastMCP
	except Exception as exc:
		print(f"[head] fastmcp not installed: {exc}")
		print("[head] Install it with: pip3 install fastmcp")
		return 1

	mcp_host = host
	mcp_port = int(port) + 600

	mcp = FastMCP.from_fastapi(app=app, name="pi_rc_bot Head MCP")
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
