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
	from fastapi import FastAPI, HTTPException, Request
	from pydantic import BaseModel, Field
	import uvicorn

	FASTAPI_AVAILABLE = True
except Exception:  # pragma: no cover
	# Stubs so annotations remain valid even when deps aren't installed yet.
	# (We bootstrap a local venv and re-exec before actually serving requests.)
	class Request:  # type: ignore[no-redef]
		pass

	class FastAPI:  # type: ignore[no-redef]
		pass

	class BaseModel:  # type: ignore[no-redef]
		pass

	def Field(*_args, **_kwargs):  # type: ignore[no-redef]
		return None

	class HTTPException(Exception):  # type: ignore[no-redef]
		def __init__(self, status_code: int, detail: str):
			super().__init__(f"{status_code}: {detail}")

	uvicorn = None  # type: ignore[assignment]


if FASTAPI_AVAILABLE:
	class SpeakRequest(BaseModel):
		"""Request payload for text-to-speech.

		The text will be spoken aloud through the robot's speakers.
		Speech is queued - if speech is already in progress, the previous
		speech is stopped and the new text is spoken.
		"""
		text: str = Field(
			description="The text to speak aloud. Supports plain text; punctuation affects pacing.",
			examples=["Hello, I am your robot assistant.", "I see an obstacle ahead.", "Moving forward now."],
			min_length=1,
			max_length=5000,
		)

	class SpeakResponse(BaseModel):
		"""Response from /speak indicating speech was started."""
		ok: bool = Field(default=True, description="Whether the operation completed without errors.")
		started: bool = Field(default=True, description="Whether the speech subprocess was started.")
		pid: int | None = Field(default=None, description="Process ID of the speech subprocess.", examples=[12345])
		stopped_previous: bool = Field(default=False, description="True if previous speech was interrupted.")
		tts_available: bool = Field(default=True, description="Whether TTS backend is available.")

	class StopResponse(BaseModel):
		"""Response from /stop."""
		ok: bool = Field(default=True, description="Whether the operation completed without errors.")
		stopped: bool = Field(description="True if ongoing speech was interrupted.")

	class StatusResponse(BaseModel):
		"""Response from /status with current playback information."""
		ok: bool = Field(default=True, description="Whether the operation completed without errors.")
		speaking: bool = Field(description="True if speech is currently playing.")
		pid: int | None = Field(default=None, description="PID of the current speech process if speaking.")
		age_ms: int | None = Field(default=None, description="How long the current speech has been playing in milliseconds.")
		text_preview: str | None = Field(default=None, description="Preview of the text being spoken (truncated to 200 chars).", examples=["Hello, I am your robot..."])


def _load_dotenv(path: str) -> None:
	"""Minimal .env loader (KEY=VALUE per line).

	Loads variables only if they are not already set in the environment.
	"""
	if not os.path.exists(path):
		return
	try:
		with open(path, "r", encoding="utf-8") as f:
			for raw in f:
				line = raw.strip()
				if not line or line.startswith("#"):
					continue
				if "=" not in line:
					continue
				k, v = line.split("=", 1)
				k = k.strip()
				if k.startswith("export "):
					k = k[len("export ") :].strip()
				v = v.strip().strip('"').strip("'")
				if not k:
					continue
				existing = os.environ.get(k)
				# Populate missing variables, and also overwrite empty ones.
				if existing is None or existing == "":
					os.environ[k] = v
	except Exception:
		# Never fail hard on dotenv parsing.
		return


def _load_yaml(path: str) -> dict[str, Any]:
	try:
		import yaml  # type: ignore
	except Exception as exc:
		raise RuntimeError(
			"PyYAML is required to load config.yaml. "
			"Install it with: pip install -r services/speak/requirements.txt"
		) from exc

	with open(path, "r", encoding="utf-8") as f:
		data = yaml.safe_load(f) or {}
	if not isinstance(data, dict):
		raise ValueError("config.yaml must contain a YAML mapping at the root")
	return data


def _ensure_local_venv(here: str) -> None:
	"""Best-effort dependency bootstrap.

	If FastAPI isn't importable (common on Debian-based systems due to PEP 668),
	create a local venv under `services/speak/.venv`, install requirements, then
	re-exec this script using the venv interpreter.
	"""
	venv_dir = os.path.join(here, ".venv")
	venv_python = os.path.join(venv_dir, "bin", "python")
	req_path = os.path.join(here, "requirements.txt")

	# If we're already running from the local venv, nothing to do.
	try:
		if os.path.realpath(sys.executable).startswith(os.path.realpath(venv_dir) + os.sep):
			return
	except Exception:
		pass

	# Prevent infinite recursion if exec fails.
	if os.environ.get("SPEAK_BOOTSTRAPPED") == "1":
		return

	try:
		import venv
		import subprocess

		created = False
		if not os.path.exists(venv_python):
			print("[speak] Creating local venv at services/speak/.venv ...")
			venv.EnvBuilder(with_pip=True).create(venv_dir)
			created = True

		# Avoid re-installing on every run; set SPEAK_UPDATE_DEPS=1 to force.
		if created or os.environ.get("SPEAK_UPDATE_DEPS") == "1":
			print("[speak] Installing dependencies into local venv ...")
			subprocess.run(
				[venv_python, "-m", "pip", "install", "-r", req_path],
				check=True,
			)

		os.environ["SPEAK_BOOTSTRAPPED"] = "1"
		os.execv(venv_python, [venv_python, os.path.abspath(__file__), *sys.argv[1:]])
	except Exception as exc:
		print(f"[speak] Dependency bootstrap failed: {exc}")
		print("[speak] Please create a venv and install: pip install -r services/speak/requirements.txt")
		return


def _kill_listeners_on_port(port: int) -> bool:
	"""Best-effort: kill any process listening on TCP `port`.

	Returns True if we *likely* killed something, False otherwise.
	"""
	port = int(port)
	if port <= 0:
		return False

	# Prefer fuser (common on Debian/RPi).
	if shutil.which("fuser") is not None:
		try:
			import subprocess

			res = subprocess.run(
				["fuser", "-k", f"{port}/tcp"],
				stdout=subprocess.DEVNULL,
				stderr=subprocess.DEVNULL,
			)
			if res.returncode == 0:
				return True

			# If we couldn't kill (possibly due to permissions), try sudo non-interactively.
			if shutil.which("sudo") is not None:
				res2 = subprocess.run(
					["sudo", "-n", "fuser", "-k", f"{port}/tcp"],
					stdout=subprocess.DEVNULL,
					stderr=subprocess.DEVNULL,
				)
				if res2.returncode == 0:
					return True
				# If explicitly enabled and interactive, allow sudo to prompt.
				if os.environ.get("SPEAK_ALLOW_SUDO_PROMPT") == "1" and sys.stdin.isatty():
					res3 = subprocess.run(["sudo", "fuser", "-k", f"{port}/tcp"], check=False)
					return res3.returncode == 0
			return False
		except Exception:
			pass

	# Fallback: lsof to list PIDs.
	if shutil.which("lsof") is not None:
		try:
			import subprocess

			res = subprocess.run(
				["lsof", "-tiTCP:%d" % port, "-sTCP:LISTEN"],
				check=False,
				stdout=subprocess.PIPE,
				stderr=subprocess.DEVNULL,
				text=True,
			)
			# If nothing visible (root-owned process), try sudo non-interactively.
			if (not (res.stdout or "").strip()) and shutil.which("sudo") is not None:
				res = subprocess.run(
					["sudo", "-n", "lsof", "-tiTCP:%d" % port, "-sTCP:LISTEN"],
					check=False,
					stdout=subprocess.PIPE,
					stderr=subprocess.DEVNULL,
					text=True,
				)
			pids = []
			for line in (res.stdout or "").splitlines():
				line = line.strip()
				if not line:
					continue
				try:
					pid = int(line)
				except Exception:
					continue
				if pid == os.getpid():
					continue
				pids.append(pid)

			killed_any = False
			for pid in pids:
				try:
					os.kill(pid, signal.SIGTERM)
					killed_any = True
				except Exception:
					pass

			# If we couldn't signal (permissions), try sudo kill.
			if not killed_any and pids and shutil.which("sudo") is not None:
				try:
					subprocess.run(
						["sudo", "-n", "kill", "-TERM", *[str(p) for p in pids]],
						check=False,
						stdout=subprocess.DEVNULL,
						stderr=subprocess.DEVNULL,
					)
					killed_any = True
				except Exception:
					pass
			return killed_any
		except Exception:
			pass

	return False


def _is_port_in_use(port: int, host: str = "0.0.0.0") -> bool:
	"""Return True if binding `host:port` fails with EADDRINUSE."""
	with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
		# SO_REUSEADDR does NOT allow binding if another process is already listening.
		s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
		try:
			s.bind((host, int(port)))
			return False
		except OSError as exc:
			return getattr(exc, "errno", None) == 98


def _ensure_port_free(port: int) -> None:
	"""Ensure `port` is free; kill existing listener if needed (best-effort)."""
	if not _is_port_in_use(port):
		return

	print(f"[speak] Port {port} is already in use; attempting to stop the existing service...")
	_kill_listeners_on_port(port)
	# Give the OS a moment to release the socket.
	for _ in range(20):
		if not _is_port_in_use(port):
			return
		time.sleep(0.1)

	print(
		f"[speak] Could not free port {port}. If this is a root-owned process, run:\n"
		f"[speak]   sudo fuser -k {port}/tcp"
	)
	raise OSError(98, f"Port {port} still in use")


def main() -> int:
	here = os.path.dirname(os.path.abspath(__file__))

	# We intentionally do NOT auto-create venvs here.
	# Install deps globally if needed (e.g. `pip3 install -r services/speak/requirements.txt`).
	if not FASTAPI_AVAILABLE or uvicorn is None:
		print("[speak] FastAPI/uvicorn not available")
		return 1

	# Load repo-level .env if present (useful for OPENAI_API_KEY).
	repo_root = os.path.dirname(os.path.dirname(here))
	_load_dotenv(os.path.join(repo_root, ".env"))
	_load_dotenv(os.path.join(here, ".env"))

	# Allow `from src import Speaker` when running from repo root.
	if here not in sys.path:
		sys.path.insert(0, here)

	from contextlib import asynccontextmanager

	from src import Speaker

	cfg_path = os.path.join(here, "config.yaml")
	cfg = _load_yaml(cfg_path)

	# Global-ish state for the service process.
	speaker_lock = threading.Lock()
	speaker: Speaker | None = None
	current_job: subprocess.Popen[bytes] | None = None
	current_job_started_ts: float | None = None
	current_job_text_preview: str | None = None

	def _stop_current_job() -> bool:
		"""Best-effort stop of any in-flight playback job."""
		nonlocal current_job, current_job_started_ts, current_job_text_preview
		p = current_job
		if p is None:
			return False
		# Already exited
		if p.poll() is not None:
			current_job = None
			current_job_started_ts = None
			current_job_text_preview = None
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
		current_job_text_preview = None
		return True

	def _init_speaker() -> Speaker:
		sp = Speaker.from_config_dict(cfg)
		if not sp.is_available:
			print(f"[speak] TTS unavailable: {sp.unavailable_reason}")
			print("[speak] Tip: set tts.backend=system or enable tts.dry_run=true")
		return sp

	@asynccontextmanager
	async def lifespan(_):
		nonlocal speaker
		speaker = _init_speaker()
		yield

	app = FastAPI(title="pi_rc_bot Speak Service", version="1.0.0", lifespan=lifespan)

	@app.get(
		"/healthz",
		operation_id="healthz_healthz_get",
		summary="Health check",
		description="Returns service health and TTS availability.",
	)
	async def healthz() -> dict[str, Any]:
		sp = speaker
		available = bool(sp and sp.is_available)
		return {
			"ok": True,
			"tts_available": available,
			"tts_unavailable_reason": (sp.unavailable_reason if sp else "Speaker not initialized"),
		}

	@app.post(
		"/speak",
		operation_id="speak",
		summary="Speak a text",
		description=(
			"Consumes a string (JSON: {text: <string>}) and speaks it out loud on the speakers so a human can hear it."
		),
		openapi_extra={
			"requestBody": {
				"required": True,
				"content": {
					"application/json": {
						"schema": {
							"type": "object",
							"properties": {"text": {"type": "string"}},
							"required": ["text"],
						},
					}
				},
			}
		},
	)
	async def speak(request: Request) -> dict[str, Any]:
		sp = speaker
		if sp is None:
			raise HTTPException(status_code=503, detail="Speaker not initialized")
		if not sp.is_available:
			raise HTTPException(status_code=503, detail=sp.unavailable_reason or "TTS unavailable")

		raw = await request.body()
		if not raw:
			raise HTTPException(status_code=422, detail="Missing text")

		text = ""
		try:
			import json

			data = json.loads(raw.decode("utf-8", errors="strict"))
			if isinstance(data, dict) and "text" in data:
				text = str(data.get("text") or "")
			elif isinstance(data, str):
				text = data
			else:
				text = ""
		except Exception:
			# If it's not JSON, treat it as plain text.
			text = raw.decode("utf-8", errors="ignore")

		text = (text or "").strip()
		if not text:
			raise HTTPException(status_code=422, detail="Missing text")

		# Serialize speech to avoid overlapping playback.
		with speaker_lock:
			# Stop any previous playback first (makes speech effectively interruptible).
			stopped_prev = _stop_current_job()

			job_path = os.path.join(here, "src", "speak_job.py")
			# Run playback in a separate process so we can interrupt it via /stop.
			try:
				p = subprocess.Popen(
					[
						sys.executable,
						os.path.abspath(job_path),
						"--config",
						os.path.abspath(cfg_path),
						"--text",
						text,
					],
					stdout=subprocess.DEVNULL,
					stderr=subprocess.DEVNULL,
					cwd=here,
				)
			except Exception as exc:
				raise HTTPException(status_code=500, detail=f"Speak failed (spawn): {exc}") from exc

			nonlocal current_job, current_job_started_ts, current_job_text_preview
			current_job = p
			current_job_started_ts = time.time()
			current_job_text_preview = (text[:200] + ("…" if len(text) > 200 else ""))

		return {
			"ok": True,
			"started": True,
			"pid": int(p.pid) if p.pid is not None else None,
			"stopped_previous": bool(stopped_prev),
			"tts_available": True,
		}

	@app.post(
		"/stop",
		operation_id="stop",
		summary="Stop current speech",
		description="Interrupt any in-progress audio playback started via /speak.",
	)
	async def stop() -> dict[str, Any]:
		with speaker_lock:
			stopped = _stop_current_job()
		return {"ok": True, "stopped": bool(stopped)}

	@app.get(
		"/status",
		operation_id="status",
		summary="Playback status",
		description="Returns whether the service is currently speaking and basic metadata.",
	)
	async def status() -> dict[str, Any]:
		with speaker_lock:
			p = current_job
			speaking = bool(p is not None and p.poll() is None)
			pid = int(p.pid) if (p is not None and p.pid is not None) else None
			started_ts = current_job_started_ts
			age_ms = int((time.time() - started_ts) * 1000) if (speaking and started_ts) else None
			preview = current_job_text_preview
		return {
			"ok": True,
			"speaking": speaking,
			"pid": pid,
			"age_ms": age_ms,
			"text_preview": preview,
		}

	# Start the HTTP API server + a proper MCP server (separate port).
	import asyncio

	try:
		from fastmcp import FastMCP
	except Exception as exc:
		print(f"[speak] fastmcp not installed: {exc}")
		print("[speak] Install it with: pip3 install fastmcp")
		return 1

	api_host = "0.0.0.0"
	api_port = 8001
	mcp_host = api_host
	mcp_port = api_port + 600

	# Convert FastAPI -> MCP (tools by default) and expose streamable HTTP at /mcp.
	mcp = FastMCP.from_fastapi(app=app, name="pi_rc_bot Speak MCP")
	mcp_app = mcp.http_app(path="/mcp")

	_ensure_port_free(api_port)
	_ensure_port_free(mcp_port)

	async def _serve_both() -> None:
		api_server = uvicorn.Server(
			uvicorn.Config(app, host=api_host, port=api_port, log_level="info", loop="asyncio")
		)
		mcp_server = uvicorn.Server(
			uvicorn.Config(mcp_app, host=mcp_host, port=mcp_port, log_level="info", loop="asyncio")
		)

		# We'll manage signals once for both servers.
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
			# Fall back to uvicorn defaults if signal handler wiring fails.
			pass

		t1 = asyncio.create_task(api_server.serve())
		t2 = asyncio.create_task(mcp_server.serve())
		done, pending = await asyncio.wait({t1, t2}, return_when=asyncio.FIRST_EXCEPTION)

		# If one server errors, stop the other.
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
