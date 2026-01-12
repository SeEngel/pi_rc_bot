from __future__ import annotations

import os
import signal
import shutil
import socket
import sys
import threading
import time
from typing import Any

FASTAPI_AVAILABLE = False

try:
	from fastapi import FastAPI, HTTPException, Request
	import uvicorn

	FASTAPI_AVAILABLE = True
except Exception:  # pragma: no cover
	# Stubs so annotations remain valid even when deps aren't installed yet.
	# (We bootstrap a local venv and re-exec before actually serving requests.)
	class Request:  # type: ignore[no-redef]
		pass

	class FastAPI:  # type: ignore[no-redef]
		pass

	class HTTPException(Exception):  # type: ignore[no-redef]
		def __init__(self, status_code: int, detail: str):
			super().__init__(f"{status_code}: {detail}")

	uvicorn = None  # type: ignore[assignment]


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
			"Install it with: pip install -r services/listening/requirements.txt"
		) from exc

	with open(path, "r", encoding="utf-8") as f:
		data = yaml.safe_load(f) or {}
	if not isinstance(data, dict):
		raise ValueError("config.yaml must contain a YAML mapping at the root")
	return data


def _ensure_local_venv(here: str) -> None:
	"""Best-effort dependency bootstrap.

	If FastAPI isn't importable (common on Debian-based systems due to PEP 668),
	create a local venv under `services/listening/.venv`, install requirements, then
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
	if os.environ.get("LISTENING_BOOTSTRAPPED") == "1":
		return

	try:
		import venv
		import subprocess

		created = False
		if not os.path.exists(venv_python):
			print("[listening] Creating local venv at services/listening/.venv ...")
			venv.EnvBuilder(with_pip=True).create(venv_dir)
			created = True

		# Avoid re-installing on every run; set LISTENING_UPDATE_DEPS=1 to force.
		if created or os.environ.get("LISTENING_UPDATE_DEPS") == "1":
			print("[listening] Installing dependencies into local venv ...")
			subprocess.run(
				[venv_python, "-m", "pip", "install", "-r", req_path],
				check=True,
			)

		os.environ["LISTENING_BOOTSTRAPPED"] = "1"
		os.execv(venv_python, [venv_python, os.path.abspath(__file__), *sys.argv[1:]])
	except Exception as exc:
		print(f"[listening] Dependency bootstrap failed: {exc}")
		print("[listening] Please create a venv and install: pip install -r services/listening/requirements.txt")
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
				if os.environ.get("LISTENING_ALLOW_SUDO_PROMPT") == "1" and sys.stdin.isatty():
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

	print(f"[listening] Port {port} is already in use; attempting to stop the existing service...")
	_kill_listeners_on_port(port)
	# Give the OS a moment to release the socket.
	for _ in range(20):
		if not _is_port_in_use(port):
			return
		time.sleep(0.1)

	print(
		f"[listening] Could not free port {port}. If this is a root-owned process, run:\n"
		f"[listening]   sudo fuser -k {port}/tcp"
	)
	raise OSError(98, f"Port {port} still in use")


def main() -> int:
	here = os.path.dirname(os.path.abspath(__file__))

	# We intentionally do NOT auto-create venvs here.
	# Install deps globally if needed (e.g. `pip3 install -r services/listening/requirements.txt`).
	if not FASTAPI_AVAILABLE or uvicorn is None:
		print("[listening] FastAPI/uvicorn not available")
		return 1

	# Load repo-level .env if present.
	repo_root = os.path.dirname(os.path.dirname(here))
	_load_dotenv(os.path.join(repo_root, ".env"))
	_load_dotenv(os.path.join(here, ".env"))

	# Allow `from src import Listener` when running from repo root.
	if here not in sys.path:
		sys.path.insert(0, here)

	from contextlib import asynccontextmanager

	from src import Listener

	cfg_path = os.path.join(here, "config.yaml")
	cfg = _load_yaml(cfg_path)

	service_cfg = cfg.get("service", {}) if isinstance(cfg, dict) else {}
	if not isinstance(service_cfg, dict):
		service_cfg = {}
	host = str(service_cfg.get("host") or "0.0.0.0")
	port = int(service_cfg.get("port") or 8002)

	listener_lock = threading.Lock()
	listener: Listener | None = None

	def _init_listener() -> Listener:
		li = Listener.from_config_dict(cfg)
		if not li.is_available:
			print(f"[listening] STT unavailable: {li.unavailable_reason}")
			print("[listening] Tip: install picar-x + robot-hat, or set stt.dry_run=true")
		return li

	@asynccontextmanager
	async def lifespan(_):
		nonlocal listener
		listener = _init_listener()
		yield

	app = FastAPI(title="pi_rc_bot Listening Service", version="1.0.0", lifespan=lifespan)

	@app.get(
		"/healthz",
		summary="Health check",
		description="Returns service health and STT availability.",
	)
	async def healthz() -> dict[str, Any]:
		li = listener
		available = bool(li and li.is_available)
		return {
			"ok": True,
			"stt_available": available,
			"stt_unavailable_reason": (li.unavailable_reason if li else "Listener not initialized"),
		}

	@app.post(
		"/listen",
		operation_id="listen",
		summary="Listen once",
		description=(
			"Records audio from the microphone and returns the recognized text.\n\n"
			"### Request body (optional JSON)\n"
			"- `stream` (bool, default: `false`)\n"
			"  - **Vosk only**: forwarded to the underlying `Vosk.listen(stream=...)` if supported.\n"
			"  - **OpenAI**: ignored.\n"
			"- `speech_pause_seconds` (number, optional)\n"
			"  - **OpenAI only**: stop recording after this many seconds of *continuous silence* **after speech has started**.\n"
			"  - If omitted: uses `stt.openai.stop_silence_seconds` from `config.yaml`.\n"
			"- `stop_silence_seconds` (number, optional)\n"
			"  - Alias for `speech_pause_seconds` (OpenAI only).\n\n"
			"### Behavior\n"
			"- Engine `vosk`: listens once via the local Vosk wrapper and returns its transcript.\n"
			"- Engine `openai`: records until either\n"
			"  1) the silence pause threshold is reached (see `speech_pause_seconds`), or\n"
			"  2) `stt.openai.record_seconds` (max duration) is reached,\n"
			"  then sends the audio to OpenAI Speech-to-Text.\n\n"
			"### Response\n"
			"Returns `{ ok: true, text: string, raw: object }`.\n"
		),
	)
	async def listen(request: Request) -> dict[str, Any]:
		li = listener
		if li is None:
			raise HTTPException(status_code=503, detail="Listener not initialized")
		if not li.is_available:
			raise HTTPException(status_code=503, detail=li.unavailable_reason or "STT unavailable")

		raw = await request.body()
		stream = False
		speech_pause_seconds: float | None = None
		try:
			if raw:
				import json

				data = json.loads(raw.decode("utf-8", errors="strict"))
				if isinstance(data, dict) and data.get("stream") is not None:
					stream = bool(data.get("stream"))
				if isinstance(data, dict):
					# OpenAI-only override: how long silence (seconds) ends the recording.
					# Accept a couple of aliases for convenience.
					val = None
					if data.get("speech_pause_seconds") is not None:
						val = data.get("speech_pause_seconds")
					elif data.get("stop_silence_seconds") is not None:
						val = data.get("stop_silence_seconds")
					if val is not None:
						try:
							speech_pause_seconds = float(val)
						except Exception:
							speech_pause_seconds = None
		except Exception:
			pass

		def _do_listen() -> tuple[dict[str, Any], str]:
			with listener_lock:
				res = li.listen_once(stream=stream, speech_pause_seconds=speech_pause_seconds)
				text = li.extract_text(res)
				return res, text

		try:
			import anyio

			res, text = await anyio.to_thread.run_sync(_do_listen)
		except Exception as exc:
			raise HTTPException(status_code=500, detail=f"Listen failed: {exc}") from exc

		return {"ok": True, "text": text, "raw": res}

	# Start the HTTP API server + a proper MCP server (separate port).
	import asyncio

	try:
		from fastmcp import FastMCP
	except Exception as exc:
		print(f"[listening] fastmcp not installed: {exc}")
		print("[listening] Install it with: pip3 install fastmcp")
		return 1

	mcp_host = host
	mcp_port = int(port) + 600

	# Convert FastAPI -> MCP (tools by default) and expose streamable HTTP at /mcp.
	mcp = FastMCP.from_fastapi(app=app, name="pi_rc_bot Listening MCP")
	mcp_app = mcp.http_app(path="/mcp")

	_ensure_port_free(port)
	_ensure_port_free(mcp_port)

	async def _serve_both() -> None:
		api_server = uvicorn.Server(uvicorn.Config(app, host=host, port=port, log_level="info", loop="asyncio"))
		mcp_server = uvicorn.Server(
			uvicorn.Config(mcp_app, host=mcp_host, port=mcp_port, log_level="info", loop="asyncio")
		)

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
