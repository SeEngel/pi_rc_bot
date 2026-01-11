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
			"Install it with: pip install -r services/observe/requirements.txt"
		) from exc

	with open(path, "r", encoding="utf-8") as f:
		data = yaml.safe_load(f) or {}
	if not isinstance(data, dict):
		raise ValueError("config.yaml must contain a YAML mapping at the root")
	return data


def _ensure_local_venv(here: str) -> None:
	"""Best-effort dependency bootstrap.

	If FastAPI isn't importable (common on Debian-based systems due to PEP 668),
	create a local venv under `services/observe/.venv`, install requirements, then
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
	if os.environ.get("OBSERVE_BOOTSTRAPPED") == "1":
		return

	try:
		import venv
		import subprocess

		created = False
		if not os.path.exists(venv_python):
			print("[observe] Creating local venv at services/observe/.venv ...")
			venv.EnvBuilder(with_pip=True).create(venv_dir)
			created = True

		# Avoid re-installing on every run; set OBSERVE_UPDATE_DEPS=1 to force.
		if created or os.environ.get("OBSERVE_UPDATE_DEPS") == "1":
			print("[observe] Installing dependencies into local venv ...")
			subprocess.run(
				[venv_python, "-m", "pip", "install", "-r", req_path],
				check=True,
			)

		os.environ["OBSERVE_BOOTSTRAPPED"] = "1"
		os.execv(venv_python, [venv_python, os.path.abspath(__file__), *sys.argv[1:]])
	except Exception as exc:
		print(f"[observe] Dependency bootstrap failed: {exc}")
		print("[observe] Please create a venv and install: pip install -r services/observe/requirements.txt")
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
				if os.environ.get("OBSERVE_ALLOW_SUDO_PROMPT") == "1" and sys.stdin.isatty():
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

	print(f"[observe] Port {port} is already in use; attempting to stop the existing service...")
	_kill_listeners_on_port(port)
	# Give the OS a moment to release the socket.
	for _ in range(20):
		if not _is_port_in_use(port):
			return
		time.sleep(0.1)

	print(
		f"[observe] Could not free port {port}. If this is a root-owned process, run:\n"
		f"[observe]   sudo fuser -k {port}/tcp"
	)
	raise OSError(98, f"Port {port} still in use")


def main() -> int:
	here = os.path.dirname(os.path.abspath(__file__))

	# Ensure FastAPI deps exist (auto-venv for Debian/PEP-668 systems).
	if not FASTAPI_AVAILABLE or uvicorn is None:
		_ensure_local_venv(here)
		# If bootstrap couldn't exec, bail out gracefully.
		print("[observe] FastAPI/uvicorn not available")
		return 1

	# Load repo-level .env if present.
	repo_root = os.path.dirname(os.path.dirname(here))
	_load_dotenv(os.path.join(repo_root, ".env"))
	_load_dotenv(os.path.join(here, ".env"))

	# Allow `from src import Observer` when running from repo root.
	if here not in sys.path:
		sys.path.insert(0, here)

	from contextlib import asynccontextmanager

	from src import Observer

	cfg_path = os.path.join(here, "config.yaml")
	cfg = _load_yaml(cfg_path)

	service_cfg = cfg.get("service", {}) if isinstance(cfg, dict) else {}
	if not isinstance(service_cfg, dict):
		service_cfg = {}
	host = str(service_cfg.get("host") or "0.0.0.0")
	port = int(service_cfg.get("port") or 8003)

	observer_lock = threading.Lock()
	observer: Observer | None = None

	def _init_observer() -> Observer:
		ob = Observer.from_config_dict(cfg)
		if not ob.is_available:
			print(f"[observe] Vision unavailable: {ob.unavailable_reason}")
			print("[observe] Tip: set vision.dry_run=true for development")
		return ob

	@asynccontextmanager
	async def lifespan(_):
		nonlocal observer
		observer = _init_observer()
		try:
			yield
		finally:
			try:
				if observer is not None:
					observer.close()
			except Exception:
				pass

	app = FastAPI(title="pi_rc_bot Observe Service", version="1.0.0", lifespan=lifespan)

	@app.get(
		"/healthz",
		summary="Health check",
		description="Returns service health and vision/camera availability.",
	)
	async def healthz() -> dict[str, Any]:
		ob = observer
		available = bool(ob and ob.is_available)
		return {
			"ok": True,
			"vision_available": available,
			"vision_unavailable_reason": (ob.unavailable_reason if ob else "Observer not initialized"),
		}

	@app.post(
		"/observe",
		summary="Observe once",
		description=(
			"Captures one image from the camera and asks a vision model to describe it.\n\n"
			"Optional request body JSON:\n"
			"- `question` (string): overrides the default question from config.\n\n"
			"### Response\n"
			"Returns only: `{ text: string }`."
		),
		openapi_extra={
			"requestBody": {
				"required": False,
				"content": {
					"application/json": {
						"schema": {
							"type": "object",
							"properties": {"question": {"type": "string"}},
						},
					}
				},
			},
			"responses": {
				"200": {
					"description": "Successful response",
					"content": {
						"application/json": {
							"schema": {
								"type": "object",
								"properties": {"text": {"type": "string"}},
								"required": ["text"],
							}
						}
					},
				}
			},
		},
	)
	async def observe(request: Request) -> dict[str, Any]:
		ob = observer
		if ob is None:
			raise HTTPException(status_code=503, detail="Observer not initialized")
		if not ob.is_available:
			raise HTTPException(status_code=503, detail=ob.unavailable_reason or "Vision unavailable")

		raw = await request.body()
		question: str | None = None
		try:
			if raw:
				import json

				data = json.loads(raw.decode("utf-8", errors="strict"))
				if isinstance(data, dict) and data.get("question") is not None:
					question = str(data.get("question") or "").strip() or None
		except Exception:
			pass

		def _do_observe() -> dict[str, Any]:
			with observer_lock:
				return ob.observe_once(question=question)

		try:
			import anyio

			res = await anyio.to_thread.run_sync(_do_observe)
		except Exception as exc:
			raise HTTPException(status_code=500, detail=f"Observe failed: {exc}") from exc

		text = ""
		try:
			text = str(res.get("text") or "")
		except Exception:
			text = ""
		return {"text": text}

	# Start the HTTP server.
	_ensure_port_free(port)
	uvicorn.run(app, host=host, port=port, log_level="info")
	return 0


if __name__ == "__main__":
	raise SystemExit(main())
