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
	from pydantic import BaseModel, Field
	import uvicorn

	FASTAPI_AVAILABLE = True
except Exception:  # pragma: no cover
	# Stubs so annotations remain valid even when deps aren't installed yet.
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
			"Install it with: pip install -r services/memory/requirements.txt"
		) from exc

	with open(path, "r", encoding="utf-8") as f:
		data = yaml.safe_load(f) or {}
	if not isinstance(data, dict):
		raise ValueError("config.yaml must contain a YAML mapping at the root")
	return data


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
				if os.environ.get("MEMORY_ALLOW_SUDO_PROMPT") == "1" and sys.stdin.isatty():
					res3 = subprocess.run(["sudo", "fuser", "-k", f"{port}/tcp"], check=False)
					return res3.returncode == 0
			return False
		except Exception:
			pass

	# Fallback: try binding check only; no extra methods.
	return False


def _is_port_in_use(port: int, host: str = "0.0.0.0") -> bool:
	"""Return True if binding `host:port` fails with EADDRINUSE."""
	with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
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

	print(f"[memory] Port {port} is already in use; attempting to stop the existing service...")
	_kill_listeners_on_port(port)
	for _ in range(20):
		if not _is_port_in_use(port):
			return
		time.sleep(0.1)

	print(
		f"[memory] Could not free port {port}. If this is a root-owned process, run:\n"
		f"[memory]   sudo fuser -k {port}/tcp"
	)
	raise OSError(98, f"Port {port} still in use")


def _get_cfg(cfg: dict[str, Any], path: str, default: Any) -> Any:
	cur: Any = cfg
	for part in path.split("."):
		if not isinstance(cur, dict) or part not in cur:
			return default
		cur = cur.get(part)
	return cur if cur is not None else default


def _has_cfg(cfg: dict[str, Any], path: str) -> bool:
	cur: Any = cfg
	for part in path.split("."):
		if not isinstance(cur, dict) or part not in cur:
			return False
		cur = cur.get(part)
	return True


class StoreMemoryRequest(BaseModel):
	content: str = Field(
		...,
		description="The memory string to store.",
		examples=["I parked the robot near the kitchen table."],
	)
	tags: list[str] = Field(
		default_factory=list,
		description="Optional list of tags to describe the memory (free-form).",
		examples=[["robot", "location", "kitchen"]],
	)


class GetTopNMemoryRequest(BaseModel):
	content: str = Field(
		...,
		description="Query string to embed and search for similar memories.",
		examples=["Where did I park the robot?"],
	)
	top_n: int = Field(
		default=2,
		ge=1,
		le=10,
		description="Number of results to return from each tier (short + long).",
		examples=[2],
	)


def main() -> int:
	here = os.path.dirname(os.path.abspath(__file__))

	if not FASTAPI_AVAILABLE or uvicorn is None:
		print("[memory] FastAPI/uvicorn not available")
		return 1

	# Load repo-level .env if present (useful for OPENAI_API_KEY).
	# We try a few common locations because this service may be launched with
	# different working directories (systemd, supervisor, manual runs, etc.).
	_load_dotenv(os.path.join(os.getcwd(), ".env"))
	repo_root = os.path.dirname(os.path.dirname(here))
	_load_dotenv(os.path.join(repo_root, ".env"))
	_load_dotenv(os.path.join(here, ".env"))

	if not os.environ.get("OPENAI_API_KEY"):
		print(
			"[memory] OPENAI_API_KEY is not set. Set it in the environment, or add it to one of:\n"
			f"[memory]   {os.path.join(os.getcwd(), '.env')}\n"
			f"[memory]   {os.path.join(repo_root, '.env')}\n"
			f"[memory]   {os.path.join(here, '.env')}"
		)

	# Allow `from src import MemoryStore` when running from repo root.
	if here not in sys.path:
		sys.path.insert(0, here)

	from src.memory_store import MemoryError, MemoryStore, OpenAIEmbedder, OpenAIEmbeddingSettings

	cfg_path = os.path.join(here, "config.yaml")
	cfg = _load_yaml(cfg_path)

	api_host = str(_get_cfg(cfg, "memory.api_host", "0.0.0.0"))
	api_port = int(_get_cfg(cfg, "memory.api_port", 8004))
	mcp_offset = int(_get_cfg(cfg, "memory.mcp_port_offset", 600))

	data_dir = os.path.join(here, str(_get_cfg(cfg, "memory.data_dir", "data")))
	# Prefer independent per-tier caps when configured; otherwise fall back to the
	# legacy global cap (prunes across both tiers combined).
	has_short_cap = _has_cfg(cfg, "memory.max_short_memory_strings")
	has_long_cap = _has_cfg(cfg, "memory.max_long_memory_strings")
	if has_short_cap or has_long_cap:
		max_short_mem = int(_get_cfg(cfg, "memory.max_short_memory_strings", 100))
		max_long_mem = int(_get_cfg(cfg, "memory.max_long_memory_strings", 1000))
		max_mem = None
	else:
		max_short_mem = None
		max_long_mem = None
		max_mem = int(_get_cfg(cfg, "memory.max_memory_strings", 1000))
	short_time = float(_get_cfg(cfg, "memory.short_time_seconds", 3600))
	long_time = float(_get_cfg(cfg, "memory.long_time_seconds", 2592000))
	prune_old = bool(_get_cfg(cfg, "memory.prune_older_than_long_time", False))

	emb_model = str(_get_cfg(cfg, "openai.embedding_model", "text-embedding-3-large"))
	emb_base_url = str(_get_cfg(cfg, "openai.base_url", ""))
	emb_timeout = float(_get_cfg(cfg, "openai.timeout_seconds", 30))

	embedder = OpenAIEmbedder(
		OpenAIEmbeddingSettings(model=emb_model, base_url=emb_base_url or None, timeout_seconds=emb_timeout)
	)
	store = MemoryStore(
		data_dir=data_dir,
		embedder=embedder,
		short_time_seconds=short_time,
		long_time_seconds=long_time,
		prune_older_than_long_time=prune_old,
		max_memory_strings=max_mem,
		max_short_memory_strings=max_short_mem,
		max_long_memory_strings=max_long_mem,
	)

	lock = threading.Lock()

	app = FastAPI(title="pi_rc_bot Memory Service", version="1.0.0")

	@app.get(
		"/healthz",
		operation_id="healthz_healthz_get",
		summary="Health check",
		description="Returns service health and basic memory store stats.",
	)
	async def healthz() -> dict[str, Any]:
		with lock:
			st = store.stats()
		return {
			"ok": True,
			"stats": st,
			"openai_api_key_present": bool(os.environ.get("OPENAI_API_KEY")),
		}

	@app.post(
		"/store_memory",
		operation_id="store_memory",
		summary="Store a memory string",
		description=(
			"Stores a memory item (content + tags) and embeds it via OpenAI embeddings. "
			"The service sets the timestamp at ingest time."
		),
	)
	async def store_memory(body: StoreMemoryRequest) -> dict[str, Any]:
		content = (body.content or "").strip()
		# Embed outside the lock to avoid blocking other calls.
		try:
			vec = store.embedder.embed(content)
		except MemoryError as exc:
			raise HTTPException(status_code=503, detail=str(exc)) from exc
		except Exception as exc:
			raise HTTPException(status_code=500, detail=f"Embedding failed: {exc}") from exc

		with lock:
			item = store.store_memory(
				content=content,
				tags=body.tags,
				vector=vec,
			)

		return {"ok": True, "item": item}

	@app.post(
		"/get_top_n_memory",
		operation_id="get_top_n_memory",
		summary="Retrieve top-N similar memories",
		description=(
			"Embeds the provided content string and runs cosine similarity against short-term and long-term memory. "
			"Returns top_n from each tier (so total up to 2*top_n)."
		),
	)
	async def get_top_n_memory(body: GetTopNMemoryRequest) -> dict[str, Any]:
		content = (body.content or "").strip()
		try:
			qvec = store.embedder.embed(content)
		except MemoryError as exc:
			raise HTTPException(status_code=503, detail=str(exc)) from exc
		except Exception as exc:
			raise HTTPException(status_code=500, detail=f"Embedding failed: {exc}") from exc

		with lock:
			res = store.get_top_n_memory(content=content, top_n=body.top_n, query_vector=qvec)
		return res

	# Start the HTTP API server + a proper MCP server (separate port).
	import asyncio

	try:
		from fastmcp import FastMCP
	except Exception as exc:
		print(f"[memory] fastmcp not installed: {exc}")
		print("[memory] Install it with: pip3 install fastmcp")
		return 1

	mcp_host = api_host
	mcp_port = api_port + mcp_offset

	mcp = FastMCP.from_fastapi(app=app, name="pi_rc_bot Memory MCP")
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
