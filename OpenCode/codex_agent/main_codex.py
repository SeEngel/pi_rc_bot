#!/usr/bin/env python3
"""
Codex Agent — MCP Tool Server
==============================
A FastAPI + FastMCP server that the main robot agent can call as a native
MCP tool (``robot_codex``).  Under the hood it runs its own OpenCode
instance (port 4097) to do the actual code work.

MCP Tools (4):
  POST /build_tool   — describe a new tool in plain text; codex builds it
  POST /repair_tool  — describe what's broken in plain text; codex fixes it
  POST /list_jobs    — high-level overview of all jobs
  POST /job_detail   — detailed status by job_id

Usage (auto-started by main.py supervisor):
    uv run python OpenCode/codex_agent/main_codex.py --port 8012
"""

from __future__ import annotations

import argparse
import asyncio
import logging
import shutil
import signal
import subprocess
from pathlib import Path

import uvicorn
import yaml

from src.client import CodexClient, wait_for_opencode
from src.jobs import JobTracker
from src.server import app, init_app
from src.tools import ToolInventory
from src.workers import BuildWorker, RepairWorker

BASE_DIR = Path(__file__).resolve().parent       # codex_agent/
OPENCODE_DIR = BASE_DIR.parent                   # OpenCode/
MY_TOOLS_DIR = OPENCODE_DIR / "my_tools"
CONFIG_PATH = BASE_DIR / "config.yaml"

LOG = logging.getLogger("codex-agent")

# ── config ──────────────────────────────────────────────────────

def _load_config() -> dict:
    if CONFIG_PATH.exists():
        with open(CONFIG_PATH) as f:
            return yaml.safe_load(f) or {}
    return {}


# ── OpenCode serve process ──────────────────────────────────────

_opencode_proc: subprocess.Popen | None = None


def _start_opencode_serve(host: str, port: int) -> None:
    global _opencode_proc
    if _opencode_proc is not None:
        return

    opencode_bin = shutil.which("opencode")
    if opencode_bin is None:
        for candidate in [
            Path.home() / ".local" / "bin" / "opencode",
            Path.home() / ".opencode" / "bin" / "opencode",
            Path("/usr/local/bin/opencode"),
        ]:
            if candidate.exists():
                opencode_bin = str(candidate)
                break
    if opencode_bin is None:
        LOG.error("opencode binary not found — AI will be unavailable")
        return

    cmd = [opencode_bin, "serve", "--port", str(port), "--hostname", host]
    LOG.info("Starting codex OpenCode: %s", " ".join(cmd))

    fh = open(BASE_DIR / "opencode_serve.log", "w", encoding="utf-8")
    _opencode_proc = subprocess.Popen(
        cmd, cwd=str(BASE_DIR), stdout=fh, stderr=subprocess.STDOUT,
    )
    _opencode_proc._serve_log_fh = fh  # type: ignore[attr-defined]
    LOG.info("Codex OpenCode started (pid %d)", _opencode_proc.pid)


def _stop_opencode_serve() -> None:
    global _opencode_proc
    if _opencode_proc is None:
        return
    proc = _opencode_proc
    _opencode_proc = None
    try:
        proc.terminate()
        proc.wait(timeout=10)
    except subprocess.TimeoutExpired:
        proc.kill()
        proc.wait(timeout=5)
    except Exception:
        pass
    fh = getattr(proc, "_serve_log_fh", None)
    if fh:
        try:
            fh.close()
        except Exception:
            pass


# ── main ────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Codex Agent MCP Server")
    parser.add_argument("--port", type=int, default=8012, help="HTTP API port")
    parser.add_argument("--host", type=str, default="0.0.0.0")
    args = parser.parse_args()

    api_port = args.port
    mcp_port = api_port + 600  # 8612

    cfg = _load_config()

    # ── Logging ─────────────────────────────────────────────────
    log_level = cfg.get("log_level", "INFO").upper()

    log_file = BASE_DIR / "log.out"
    file_handler = logging.FileHandler(log_file, mode="w", encoding="utf-8")
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(logging.Formatter(
        "%(asctime)s [%(levelname)-7s] %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    ))

    console_handler = logging.StreamHandler()
    console_handler.setLevel(getattr(logging, log_level, logging.INFO))
    console_handler.setFormatter(logging.Formatter(
        "%(asctime)s [%(levelname)-7s] %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    ))

    root_logger = logging.getLogger()
    root_logger.setLevel(logging.DEBUG)
    root_logger.addHandler(file_handler)
    root_logger.addHandler(console_handler)

    # ── Boot OpenCode instance ──────────────────────────────────
    oc_cfg = cfg.get("opencode", {})
    oc_host = oc_cfg.get("host", "127.0.0.1")
    oc_port = oc_cfg.get("port", 4097)
    oc_timeout = oc_cfg.get("request_timeout", 600)

    LOG.info("Booting codex agent on port %d (MCP: %d)", api_port, mcp_port)
    _start_opencode_serve(oc_host, oc_port)
    wait_for_opencode(oc_host, oc_port, retries=30, delay=2.0)

    # ── Wire up dependencies ───────────────────────────────────
    tracker = JobTracker()
    inventory = ToolInventory(MY_TOOLS_DIR, cfg)

    def client_factory() -> CodexClient:
        return CodexClient(host=oc_host, port=oc_port, timeout=oc_timeout)

    build_worker = BuildWorker(inventory, client_factory)
    repair_worker = RepairWorker(inventory, client_factory)

    init_app(tracker, build_worker, repair_worker, MY_TOOLS_DIR)

    # ── Serve FastAPI + MCP ─────────────────────────────────────
    from fastmcp import FastMCP

    mcp = FastMCP.from_fastapi(app=app, name=f"robot_codex (port {api_port})")
    mcp_app = mcp.http_app(path="/mcp")

    async def serve():
        api_server = uvicorn.Server(uvicorn.Config(
            app, host=args.host, port=api_port, log_level="info",
        ))
        mcp_server = uvicorn.Server(uvicorn.Config(
            mcp_app, host=args.host, port=mcp_port, log_level="info",
        ))
        api_server.install_signal_handlers = lambda: None
        mcp_server.install_signal_handlers = lambda: None

        def shutdown():
            api_server.should_exit = True
            mcp_server.should_exit = True

        loop = asyncio.get_running_loop()
        for sig in (signal.SIGINT, signal.SIGTERM):
            try:
                loop.add_signal_handler(sig, shutdown)
            except NotImplementedError:
                signal.signal(sig, lambda *_: shutdown())

        t1 = asyncio.create_task(api_server.serve())
        t2 = asyncio.create_task(mcp_server.serve())
        await asyncio.wait({t1, t2}, return_when=asyncio.FIRST_EXCEPTION)
        shutdown()

    try:
        asyncio.run(serve())
    finally:
        _stop_opencode_serve()


if __name__ == "__main__":
    main()
