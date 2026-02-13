#!/usr/bin/env python3
"""
Repair Agent — MCP Tool Server
================================
A FastAPI + FastMCP server that the main robot agent can call as a native
MCP tool (``robot_repair``).  Under the hood it runs its own OpenCode
instance (port 4097) to do the actual code diagnosis and fixing.

Endpoints (each becomes an MCP tool):
  POST /diagnose    — read a tool's server.log + server.py, return diagnosis
  POST /repair      — diagnose AND fix a broken tool, restart it, re-register
  POST /scan_all    — scan every tool in my_tools/, return list of broken ones
  GET  /healthz     — liveness check

Usage (auto-started by main.py supervisor):
    uv run python OpenCode/repair_agent/main_repair.py --port 8012
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import os
import re
import shutil
import signal
import subprocess
import sys
import textwrap
import time
from pathlib import Path
from typing import Any

import yaml

try:
    from fastapi import FastAPI
    from pydantic import BaseModel, Field
    import uvicorn
    from fastmcp import FastMCP
except ImportError as exc:
    sys.exit(
        f"Missing dependency: {exc}\n"
        "Install with:  cd /home/engelbot/Desktop/pi_rc_bot && uv add fastapi uvicorn fastmcp pydantic"
    )

try:
    import requests
except ImportError:
    sys.exit("ERROR: 'requests' is required.  Install: uv add requests")

# ──────────────────────────── paths ──────────────────────────────

BASE_DIR = Path(__file__).resolve().parent          # repair_agent/
OPENCODE_DIR = BASE_DIR.parent                      # OpenCode/
MY_TOOLS_DIR = OPENCODE_DIR / "my_tools"
CONFIG_PATH = BASE_DIR / "config.yaml"
PROJECT_ROOT = OPENCODE_DIR.parent                  # pi_rc_bot/

LOG = logging.getLogger("repair-agent")

# ──────────────────────────── config ─────────────────────────────

def _load_config() -> dict:
    cfg: dict = {}
    if CONFIG_PATH.exists():
        with open(CONFIG_PATH) as f:
            cfg = yaml.safe_load(f) or {}
    return cfg

_CFG = _load_config()

# ──────────────────────────── OpenCode serve (for repair AI) ─────

_opencode_proc: subprocess.Popen | None = None


def _find_opencode_bin() -> str | None:
    b = shutil.which("opencode")
    if b:
        return b
    for candidate in [
        Path.home() / ".local" / "bin" / "opencode",
        Path.home() / ".opencode" / "bin" / "opencode",
        Path("/usr/local/bin/opencode"),
    ]:
        if candidate.exists():
            return str(candidate)
    return None


def _start_opencode_serve() -> None:
    """Boot a dedicated OpenCode instance for the repair agent."""
    global _opencode_proc
    if _opencode_proc is not None:
        return

    oc_cfg = _CFG.get("opencode", {})
    host = oc_cfg.get("host", "127.0.0.1")
    port = oc_cfg.get("port", 4097)

    opencode_bin = _find_opencode_bin()
    if opencode_bin is None:
        LOG.error("opencode binary not found — repair AI will be unavailable")
        return

    cmd = [opencode_bin, "serve", "--port", str(port), "--hostname", host]
    LOG.info("Starting repair OpenCode: %s", " ".join(cmd))

    serve_log = BASE_DIR / "opencode_serve.log"
    fh = open(serve_log, "w", encoding="utf-8")

    _opencode_proc = subprocess.Popen(
        cmd,
        cwd=str(BASE_DIR),         # picks up repair_agent/opencode.json
        stdout=fh,
        stderr=subprocess.STDOUT,
    )
    _opencode_proc._serve_log_fh = fh  # type: ignore[attr-defined]
    LOG.info("Repair OpenCode started (pid %d)", _opencode_proc.pid)


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


def _wait_for_opencode(retries: int = 30, delay: float = 2.0) -> bool:
    oc = _CFG.get("opencode", {})
    base = f"http://{oc.get('host', '127.0.0.1')}:{oc.get('port', 4097)}"
    for i in range(retries):
        try:
            r = requests.get(f"{base}/global/health", timeout=5)
            if r.ok and r.json().get("healthy", False):
                LOG.info("Repair OpenCode healthy at %s", base)
                return True
        except Exception:
            pass
        LOG.info("Waiting for repair OpenCode (%d/%d)…", i + 1, retries)
        time.sleep(delay)
    LOG.error("Repair OpenCode not reachable after %d attempts", retries)
    return False


# ──────────────────────────── OpenCode repair client ─────────────

class _RepairClient:
    """Send a repair prompt to the dedicated OpenCode instance."""

    def __init__(self):
        oc = _CFG.get("opencode", {})
        self.base = f"http://{oc.get('host', '127.0.0.1')}:{oc.get('port', 4097)}"
        self.timeout = oc.get("request_timeout", 90)
        self._session_id: str | None = None

    @property
    def session_id(self) -> str:
        if self._session_id is None:
            r = requests.post(
                f"{self.base}/session",
                json={"title": "repair"},
                timeout=self.timeout,
            )
            r.raise_for_status()
            data = r.json()
            self._session_id = data.get("id") or data.get("ID") or data.get("sessionID")
        return self._session_id  # type: ignore[return-value]

    def new_session(self) -> None:
        self._session_id = None
        _ = self.session_id

    def send(self, prompt: str) -> str:
        url = f"{self.base}/session/{self.session_id}/message"
        r = requests.post(
            url,
            json={
                "parts": [{"type": "text", "text": prompt}],
                "agent": "repair",
            },
            timeout=self.timeout,
        )
        r.raise_for_status()
        raw = r.text.strip()
        if not raw:
            return ""
        try:
            data = r.json()
        except Exception:
            return raw[:2000]
        parts = data.get("parts", [])
        texts = []
        for p in parts:
            if isinstance(p, dict):
                if p.get("type") == "text":
                    texts.append(p.get("text", ""))
                elif "content" in p:
                    texts.append(str(p["content"]))
        return "\n".join(texts) if texts else raw[:2000]


_repair_client: _RepairClient | None = None
_repair_turn_count = 0


def _get_client() -> _RepairClient:
    global _repair_client, _repair_turn_count
    if _repair_client is None:
        _repair_client = _RepairClient()
    _repair_turn_count += 1
    if _repair_turn_count % 5 == 0:
        try:
            _repair_client.new_session()
        except Exception:
            pass
    return _repair_client


# ──────────────────────────── error helpers ──────────────────────

_ERROR_PATTERNS = [
    re.compile(r"Traceback \(most recent call last\)", re.IGNORECASE),
    re.compile(
        r"^\s*(ModuleNotFoundError|ImportError|SyntaxError|NameError|TypeError|"
        r"AttributeError|ValueError|KeyError|RuntimeError|IndentationError|"
        r"TabError|FileNotFoundError|PermissionError|OSError|ConnectionError|"
        r"TimeoutError):",
        re.MULTILINE,
    ),
    re.compile(r"ERROR:\s+.*?(?:exception|error|failed|crash)", re.IGNORECASE),
    re.compile(r"Process finished with exit code [^0]", re.IGNORECASE),
    re.compile(r"uvicorn.*?ERROR", re.IGNORECASE),
]


def _has_errors(text: str) -> bool:
    return any(p.search(text) for p in _ERROR_PATTERNS)


def _extract_error_block(text: str) -> str:
    lines = text.splitlines()
    last_tb = -1
    for i, line in enumerate(lines):
        if "Traceback (most recent call last)" in line:
            last_tb = i
    if last_tb >= 0:
        return "\n".join(lines[last_tb:])
    err = [l for l in lines if re.search(r"error|exception|failed", l, re.IGNORECASE)]
    if err:
        return "\n".join(err[-20:])
    return "\n".join(lines[-30:])


def _read_tool(tool_name: str) -> dict:
    """Gather all info about a tool: source, log tail, port, status."""
    tool_dir = MY_TOOLS_DIR / tool_name
    if not tool_dir.is_dir():
        return {"error": f"Tool '{tool_name}' not found in my_tools/"}

    server_py = tool_dir / "server.py"
    log_file = tool_dir / "server.log"
    port_file = tool_dir / "port.txt"

    source = server_py.read_text("utf-8") if server_py.exists() else "(no server.py)"
    tail_lines = _CFG.get("log_tail_lines", 80)
    if log_file.exists():
        all_lines = log_file.read_text("utf-8", errors="replace").splitlines()
        log_tail = "\n".join(all_lines[-tail_lines:])
    else:
        log_tail = "(no server.log)"
    port = int(port_file.read_text().strip()) if port_file.exists() else 9100

    # Check process
    pid_file = tool_dir / "server.pid"
    process_alive = False
    if pid_file.exists():
        try:
            os.kill(int(pid_file.read_text().strip()), 0)
            process_alive = True
        except (OSError, ValueError):
            pass

    # Check healthz
    healthy = False
    try:
        r = requests.get(f"http://127.0.0.1:{port}/healthz", timeout=3)
        healthy = r.ok
    except Exception:
        pass

    has_err = _has_errors(log_tail)
    error_block = _extract_error_block(log_tail) if has_err else ""

    return {
        "tool_name": tool_name,
        "tool_dir": str(tool_dir),
        "port": port,
        "mcp_port": port + 600,
        "process_alive": process_alive,
        "healthy": healthy,
        "has_errors": has_err,
        "error_block": error_block,
        "log_tail": log_tail,
        "source": source,
    }


def _build_repair_prompt(info: dict) -> str:
    return textwrap.dedent(f"""\
        [REPAIR] A custom MCP tool server is broken. Fix it.

        ## Tool info
        - Name: {info['tool_name']}
        - Path: {info['tool_dir']}
        - Port: {info['port']} (MCP: {info['mcp_port']})
        - Process alive: {info['process_alive']}
        - Healthy: {info['healthy']}

        ## server.py (full source)
        ```python
        {info['source']}
        ```

        ## server.log (error section)
        ```
        {info['error_block']}
        ```

        ## Full log tail (last lines)
        ```
        {info['log_tail'][-3000:]}
        ```

        ## Your task
        1. Diagnose the error from the log.
        2. Write the fixed server.py using bash: cat > {info['tool_dir']}/server.py << 'PYEOF' ... PYEOF
        3. If a missing package caused the error: cd /home/engelbot/Desktop/pi_rc_bot && uv add PACKAGE
        4. Verify: python3 -c "import ast; ast.parse(open('{info['tool_dir']}/server.py').read())"
        5. DO NOT start the server or register it — the caller handles that.

        Respond with FIXED: <description> or UNFIXABLE: <reason>
    """)


def _restart_tool(tool_name: str, port: int) -> bool:
    """Kill old process, start fresh, return True if healthy."""
    tool_dir = MY_TOOLS_DIR / tool_name
    pid_file = tool_dir / "server.pid"

    # Kill old
    if pid_file.exists():
        try:
            os.kill(int(pid_file.read_text().strip()), signal.SIGTERM)
            time.sleep(1)
        except (OSError, ValueError):
            pass
        pid_file.unlink(missing_ok=True)

    # Clear log and start fresh
    log_file = tool_dir / "server.log"
    log_fh = open(log_file, "w", encoding="utf-8")

    try:
        proc = subprocess.Popen(
            [sys.executable, str(tool_dir / "server.py"), "--port", str(port)],
            cwd=str(tool_dir),
            stdout=log_fh,
            stderr=subprocess.STDOUT,
        )
        pid_file.write_text(str(proc.pid))
        LOG.info("Restarted %s on port %d (pid %d)", tool_name, port, proc.pid)
    except Exception as exc:
        LOG.error("Failed to restart %s: %s", tool_name, exc)
        return False

    # Wait and check health
    time.sleep(3)
    try:
        r = requests.get(f"http://127.0.0.1:{port}/healthz", timeout=5)
        if r.ok:
            LOG.info("✅ %s healthy after restart", tool_name)
            return True
    except Exception:
        pass
    LOG.warning("❌ %s still unhealthy after restart", tool_name)
    return False


def _re_register_tool(tool_name: str, mcp_port: int) -> bool:
    """Hot-register with the MAIN OpenCode (port 4096)."""
    main_port = _CFG.get("main_opencode_port", 4096)
    name = f"my_{tool_name}"
    try:
        r = requests.post(
            f"http://127.0.0.1:{main_port}/mcp",
            json={
                "name": name,
                "config": {
                    "type": "remote",
                    "url": f"http://127.0.0.1:{mcp_port}/mcp",
                    "enabled": True,
                },
            },
            timeout=10,
        )
        return r.ok
    except Exception:
        return False


# ══════════════════════════════════════════════════════════════════
#  FastAPI  Application
# ══════════════════════════════════════════════════════════════════

app = FastAPI(title="robot_repair", version="0.2.0")


@app.get("/healthz")
async def healthz():
    return {"ok": True, "service": "robot_repair"}


# ── POST /diagnose ──────────────────────────────────────────────

class DiagnoseRequest(BaseModel):
    tool_name: str = Field(description="Name of the tool folder inside my_tools/ (e.g. 'my_web_search')")

class DiagnoseResponse(BaseModel):
    ok: bool = True
    tool_name: str = ""
    healthy: bool = False
    process_alive: bool = False
    has_errors: bool = False
    error_block: str = ""
    log_tail: str = ""
    diagnosis: str = ""


@app.post("/diagnose", response_model=DiagnoseResponse)
async def diagnose(req: DiagnoseRequest):
    """Read a tool's logs and source code, return a diagnosis of what's wrong.
    Does NOT fix or restart the tool — use /repair for that."""
    info = _read_tool(req.tool_name)
    if "error" in info:
        return DiagnoseResponse(ok=False, tool_name=req.tool_name, diagnosis=info["error"])

    if info["healthy"] and not info["has_errors"]:
        return DiagnoseResponse(
            ok=True,
            tool_name=req.tool_name,
            healthy=True,
            process_alive=info["process_alive"],
            has_errors=False,
            diagnosis="Tool is healthy — no errors found.",
        )

    # Use OpenCode to diagnose
    prompt = (
        f"[DIAGNOSE ONLY — do NOT write files or fix anything]\n\n"
        f"Tool: {info['tool_name']} at {info['tool_dir']}\n"
        f"Port: {info['port']}, process_alive={info['process_alive']}, healthy={info['healthy']}\n\n"
        f"## server.py\n```python\n{info['source']}\n```\n\n"
        f"## Error from server.log\n```\n{info['error_block']}\n```\n\n"
        f"## Full log tail\n```\n{info['log_tail'][-2000:]}\n```\n\n"
        f"Explain what's wrong in 1-3 sentences. Start with 'DIAGNOSIS:'"
    )
    try:
        client = _get_client()
        reply = client.send(prompt)
    except Exception as exc:
        reply = f"Could not reach repair AI: {exc}"

    return DiagnoseResponse(
        ok=True,
        tool_name=req.tool_name,
        healthy=info["healthy"],
        process_alive=info["process_alive"],
        has_errors=info["has_errors"],
        error_block=info["error_block"][:500],
        log_tail=info["log_tail"][-500:],
        diagnosis=reply[:1000],
    )


# ── POST /repair ────────────────────────────────────────────────

class RepairRequest(BaseModel):
    tool_name: str = Field(description="Name of the tool folder inside my_tools/ (e.g. 'my_web_search')")

class RepairResponse(BaseModel):
    ok: bool = True
    tool_name: str = ""
    was_broken: bool = False
    fix_applied: bool = False
    restarted: bool = False
    re_registered: bool = False
    healthy_after: bool = False
    summary: str = ""


@app.post("/repair", response_model=RepairResponse)
async def repair(req: RepairRequest):
    """Diagnose, fix, restart, and re-register a broken tool.
    Uses an AI (OpenCode) under the hood to read the logs, understand the error,
    and write a fixed server.py. Then restarts the process and re-registers with
    the main agent's OpenCode."""
    info = _read_tool(req.tool_name)
    if "error" in info:
        return RepairResponse(ok=False, tool_name=req.tool_name, summary=info["error"])

    if info["healthy"] and not info["has_errors"]:
        return RepairResponse(
            ok=True,
            tool_name=req.tool_name,
            was_broken=False,
            healthy_after=True,
            summary="Tool is already healthy — nothing to repair.",
        )

    # 1. Send to OpenCode for fix
    prompt = _build_repair_prompt(info)
    try:
        client = _get_client()
        reply = client.send(prompt)
        LOG.info("Repair AI reply for %s: %s", req.tool_name, reply[:300])
    except Exception as exc:
        return RepairResponse(
            ok=False,
            tool_name=req.tool_name,
            was_broken=True,
            summary=f"Repair AI failed: {exc}",
        )

    if "UNFIXABLE" in reply.upper():
        return RepairResponse(
            ok=True,
            tool_name=req.tool_name,
            was_broken=True,
            fix_applied=False,
            summary=f"AI says unfixable: {reply[:500]}",
        )

    fix_applied = "FIXED" in reply.upper()

    # 2. Restart the tool
    restarted = _restart_tool(req.tool_name, info["port"])

    # 3. Re-register with main OpenCode
    re_registered = False
    if restarted:
        re_registered = _re_register_tool(req.tool_name, info["mcp_port"])

    # 4. Final health check
    healthy_after = False
    try:
        r = requests.get(f"http://127.0.0.1:{info['port']}/healthz", timeout=5)
        healthy_after = r.ok
    except Exception:
        pass

    # 5. Write repair log
    repair_log = MY_TOOLS_DIR / req.tool_name / "repair.log"
    try:
        with open(repair_log, "a", encoding="utf-8") as f:
            f.write(f"\n{'='*60}\n")
            f.write(f"Time: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"Result: {'FIXED' if healthy_after else 'ATTEMPTED'}\n")
            f.write(f"Reply: {reply[:500]}\n")
    except Exception:
        pass

    summary_parts = []
    if fix_applied:
        summary_parts.append("Fix applied by AI")
    if restarted:
        summary_parts.append("process restarted")
    if re_registered:
        summary_parts.append("re-registered with OpenCode")
    if healthy_after:
        summary_parts.append("✅ tool is now healthy!")
    else:
        summary_parts.append("❌ tool still unhealthy after repair attempt")
    summary_parts.append(f"AI said: {reply[:300]}")

    return RepairResponse(
        ok=True,
        tool_name=req.tool_name,
        was_broken=True,
        fix_applied=fix_applied,
        restarted=restarted,
        re_registered=re_registered,
        healthy_after=healthy_after,
        summary=" | ".join(summary_parts),
    )


# ── POST /scan_all ──────────────────────────────────────────────

class ScanAllResponse(BaseModel):
    ok: bool = True
    total_tools: int = 0
    healthy_tools: list[str] = []
    broken_tools: list[str] = []
    details: dict[str, str] = {}


@app.post("/scan_all", response_model=ScanAllResponse)
async def scan_all():
    """Scan all tools in my_tools/ and return which are healthy vs broken."""
    if not MY_TOOLS_DIR.exists():
        return ScanAllResponse()

    healthy: list[str] = []
    broken: list[str] = []
    details: dict[str, str] = {}

    for tool_dir in sorted(MY_TOOLS_DIR.iterdir()):
        if not tool_dir.is_dir() or tool_dir.name.startswith(("_", ".")):
            continue
        if not (tool_dir / "server.py").exists():
            continue

        info = _read_tool(tool_dir.name)
        if "error" in info:
            continue

        if info["healthy"] and not info["has_errors"]:
            healthy.append(tool_dir.name)
            details[tool_dir.name] = "healthy"
        else:
            broken.append(tool_dir.name)
            reason = []
            if not info["process_alive"]:
                reason.append("process dead")
            if not info["healthy"]:
                reason.append("healthz failed")
            if info["has_errors"]:
                reason.append("errors in log")
            details[tool_dir.name] = ", ".join(reason) if reason else "unknown issue"

    return ScanAllResponse(
        total_tools=len(healthy) + len(broken),
        healthy_tools=healthy,
        broken_tools=broken,
        details=details,
    )


# ══════════════════════════════════════════════════════════════════
#  Server bootstrap
# ══════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(description="Repair Agent MCP Server")
    parser.add_argument("--port", type=int, default=8012, help="HTTP API port")
    parser.add_argument("--host", type=str, default="0.0.0.0")
    args = parser.parse_args()

    api_port = args.port
    mcp_port = api_port + 600   # 8612

    # Setup logging
    log_file = BASE_DIR / "log.out"
    logging.basicConfig(
        level=logging.DEBUG,
        format="%(asctime)s [%(levelname)-7s] %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        handlers=[
            logging.FileHandler(log_file, mode="w", encoding="utf-8"),
            logging.StreamHandler(),
        ],
    )

    # Start the dedicated OpenCode instance for AI-powered repairs
    LOG.info("Booting repair agent MCP server on port %d (MCP: %d)", api_port, mcp_port)
    _start_opencode_serve()
    _wait_for_opencode(retries=30, delay=2.0)

    # Build MCP app
    mcp = FastMCP.from_fastapi(app=app, name=f"robot_repair (port {api_port})")
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
