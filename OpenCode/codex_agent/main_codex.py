#!/usr/bin/env python3
"""
Codex Agent — MCP Tool Server
================================
A FastAPI + FastMCP server that the main robot agent can call as a native
MCP tool (``robot_codex``).  Under the hood it runs its own OpenCode
instance (port 4097) to do the actual code diagnosis and fixing.

Endpoints (each becomes an MCP tool):
  POST /diagnose    — read a tool's server.log + server.py, return diagnosis
  POST /repair      — diagnose AND fix a broken tool, restart it, re-register
  POST /scan_all    — scan every tool in my_tools/, return list of broken ones
  GET  /healthz     — liveness check

Usage (auto-started by main.py supervisor):
    uv run python OpenCode/codex_agent/main_codex.py --port 8012
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
import threading
import time
import uuid
from dataclasses import dataclass, field as dc_field
from enum import Enum
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

BASE_DIR = Path(__file__).resolve().parent          # codex_agent/
OPENCODE_DIR = BASE_DIR.parent                      # OpenCode/
MY_TOOLS_DIR = OPENCODE_DIR / "my_tools"
CONFIG_PATH = BASE_DIR / "config.yaml"
PROJECT_ROOT = OPENCODE_DIR.parent                  # pi_rc_bot/

LOG = logging.getLogger("codex-agent")


_SECRET_RE = re.compile(r"sk-[A-Za-z0-9]{10,}")


def _redact_secrets(text: str) -> str:
    """Best-effort redaction for logs (avoid leaking API keys into log files)."""
    if not text:
        return text
    text = _SECRET_RE.sub("sk-***", text)
    text = re.sub(r"(OPENAI_API_KEY\s*=\s*)(\S+)", r"\1***", text)
    return text

# ──────────────────────────── config ─────────────────────────────

def _load_config() -> dict:
    cfg: dict = {}
    if CONFIG_PATH.exists():
        with open(CONFIG_PATH) as f:
            cfg = yaml.safe_load(f) or {}
    return cfg

_CFG = _load_config()

# ──────────────────────────── job tracker ────────────────────────


class JobKind(str, Enum):
    BUILD = "build"
    REPAIR = "repair"


class JobState(str, Enum):
    QUEUED = "queued"
    RUNNING = "running"
    DONE = "done"
    FAILED = "failed"


@dataclass
class Job:
    job_id: str
    kind: JobKind
    tool_name: str
    state: JobState = JobState.QUEUED
    phase: str = "queued"          # human-readable current step
    progress: list[str] = dc_field(default_factory=list)  # log of completed phases
    result: dict[str, Any] = dc_field(default_factory=dict)
    created_at: float = dc_field(default_factory=time.time)
    finished_at: float | None = None

    def log(self, msg: str) -> None:
        ts = time.strftime("%H:%M:%S")
        entry = f"[{ts}] {msg}"
        self.progress.append(entry)
        LOG.info("Job %s (%s/%s): %s", self.job_id[:8], self.kind.value, self.tool_name, msg)

    def elapsed(self) -> float:
        end = self.finished_at or time.time()
        return round(end - self.created_at, 1)


_jobs: dict[str, Job] = {}   # job_id → Job
_jobs_lock = threading.Lock()


def _create_job(kind: JobKind, tool_name: str) -> Job:
    jid = uuid.uuid4().hex[:12]
    job = Job(job_id=jid, kind=kind, tool_name=tool_name)
    with _jobs_lock:
        _jobs[jid] = job
    return job


def _get_job(job_id: str) -> Job | None:
    with _jobs_lock:
        return _jobs.get(job_id)


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
        cwd=str(BASE_DIR),         # picks up codex_agent/opencode.json
        stdout=fh,
        stderr=subprocess.STDOUT,
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


def _wait_for_opencode(retries: int = 30, delay: float = 2.0) -> bool:
    oc = _CFG.get("opencode", {})
    base = f"http://{oc.get('host', '127.0.0.1')}:{oc.get('port', 4097)}"
    for i in range(retries):
        try:
            r = requests.get(f"{base}/global/health", timeout=5)
            if r.ok and r.json().get("healthy", False):
                LOG.info("Codex OpenCode healthy at %s", base)
                return True
        except Exception:
            pass
        LOG.info("Waiting for codex OpenCode (%d/%d)…", i + 1, retries)
        time.sleep(delay)
    LOG.error("Codex OpenCode not reachable after %d attempts", retries)
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
                "agent": "codex",
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
        texts: list[str] = []
        tool_calls: list[str] = []
        tool_results: list[str] = []

        def _is_tool_call(part: dict) -> bool:
            t = str(part.get("type", ""))
            if t in {"tool-invocation", "tool_invocation", "tool-call", "tool_call", "tool"}:
                return True
            # Some builds omit type but include toolName/tool + args
            if ("toolName" in part or "tool" in part) and any(k in part for k in ("input", "args", "parameters")):
                return True
            return False

        def _is_tool_result(part: dict) -> bool:
            t = str(part.get("type", ""))
            if t in {"tool-result", "tool_result", "toolResult"}:
                return True
            # Heuristic: has output/result/content and also references a tool
            if any(k in part for k in ("output", "result")) and ("toolName" in part or "tool" in part or "name" in part):
                return True
            return False

        for p in parts:
            if not isinstance(p, dict):
                continue
            if p.get("type") == "text":
                texts.append(p.get("text", ""))
                continue
            if _is_tool_call(p):
                tool_name = p.get("toolName") or p.get("tool") or p.get("name") or "unknown_tool"
                tool_input = p.get("input") or p.get("args") or p.get("parameters") or {}
                summary = ""
                if isinstance(tool_input, dict):
                    cmd = tool_input.get("command") or tool_input.get("cmd") or tool_input.get("shell") or ""
                    file_path = tool_input.get("filePath") or tool_input.get("file") or tool_input.get("path") or ""
                    if cmd:
                        summary = f"🔧 {tool_name}: {cmd}"
                    elif file_path:
                        summary = f"🔧 {tool_name}: {file_path}"
                    else:
                        summary = f"🔧 {tool_name}: {tool_input}"
                else:
                    summary = f"🔧 {tool_name}: {tool_input}"
                summary = _redact_secrets(str(summary))[:220]
                tool_calls.append(summary)
                if _CFG.get("log_ai_details", True):
                    LOG.info("AI tool call: %s", summary)
                continue
            if _is_tool_result(p):
                output = p.get("output") or p.get("result") or p.get("content") or ""
                if isinstance(output, list):
                    output = " ".join(str(x) for x in output)
                out_s = _redact_secrets(str(output)).strip()
                if out_s:
                    tool_results.append(out_s[:300])
                    if _CFG.get("log_ai_details", True):
                        LOG.debug("AI tool result: %s", out_s[:200])
                continue
            if "content" in p:
                texts.append(str(p["content"]))

        # Store for callers to access (job logs, status polling, etc.)
        self._last_response = data
        self._last_tool_calls = tool_calls
        self._last_tool_results = tool_results
        joined = "\n".join(texts) if texts else raw[:2000]
        self._last_text = joined
        return joined

    @property
    def last_tool_calls(self) -> list[str]:
        """Tool calls from the most recent send() — bash commands, file edits, etc."""
        return getattr(self, "_last_tool_calls", [])

    @property
    def last_tool_results(self) -> list[str]:
        """Tool outputs from the most recent send() (truncated)."""
        return getattr(self, "_last_tool_results", [])

    @property
    def last_text(self) -> str:
        """Assistant text from the most recent send()."""
        return getattr(self, "_last_text", "")


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

app = FastAPI(title="robot_codex", version="0.2.0")


@app.get("/healthz")
async def healthz():
    return {"ok": True, "service": "robot_codex"}


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
    job_id: str = ""


def _run_repair_sync(job: Job, tool_name: str) -> None:
    """Background worker for repair — runs in a thread."""
    try:
        job.state = JobState.RUNNING

        info = _read_tool(tool_name)
        if "error" in info:
            job.phase = "tool not found"
            job.log(info["error"])
            job.state = JobState.FAILED
            job.result = {"ok": False, "tool_name": tool_name, "summary": info["error"]}
            job.finished_at = time.time()
            return

        if info["healthy"] and not info["has_errors"]:
            job.phase = "already healthy"
            job.log("Tool is already healthy — nothing to repair.")
            job.state = JobState.DONE
            job.result = {"ok": True, "tool_name": tool_name, "was_broken": False,
                          "healthy_after": True, "summary": "Tool is already healthy — nothing to repair."}
            job.finished_at = time.time()
            return

        # 1. Send to OpenCode for fix
        job.phase = "sending repair prompt to AI"
        job.log("Reading logs and source, sending to code AI…")
        prompt = _build_repair_prompt(info)
        try:
            client = _get_client()
            reply = client.send(prompt)
            LOG.info("Repair AI reply for %s: %s", tool_name, reply[:300])
            # Log what the AI did (tool calls = bash commands, file edits)
            for tc in client.last_tool_calls:
                job.log(tc)
            # Log the AI's reasoning/conclusion
            if reply.strip():
                job.log(f"AI says: {reply[:300]}")
            else:
                job.log("AI replied (no text — only tool actions)")
        except Exception as exc:
            job.phase = "AI error"
            job.log(f"Repair AI failed: {exc}")
            job.state = JobState.FAILED
            job.result = {"ok": False, "tool_name": tool_name, "was_broken": True,
                          "summary": f"Repair AI failed: {exc}"}
            job.finished_at = time.time()
            return

        if "UNFIXABLE" in reply.upper():
            job.phase = "unfixable"
            job.log(f"AI says unfixable: {reply[:200]}")
            job.state = JobState.FAILED
            job.result = {"ok": True, "tool_name": tool_name, "was_broken": True,
                          "fix_applied": False, "summary": f"AI says unfixable: {reply[:500]}"}
            job.finished_at = time.time()
            return

        fix_applied = "FIXED" in reply.upper()
        if fix_applied:
            job.log("AI applied a fix")

        # 2. Restart the tool
        job.phase = "restarting server"
        job.log("Restarting tool server…")
        restarted = _restart_tool(tool_name, info["port"])
        if restarted:
            job.log("✅ Server restarted")
        else:
            job.log("❌ Server failed to restart")

        # 3. Re-register with main OpenCode
        re_registered = False
        if restarted:
            job.phase = "re-registering with OpenCode"
            job.log("Re-registering with main OpenCode…")
            re_registered = _re_register_tool(tool_name, info["mcp_port"])
            if re_registered:
                job.log("✅ Re-registered")

        # 4. Final health check
        job.phase = "health check"
        healthy_after = False
        try:
            r = requests.get(f"http://127.0.0.1:{info['port']}/healthz", timeout=5)
            healthy_after = r.ok
        except Exception:
            pass
        if healthy_after:
            job.log("✅ Tool is now healthy!")
        else:
            job.log("❌ Tool still unhealthy after repair attempt")

        # 5. Write repair log
        repair_log = MY_TOOLS_DIR / tool_name / "repair.log"
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

        job.phase = "done" if healthy_after else "done (still broken)"
        job.state = JobState.DONE if healthy_after else JobState.FAILED
        job.result = {
            "ok": True, "tool_name": tool_name, "was_broken": True,
            "fix_applied": fix_applied, "restarted": restarted,
            "re_registered": re_registered, "healthy_after": healthy_after,
            "summary": " | ".join(summary_parts),
        }
        job.finished_at = time.time()
        job.log(f"Repair finished in {job.elapsed()}s — {'FIXED' if healthy_after else 'STILL BROKEN'}")

    except Exception as exc:
        job.phase = "unexpected error"
        job.log(f"Unexpected error: {exc}")
        job.state = JobState.FAILED
        job.result = {"ok": False, "tool_name": tool_name, "summary": f"Unexpected error: {exc}"}
        job.finished_at = time.time()


@app.post("/repair", response_model=RepairResponse)
async def repair(req: RepairRequest):
    """Diagnose, fix, restart, and re-register a broken tool.

    Runs **asynchronously** — returns a job_id immediately.
    Poll ``/build_status`` with the job_id to track progress.
    """
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

    job = _create_job(JobKind.REPAIR, req.tool_name)
    job.log(f"Repair queued — tool={req.tool_name}")

    thread = threading.Thread(
        target=_run_repair_sync,
        args=(job, req.tool_name),
        daemon=True,
    )
    thread.start()

    return RepairResponse(
        ok=True,
        tool_name=req.tool_name,
        was_broken=True,
        summary=f"Repair started — job_id={job.job_id}. Poll build_status for progress.",
        job_id=job.job_id,
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


# ── POST /build_tool ────────────────────────────────────────────

def _next_free_port() -> int:
    """Find the next unused port starting at 9100 by reading existing port.txt files."""
    used: set[int] = set()
    if MY_TOOLS_DIR.exists():
        for d in MY_TOOLS_DIR.iterdir():
            pf = d / "port.txt"
            if pf.exists():
                try:
                    used.add(int(pf.read_text().strip()))
                except ValueError:
                    pass
    port = 9100
    while port in used:
        port += 1
    return port


def _build_tool_prompt(tool_name: str, description: str, port: int, tool_dir: str) -> str:
    return textwrap.dedent(f"""\
        [BUILD NEW MCP TOOL]

        Create a fully working FastAPI + FastMCP tool server.

        ## Requirements
        - Tool name: {tool_name}
        - Description: {description}
        - API port: {port}
        - MCP port: {port + 600}
        - Tool directory: {tool_dir}

        ## Steps
        1. Think about what Python packages are needed. Install them:
           cd /home/engelbot/Desktop/pi_rc_bot && uv add PACKAGE_NAME
        2. Write the complete server.py:
           cat > {tool_dir}/server.py << 'PYEOF'
           ... (full code) ...
           PYEOF
        3. The server.py MUST follow this exact structure:
           - imports at top
           - FastAPI app = FastAPI(title="{tool_name}", version="0.1.0")
           - @app.get("/healthz") returning {{"ok": True, "service": "{tool_name}"}}
           - One or more @app.post() endpoints, each with Pydantic request/response models
           - Every POST endpoint becomes an MCP tool — give them clear names and Field descriptions
           - The bootstrap section at the bottom (argparse + FastMCP.from_fastapi + dual uvicorn)
        4. Verify syntax: python3 -c "import ast; ast.parse(open('{tool_dir}/server.py').read())"
        5. DO NOT start the server — the caller handles that.

        ## Bootstrap template (MUST be at bottom of server.py, copy exactly):
        ```python
        def main():
            parser = argparse.ArgumentParser()
            parser.add_argument("--port", type=int, default={port})
            parser.add_argument("--host", type=str, default="0.0.0.0")
            args = parser.parse_args()
            mcp = FastMCP.from_fastapi(app=app, name=f"{tool_name} (port {{args.port}})")
            mcp_app = mcp.http_app(path="/mcp")
            async def serve():
                api = uvicorn.Server(uvicorn.Config(app, host=args.host, port=args.port, log_level="info"))
                mcp_srv = uvicorn.Server(uvicorn.Config(mcp_app, host=args.host, port=args.port+600, log_level="info"))
                api.install_signal_handlers = lambda: None
                mcp_srv.install_signal_handlers = lambda: None
                def shutdown():
                    api.should_exit = True
                    mcp_srv.should_exit = True
                loop = asyncio.get_running_loop()
                for sig in (signal.SIGINT, signal.SIGTERM):
                    try:
                        loop.add_signal_handler(sig, shutdown)
                    except NotImplementedError:
                        signal.signal(sig, lambda *_: shutdown())
                t1 = asyncio.create_task(api.serve())
                t2 = asyncio.create_task(mcp_srv.serve())
                await asyncio.wait({{t1, t2}}, return_when=asyncio.FIRST_EXCEPTION)
                shutdown()
            asyncio.run(serve())

        if __name__ == "__main__":
            main()
        ```

        ## IMPORTANT
        - Python only. Use uv add for packages (NEVER pip).
        - Make the tool actually useful and functional — not a stub.
        - Use httpx for HTTP requests (not requests).
        - Handle errors gracefully — return error info in the response model.
        - Project root: /home/engelbot/Desktop/pi_rc_bot

        Respond with BUILT: <one-line description of what the tool does>
        Or FAILED: <reason> if you can't build it.
    """)


class BuildToolRequest(BaseModel):
    tool_name: str = Field(description="Name for the new tool (lowercase, underscores, e.g. 'web_search')")
    description: str = Field(description="What the tool should do — be specific about the API, data sources, endpoints needed")

class BuildToolResponse(BaseModel):
    ok: bool = True
    tool_name: str = ""
    port: int = 0
    mcp_port: int = 0
    built: bool = False
    started: bool = False
    registered: bool = False
    healthy: bool = False
    summary: str = ""
    job_id: str = ""


def _run_build_tool_sync(job: Job, name: str, description: str, port: int, mcp_port: int, tool_dir: Path) -> None:
    """Background worker for build_tool — runs in a thread."""
    try:
        job.state = JobState.RUNNING

        # 1. Send build prompt to OpenCode (strong code model)
        job.phase = "sending build prompt to AI"
        job.log("Sending build prompt to code AI…")
        prompt = _build_tool_prompt(name, description, port, str(tool_dir))
        try:
            client = _get_client()
            client.new_session()  # fresh session for each build
            reply = client.send(prompt)
            LOG.info("Build AI reply for %s: %s", name, reply[:400])
            # Log what the AI did (tool calls = bash commands, file edits)
            for tc in client.last_tool_calls:
                job.log(tc)
            # Log the AI's reasoning/conclusion
            if reply.strip():
                job.log(f"AI says: {reply[:300]}")
            else:
                job.log("AI replied (no text — only tool actions)")
        except Exception as exc:
            job.phase = "AI error"
            job.log(f"Build AI failed: {exc}")
            job.state = JobState.FAILED
            job.result = {"ok": False, "tool_name": name, "port": port, "mcp_port": mcp_port,
                          "summary": f"Build AI failed: {exc}"}
            job.finished_at = time.time()
            return

        if "FAILED" in reply.upper()[:100]:
            job.phase = "AI could not build"
            job.log(f"AI refused to build: {reply[:200]}")
            job.state = JobState.FAILED
            job.result = {"ok": True, "tool_name": name, "port": port, "mcp_port": mcp_port,
                          "built": False, "summary": f"AI could not build: {reply[:500]}"}
            job.finished_at = time.time()
            return

        # 2. Check if server.py was actually created
        job.phase = "checking generated code"
        job.log("Checking if server.py was created…")
        server_py = tool_dir / "server.py"
        if not server_py.exists():
            job.phase = "no server.py created"
            job.log("AI did not create server.py — build failed")
            job.state = JobState.FAILED
            job.result = {"ok": False, "tool_name": name, "port": port, "mcp_port": mcp_port,
                          "built": False, "summary": "AI did not create server.py — build failed"}
            job.finished_at = time.time()
            return

        # 3. Verify syntax
        job.phase = "verifying syntax"
        job.log("Verifying Python syntax…")
        try:
            import ast
            ast.parse(server_py.read_text("utf-8"))
            job.log("✅ Syntax OK")
        except SyntaxError as exc:
            job.phase = "fixing syntax error"
            job.log(f"Syntax error: {exc} — attempting auto-repair…")
            repair_info = _read_tool(name)
            repair_prompt = _build_repair_prompt(repair_info)
            try:
                reply2 = client.send(repair_prompt)
                LOG.info("Repair attempt after build: %s", reply2[:200])
                for tc in client.last_tool_calls:
                    job.log(tc)
                if reply2.strip():
                    job.log(f"AI fix says: {reply2[:200]}")
            except Exception:
                pass
            # Re-check
            try:
                ast.parse(server_py.read_text("utf-8"))
                job.log("✅ Syntax OK after fix")
            except SyntaxError:
                job.phase = "syntax error unfixable"
                job.log(f"Syntax still broken: {exc}")
                job.state = JobState.FAILED
                job.result = {"ok": False, "tool_name": name, "port": port, "mcp_port": mcp_port,
                              "built": False, "summary": f"server.py has syntax errors even after repair: {exc}"}
                job.finished_at = time.time()
                return

        built = True
        job.log("✅ Code generated successfully")

        # 4. Start the tool
        job.phase = "starting server"
        job.log(f"Starting server on port {port}…")
        started = _restart_tool(name, port)
        if started:
            job.log("✅ Server started")
        else:
            job.log("❌ Server failed to start")

        # 5. Register with main OpenCode
        registered = False
        if started:
            job.phase = "registering with OpenCode"
            job.log("Registering with main OpenCode…")
            registered = _re_register_tool(name, mcp_port)
            if registered:
                job.log("✅ Registered with OpenCode")
            else:
                job.log("❌ Registration failed")

        # 6. Final health check
        healthy = False
        if started:
            job.phase = "health check"
            try:
                r = requests.get(f"http://127.0.0.1:{port}/healthz", timeout=5)
                healthy = r.ok
            except Exception:
                pass
            if healthy:
                job.log(f"✅ Healthy on port {port} (MCP: {mcp_port})")
            else:
                job.log("❌ Not healthy yet")

        # 7. Build log
        build_log = tool_dir / "build.log"
        try:
            with open(build_log, "a", encoding="utf-8") as f:
                f.write(f"\n{'='*60}\n")
                f.write(f"Time: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
                f.write(f"Description: {description}\n")
                f.write(f"Port: {port}, MCP: {mcp_port}\n")
                f.write(f"Result: {'SUCCESS' if healthy else 'PARTIAL'}\n")
                f.write(f"AI reply: {reply[:500]}\n")
        except Exception:
            pass

        parts = []
        if built:
            parts.append("✅ Code generated")
        if started:
            parts.append("✅ Server started")
        if registered:
            parts.append("✅ Registered with OpenCode")
        if healthy:
            parts.append(f"✅ Healthy on port {port} (MCP: {mcp_port})")
        else:
            parts.append("❌ Not healthy yet")
        parts.append(f"AI: {reply[:300]}")

        job.phase = "done" if healthy else "done (unhealthy)"
        job.state = JobState.DONE if healthy else JobState.FAILED
        job.result = {
            "ok": True, "tool_name": name, "port": port, "mcp_port": mcp_port,
            "built": built, "started": started, "registered": registered, "healthy": healthy,
            "summary": " | ".join(parts),
        }
        job.finished_at = time.time()
        job.log(f"Build finished in {job.elapsed()}s — {'SUCCESS' if healthy else 'PARTIAL'}")

    except Exception as exc:
        job.phase = "unexpected error"
        job.log(f"Unexpected error: {exc}")
        job.state = JobState.FAILED
        job.result = {"ok": False, "tool_name": name, "summary": f"Unexpected error: {exc}"}
        job.finished_at = time.time()


@app.post("/build_tool", response_model=BuildToolResponse)
async def build_tool(req: BuildToolRequest):
    """Build a brand-new MCP tool server from a description.

    Runs **asynchronously** — returns a job_id immediately.
    Poll ``/build_status`` with the job_id to track progress.
    """
    # Sanitize name
    name = re.sub(r"[^a-z0-9_]", "_", req.tool_name.lower().strip())
    if not name:
        return BuildToolResponse(ok=False, summary="Invalid tool name")

    tool_dir = MY_TOOLS_DIR / name
    port = _next_free_port()
    mcp_port = port + 600

    # Create directory
    tool_dir.mkdir(parents=True, exist_ok=True)

    # Write port file
    (tool_dir / "port.txt").write_text(str(port))

    LOG.info("🔨 Building tool '%s' on port %d — launching async job…", name, port)

    # Create a job and run in background thread
    job = _create_job(JobKind.BUILD, name)
    job.log(f"Build queued — tool={name}, port={port}")

    thread = threading.Thread(
        target=_run_build_tool_sync,
        args=(job, name, req.description, port, mcp_port, tool_dir),
        daemon=True,
    )
    thread.start()

    return BuildToolResponse(
        ok=True,
        tool_name=name,
        port=port,
        mcp_port=mcp_port,
        summary=f"Build started — job_id={job.job_id}. Poll build_status for progress.",
        job_id=job.job_id,
    )


# ── POST /build_status ─────────────────────────────────────────

class BuildStatusRequest(BaseModel):
    job_id: str = Field(description="The job_id returned by build_tool or repair")

class BuildStatusResponse(BaseModel):
    ok: bool = True
    job_id: str = ""
    kind: str = ""
    tool_name: str = ""
    state: str = ""
    phase: str = ""
    elapsed_seconds: float = 0.0
    progress: list[str] = []
    result: dict[str, Any] = {}


@app.post("/build_status", response_model=BuildStatusResponse)
async def build_status(req: BuildStatusRequest):
    """Check the current status of a build or repair job.

    Returns the current phase (e.g. 'sending build prompt to AI',
    'starting server', 'done'), a progress log of completed steps,
    and the final result once the job finishes.
    Call this to monitor long-running builds.
    """
    job = _get_job(req.job_id)
    if job is None:
        return BuildStatusResponse(ok=False, job_id=req.job_id, state="not_found",
                                   phase="Unknown job_id")
    return BuildStatusResponse(
        ok=True,
        job_id=job.job_id,
        kind=job.kind.value,
        tool_name=job.tool_name,
        state=job.state.value,
        phase=job.phase,
        elapsed_seconds=job.elapsed(),
        progress=list(job.progress),
        result=dict(job.result) if job.result else {},
    )


# ── POST /list_jobs ─────────────────────────────────────────────

class ListJobsResponse(BaseModel):
    ok: bool = True
    jobs: list[dict[str, Any]] = []


@app.post("/list_jobs", response_model=ListJobsResponse)
async def list_jobs():
    """List all recent build and repair jobs with their current status.

    Shows job_id, kind (build/repair), tool_name, state (queued/running/done/failed),
    current phase, and elapsed time. Useful to see what the codex agent is working on.
    """
    with _jobs_lock:
        items = sorted(_jobs.values(), key=lambda j: j.created_at, reverse=True)
    return ListJobsResponse(
        jobs=[
            {
                "job_id": j.job_id,
                "kind": j.kind.value,
                "tool_name": j.tool_name,
                "state": j.state.value,
                "phase": j.phase,
                "elapsed_seconds": j.elapsed(),
            }
            for j in items[:20]  # last 20 jobs
        ]
    )


def main():
    parser = argparse.ArgumentParser(description="Codex Agent MCP Server")
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
