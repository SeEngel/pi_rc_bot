#!/usr/bin/env python3
"""
Codex Agent — MCP Tool Server (simplified)
============================================
A FastAPI + FastMCP server that the main robot agent can call as a native
MCP tool (``robot_codex``).  Under the hood it runs its own OpenCode
instance (port 4097) to do the actual code work.

MCP Tools (4):
  POST /build_tool   — describe a new tool in plain text; codex builds it
  POST /repair_tool  — describe what's broken in plain text; codex fixes it
  POST /list_jobs    — high-level overview of all jobs
  POST /job_detail   — detailed status by job_id

Plus:
  GET  /healthz      — liveness check

Usage (auto-started by main.py supervisor):
    uv run python OpenCode/codex_agent/main_codex.py --port 8012
"""

from __future__ import annotations

import argparse
import asyncio
import hashlib
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
    """Best-effort redaction for logs (avoid leaking API keys)."""
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


def _file_sha256(path: Path) -> str:
    """Return hex SHA-256 of a file, or empty string if missing."""
    if not path.exists():
        return ""
    return hashlib.sha256(path.read_bytes()).hexdigest()


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
    description: str = ""          # original user request (plain text)
    state: JobState = JobState.QUEUED
    phase: str = "queued"          # human-readable current step
    progress: list[str] = dc_field(default_factory=list)
    result: dict[str, Any] = dc_field(default_factory=dict)
    created_at: float = dc_field(default_factory=time.time)
    finished_at: float | None = None
    # internal context
    session_id: str | None = None
    turns_completed: int = 0
    source_hash_before: str = ""

    def log(self, msg: str) -> None:
        ts = time.strftime("%H:%M:%S")
        entry = f"[{ts}] {msg}"
        self.progress.append(entry)
        LOG.info("Job %s (%s/%s): %s", self.job_id[:8], self.kind.value, self.tool_name, msg)

    def elapsed(self) -> float:
        end = self.finished_at or time.time()
        return round(end - self.created_at, 1)


_jobs: dict[str, Job] = {}   # job_id -> Job
_jobs_lock = threading.Lock()


def _create_job(kind: JobKind, tool_name: str, description: str = "") -> Job:
    jid = uuid.uuid4().hex[:12]
    job = Job(job_id=jid, kind=kind, tool_name=tool_name, description=description)
    with _jobs_lock:
        _jobs[jid] = job
    return job


def _get_job(job_id: str) -> Job | None:
    with _jobs_lock:
        return _jobs.get(job_id)


def _has_running_job_for(tool_name: str) -> Job | None:
    """Return a running/queued job for this tool, or None."""
    with _jobs_lock:
        for j in _jobs.values():
            if j.tool_name == tool_name and j.state in (JobState.QUEUED, JobState.RUNNING):
                return j
    return None


# ──────────────────────────── OpenCode serve ─────────────────────

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
    """Boot a dedicated OpenCode instance for the codex agent."""
    global _opencode_proc
    if _opencode_proc is not None:
        return

    oc_cfg = _CFG.get("opencode", {})
    host = oc_cfg.get("host", "127.0.0.1")
    port = oc_cfg.get("port", 4097)

    opencode_bin = _find_opencode_bin()
    if opencode_bin is None:
        LOG.error("opencode binary not found — AI will be unavailable")
        return

    cmd = [opencode_bin, "serve", "--port", str(port), "--hostname", host]
    LOG.info("Starting codex OpenCode: %s", " ".join(cmd))

    serve_log = BASE_DIR / "opencode_serve.log"
    fh = open(serve_log, "w", encoding="utf-8")

    _opencode_proc = subprocess.Popen(
        cmd,
        cwd=str(BASE_DIR),
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
        LOG.info("Waiting for codex OpenCode (%d/%d)...", i + 1, retries)
        time.sleep(delay)
    LOG.error("Codex OpenCode not reachable after %d attempts", retries)
    return False


# ──────────────────────────── OpenCode client ────────────────────

class _CodexClient:
    """Send prompts to the dedicated OpenCode instance."""

    def __init__(self):
        oc = _CFG.get("opencode", {})
        self.base = f"http://{oc.get('host', '127.0.0.1')}:{oc.get('port', 4097)}"
        self.timeout = oc.get("request_timeout", 600)
        self._session_id: str | None = None

    @property
    def session_id(self) -> str:
        if self._session_id is None:
            r = requests.post(
                f"{self.base}/session",
                json={"title": "codex"},
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

        for p in parts:
            if not isinstance(p, dict):
                continue
            if p.get("type") == "text":
                texts.append(p.get("text", ""))
                continue
            # Tool call
            t = str(p.get("type", ""))
            if t in {"tool-invocation", "tool_invocation", "tool-call", "tool_call", "tool"} or (
                ("toolName" in p or "tool" in p) and any(k in p for k in ("input", "args", "parameters"))
            ):
                tool_name = p.get("toolName") or p.get("tool") or p.get("name") or "unknown"
                tool_input = p.get("input") or p.get("args") or p.get("parameters") or {}
                if isinstance(tool_input, dict):
                    cmd = tool_input.get("command") or tool_input.get("cmd") or ""
                    file_path = tool_input.get("filePath") or tool_input.get("file") or ""
                    summary = f"🔧 {tool_name}: {cmd or file_path or tool_input}"
                else:
                    summary = f"🔧 {tool_name}: {tool_input}"
                summary = _redact_secrets(str(summary))[:220]
                tool_calls.append(summary)
                LOG.info("AI tool call: %s", summary)
                continue
            if "content" in p:
                texts.append(str(p["content"]))

        self._last_tool_calls = tool_calls
        joined = "\n".join(texts) if texts else raw[:2000]
        self._last_text = joined
        return joined

    @property
    def last_tool_calls(self) -> list[str]:
        return getattr(self, "_last_tool_calls", [])

    @property
    def last_text(self) -> str:
        return getattr(self, "_last_text", "")


_codex_client: _CodexClient | None = None


def _get_client() -> _CodexClient:
    global _codex_client
    if _codex_client is None:
        _codex_client = _CodexClient()
    return _codex_client


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


# ──────────────────────────── tool helpers ───────────────────────

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
            LOG.info("%s healthy after restart", tool_name)
            return True
    except Exception:
        pass
    LOG.warning("%s still unhealthy after restart", tool_name)
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


def _next_free_port() -> int:
    """Find the next unused port starting at 9100."""
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


def _build_tool_inventory_for_ai() -> str:
    """Build a human-readable summary of all existing tools for the AI."""
    if not MY_TOOLS_DIR.exists():
        return "(no my_tools/ directory — no tools exist yet)"

    sections: list[str] = []
    for tool_dir in sorted(MY_TOOLS_DIR.iterdir()):
        if not tool_dir.is_dir() or tool_dir.name.startswith(("_", ".")):
            continue
        name = tool_dir.name
        server_py = tool_dir / "server.py"
        port_file = tool_dir / "port.txt"

        if not server_py.exists():
            sections.append(f"### {name}\n- Status: EMPTY (no server.py)\n")
            continue

        port = int(port_file.read_text().strip()) if port_file.exists() else 0
        source = server_py.read_text("utf-8", errors="replace")

        # Extract endpoint info
        endpoints: list[str] = []
        for line in source.splitlines():
            stripped = line.strip()
            if stripped.startswith("@app.post(") or stripped.startswith("@app.get("):
                endpoints.append(stripped)

        info = _read_tool(name)
        status = "healthy" if info.get("healthy") else ("running" if info.get("process_alive") else "stopped")

        section = f"### {name}\n"
        section += f"- Port: {port}, Status: {status}\n"
        section += f"- Endpoints: {', '.join(endpoints[:10]) if endpoints else '(none)'}\n"
        source_preview = "\n".join(source.splitlines()[:80])
        section += f"- Source preview:\n```python\n{source_preview}\n```\n"
        sections.append(section)

    if not sections:
        return "(no tools found in my_tools/)"

    return "\n".join(sections)


# ──────────────────────────── prompt builders ────────────────────

def _build_build_prompt(tool_name: str, description: str, port: int,
                        tool_dir: str, inventory: str) -> str:
    """Prompt for building a new tool OR extending an existing one."""
    return textwrap.dedent(f"""\
        [BUILD / EXTEND MCP TOOL]

        The robot needs a new capability. Your job is to decide whether an
        existing tool can be extended or a new one should be built, then DO it.

        ## Requested capability
        {description}

        ## Suggested tool name (use this if building new): {tool_name}
        ## Assigned port (if building new): {port} (MCP: {port + 600})
        ## Target directory (if building new): {tool_dir}

        ## Existing tools in my_tools/
        {inventory}

        ## Decision process
        1. If an existing tool already covers this: respond `EXISTS: <tool_name> — <reason>`
        2. If an existing tool is close and just needs new endpoints ADDED:
           - Respond first: `EXTENDING: <tool_name>`
           - Then write the FULL updated server.py preserving ALL existing endpoints.
           - Use: `cat > <existing_tool_dir>/server.py << 'PYEOF' ... PYEOF`
           - Respond: `EXTENDED: <what was added>`
        3. If an existing tool needs to be FUNDAMENTALLY REDESIGNED (different
           endpoints, changed behavior, removed features, new purpose):
           - Respond first: `REWRITING: <tool_name>`
           - Then write a completely NEW server.py — you MAY remove, rename,
             or replace any existing endpoints.
           - Use: `cat > <existing_tool_dir>/server.py << 'PYEOF' ... PYEOF`
           - Respond: `REWRITTEN: <what changed and why>`
        4. If a new tool is needed:
           - Use the suggested name, port, and directory above.
           - Write the complete server.py following the template below.
           - Respond: `BUILT: <what the tool does>`

        ## Template for NEW tools (server.py structure)
        Follow this EXACT structure. Every section is mandatory.

        ```python
        #!/usr/bin/env python3
        \"\"\"{tool_name} — MCP Tool Server

        <One-line description.>
        \"\"\"

        from __future__ import annotations

        import argparse
        import asyncio
        import logging
        import signal
        from typing import Any, Optional

        import uvicorn
        from fastapi import FastAPI
        from fastmcp import FastMCP
        from pydantic import BaseModel, Field

        # domain-specific imports here (e.g. import httpx)

        LOG = logging.getLogger("{tool_name}")

        app = FastAPI(title="{tool_name}", version="0.1.0")

        @app.get("/healthz")
        def healthz() -> dict[str, Any]:
            return {{"ok": True, "service": "{tool_name}"}}

        # --- Pydantic models (every Field needs description=) ---

        # --- @app.post() endpoints ---

        def main():
            parser = argparse.ArgumentParser(description="{tool_name} MCP server")
            parser.add_argument("--port", type=int, default={port})
            parser.add_argument("--host", type=str, default="0.0.0.0")
            args = parser.parse_args()

            logging.basicConfig(
                level=logging.INFO,
                format="%(asctime)s [%(levelname)-7s] %(name)s: %(message)s",
                datefmt="%H:%M:%S",
            )

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

        ## Rules
        - Python only. Use `uv add` for packages (NEVER pip).
        - Use `httpx` for HTTP requests (not requests).
        - FastAPI + FastMCP pattern. Every POST endpoint = MCP tool.
        - **Logging**: Use `LOG = logging.getLogger("<name>")` + `LOG.info/warning/error/debug`. NEVER use `print()`.
        - **Imports**: Group as: `from __future__` → stdlib → third-party → domain-specific.
        - **Docstring**: Every server.py starts with `#!/usr/bin/env python3` + module docstring.
        - **Pydantic**: Every `Field()` must have `description=`.
        - **`logging.basicConfig(...)`** must be in `main()` — configure format + level there.
        - Verify syntax: `python3 -c "import ast; ast.parse(open('<path>').read())"`
        - DO NOT start the server — the caller handles that.
        - Write files using bash: `cat > /path/to/server.py << 'PYEOF' ... PYEOF`
        - Project root: /home/engelbot/Desktop/pi_rc_bot
    """)


def _build_repair_prompt(description: str, info: dict | None = None,
                         tool_name: str = "") -> str:
    """Prompt for repairing a tool or system issue."""
    context_section = ""
    if info and "error" not in info:
        context_section = textwrap.dedent(f"""\
            ## Tool info (auto-detected)
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
        """)

    inventory = _build_tool_inventory_for_ai()

    return textwrap.dedent(f"""\
        [REPAIR] Something is broken. Fix it.

        ## Problem description (from the operator)
        {description}

        {context_section}

        ## All existing tools (for context)
        {inventory}

        ## Your task
        1. Figure out what is broken based on the description above.
           - If a specific tool is mentioned or auto-detected, fix its server.py.
           - If it is a system-level issue (e.g. missing package, config), fix that.
        2. Write the fix using bash: `cat > /path/to/server.py << 'PYEOF' ... PYEOF`
        3. If a missing package caused the error: `cd /home/engelbot/Desktop/pi_rc_bot && uv add PACKAGE`
        4. Verify syntax: `python3 -c "import ast; ast.parse(open('/path/to/server.py').read())"`
        5. DO NOT start the server or register it — the caller handles that.
        6. While fixing, also fix any style violations you notice:
           - Add `#!/usr/bin/env python3` shebang + module docstring if missing
           - Add `from __future__ import annotations` if missing
           - Replace any `print()` calls with `LOG.info/warning/error/debug`
           - Add `LOG = logging.getLogger("<tool_name>")` if missing
           - Ensure `logging.basicConfig(...)` is in `main()`
           - Add `description=` to any Pydantic Field that lacks it
           - Fix import ordering: future → stdlib → third-party → domain

        Respond with `FIXED: <tool_name> — <description>` or `UNFIXABLE: <reason>`
    """)


# ══════════════════════════════════════════════════════════════════
#  Background workers
# ══════════════════════════════════════════════════════════════════

def _run_build_sync(job: Job) -> None:
    """Background worker for build_tool — runs in a thread."""
    try:
        job.state = JobState.RUNNING
        description = job.description
        suggested_name = job.tool_name

        # 1. Scan existing tools
        job.phase = "scanning existing tools"
        job.log("Scanning existing tools...")
        inventory = _build_tool_inventory_for_ai()

        # Prepare directory + port for potential new build
        tool_dir = MY_TOOLS_DIR / suggested_name
        tool_dir.mkdir(parents=True, exist_ok=True)
        port_file = tool_dir / "port.txt"
        if port_file.exists():
            port = int(port_file.read_text().strip())
        else:
            port = _next_free_port()
            port_file.write_text(str(port))
        mcp_port = port + 600

        # 2. Send to AI
        job.phase = "sending build prompt to AI"
        job.log("Sending build prompt to code AI...")
        prompt = _build_build_prompt(suggested_name, description, port,
                                     str(tool_dir), inventory)

        client = _get_client()
        client.new_session()

        try:
            reply = client.send(prompt)
            job.turns_completed = 1
            job.session_id = client._session_id
            LOG.info("Build AI reply for %s: %s", suggested_name, reply[:400])
            for tc in client.last_tool_calls:
                job.log(tc)
            if reply.strip():
                job.log(f"AI says: {reply[:300]}")
            else:
                job.log("AI replied (tool actions only)")
        except Exception as exc:
            job.phase = "AI error"
            job.log(f"Build AI failed: {exc}")
            job.state = JobState.FAILED
            job.result = {"ok": False, "summary": f"Build AI failed: {exc}"}
            job.finished_at = time.time()
            return

        # 3. Parse AI decision
        reply_upper = reply.upper()

        # EXISTS — tool already covers this
        if "EXISTS:" in reply_upper:
            m = re.search(r"EXISTS:\s*(\S+)", reply, re.IGNORECASE)
            existing_name = m.group(1).strip(" -") if m else suggested_name
            job.phase = "done (already exists)"
            job.log(f"AI says tool already exists: {existing_name}")
            job.state = JobState.DONE
            job.result = {
                "ok": True, "decision": "exists", "tool_name": existing_name,
                "summary": f"Capability already provided by '{existing_name}'",
            }
            # Clean up the empty dir we created if different
            if existing_name != suggested_name and not (tool_dir / "server.py").exists():
                shutil.rmtree(tool_dir, ignore_errors=True)
            job.finished_at = time.time()
            return

        # Determine which tool was actually built/extended/rewritten
        actual_tool_name = suggested_name
        actual_tool_dir = tool_dir
        actual_port = port
        actual_mcp_port = mcp_port

        is_rewrite = "REWRITTEN:" in reply_upper or "REWRITING:" in reply_upper
        is_extend = "EXTENDED:" in reply_upper or "EXTENDING:" in reply_upper

        if is_rewrite or is_extend:
            pattern = r"REWRIT(?:ING|TEN):\s*(\S+)" if is_rewrite else r"EXTEND(?:ING|ED):\s*(\S+)"
            m = re.search(pattern, reply, re.IGNORECASE)
            if m:
                ext_name = m.group(1).strip(" -")
                ext_dir = MY_TOOLS_DIR / ext_name
                if ext_dir.is_dir():
                    actual_tool_name = ext_name
                    actual_tool_dir = ext_dir
                    ext_port_file = ext_dir / "port.txt"
                    if ext_port_file.exists():
                        actual_port = int(ext_port_file.read_text().strip())
                        actual_mcp_port = actual_port + 600
            # Clean up the empty dir for the suggested name
            if actual_tool_name != suggested_name and not (tool_dir / "server.py").exists():
                shutil.rmtree(tool_dir, ignore_errors=True)
            job.tool_name = actual_tool_name
            if is_rewrite:
                job.log(f"AI is rewriting tool: {actual_tool_name}")

        if "FAILED:" in reply_upper[:100]:
            job.phase = "AI could not build"
            job.log(f"AI refused: {reply[:200]}")
            job.state = JobState.FAILED
            job.result = {"ok": False, "summary": f"AI could not build: {reply[:500]}"}
            job.finished_at = time.time()
            return

        # 4. Verify server.py exists
        job.phase = "checking generated code"
        server_py = actual_tool_dir / "server.py"
        if not server_py.exists():
            # Check other dirs
            for d in MY_TOOLS_DIR.iterdir():
                if d.is_dir() and (d / "server.py").exists():
                    m2 = re.search(rf"my_tools/{d.name}/server\.py", reply)
                    if m2:
                        actual_tool_name = d.name
                        actual_tool_dir = d
                        server_py = d / "server.py"
                        pf = d / "port.txt"
                        if pf.exists():
                            actual_port = int(pf.read_text().strip())
                            actual_mcp_port = actual_port + 600
                        break
            if not server_py.exists():
                job.phase = "no server.py created"
                job.log("AI did not create server.py")
                job.state = JobState.FAILED
                job.result = {"ok": False, "summary": "AI did not create server.py"}
                job.finished_at = time.time()
                return

        # 5. Verify syntax
        job.phase = "verifying syntax"
        job.log("Verifying Python syntax...")
        try:
            import ast
            ast.parse(server_py.read_text("utf-8"))
            job.log("Syntax OK")
        except SyntaxError as exc:
            job.log(f"Syntax error: {exc} — attempting auto-repair...")
            try:
                repair_info = _read_tool(actual_tool_name)
                repair_reply = client.send(_build_repair_prompt(
                    f"Fix syntax error in {actual_tool_name}: {exc}",
                    repair_info, actual_tool_name,
                ))
                for tc in client.last_tool_calls:
                    job.log(tc)
            except Exception:
                pass
            try:
                ast.parse(server_py.read_text("utf-8"))
                job.log("Syntax OK after fix")
            except SyntaxError as exc2:
                job.state = JobState.FAILED
                job.result = {"ok": False, "summary": f"Syntax error unfixable: {exc2}"}
                job.finished_at = time.time()
                return

        # 6. Start the tool
        job.phase = "starting server"
        job.log(f"Starting server on port {actual_port}...")
        started = _restart_tool(actual_tool_name, actual_port)
        job.log("Server started" if started else "Server failed to start")

        # 7. Register with main OpenCode
        registered = False
        if started:
            job.phase = "registering with OpenCode"
            registered = _re_register_tool(actual_tool_name, actual_mcp_port)
            job.log("Registered" if registered else "Registration failed")

        # 8. Health check
        healthy = False
        if started:
            try:
                r = requests.get(f"http://127.0.0.1:{actual_port}/healthz", timeout=5)
                healthy = r.ok
            except Exception:
                pass
            job.log("Healthy!" if healthy else "Not healthy")

        # 9. Build log
        build_log = actual_tool_dir / "build.log"
        try:
            with open(build_log, "a", encoding="utf-8") as f:
                f.write(f"\n{'='*60}\n")
                f.write(f"Time: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
                f.write(f"Description: {description}\n")
                f.write(f"Port: {actual_port}, MCP: {actual_mcp_port}\n")
                f.write(f"Result: {'SUCCESS' if healthy else 'PARTIAL'}\n")
        except Exception:
            pass

        decision = "rewrite" if is_rewrite else ("extend" if is_extend else "build")
        job.phase = "done" if healthy else "done (unhealthy)"
        job.state = JobState.DONE if healthy else JobState.FAILED
        job.result = {
            "ok": True,
            "decision": decision,
            "tool_name": actual_tool_name,
            "port": actual_port,
            "mcp_port": actual_mcp_port,
            "built": True,
            "started": started,
            "registered": registered,
            "healthy": healthy,
            "summary": f"{actual_tool_name} on port {actual_port} ({'healthy' if healthy else 'unhealthy'})",
        }
        job.finished_at = time.time()
        job.log(f"Build finished in {job.elapsed()}s")

    except Exception as exc:
        job.phase = "unexpected error"
        job.log(f"Unexpected error: {exc}")
        job.state = JobState.FAILED
        job.result = {"ok": False, "summary": f"Unexpected error: {exc}"}
        job.finished_at = time.time()


def _run_repair_sync(job: Job) -> None:
    """Background worker for repair_tool — runs in a thread."""
    try:
        job.state = JobState.RUNNING
        description = job.description

        # 1. Try to auto-detect which tool is mentioned
        job.phase = "analyzing repair request"
        job.log(f"Analyzing: {description[:200]}")

        detected_tool: str | None = None
        detected_info: dict | None = None

        # Check if any known tool name appears in the description
        if MY_TOOLS_DIR.exists():
            for tool_dir in sorted(MY_TOOLS_DIR.iterdir()):
                if not tool_dir.is_dir() or tool_dir.name.startswith(("_", ".")):
                    continue
                if not (tool_dir / "server.py").exists():
                    continue
                if tool_dir.name.lower() in description.lower():
                    detected_tool = tool_dir.name
                    detected_info = _read_tool(detected_tool)
                    job.log(f"Auto-detected tool: {detected_tool}")
                    break

        # If no tool auto-detected, scan for broken ones
        if detected_tool is None and MY_TOOLS_DIR.exists():
            job.log("No specific tool mentioned — scanning for broken tools...")
            for tool_dir in sorted(MY_TOOLS_DIR.iterdir()):
                if not tool_dir.is_dir() or tool_dir.name.startswith(("_", ".")):
                    continue
                if not (tool_dir / "server.py").exists():
                    continue
                info = _read_tool(tool_dir.name)
                if info.get("has_errors") or not info.get("healthy"):
                    detected_tool = tool_dir.name
                    detected_info = info
                    job.log(f"Found broken tool: {detected_tool}")
                    break

        if detected_tool:
            job.tool_name = detected_tool

        # Record hash before
        if detected_tool:
            server_py = MY_TOOLS_DIR / detected_tool / "server.py"
            job.source_hash_before = _file_sha256(server_py)

        # 2. Send to AI
        job.phase = "sending repair prompt to AI"
        job.log("Sending repair prompt to code AI...")
        prompt = _build_repair_prompt(description, detected_info,
                                      detected_tool or "")

        client = _get_client()
        client.new_session()

        try:
            reply = client.send(prompt)
            job.turns_completed = 1
            job.session_id = client._session_id
            LOG.info("Repair AI reply: %s", reply[:400])
            for tc in client.last_tool_calls:
                job.log(tc)
            if reply.strip():
                job.log(f"AI says: {reply[:300]}")
            else:
                job.log("AI replied (tool actions only)")
        except Exception as exc:
            job.phase = "AI error"
            job.log(f"Repair AI failed: {exc}")
            job.state = JobState.FAILED
            job.result = {"ok": False, "summary": f"Repair AI failed: {exc}"}
            job.finished_at = time.time()
            return

        if "UNFIXABLE" in reply.upper():
            job.phase = "unfixable"
            job.log(f"AI says unfixable: {reply[:200]}")
            job.state = JobState.FAILED
            job.result = {"ok": False, "summary": f"AI says unfixable: {reply[:500]}"}
            job.finished_at = time.time()
            return

        # 3. Figure out which tool was fixed (from AI reply)
        fixed_tool = detected_tool
        m = re.search(r"FIXED:\s*(\S+)", reply, re.IGNORECASE)
        if m:
            candidate = m.group(1).strip(" -")
            if (MY_TOOLS_DIR / candidate).is_dir():
                fixed_tool = candidate
        if not fixed_tool:
            m2 = re.search(r"my_tools/(\w+)/server\.py", reply)
            if m2:
                fixed_tool = m2.group(1)

        if not fixed_tool:
            # System-level repair (no specific tool to restart)
            job.phase = "done (system repair)"
            job.log("System-level repair completed (no tool to restart)")
            job.state = JobState.DONE
            job.result = {
                "ok": True,
                "summary": f"System repair done: {reply[:300]}",
            }
            job.finished_at = time.time()
            return

        job.tool_name = fixed_tool

        # 4. Verify source changed
        server_py = MY_TOOLS_DIR / fixed_tool / "server.py"
        new_hash = _file_sha256(server_py)
        if new_hash == job.source_hash_before and "FIXED" not in reply.upper():
            job.phase = "no changes made"
            job.log("AI did not modify server.py")
            job.state = JobState.FAILED
            job.result = {"ok": False, "summary": f"AI did not change server.py. Reply: {reply[:300]}"}
            job.finished_at = time.time()
            return

        if new_hash != job.source_hash_before:
            job.log("server.py was modified")

        # 5. Restart
        info = _read_tool(fixed_tool)
        port = info.get("port", 9100) if "error" not in info else 9100
        mcp_port = port + 600

        job.phase = "restarting server"
        job.log("Restarting tool server...")
        restarted = _restart_tool(fixed_tool, port)
        job.log("Restarted" if restarted else "Restart failed")

        # 6. Re-register
        registered = False
        if restarted:
            job.phase = "re-registering"
            registered = _re_register_tool(fixed_tool, mcp_port)
            job.log("Re-registered" if registered else "Registration failed")

        # 7. Health check
        healthy = False
        if restarted:
            try:
                r = requests.get(f"http://127.0.0.1:{port}/healthz", timeout=5)
                healthy = r.ok
            except Exception:
                pass
            job.log("Healthy!" if healthy else "Still unhealthy")

        # 8. Repair log
        repair_log = MY_TOOLS_DIR / fixed_tool / "repair.log"
        try:
            with open(repair_log, "a", encoding="utf-8") as f:
                f.write(f"\n{'='*60}\n")
                f.write(f"Time: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
                f.write(f"Description: {description[:200]}\n")
                f.write(f"Result: {'FIXED' if healthy else 'ATTEMPTED'}\n")
        except Exception:
            pass

        job.phase = "done" if healthy else "done (still broken)"
        job.state = JobState.DONE if healthy else JobState.FAILED
        job.result = {
            "ok": True,
            "tool_name": fixed_tool,
            "port": port,
            "mcp_port": mcp_port,
            "fix_applied": new_hash != job.source_hash_before,
            "restarted": restarted,
            "registered": registered,
            "healthy": healthy,
            "summary": f"{'FIXED' if healthy else 'STILL BROKEN'}: {fixed_tool}",
        }
        job.finished_at = time.time()
        job.log(f"Repair finished in {job.elapsed()}s")

    except Exception as exc:
        job.phase = "unexpected error"
        job.log(f"Unexpected error: {exc}")
        job.state = JobState.FAILED
        job.result = {"ok": False, "summary": f"Unexpected error: {exc}"}
        job.finished_at = time.time()


# ══════════════════════════════════════════════════════════════════
#  FastAPI  Application
# ══════════════════════════════════════════════════════════════════

app = FastAPI(title="robot_codex", version="1.0.0")


@app.get("/healthz")
async def healthz():
    return {"ok": True, "service": "robot_codex"}


# ── POST /build_tool ────────────────────────────────────────────

class BuildToolRequest(BaseModel):
    description: str = Field(
        description=(
            "Plain-text description of what you need. "
            "Be specific about the desired functionality. "
            "The AI will decide whether to build a new tool or extend an existing one."
        )
    )
    suggested_name: str = Field(
        default="",
        description=(
            "Optional: suggested name for the tool (lowercase, underscores). "
            "If empty, a name will be derived from the description."
        )
    )

class BuildToolResponse(BaseModel):
    ok: bool = True
    job_id: str = ""
    tool_name: str = ""
    summary: str = ""


@app.post("/build_tool", response_model=BuildToolResponse)
async def build_tool(req: BuildToolRequest):
    """Describe a new tool in plain text. The codex agent builds it as MCP in my_tools/.

    The AI automatically checks existing tools and decides whether to:
    - Build a completely new tool
    - Extend an existing tool with new endpoints
    - Report that the capability already exists

    Returns a job_id immediately. Use /job_detail to track progress.
    Max runtime: ~10 minutes.
    """
    # Derive name from description if not provided
    if req.suggested_name:
        name = re.sub(r"[^a-z0-9_]", "_", req.suggested_name.lower().strip())
    else:
        words = re.sub(r"[^a-z0-9 ]", "", req.description.lower()).split()[:3]
        name = "_".join(words) if words else "new_tool"
    if not name:
        name = "new_tool"

    # Check for duplicate running jobs
    existing = _has_running_job_for(name)
    if existing:
        return BuildToolResponse(
            ok=True, job_id=existing.job_id, tool_name=name,
            summary=f"A job is already running for '{name}' — job_id={existing.job_id}",
        )

    job = _create_job(JobKind.BUILD, name, req.description)
    job.log(f"Build queued: {req.description[:200]}")

    thread = threading.Thread(target=_run_build_sync, args=(job,), daemon=True)
    thread.start()

    return BuildToolResponse(
        ok=True, job_id=job.job_id, tool_name=name,
        summary=f"Build started — job_id={job.job_id}. Use job_detail to track progress.",
    )


# ── POST /repair_tool ──────────────────────────────────────────

class RepairToolRequest(BaseModel):
    description: str = Field(
        description=(
            "Plain-text description of what is broken. "
            "Can mention a specific tool name, or just describe the problem. "
            "The AI will figure out what to fix (my_tools or system-level)."
        )
    )

class RepairToolResponse(BaseModel):
    ok: bool = True
    job_id: str = ""
    tool_name: str = ""
    summary: str = ""


@app.post("/repair_tool", response_model=RepairToolResponse)
async def repair_tool(req: RepairToolRequest):
    """Describe what is broken in plain text. The codex agent figures out what to fix.

    The AI will:
    - Auto-detect the broken tool from your description
    - If no tool mentioned, scan for broken tools
    - Fix the code, restart the server, re-register with OpenCode

    Can also handle system-level repairs (missing packages, config issues).
    Returns a job_id immediately. Use /job_detail to track progress.
    Max runtime: ~10 minutes.
    """
    # Try to detect tool name for duplicate check
    tool_hint = "unknown"
    if MY_TOOLS_DIR.exists():
        for d in sorted(MY_TOOLS_DIR.iterdir()):
            if d.is_dir() and d.name.lower() in req.description.lower():
                tool_hint = d.name
                break

    existing = _has_running_job_for(tool_hint)
    if existing and tool_hint != "unknown":
        return RepairToolResponse(
            ok=True, job_id=existing.job_id, tool_name=tool_hint,
            summary=f"A repair job is already running for '{tool_hint}' — job_id={existing.job_id}",
        )

    job = _create_job(JobKind.REPAIR, tool_hint, req.description)
    job.log(f"Repair queued: {req.description[:200]}")

    thread = threading.Thread(target=_run_repair_sync, args=(job,), daemon=True)
    thread.start()

    return RepairToolResponse(
        ok=True, job_id=job.job_id, tool_name=tool_hint,
        summary=f"Repair started — job_id={job.job_id}. Use job_detail to track progress.",
    )


# ── POST /list_jobs ─────────────────────────────────────────────

class JobSummary(BaseModel):
    job_id: str = Field(description="Unique job identifier")
    kind: str = Field(description="build or repair")
    tool_name: str = Field(description="Tool being built or repaired")
    state: str = Field(description="queued / running / done / failed")
    phase: str = Field(description="Current step")
    elapsed_seconds: float = Field(description="Time since job was created")

class ListJobsResponse(BaseModel):
    ok: bool = True
    total: int = 0
    jobs: list[JobSummary] = []


@app.post("/list_jobs", response_model=ListJobsResponse)
async def list_jobs():
    """List all build and repair jobs with high-level info.

    Returns job_id, type, tool name, state, current phase, and elapsed time.
    Use job_detail with a job_id from this list to get full details.
    """
    with _jobs_lock:
        items = sorted(_jobs.values(), key=lambda j: j.created_at, reverse=True)
    return ListJobsResponse(
        total=len(items),
        jobs=[
            JobSummary(
                job_id=j.job_id,
                kind=j.kind.value,
                tool_name=j.tool_name,
                state=j.state.value,
                phase=j.phase,
                elapsed_seconds=j.elapsed(),
            )
            for j in items[:30]
        ],
    )


# ── POST /job_detail ────────────────────────────────────────────

class JobDetailRequest(BaseModel):
    job_id: str = Field(description="The job_id from list_jobs")

class JobDetailResponse(BaseModel):
    ok: bool = True
    job_id: str = ""
    kind: str = ""
    tool_name: str = ""
    description: str = Field(default="", description="Original request text")
    state: str = ""
    phase: str = ""
    elapsed_seconds: float = 0.0
    turns_completed: int = 0
    progress: list[str] = Field(default_factory=list, description="Step-by-step log")
    result: dict[str, Any] = Field(default_factory=dict, description="Final result")


@app.post("/job_detail", response_model=JobDetailResponse)
async def job_detail(req: JobDetailRequest):
    """Get detailed status of a specific job by job_id.

    Returns the full progress log (every step the AI took),
    the original request description, current phase, and the final
    result once the job finishes.
    """
    job = _get_job(req.job_id)
    if job is None:
        return JobDetailResponse(
            ok=False, job_id=req.job_id, state="not_found",
            phase=f"Job '{req.job_id}' not found",
        )
    return JobDetailResponse(
        ok=True,
        job_id=job.job_id,
        kind=job.kind.value,
        tool_name=job.tool_name,
        description=job.description,
        state=job.state.value,
        phase=job.phase,
        elapsed_seconds=job.elapsed(),
        turns_completed=job.turns_completed,
        progress=list(job.progress),
        result=dict(job.result) if job.result else {},
    )


# ══════════════════════════════════════════════════════════════════
#  Main
# ══════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(description="Codex Agent MCP Server")
    parser.add_argument("--port", type=int, default=8012, help="HTTP API port")
    parser.add_argument("--host", type=str, default="0.0.0.0")
    args = parser.parse_args()

    api_port = args.port
    mcp_port = api_port + 600   # 8612

    # Setup logging
    log_file = BASE_DIR / "log.out"
    cfg_log_level = _CFG.get("log_level", "INFO").upper()

    file_handler = logging.FileHandler(log_file, mode="w", encoding="utf-8")
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(logging.Formatter(
        "%(asctime)s [%(levelname)-7s] %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    ))

    console_handler = logging.StreamHandler()
    console_handler.setLevel(getattr(logging, cfg_log_level, logging.INFO))
    console_handler.setFormatter(logging.Formatter(
        "%(asctime)s [%(levelname)-7s] %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    ))

    root_logger = logging.getLogger()
    root_logger.setLevel(logging.DEBUG)
    root_logger.addHandler(file_handler)
    root_logger.addHandler(console_handler)

    # Start the dedicated OpenCode instance
    LOG.info("Booting codex agent on port %d (MCP: %d)", api_port, mcp_port)
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
