"""Tool inventory — read, restart, register, health-check for my_tools."""

from __future__ import annotations

import hashlib
import logging
import os
import re
import signal
import subprocess
import sys
import time
from pathlib import Path

import requests

LOG = logging.getLogger("codex-agent")

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


class ToolInventory:
    """Read, restart, register, and inspect custom MCP tools in ``my_tools/``."""

    def __init__(self, my_tools_dir: Path, cfg: dict):
        self.my_tools_dir = my_tools_dir
        self._cfg = cfg

    # ── read tool info ──────────────────────────────────────────

    def read(self, tool_name: str) -> dict:
        """Gather all info about a tool: source, log tail, port, status."""
        tool_dir = self.my_tools_dir / tool_name
        if not tool_dir.is_dir():
            return {"error": f"Tool '{tool_name}' not found in my_tools/"}

        server_py = tool_dir / "server.py"
        log_file = tool_dir / "server.log"
        port_file = tool_dir / "port.txt"

        source = server_py.read_text("utf-8") if server_py.exists() else "(no server.py)"
        tail_lines = self._cfg.get("log_tail_lines", 80)
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

        has_err = self._has_errors(log_tail)
        error_block = self._extract_error_block(log_tail) if has_err else ""

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

    # ── restart tool ────────────────────────────────────────────

    def restart(self, tool_name: str, port: int) -> bool:
        """Kill old process, start fresh, return True if healthy."""
        tool_dir = self.my_tools_dir / tool_name
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
        log_fh = open(tool_dir / "server.log", "w", encoding="utf-8")
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

    # ── register with main opencode ─────────────────────────────

    def re_register(self, tool_name: str, mcp_port: int) -> bool:
        """Hot-register with the MAIN OpenCode instance."""
        main_port = self._cfg.get("main_opencode_port", 4096)
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

    # ── next free port ──────────────────────────────────────────

    def next_free_port(self) -> int:
        used: set[int] = set()
        if self.my_tools_dir.exists():
            for d in self.my_tools_dir.iterdir():
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

    # ── build inventory for AI ──────────────────────────────────

    def build_inventory_for_ai(self) -> str:
        """Build a human-readable summary of all existing tools for the AI."""
        if not self.my_tools_dir.exists():
            return "(no my_tools/ directory — no tools exist yet)"

        sections: list[str] = []
        for tool_dir in sorted(self.my_tools_dir.iterdir()):
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

            endpoints: list[str] = []
            for line in source.splitlines():
                stripped = line.strip()
                if stripped.startswith("@app.post(") or stripped.startswith("@app.get("):
                    endpoints.append(stripped)

            info = self.read(name)
            status = ("healthy" if info.get("healthy")
                      else ("running" if info.get("process_alive") else "stopped"))

            section = f"### {name}\n"
            section += f"- Port: {port}, Status: {status}\n"
            section += f"- Endpoints: {', '.join(endpoints[:10]) if endpoints else '(none)'}\n"
            source_preview = "\n".join(source.splitlines()[:80])
            section += f"- Source preview:\n```python\n{source_preview}\n```\n"
            sections.append(section)

        return "\n".join(sections) if sections else "(no tools found in my_tools/)"

    # ── hash helper ─────────────────────────────────────────────

    @staticmethod
    def file_sha256(path: Path) -> str:
        if not path.exists():
            return ""
        return hashlib.sha256(path.read_bytes()).hexdigest()

    # ── error detection ─────────────────────────────────────────

    @staticmethod
    def _has_errors(text: str) -> bool:
        return any(p.search(text) for p in _ERROR_PATTERNS)

    @staticmethod
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
