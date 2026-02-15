"""Child-process management for OpenCode serve, my_tools, and the codex agent."""

from __future__ import annotations

import atexit
import logging
import os
import shutil
import signal
import subprocess
import sys
import time
from pathlib import Path

import requests
import yaml

from .config import deep_get

LOG = logging.getLogger("supervisor")


# ═══════════════════════════════════════════════════════════════
#  OpenCode serve process
# ═══════════════════════════════════════════════════════════════

class OpenCodeProcess:
    """Manage the ``opencode serve`` child process."""

    def __init__(self, base_dir: Path):
        self._base_dir = base_dir
        self._proc: subprocess.Popen | None = None
        atexit.register(self.stop)

    # ── locate binary ───────────────────────────────────────────

    @staticmethod
    def _find_binary() -> str | None:
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

    # ── start / stop ────────────────────────────────────────────

    def start(self, host: str, port: int) -> None:
        opencode_bin = self._find_binary()
        if opencode_bin is None:
            raise FileNotFoundError(
                "'opencode' not found in PATH. Install it: "
                "curl -fsSL https://opencode.ai/install | bash"
            )

        cmd = [opencode_bin, "serve", "--port", str(port), "--hostname", host]
        LOG.info("Starting OpenCode server: %s", " ".join(cmd))

        serve_log_path = self._base_dir / "opencode_serve.log"
        fh = open(serve_log_path, "w", encoding="utf-8")  # noqa: SIM115
        LOG.info("OpenCode server output → %s", serve_log_path)

        self._proc = subprocess.Popen(
            cmd,
            cwd=str(self._base_dir),
            stdout=fh,
            stderr=subprocess.STDOUT,
        )
        self._proc._serve_log_fh = fh  # type: ignore[attr-defined]
        LOG.info("OpenCode server started (pid %d)", self._proc.pid)

    def stop(self) -> None:
        if self._proc is None:
            return
        proc = self._proc
        self._proc = None
        LOG.info("Stopping OpenCode server (pid %d)…", proc.pid)
        try:
            proc.terminate()
            try:
                proc.wait(timeout=10)
                LOG.info("OpenCode server stopped gracefully")
            except subprocess.TimeoutExpired:
                LOG.warning("OpenCode server did not stop in 10s — killing")
                proc.kill()
                proc.wait(timeout=5)
        except Exception as exc:
            LOG.error("Error stopping OpenCode server: %s", exc)
        finally:
            fh = getattr(proc, "_serve_log_fh", None)
            if fh:
                try:
                    fh.close()
                except Exception:
                    pass


# ═══════════════════════════════════════════════════════════════
#  my_tools process manager
# ═══════════════════════════════════════════════════════════════

MY_TOOLS_BASE_PORT = 9100


class MyToolsManager:
    """Start, stop, register, and health-check custom MCP tool servers."""

    def __init__(self, base_dir: Path):
        self.tools_dir = base_dir / "my_tools"
        self._procs: list[subprocess.Popen] = []
        self._known: set[str] = set()

        # Health-check cache
        self._last_check: float = 0.0
        self._last_result: list[str] = []
        self._first_done = False
        self._check_interval = 300.0  # 5 min

    # ── start all tools ─────────────────────────────────────────

    def start_all(self, cfg: dict) -> None:
        if not self.tools_dir.exists():
            return
        self._known = self._live_tool_names()
        self._cleanup_orphans(cfg)

        for tool_dir in self._tool_dirs():
            port = self._port_for(tool_dir)
            pid_file = tool_dir / "server.pid"

            if pid_file.exists():
                try:
                    old_pid = int(pid_file.read_text().strip())
                    os.kill(old_pid, 0)
                    LOG.info("my_tools/%s already running (pid %d) on port %d",
                             tool_dir.name, old_pid, port)
                    continue
                except (OSError, ValueError):
                    pid_file.unlink(missing_ok=True)

            self._launch(tool_dir, port)

        # Give servers a moment to boot, then hot-register
        if list(self._tool_dirs()):
            time.sleep(2)
            self.register_with_opencode(cfg)

    # ── stop all tools ──────────────────────────────────────────

    def stop_all(self) -> None:
        for proc in self._procs:
            try:
                proc.terminate()
                proc.wait(timeout=5)
            except Exception:
                try:
                    proc.kill()
                except Exception:
                    pass
            fh = getattr(proc, "_log_fh", None)
            if fh:
                try:
                    fh.close()
                except Exception:
                    pass
        self._procs.clear()
        if self.tools_dir.exists():
            for d in self.tools_dir.iterdir():
                if d.is_dir():
                    (d / "server.pid").unlink(missing_ok=True)

    # ── register with opencode ──────────────────────────────────

    def register_with_opencode(
        self, cfg: dict, tool_dirs: list[Path] | None = None,
    ) -> None:
        oc_cfg = cfg.get("opencode", {})
        oc_base = f"http://{oc_cfg.get('host', '127.0.0.1')}:{oc_cfg.get('port', 4096)}"

        if tool_dirs is None:
            tool_dirs = list(self._tool_dirs())

        for tool_dir in tool_dirs:
            port = self._port_for(tool_dir)
            mcp_port = port + 600
            name = f"my_{tool_dir.name}"
            try:
                r = requests.post(
                    f"{oc_base}/mcp",
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
                if r.ok:
                    LOG.info("Hot-registered MCP '%s' at port %d", name, mcp_port)
                else:
                    LOG.warning("Failed to register MCP '%s': %d %s",
                                name, r.status_code, r.text[:200])
            except Exception as exc:
                LOG.warning("Failed to register MCP '%s': %s", name, exc)

    # ── health check (cached) ──────────────────────────────────

    def check_health(self, cfg: dict, force: bool = False) -> list[str]:
        """Return list of broken tool names.  Cached for 5 min."""
        now = time.monotonic()
        if (
            not force
            and self._first_done
            and (now - self._last_check) < self._check_interval
        ):
            return list(self._last_result)

        broken: list[str] = []
        if not self.tools_dir.exists():
            self._last_check = now
            self._last_result = broken
            self._first_done = True
            return broken

        for tool_dir in self._tool_dirs():
            port_file = tool_dir / "port.txt"
            if not port_file.exists():
                continue
            port = int(port_file.read_text().strip())
            try:
                r = requests.get(f"http://127.0.0.1:{port}/healthz", timeout=3)
                if not r.ok:
                    broken.append(tool_dir.name)
            except Exception:
                broken.append(tool_dir.name)

        self._last_check = now
        self._last_result = broken
        self._first_done = True
        return broken

    # ── orphan cleanup ──────────────────────────────────────────

    def cleanup_orphans(self, cfg: dict) -> list[str]:
        return self._cleanup_orphans(cfg)

    def _cleanup_orphans(self, cfg: dict) -> list[str]:
        cleaned: list[str] = []
        if not self.tools_dir.exists():
            for name in list(self._known):
                self._unregister(cfg, name)
                cleaned.append(name)
            self._known.clear()
            return cleaned

        current_dirs = {
            d.name for d in self.tools_dir.iterdir()
            if d.is_dir() and not d.name.startswith(("_", "."))
        }

        # Dirs that lost their server.py
        for tool_dir in sorted(self.tools_dir.iterdir()):
            if not tool_dir.is_dir() or tool_dir.name.startswith(("_", ".")):
                continue
            pid_file = tool_dir / "server.pid"
            server_py = tool_dir / "server.py"
            if pid_file.exists() and not server_py.exists():
                try:
                    old_pid = int(pid_file.read_text().strip())
                    os.kill(old_pid, signal.SIGTERM)
                    LOG.info("Killed orphan pid=%d for %s", old_pid, tool_dir.name)
                except (OSError, ValueError):
                    pass
                pid_file.unlink(missing_ok=True)
                self._unregister(cfg, tool_dir.name)
                cleaned.append(tool_dir.name)

        # Fully removed folders
        removed = self._known - current_dirs
        for name in removed:
            LOG.info("Tool '%s' was removed from disk — cleaning up", name)
            self._unregister(cfg, name)
            cleaned.append(name)

        self._known = self._live_tool_names()
        if cleaned:
            LOG.info("Cleaned up %d orphaned tool(s): %s", len(cleaned), ", ".join(cleaned))
        return cleaned

    # ── internals ───────────────────────────────────────────────

    def _tool_dirs(self):
        if not self.tools_dir.exists():
            return
        for d in sorted(self.tools_dir.iterdir()):
            if (
                d.is_dir()
                and not d.name.startswith(("_", "."))
                and (d / "server.py").exists()
            ):
                yield d

    def _live_tool_names(self) -> set[str]:
        return {d.name for d in self._tool_dirs()} if self.tools_dir.exists() else set()

    @staticmethod
    def _port_for(tool_dir: Path) -> int:
        pf = tool_dir / "port.txt"
        return int(pf.read_text().strip()) if pf.exists() else MY_TOOLS_BASE_PORT

    def _launch(self, tool_dir: Path, port: int) -> None:
        log_fh = open(tool_dir / "server.log", "a", encoding="utf-8")
        proc = subprocess.Popen(
            [sys.executable, str(tool_dir / "server.py"), "--port", str(port)],
            cwd=str(tool_dir),
            stdout=log_fh,
            stderr=subprocess.STDOUT,
        )
        proc._log_fh = log_fh  # type: ignore[attr-defined]
        (tool_dir / "server.pid").write_text(str(proc.pid))
        self._procs.append(proc)
        mcp_port = port + 600
        LOG.info("Started my_tools/%s on port %d (MCP: %d), pid=%d",
                 tool_dir.name, port, mcp_port, proc.pid)

    def _unregister(self, cfg: dict, tool_name: str) -> bool:
        oc_cfg = cfg.get("opencode", {})
        oc_base = f"http://{oc_cfg.get('host', '127.0.0.1')}:{oc_cfg.get('port', 4096)}"
        name = f"my_{tool_name}"
        try:
            r = requests.post(
                f"{oc_base}/mcp",
                json={
                    "name": name,
                    "config": {
                        "type": "remote",
                        "url": "http://127.0.0.1:1/mcp",
                        "enabled": False,
                    },
                },
                timeout=10,
            )
            if r.ok:
                LOG.info("Unregistered orphaned MCP '%s'", name)
                return True
            else:
                LOG.warning("Failed to unregister '%s': %d %s",
                            name, r.status_code, r.text[:200])
        except Exception as exc:
            LOG.warning("Failed to unregister '%s': %s", name, exc)
        return False


# ═══════════════════════════════════════════════════════════════
#  Codex Agent child process
# ═══════════════════════════════════════════════════════════════

CODEX_API_PORT = 8012
CODEX_MCP_PORT = 8612


class CodexAgentProcess:
    """Manage the codex-agent FastAPI + FastMCP child process."""

    def __init__(self, base_dir: Path):
        self._base_dir = base_dir
        self._codex_dir = base_dir / "codex_agent"
        self._proc: subprocess.Popen | None = None

    # ── start / stop ────────────────────────────────────────────

    def start(self) -> None:
        script = self._codex_dir / "main_codex.py"
        if not script.exists():
            LOG.warning("Codex agent not found at %s — skipping", script)
            return

        if self._proc is not None:
            try:
                self._proc.poll()
                if self._proc.returncode is None:
                    LOG.info("Codex agent already running (pid %d)", self._proc.pid)
                    return
            except Exception:
                pass

        log_fh = open(self._codex_dir / "codex_supervisor.log", "w", encoding="utf-8")
        self._proc = subprocess.Popen(
            [sys.executable, str(script), "--port", str(CODEX_API_PORT)],
            cwd=str(self._codex_dir),
            stdout=log_fh,
            stderr=subprocess.STDOUT,
        )
        self._proc._log_fh = log_fh  # type: ignore[attr-defined]
        LOG.info(
            "Started codex agent (pid %d) on port %d (MCP: %d)",
            self._proc.pid, CODEX_API_PORT, CODEX_MCP_PORT,
        )

    def stop(self) -> None:
        if self._proc is None:
            return
        proc = self._proc
        self._proc = None
        LOG.info("Stopping codex agent (pid %d)…", proc.pid)
        try:
            proc.terminate()
            try:
                proc.wait(timeout=10)
            except subprocess.TimeoutExpired:
                proc.kill()
                proc.wait(timeout=5)
        except Exception as exc:
            LOG.error("Error stopping codex agent: %s", exc)
        finally:
            fh = getattr(proc, "_log_fh", None)
            if fh:
                try:
                    fh.close()
                except Exception:
                    pass

    # ── wait / register ─────────────────────────────────────────

    def wait_until_ready(self, retries: int = 15, delay: float = 2.0) -> bool:
        url = f"http://127.0.0.1:{CODEX_API_PORT}/healthz"
        for i in range(retries):
            try:
                r = requests.get(url, timeout=3)
                if r.ok:
                    LOG.info("Codex agent healthy at port %d", CODEX_API_PORT)
                    return True
            except Exception:
                pass
            LOG.info("Waiting for codex agent (%d/%d)…", i + 1, retries)
            time.sleep(delay)
        LOG.warning("Codex agent not reachable after %d attempts", retries)
        return False

    def register_with_opencode(self, cfg: dict) -> None:
        oc_cfg = cfg.get("opencode", {})
        oc_base = f"http://{oc_cfg.get('host', '127.0.0.1')}:{oc_cfg.get('port', 4096)}"
        try:
            r = requests.post(
                f"{oc_base}/mcp",
                json={
                    "name": "robot_codex",
                    "config": {
                        "type": "remote",
                        "url": f"http://127.0.0.1:{CODEX_MCP_PORT}/mcp",
                        "enabled": True,
                    },
                },
                timeout=10,
            )
            if r.ok:
                LOG.info("Hot-registered robot_codex MCP at port %d", CODEX_MCP_PORT)
            else:
                LOG.warning("Failed to register robot_codex: %d %s",
                            r.status_code, r.text[:200])
        except Exception as exc:
            LOG.warning("Failed to register robot_codex: %s", exc)

    # ── repair broken tools via codex ───────────────────────────

    def queue_repairs(self, cfg: dict, tools_mgr: MyToolsManager) -> bool:
        """Queue repair jobs for broken my_tools (non-blocking, fire-and-forget).

        Returns True if repairs were queued (or none needed).
        The codex agent handles repairs in the background — the main
        supervisor loop can start immediately.
        """
        broken = tools_mgr.check_health(cfg, force=True)
        if not broken:
            LOG.info("All custom tools are healthy — nothing to repair")
            return True

        LOG.info("Found %d broken tool(s): %s — queuing background repairs",
                 len(broken), ", ".join(broken))

        for tool_name in broken:
            try:
                r = requests.post(
                    f"http://127.0.0.1:{CODEX_API_PORT}/repair_tool",
                    json={"description": f"Tool '{tool_name}' is broken or not responding. Fix it."},
                    timeout=30,
                )
                if r.ok:
                    job_id = r.json().get("job_id", "")
                    if job_id:
                        LOG.info("  🔧 Repair queued for %s — job_id=%s (background)",
                                 tool_name, job_id)
            except Exception as exc:
                LOG.warning("  Failed to queue repair for %s: %s", tool_name, exc)

        return True

    # ── thorough mode ───────────────────────────────────────────

    def enable_thorough_mode(self) -> None:
        codex_config = self._codex_dir / "config.yaml"
        if not codex_config.exists():
            LOG.warning("Codex config not found — cannot enable thorough mode")
            return
        try:
            with open(codex_config) as f:
                cfg = yaml.safe_load(f) or {}
            cfg.setdefault("thorough_mode", {})["enabled"] = True
            with open(codex_config, "w") as f:
                yaml.safe_dump(cfg, f, default_flow_style=False, sort_keys=False)
            LOG.info("🧠 Thorough codex mode ENABLED")
        except Exception as exc:
            LOG.error("Failed to enable thorough codex mode: %s", exc)
