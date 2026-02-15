"""Background workers for build and repair jobs."""

from __future__ import annotations

import ast
import logging
import re
import shutil
import time
from pathlib import Path

import requests

from .client import CodexClient
from .jobs import Job, JobState
from .prompts import build_prompt, repair_prompt
from .tools import ToolInventory

LOG = logging.getLogger("codex-agent")


class BuildWorker:
    """Execute a build job synchronously (runs in a background thread)."""

    def __init__(self, inventory: ToolInventory, client_factory):
        self._inventory = inventory
        self._client_factory = client_factory

    def run(self, job: Job) -> None:
        try:
            self._do_build(job)
        except Exception as exc:
            job.phase = "unexpected error"
            job.log(f"Unexpected error: {exc}")
            job.state = JobState.FAILED
            job.result = {"ok": False, "summary": f"Unexpected error: {exc}"}
            job.finished_at = time.time()

    def _do_build(self, job: Job) -> None:
        inv = self._inventory
        job.state = JobState.RUNNING
        description = job.description
        suggested_name = job.tool_name

        # 1. Scan existing tools
        job.phase = "scanning existing tools"
        job.log("Scanning existing tools...")
        inventory_text = inv.build_inventory_for_ai()

        # Prepare directory + port
        tool_dir = inv.my_tools_dir / suggested_name
        tool_dir.mkdir(parents=True, exist_ok=True)
        port_file = tool_dir / "port.txt"
        if port_file.exists():
            port = int(port_file.read_text().strip())
        else:
            port = inv.next_free_port()
            port_file.write_text(str(port))
        mcp_port = port + 600

        # 2. Send to AI
        job.phase = "sending build prompt to AI"
        job.log("Sending build prompt to code AI...")
        prompt = build_prompt(suggested_name, description, port,
                              str(tool_dir), inventory_text)

        client: CodexClient = self._client_factory()
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

        # EXISTS
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
            if existing_name != suggested_name and not (tool_dir / "server.py").exists():
                shutil.rmtree(tool_dir, ignore_errors=True)
            job.finished_at = time.time()
            return

        # Determine actual tool
        actual_tool_name = suggested_name
        actual_tool_dir = tool_dir
        actual_port = port
        actual_mcp_port = mcp_port

        is_rewrite = "REWRITTEN:" in reply_upper or "REWRITING:" in reply_upper
        is_extend = "EXTENDED:" in reply_upper or "EXTENDING:" in reply_upper

        if is_rewrite or is_extend:
            pattern = (r"REWRIT(?:ING|TEN):\s*(\S+)" if is_rewrite
                       else r"EXTEND(?:ING|ED):\s*(\S+)")
            m = re.search(pattern, reply, re.IGNORECASE)
            if m:
                ext_name = m.group(1).strip(" -")
                ext_dir = inv.my_tools_dir / ext_name
                if ext_dir.is_dir():
                    actual_tool_name = ext_name
                    actual_tool_dir = ext_dir
                    ext_port_file = ext_dir / "port.txt"
                    if ext_port_file.exists():
                        actual_port = int(ext_port_file.read_text().strip())
                        actual_mcp_port = actual_port + 600
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
            for d in inv.my_tools_dir.iterdir():
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
            ast.parse(server_py.read_text("utf-8"))
            job.log("Syntax OK")
        except SyntaxError as exc:
            job.log(f"Syntax error: {exc} — attempting auto-repair...")
            try:
                repair_info = inv.read(actual_tool_name)
                client.send(repair_prompt(
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

        # 6–9. Start, register, health-check, log
        self._finalize(
            job, actual_tool_name, actual_tool_dir,
            actual_port, actual_mcp_port,
            description, is_rewrite, is_extend,
        )

    def _finalize(self, job, tool_name, tool_dir, port, mcp_port,
                  description, is_rewrite, is_extend) -> None:
        inv = self._inventory

        job.phase = "starting server"
        job.log(f"Starting server on port {port}...")
        started = inv.restart(tool_name, port)
        job.log("Server started" if started else "Server failed to start")

        registered = False
        if started:
            job.phase = "registering with OpenCode"
            registered = inv.re_register(tool_name, mcp_port)
            job.log("Registered" if registered else "Registration failed")

        healthy = False
        if started:
            try:
                r = requests.get(f"http://127.0.0.1:{port}/healthz", timeout=5)
                healthy = r.ok
            except Exception:
                pass
            job.log("Healthy!" if healthy else "Not healthy")

        # Build log
        build_log = tool_dir / "build.log"
        try:
            with open(build_log, "a", encoding="utf-8") as f:
                f.write(f"\n{'='*60}\n")
                f.write(f"Time: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
                f.write(f"Description: {description}\n")
                f.write(f"Port: {port}, MCP: {mcp_port}\n")
                f.write(f"Result: {'SUCCESS' if healthy else 'PARTIAL'}\n")
        except Exception:
            pass

        decision = "rewrite" if is_rewrite else ("extend" if is_extend else "build")
        job.phase = "done" if healthy else "done (unhealthy)"
        job.state = JobState.DONE if healthy else JobState.FAILED
        job.result = {
            "ok": True,
            "decision": decision,
            "tool_name": tool_name,
            "port": port,
            "mcp_port": mcp_port,
            "built": True,
            "started": started,
            "registered": registered,
            "healthy": healthy,
            "summary": f"{tool_name} on port {port} ({'healthy' if healthy else 'unhealthy'})",
        }
        job.finished_at = time.time()
        job.log(f"Build finished in {job.elapsed()}s")


class RepairWorker:
    """Execute a repair job synchronously (runs in a background thread)."""

    def __init__(self, inventory: ToolInventory, client_factory):
        self._inventory = inventory
        self._client_factory = client_factory

    def run(self, job: Job) -> None:
        try:
            self._do_repair(job)
        except Exception as exc:
            job.phase = "unexpected error"
            job.log(f"Unexpected error: {exc}")
            job.state = JobState.FAILED
            job.result = {"ok": False, "summary": f"Unexpected error: {exc}"}
            job.finished_at = time.time()

    def _do_repair(self, job: Job) -> None:
        inv = self._inventory
        job.state = JobState.RUNNING
        description = job.description

        # 1. Auto-detect tool
        job.phase = "analyzing repair request"
        job.log(f"Analyzing: {description[:200]}")

        detected_tool: str | None = None
        detected_info: dict | None = None

        if inv.my_tools_dir.exists():
            for tool_dir in sorted(inv.my_tools_dir.iterdir()):
                if not tool_dir.is_dir() or tool_dir.name.startswith(("_", ".")):
                    continue
                if not (tool_dir / "server.py").exists():
                    continue
                if tool_dir.name.lower() in description.lower():
                    detected_tool = tool_dir.name
                    detected_info = inv.read(detected_tool)
                    job.log(f"Auto-detected tool: {detected_tool}")
                    break

        if detected_tool is None and inv.my_tools_dir.exists():
            job.log("No specific tool mentioned — scanning for broken tools...")
            for tool_dir in sorted(inv.my_tools_dir.iterdir()):
                if not tool_dir.is_dir() or tool_dir.name.startswith(("_", ".")):
                    continue
                if not (tool_dir / "server.py").exists():
                    continue
                info = inv.read(tool_dir.name)
                if info.get("has_errors") or not info.get("healthy"):
                    detected_tool = tool_dir.name
                    detected_info = info
                    job.log(f"Found broken tool: {detected_tool}")
                    break

        if detected_tool:
            job.tool_name = detected_tool

        # Record hash before
        if detected_tool:
            server_py = inv.my_tools_dir / detected_tool / "server.py"
            job.source_hash_before = inv.file_sha256(server_py)

        # 2. Send to AI
        job.phase = "sending repair prompt to AI"
        job.log("Sending repair prompt to code AI...")
        prompt = repair_prompt(description, detected_info, detected_tool or "")

        client: CodexClient = self._client_factory()
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

        # 3. Figure out which tool was fixed
        fixed_tool = detected_tool
        m = re.search(r"FIXED:\s*(\S+)", reply, re.IGNORECASE)
        if m:
            candidate = m.group(1).strip(" -")
            if (inv.my_tools_dir / candidate).is_dir():
                fixed_tool = candidate
        if not fixed_tool:
            m2 = re.search(r"my_tools/(\w+)/server\.py", reply)
            if m2:
                fixed_tool = m2.group(1)

        if not fixed_tool:
            job.phase = "done (system repair)"
            job.log("System-level repair completed (no tool to restart)")
            job.state = JobState.DONE
            job.result = {"ok": True, "summary": f"System repair done: {reply[:300]}"}
            job.finished_at = time.time()
            return

        job.tool_name = fixed_tool

        # 4. Verify source changed
        server_py = inv.my_tools_dir / fixed_tool / "server.py"
        new_hash = inv.file_sha256(server_py)
        if new_hash == job.source_hash_before and "FIXED" not in reply.upper():
            job.phase = "no changes made"
            job.log("AI did not modify server.py")
            job.state = JobState.FAILED
            job.result = {"ok": False,
                          "summary": f"AI did not change server.py. Reply: {reply[:300]}"}
            job.finished_at = time.time()
            return

        if new_hash != job.source_hash_before:
            job.log("server.py was modified")

        # 5. Restart
        info = inv.read(fixed_tool)
        port = info.get("port", 9100) if "error" not in info else 9100
        mcp_port = port + 600

        job.phase = "restarting server"
        job.log("Restarting tool server...")
        restarted = inv.restart(fixed_tool, port)
        job.log("Restarted" if restarted else "Restart failed")

        # 6. Re-register
        registered = False
        if restarted:
            job.phase = "re-registering"
            registered = inv.re_register(fixed_tool, mcp_port)
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
        repair_log_file = inv.my_tools_dir / fixed_tool / "repair.log"
        try:
            with open(repair_log_file, "a", encoding="utf-8") as f:
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
