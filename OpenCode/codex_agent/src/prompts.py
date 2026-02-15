"""Prompt templates for the codex agent (build & repair)."""

from __future__ import annotations

import textwrap


def build_prompt(
    tool_name: str,
    description: str,
    port: int,
    tool_dir: str,
    inventory: str,
) -> str:
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
        \\\"\\\"\\\"{tool_name} — MCP Tool Server

        <One-line description.>
        \\\"\\\"\\\"

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


def repair_prompt(
    description: str,
    info: dict | None = None,
    tool_name: str = "",
) -> str:
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

    return textwrap.dedent(f"""\
        [REPAIR] Something is broken. Fix it.

        ## Problem description (from the operator)
        {description}

        {context_section}

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
