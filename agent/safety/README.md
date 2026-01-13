# Safety agent

Agent Framework-based agent that manages e-stop and guarded driving via `services/safety` MCP.

## Run

- MCP smoke test:
  - `uv run python agent/safety/main.py --mcp-smoke-test`
- Direct check (no LLM):
  - `uv run python agent/safety/main.py --direct --check`
- Direct guarded drive:
  - `uv run python agent/safety/main.py --direct --speed 20 --duration-s 0.5`
