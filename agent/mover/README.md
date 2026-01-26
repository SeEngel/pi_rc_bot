# Mover agent

Agent Framework-based agent that controls chassis motion via `services/move` MCP.

## Run

- MCP smoke test:
  - `uv run python agent/mover/main.py --mcp-smoke-test`
- Direct drive (no LLM):
  - `uv run python agent/mover/main.py --direct --speed 30 --duration-s 0.5`
- LLM mode:
  - `uv run python agent/mover/main.py --speed 30 --duration-s 0.5`
