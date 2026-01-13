# Head agent

Agent Framework-based agent that controls pan/tilt via `services/head` MCP.

## Run

- MCP smoke test:
  - `uv run python agent/head/main.py --mcp-smoke-test`
- Direct set angles (no LLM):
  - `uv run python agent/head/main.py --direct --pan-deg -20 --tilt-deg 10`
- LLM mode:
  - `uv run python agent/head/main.py --pan-deg -20 --tilt-deg 10`
