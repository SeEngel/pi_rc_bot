# Proximity agent

Agent Framework-based agent that reads ultrasonic distance via `services/proximity` MCP.

## Run

- MCP smoke test:
  - `uv run python agent/proximity/main.py --mcp-smoke-test`
- Direct distance (no LLM):
  - `uv run python agent/proximity/main.py --direct`
