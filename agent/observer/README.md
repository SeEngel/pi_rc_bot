# Observer agent

This agent uses **Microsoft Agent Framework** with the **MCP server** exposed by `services/observe`.

## Configure

Edit `agent/observer/config.yaml`:

- `openai.model`: model id for the agent LLM
- `openai.base_url`: optional OpenAI-compatible base URL
- `mcp.observe_mcp_url`: observe MCP endpoint (default: `http://127.0.0.1:8603/mcp`)

The OpenAI API key is read from the repo root `.env` as `OPENAI_API_KEY`.

## Run

In one terminal (if not already running):

```bash
uv run services/observe/main.py
```

In another terminal:

```bash
uv run python agent/observer/main.py --mcp-smoke-test
uv run python agent/observer/main.py --mode describe --question "What is in front of the robot?"
uv run python agent/observer/main.py --mode direction --question "Move toward the nearest safe open space."
```
