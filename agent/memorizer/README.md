# Memorizer agent

This agent uses **Microsoft Agent Framework** with the **MCP server** exposed by `services/memory`.

It has two main jobs:

1. Decide whether a piece of information should be stored as a memory (and tag it).
2. Retrieve relevant memories for a query and summarize them.

## Configure

Edit `agent/memorizer/config.yaml`:

- `openai.model`: model id for the agent LLM
- `openai.base_url`: optional OpenAI-compatible base URL
- `mcp.memory_mcp_url`: memory MCP endpoint (default: `http://127.0.0.1:8604/mcp`)

The OpenAI API key is read from the repo root `.env` as `OPENAI_API_KEY`.

## Run

In one terminal (if not already running):

```bash
uv run services/memory/main.py
```

In another terminal:

```bash
# Prove the MCP endpoint is reachable
uv run python agent/memorizer/main.py --mcp-smoke-test

# Let the LLM decide whether to store a memory
uv run python agent/memorizer/main.py --mode ingest --text "The robot is parked next to the kitchen table."

# Retrieve and summarize memories
uv run python agent/memorizer/main.py --mode recall --query "Where did I park the robot?" --top-n 3

# Create and store a compact summary memory for a topic
uv run python agent/memorizer/main.py --mode compact --topic "robot parking locations" --top-n 5

# Direct (no LLM): store a raw memory
uv run python agent/memorizer/main.py --direct --mode ingest --text "Battery is low" --tags "battery,warning"
```
