# Memory service

A tiny FastAPI service that stores and retrieves short-term and long-term memories using OpenAI embeddings (cosine similarity search).

- HTTP API: `POST /store_memory`, `POST /get_top_n_memory`
- MCP server: runs on a second port and exposes the same endpoints as tools via `/mcp`

## Requirements

Install deps (recommended in a venv):

- `pip3 install -r services/memory/requirements.txt`

Create a repo-level `.env` with:

- `OPENAI_API_KEY=...`
- optionally: `OPENAI_BASE_URL=...`

## Run

From repo root:

- `python3 services/memory/main.py`

Defaults:

- API: http://0.0.0.0:8004
- MCP: http://0.0.0.0:8604/mcp
