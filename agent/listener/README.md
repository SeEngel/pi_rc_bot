# Listener agent

An Agent Framework-based agent that uses the `services/listening` MCP server to do speech-to-text.

## Config

- `config.yaml` controls:
  - the agent name + instructions
  - the OpenAI model used by the Agent Framework chat client
  - the MCP URL for the listening service

The OpenAI API key is read from the repo root `.env` (`OPENAI_API_KEY=...`) when running the LLM-driven mode.

## Run

### 1) Health check the MCP server (no microphone)

```bash
uv run python agent/listener/main.py --mcp-smoke-test
```

### 2) Direct listen (no LLM; uses the MCP tool directly)

```bash
uv run python agent/listener/main.py --direct
```

### 3) LLM-driven agent mode

```bash
uv run python agent/listener/main.py
```
