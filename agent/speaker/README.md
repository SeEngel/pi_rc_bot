# Speaker agent

An Agent Framework-based agent that uses the `services/speak` MCP server to do text-to-speech.

## Config

- `config.yaml` controls:
  - the agent name + instructions
  - the OpenAI model used by the Agent Framework chat client
  - the MCP URL for the speak service

The OpenAI API key is read from the repo root `.env` (`OPENAI_API_KEY=...`) when running the LLM-driven mode.

## Run

### 1) Health check the MCP server (no audio output)

```bash
uv run python agent/speaker/main.py --mcp-smoke-test
```

### 2) Direct speak (no LLM; uses the MCP tool directly)

```bash
uv run python agent/speaker/main.py --direct --text "Hallo!"
```

### 3) LLM-driven agent mode

```bash
uv run python agent/speaker/main.py --text "Hallo!"
```
