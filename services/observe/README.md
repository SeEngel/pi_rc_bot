# Observe service

A tiny FastAPI service that captures a camera image and asks a vision model to describe what it sees.

This mirrors the style of `services/listening` and `services/speak`.

## Files

- `src/observer.py`: `Observer` wrapper (camera + vision)
- `config.yaml`: configuration (commented)
- `main.py`: service entrypoint (auto-creates a local venv if needed)

## Requirements

- Raspberry Pi camera setup with `picamera2` (typically installed via apt on Raspberry Pi OS)
- An OpenAI-compatible API endpoint and an API key in `OPENAI_API_KEY`

## Run

```bash
python services/observe/main.py
```

## API

- `GET /healthz` → service status
- `POST /observe` → captures + describes and returns only `{ text }`

Example:

```bash
curl -s http://localhost:8003/healthz | jq
curl -s -X POST http://localhost:8003/observe | jq
curl -s -X POST http://localhost:8003/observe -H 'content-type: application/json' -d '{"question":"What objects are visible?"}' | jq
```

## OpenAI-compatible base URL

In `services/observe/config.yaml`, set:

- `vision.openai.base_url: ""` to use OpenAI's default API base URL.
- Or set it to an OpenAI-compatible base URL (often ending with `/v1`), e.g. `http://localhost:8000/v1`.
