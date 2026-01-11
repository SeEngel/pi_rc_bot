# Listening service

A tiny speech-to-text (STT) HTTP service modeled after PiCar-X `example/16.voice_controlled_car.py`.

It exposes a FastAPI server that records from the microphone and returns the recognized text as a string.

## Files

- `src/listener.py`: `Listener` wrapper (STT engine: `vosk` or `openai`)
- `config.yaml`: configuration (commented)
- `main.py`: service entrypoint (auto-creates a local venv if needed)

## Install (if needed)

This service can auto-create and use a local virtualenv at `services/listening/.venv` (recommended on Raspberry Pi / Debian).

If you run `python services/listening/main.py` and dependencies are missing, it will create the venv and install `requirements.txt` automatically.

Manual venv install (optional):

```bash
python3 -m venv services/listening/.venv
services/listening/.venv/bin/pip install -r services/listening/requirements.txt
```

Avoid `sudo pip3 install ... --break-system-packages` if you can.
It commonly fails with `uninstall-no-record-file` for system-managed packages like `typing_extensions`.

If you *must* install system-wide anyway, a last-resort workaround is:

```bash
sudo PIP_IGNORE_INSTALLED=1 pip3 install -r requirements.txt --break-system-packages
```

(
Note: This can leave multiple copies of packages on your system; prefer the venv.
)

## Run

```bash
python services/listening/main.py
```

If `stt.engine: openai`, set `OPENAI_API_KEY` (e.g. in the repo `.env`).

## Quick self-check (no microphone)

```bash
python services/listening/self_check.py
```

## API

- `GET /healthz` → service status + STT availability
- `POST /listen` → records + transcribes and returns `{ ok, text, raw }`

Example:

```bash
curl -s http://localhost:8002/healthz | jq
curl -s -X POST http://localhost:8002/listen | jq
```

Optional request body (JSON):

- `stream` (bool, default `false`)
	- Vosk only: forwarded to `Vosk.listen(stream=...)` if supported.
- `speech_pause_seconds` (number, optional)
	- OpenAI only: stop recording after this many seconds of continuous silence **after speech has started**.
	- If omitted: uses `stt.openai.stop_silence_seconds` from `config.yaml`.

Examples:

```bash
# OpenAI: allow longer pauses between sentences
curl -s -X POST http://localhost:8002/listen \
	-H 'content-type: application/json' \
	-d '{"speech_pause_seconds": 2.0}' | jq

# Vosk: request stream mode (if supported by your robot_hat/picarx version)
curl -s -X POST http://localhost:8002/listen \
	-H 'content-type: application/json' \
	-d '{"stream": true}' | jq
```
