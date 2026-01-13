# Perception service

Lightweight visual perception (faces/people) via HTTP + MCP.

- Uses `picamera2` if available to capture a frame.
- Uses OpenCV (`cv2`) if available for detectors.
- If unavailable, returns structured `available=false` with a reason (or uses `dry_run`).

Designed to run under system `python3` (started via `services/main.sh`).

## MCP tools

- `healthz_healthz_get`
- `detect`
- `status`

Default ports:
- HTTP: 8008
- MCP:  8608
