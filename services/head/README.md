# Head service

Controls the PiCar-X camera pan/tilt ("head") via HTTP + MCP.

Designed to run under system `python3` (started via `services/main.sh`).

## MCP tools

- `healthz_healthz_get`
- `set_angles`
- `center`
- `scan`
- `stop`
- `status`

Default ports:
- HTTP: 8006
- MCP:  8606
