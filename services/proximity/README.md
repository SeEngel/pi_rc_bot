# Proximity service

Reads proximity sensors (currently: PiCar-X ultrasonic distance) and exposes them via HTTP + MCP.

Designed to run under system `python3` (started via `services/main.sh`).

## MCP tools

- `healthz_healthz_get`
- `distance_cm`
- `is_obstacle`

Default ports:
- HTTP: 8007
- MCP:  8607
