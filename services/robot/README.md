# robot service (shared PiCar-X hardware)

This service owns the single `picarx.Picarx()` instance (GPIO) and exposes a small HTTP + MCP API.

Why: multiple processes instantiating `Picarx()` at the same time often fails with `GPIO busy`. Instead, other services (`move`, `head`, `proximity`) proxy to this service.

## Ports

- HTTP API: `8010`
- MCP: `8610` (`HTTP + 600`)

## Endpoints

- `GET /healthz`
- `GET /status`
- `POST /drive` `{speed, steer_deg}`
- `POST /stop`
- `POST /head/set_angles` `{pan_deg, tilt_deg}`
- `POST /head/center`
- `GET /ultrasonic/distance`

## Notes

- If the camera is in use, that does not affect this service.
- If GPIO is in use by *another* process outside this stack, this service will report unavailable.
