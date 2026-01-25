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

If you can `POST /drive` and get `200 OK` but the car does not move, check `config.yaml`:

- `robot.dry_run: true` will intentionally **disable all hardware movement**.

### Motor speed trim (left/right)

If your left/right drive motors run at slightly different speeds for the same command, you can compensate in software via `config.yaml`:

- `robot.left_speed_offset`
- `robot.right_speed_offset`

These offsets are applied to each motor command (after steering scaling), preserving direction (forward/backward). For example, if you command `speed=10` and set `left_speed_offset: 5`, the left motor will be driven as `15`.
