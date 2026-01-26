# Safety service

Central safety/guardrails service (e-stop + guarded motion) via HTTP + MCP.

It does **not** directly control hardware. Instead, it:
- checks `services/proximity` for obstacles
- delegates motion to `services/move`

Designed to run under system `python3` (started via `services/main.sh`).

## MCP tools

- `healthz_healthz_get`
- `status`
- `estop_on`
- `estop_off`
- `check`
- `guarded_drive`
- `stop`

Default ports:
- HTTP: 8009
- MCP:  8609
