# Move service

A small HTTP + MCP service that controls the PiCar-X chassis (drive + steering).

This service is designed to be started via `services/main.sh` using the system interpreter (`python3`) so it can access robot hardware libraries.

## MCP tools

- `healthz_healthz_get`: health check
- `drive`: drive with optional duration
- `stop`: stop immediately
- `status`: current motion state

## Run

```bash
python3 services/move/main.py
```

The HTTP API runs on port **8005** by default and the MCP endpoint on **8605** (`/mcp`).
