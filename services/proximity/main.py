from __future__ import annotations

import threading
from pathlib import Path
from typing import Any

import yaml
from fastapi import FastAPI, Query
from pydantic import BaseModel, Field
from fastmcp import FastMCP

from src.proximity import ProximitySensor


def load_config() -> dict[str, Any]:
	cfg_path = Path(__file__).with_name("config.yaml")
	if not cfg_path.exists():
		return {}
	try:
		with cfg_path.open("r", encoding="utf-8") as f:
			data = yaml.safe_load(f) or {}
			return data if isinstance(data, dict) else {}
	except Exception:
		return {}


_cfg = load_config()
_service_cfg = _cfg.get("service", {}) if isinstance(_cfg, dict) else {}
HOST = str(_service_cfg.get("host", "0.0.0.0"))
PORT = int(_service_cfg.get("port", 8007))
MCP_PORT = PORT + 600

sensor = ProximitySensor.from_config_dict(_cfg)

app = FastAPI(title="proximity")

_state_lock = threading.Lock()


class ServiceInfo(BaseModel):
	host: str
	port: int
	mcp_port: int


class StatusResponse(BaseModel):
	"""Best-effort sensor status payload.

	The exact keys depend on the configured hardware backend.
	"""
	# Keep flexible: different proximity backends may report different keys.
	model_config = {"extra": "allow"}


class HealthzResponse(StatusResponse):
	"""Health/status response including service networking info."""
	service: ServiceInfo


class DistanceResponse(BaseModel):
	ok: bool = True
	distance_cm: float | None = Field(default=None, description="Measured distance in centimeters. Null if unavailable.")
	status: dict[str, Any]


class ObstacleResponse(BaseModel):
	ok: bool = True
	obstacle: bool = Field(description="True if distance_cm <= threshold_cm (and distance is available).")
	distance_cm: float | None
	threshold_cm: float
	status: dict[str, Any]


@app.get(
	"/healthz",
	operation_id="healthz_healthz_get",
	summary="Health check",
	description="Returns service health and ultrasonic proximity sensor availability.",
	response_model=HealthzResponse,
)
def healthz() -> dict[str, Any]:
	with _state_lock:
		st = sensor.status()
	st["service"] = {"host": HOST, "port": PORT, "mcp_port": MCP_PORT}
	return st


@app.get(
	"/distance",
	operation_id="distance_cm",
	summary="Read distance",
	description="Reads the current distance measurement in centimeters.",
	response_model=DistanceResponse,
)
def distance_cm() -> dict[str, Any]:
	with _state_lock:
		d = sensor.read_distance_cm()
		st = sensor.status()
	return {"ok": True, "distance_cm": d, "status": st}


@app.get(
	"/obstacle",
	operation_id="is_obstacle",
	summary="Obstacle check",
	description="Convenience endpoint that compares the current distance to a threshold.",
	response_model=ObstacleResponse,
)
def is_obstacle(
	threshold_cm: float | None = Query(
		default=None,
		description="Override obstacle threshold (cm). If omitted, uses config/default status threshold.",
	),
) -> dict[str, Any]:
	with _state_lock:
		d = sensor.read_distance_cm()
		st = sensor.status()
		thr = float(threshold_cm) if threshold_cm is not None else float(st["obstacle_threshold_cm"])
		obs = (d is not None) and (d <= thr)
	return {"ok": True, "obstacle": obs, "distance_cm": d, "threshold_cm": thr, "status": st}


@app.get(
	"/status",
	operation_id="status",
	summary="Status",
	description="Returns the last-known configuration/status for the proximity sensor backend.",
	response_model=StatusResponse,
)
def status() -> dict[str, Any]:
	with _state_lock:
		return sensor.status()


mcp = FastMCP.from_fastapi(app, name="proximity")


if __name__ == "__main__":
	import uvicorn

	# HTTP API
	api_thread = threading.Thread(
		target=lambda: uvicorn.run(app, host=HOST, port=PORT, log_level="info"),
		daemon=True,
	)
	api_thread.start()

	# MCP
	uvicorn.run(mcp.http_app(path="/mcp"), host=HOST, port=MCP_PORT, log_level="info")
