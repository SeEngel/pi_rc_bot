from __future__ import annotations

import threading
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

import yaml
from fastapi import Body, FastAPI, Query
from pydantic import BaseModel, Field
from fastmcp import FastMCP

from src.safety import SafetyController


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
PORT = int(_service_cfg.get("port", 8009))
MCP_PORT = PORT + 600

_lock = threading.Lock()
controller: SafetyController | None = None


def _get_controller() -> SafetyController:
	global controller
	if controller is None:
		controller = SafetyController.from_config_dict(_cfg)
	return controller


@asynccontextmanager
async def lifespan(_: FastAPI):
	with _lock:
		_get_controller()
	yield


app = FastAPI(title="safety", lifespan=lifespan)


class ServiceInfo(BaseModel):
	host: str
	port: int
	mcp_port: int


class StatusResponse(BaseModel):
	"""Best-effort safety controller status payload."""
	model_config = {"extra": "allow"}


class HealthzResponse(StatusResponse):
	service: ServiceInfo


class EstopResponse(BaseModel):
	ok: bool = True
	estop: bool = Field(description="Whether emergency stop is currently active.")
	model_config = {"extra": "allow"}


class CheckResponse(BaseModel):
	ok: bool = True
	safe: bool | None = Field(default=None, description="Whether it is safe to drive forward under current conditions.")
	blocked: bool | None = Field(default=None, description="Whether driving is blocked (e.g. obstacle detected or estop).")
	threshold_cm: float | None = None
	distance_cm: float | None = None
	model_config = {"extra": "allow"}


class GuardedDriveRequest(BaseModel):
	speed: int = Field(description="Signed speed (-100..100). Negative = backward.")
	steer_deg: int = Field(default=0, description="Steering degrees (-35..35).")
	duration_s: float | None = Field(default=None, description="If provided, run for this duration then stop.")
	threshold_cm: float | None = Field(default=None, description="Obstacle threshold override (cm).")


class GuardedDriveResponse(BaseModel):
	ok: bool = True
	blocked: bool | None = None
	model_config = {"extra": "allow"}


class StopResponse(BaseModel):
	ok: bool = True
	stopped: bool | None = None
	model_config = {"extra": "allow"}


@app.get(
	"/healthz",
	operation_id="healthz_healthz_get",
	summary="Health check",
	description="Returns safety controller status and service networking info.",
	response_model=HealthzResponse,
)
def healthz() -> dict[str, Any]:
	with _lock:
		c = _get_controller()
		st = c.status()
	st["service"] = {"host": HOST, "port": PORT, "mcp_port": MCP_PORT}
	return st


@app.get(
	"/status",
	operation_id="status",
	summary="Status",
	description="Returns the current safety controller status/config (best-effort).",
	response_model=StatusResponse,
)
def status() -> dict[str, Any]:
	with _lock:
		c = _get_controller()
		return c.status()


@app.post(
	"/estop/on",
	operation_id="estop_on",
	summary="Emergency stop ON",
	description="Engages the emergency stop (blocks motion).",
	response_model=EstopResponse,
)
def estop_on() -> dict[str, Any]:
	with _lock:
		c = _get_controller()
		return c.estop_on()


@app.post(
	"/estop/off",
	operation_id="estop_off",
	summary="Emergency stop OFF",
	description="Releases the emergency stop (does not start motion by itself).",
	response_model=EstopResponse,
)
def estop_off() -> dict[str, Any]:
	with _lock:
		c = _get_controller()
		return c.estop_off()


@app.get(
	"/check",
	operation_id="check",
	summary="Safety check",
	description="Checks whether forward motion is currently safe given obstacle distance and estop state.",
	response_model=CheckResponse,
)
def check(
	threshold_cm: float | None = Query(default=None, description="Override obstacle threshold (cm)."),
) -> dict[str, Any]:
	with _lock:
		c = _get_controller()
		return c.check(threshold_cm=threshold_cm)


@app.post(
	"/guarded_drive",
	operation_id="guarded_drive",
	summary="Guarded drive",
	description=(
		"Drives the robot through the safety controller. If motion is unsafe (obstacle/estop), it will block. "
		"This is the recommended way to drive from higher-level agents."
	),
	response_model=GuardedDriveResponse,
)
async def guarded_drive(payload: GuardedDriveRequest = Body(...)) -> dict[str, Any]:
	speed = int(payload.speed)
	steer = int(payload.steer_deg or 0)
	duration_s = float(payload.duration_s) if payload.duration_s is not None else None
	threshold_cm = float(payload.threshold_cm) if payload.threshold_cm is not None else None

	with _lock:
		c = _get_controller()
		return c.guarded_drive(speed=speed, steer_deg=steer, duration_s=duration_s, threshold_cm=threshold_cm)


@app.post(
	"/stop",
	operation_id="stop",
	summary="Stop",
	description="Stops motion through the safety controller (best-effort).",
	response_model=StopResponse,
)
def stop() -> dict[str, Any]:
	with _lock:
		c = _get_controller()
		return c.stop()


mcp = FastMCP.from_fastapi(app, name="safety")


if __name__ == "__main__":
	import uvicorn

	api_thread = threading.Thread(
		target=lambda: uvicorn.run(app, host=HOST, port=PORT, log_level="info"),
		daemon=True,
	)
	api_thread.start()

	uvicorn.run(mcp.http_app(path="/mcp"), host=HOST, port=MCP_PORT, log_level="info")
