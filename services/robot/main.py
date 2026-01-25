from __future__ import annotations

import json
import logging
import threading
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

import yaml
from fastapi import Body, FastAPI, HTTPException
from pydantic import BaseModel, Field
from fastmcp import FastMCP

from src.robot_controller import RobotController


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
PORT = int(_service_cfg.get("port", 8010))
MCP_PORT = PORT + 600

_robot_lock = threading.Lock()
robot: RobotController | None = None

logger = logging.getLogger("robot")


def _get_robot() -> RobotController:
	global robot
	if robot is None:
		robot = RobotController.from_config_dict(_cfg)
	return robot


@asynccontextmanager
async def lifespan(_: FastAPI):
	# Init once on startup so GPIO ownership is explicit and stable.
	with _robot_lock:
		r = _get_robot()
		if r.settings.dry_run:
			logger.warning(
				"robot service is running with robot.dry_run=true; motors will NOT move (API calls will still return 200)."
			)
		elif not r.is_available:
			logger.error("robot service hardware unavailable: %s", r.unavailable_reason)
	yield
	with _robot_lock:
		try:
			if robot is not None:
				robot.stop()
		except Exception:
			pass


app = FastAPI(title="robot", lifespan=lifespan)


class ServiceInfo(BaseModel):
	host: str
	port: int
	mcp_port: int


class StatusResponse(BaseModel):
	"""Best-effort robot status payload.

	The exact keys depend on the configured hardware backend.
	"""
	model_config = {"extra": "allow"}


class HealthzResponse(StatusResponse):
	service: ServiceInfo


class DriveRequest(BaseModel):
	"""Request payload to drive the robot directly.

	NOTE: For safer motion with obstacle detection, prefer using the safety service's /guarded_drive endpoint.
	This endpoint bypasses safety checks.
	"""
	speed: int = Field(
		description="Signed speed percentage (-100 to 100). Positive = forward, negative = backward. 0 = stop.",
		examples=[30, -25, 50, 0],
		ge=-100,
		le=100,
	)
	steer_deg: int = Field(
		default=0,
		description="Steering angle in degrees (-35 to 35). Negative = turn left, positive = turn right.",
		examples=[0, -15, 15],
		ge=-35,
		le=35,
	)


class DriveResponse(BaseModel):
	"""Response from /drive confirming the motion command."""
	ok: bool = Field(default=True, description="Whether the operation completed without errors.")
	speed: int = Field(description="The speed that was applied.", examples=[30])
	steer_deg: int = Field(description="The steering angle that was applied.", examples=[0])


class StopResponse(BaseModel):
	"""Response from /stop."""
	ok: bool = Field(default=True, description="Whether the operation completed without errors.")
	stopped: bool = Field(default=True, description="True if the stop command was sent.")


class HeadSetAnglesRequest(BaseModel):
	"""Request payload to set head servo positions."""
	pan_deg: int | None = Field(
		default=None,
		description="Pan (horizontal) angle in degrees. 0 = center, negative = left, positive = right.",
		examples=[0, -45, 45],
	)
	tilt_deg: int | None = Field(
		default=None,
		description="Tilt (vertical) angle in degrees. 0 = level, negative = down, positive = up.",
		examples=[0, -20, 15],
	)


class HeadSetAnglesResponse(BaseModel):
	"""Response from /head/set_angles."""
	ok: bool = Field(default=True, description="Whether the operation completed without errors.")
	pan_deg: int = Field(description="Current pan angle after the operation.", examples=[0])
	tilt_deg: int = Field(description="Current tilt angle after the operation.", examples=[0])


class UltrasonicDistanceResponse(BaseModel):
	"""Response from /ultrasonic/distance with the sensor reading."""
	ok: bool = Field(description="Whether the sensor read was successful.")
	available: bool = Field(description="Whether the ultrasonic sensor is available.")
	distance_cm: float | None = Field(default=None, description="Measured distance in centimeters. Null if unavailable.", examples=[25.5, 100.0])
	reason: str | None = Field(default=None, description="Reason for unavailability if applicable.")


@app.get(
	"/healthz",
	operation_id="healthz_healthz_get",
	summary="Health check",
	description="Returns robot controller status and service networking info.",
	response_model=HealthzResponse,
)
def healthz() -> dict[str, Any]:
	with _robot_lock:
		r = _get_robot()
		st = r.status()
	st["service"] = {"host": HOST, "port": PORT, "mcp_port": MCP_PORT}
	return st


@app.get(
	"/status",
	operation_id="status",
	summary="Status",
	description="Returns the robot controller status/config (best-effort).",
	response_model=StatusResponse,
)
def status() -> dict[str, Any]:
	with _robot_lock:
		r = _get_robot()
		st = r.status()
	st["service"] = {"host": HOST, "port": PORT, "mcp_port": MCP_PORT}
	return st


@app.post(
	"/drive",
	operation_id="drive",
	summary="Drive",
	description="Drives the robot with a signed speed and steering angle. For safer motion prefer using the safety service's guarded_drive.",
	response_model=DriveResponse,
)
async def drive(payload: DriveRequest = Body(...)) -> dict[str, Any]:
	speed = int(payload.speed)
	steer = int(payload.steer_deg or 0)

	with _robot_lock:
		r = _get_robot()
		if not (r.is_available or r.settings.dry_run):
			raise HTTPException(status_code=503, detail=r.unavailable_reason or "Robot unavailable")
		try:
			r.drive(speed=speed, steer_deg=steer)
		except Exception as exc:
			raise HTTPException(status_code=500, detail=str(exc)) from exc

	return {"ok": True, "speed": speed, "steer_deg": steer}


@app.post(
	"/stop",
	operation_id="stop",
	summary="Stop",
	description="Stops all robot motion (best-effort).",
	response_model=StopResponse,
)
def stop() -> dict[str, Any]:
	with _robot_lock:
		r = _get_robot()
		try:
			r.stop()
		except Exception:
			pass
	return {"ok": True, "stopped": True}


@app.post(
	"/head/set_angles",
	operation_id="head_set_angles",
	summary="Set head angles",
	description="Sets head pan/tilt angles (degrees).",
	response_model=HeadSetAnglesResponse,
)
async def head_set_angles(payload: HeadSetAnglesRequest = Body(...)) -> dict[str, Any]:
	pan_i = int(payload.pan_deg) if payload.pan_deg is not None else None
	tilt_i = int(payload.tilt_deg) if payload.tilt_deg is not None else None

	with _robot_lock:
		r = _get_robot()
		if not (r.is_available or r.settings.dry_run):
			raise HTTPException(status_code=503, detail=r.unavailable_reason or "Robot unavailable")
		try:
			r.set_head_angles(pan_deg=pan_i, tilt_deg=tilt_i)
		except Exception as exc:
			raise HTTPException(status_code=500, detail=str(exc)) from exc

	return {"ok": True, "pan_deg": r.pan_deg, "tilt_deg": r.tilt_deg}


@app.post(
	"/head/center",
	operation_id="head_center",
	summary="Center head",
	description="Centers head pan/tilt.",
)
def head_center() -> dict[str, Any]:
	with _robot_lock:
		r = _get_robot()
		if not (r.is_available or r.settings.dry_run):
			raise HTTPException(status_code=503, detail=r.unavailable_reason or "Robot unavailable")
		try:
			r.center_head()
		except Exception as exc:
			raise HTTPException(status_code=500, detail=str(exc)) from exc
	return {"ok": True, "pan_deg": 0, "tilt_deg": 0}


@app.get(
	"/ultrasonic/distance",
	operation_id="ultrasonic_distance_cm",
	summary="Read ultrasonic distance",
	description="Reads the forward ultrasonic distance sensor (cm) if available.",
	response_model=UltrasonicDistanceResponse,
)
def ultrasonic_distance() -> dict[str, Any]:
	with _robot_lock:
		r = _get_robot()
		if not (r.is_available or r.settings.dry_run):
			return {"ok": False, "available": False, "reason": r.unavailable_reason or "Robot unavailable"}
		d = r.read_distance_cm()
	return {"ok": True, "available": True, "distance_cm": d}


mcp = FastMCP.from_fastapi(app, name="robot")


if __name__ == "__main__":
	import uvicorn

	api_thread = threading.Thread(
		target=lambda: uvicorn.run(app, host=HOST, port=PORT, log_level="info"),
		daemon=True,
	)
	api_thread.start()

	uvicorn.run(mcp.http_app(path="/mcp"), host=HOST, port=MCP_PORT, log_level="info")
