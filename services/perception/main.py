from __future__ import annotations

import threading
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

import yaml
from fastapi import FastAPI
from pydantic import BaseModel, Field
from fastmcp import FastMCP

from src.perception import Perception


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
PORT = int(_service_cfg.get("port", 8008))
MCP_PORT = PORT + 600

_perception_lock = threading.Lock()
perception: Perception | None = None


def _get_perception() -> Perception:
	global perception
	if perception is None:
		perception = Perception.from_config_dict(_cfg)
	return perception


@asynccontextmanager
async def lifespan(_: FastAPI):
	yield
	with _perception_lock:
		try:
			if perception is not None:
				perception.close()
		except Exception:
			pass


app = FastAPI(title="perception", lifespan=lifespan)


class ServiceInfo(BaseModel):
	host: str
	port: int
	mcp_port: int


class StatusResponse(BaseModel):
	"""Best-effort perception status payload.

	The exact keys depend on the configured perception backend.
	"""
	model_config = {"extra": "allow"}


class HealthzResponse(StatusResponse):
	service: ServiceInfo


class DetectResponse(BaseModel):
	"""Single detection result.

	This endpoint returns detected entities from the perception backend.
	Typical keys include `faces` (list of detected faces) and `people` (list of detected people).
	The exact structure depends on the configured perception backend.
	"""
	ok: bool = Field(default=True, description="Whether the operation completed without errors.")
	available: bool | None = Field(
		default=None,
		description="Whether the perception backend reports as available.",
	)
	model_config = {"extra": "allow"}


@app.get(
	"/healthz",
	operation_id="healthz_healthz_get",
	summary="Health check",
	description="Returns service health and perception backend availability.",
	response_model=HealthzResponse,
)
def healthz() -> dict[str, Any]:
	with _perception_lock:
		p = _get_perception()
		st = p.status()
	st["service"] = {"host": HOST, "port": PORT, "mcp_port": MCP_PORT}
	return st


@app.get(
	"/detect",
	operation_id="detect",
	summary="Detect once",
	description="Runs a single perception pass and returns detected entities (best-effort).",
	response_model=DetectResponse,
)
def detect() -> dict[str, Any]:
	with _perception_lock:
		p = _get_perception()
		return p.detect_once()


@app.get(
	"/status",
	operation_id="status",
	summary="Status",
	description="Returns the current perception backend status/config (best-effort).",
	response_model=StatusResponse,
)
def status() -> dict[str, Any]:
	with _perception_lock:
		p = _get_perception()
		return p.status()


mcp = FastMCP.from_fastapi(app, name="perception")


if __name__ == "__main__":
	import uvicorn

	api_thread = threading.Thread(
		target=lambda: uvicorn.run(app, host=HOST, port=PORT, log_level="info"),
		daemon=True,
	)
	api_thread.start()

	uvicorn.run(mcp.http_app(path="/mcp"), host=HOST, port=MCP_PORT, log_level="info")
