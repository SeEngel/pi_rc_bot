from __future__ import annotations

import os
from dataclasses import dataclass

from agent.src.base_agent import BaseAgentConfig, BaseWorkbenchChatAgent
from agent.src.config import MCPServerConfig, OpenAIClientConfig, load_yaml, resolve_repo_root


@dataclass(frozen=True)
class MoverAgentSettings:
	name: str
	instructions: str
	openai_model: str
	openai_base_url: str | None
	move_mcp_url: str
	env_file_path: str | None

	default_speed: int
	default_steer_deg: int
	default_duration_s: float


class MoverAgent(BaseWorkbenchChatAgent):
	"""Agent Framework wrapper for the `services/move` MCP server."""

	settings: MoverAgentSettings

	def __init__(self, settings: MoverAgentSettings):
		self.settings = settings
		agent_cfg = BaseAgentConfig(
			name=settings.name,
			instructions=settings.instructions,
			openai=OpenAIClientConfig(
				model=settings.openai_model,
				base_url=settings.openai_base_url,
				env_file_path=settings.env_file_path,
			),
			mcp_servers=[MCPServerConfig(name="pi_rc_bot Move MCP", url=settings.move_mcp_url)],
		)
		super().__init__(agent_cfg)

	@classmethod
	def from_config_yaml(cls, path: str) -> "MoverAgent":
		cfg = load_yaml(path)
		here = os.path.dirname(os.path.abspath(path))
		repo_root = resolve_repo_root(here)
		env_file_path = os.path.join(repo_root, ".env") if os.path.isfile(os.path.join(repo_root, ".env")) else None

		agent_cfg = cfg.get("agent", {}) if isinstance(cfg, dict) else {}
		openai_cfg = cfg.get("openai", {}) if isinstance(cfg, dict) else {}
		mcp_cfg = cfg.get("mcp", {}) if isinstance(cfg, dict) else {}

		name = str((agent_cfg or {}).get("name") or "MoverAgent")
		instructions = str((agent_cfg or {}).get("instructions") or "")

		default_speed = int((agent_cfg or {}).get("default_speed") or 20)
		default_steer_deg = int((agent_cfg or {}).get("default_steer_deg") or 0)
		default_duration_s = float((agent_cfg or {}).get("default_duration_s") or 0.6)

		openai_model = str((openai_cfg or {}).get("model") or "gpt-4o-mini")
		base_url_raw = str((openai_cfg or {}).get("base_url") or "").strip()
		openai_base_url = base_url_raw or None

		move_mcp_url = str((mcp_cfg or {}).get("move_mcp_url") or "http://127.0.0.1:8605/mcp").strip()

		if not instructions.strip():
			instructions = (
				"You are the robot's motion controller. You MUST use MCP tools to move the robot. "
				"Never claim you moved unless you called the tool.\n\n"
				"Available tools:\n"
				"- drive {speed:int, steer_deg:int, duration_s?:float}\n"
				"- stop {}\n"
				"- status {}\n\n"
				"Tool reliability policy:\n"
				"- If a tool call fails, retry ONCE with minimal valid inputs.\n"
				"- If the retry fails, return 'error: ...' then 'hint: ...'.\n\n"
				"When asked to move, call `drive` then return ONLY the tool result."
			)

		return cls(
			MoverAgentSettings(
				name=name,
				instructions=instructions,
				openai_model=openai_model,
				openai_base_url=openai_base_url,
				move_mcp_url=move_mcp_url,
				env_file_path=env_file_path,
				default_speed=default_speed,
				default_steer_deg=default_steer_deg,
				default_duration_s=default_duration_s,
			)
		)

	async def drive(self, *, speed: int, steer_deg: int = 0, duration_s: float | None = None) -> str:
		user_input = (
			"Call MCP tool `drive` now with these fields: speed, steer_deg, duration_s (if provided). "
			"Return ONLY the tool result.\n\n"
			f"speed={int(speed)}\nsteer_deg={int(steer_deg)}\nduration_s={duration_s if duration_s is not None else ''}"
		)
		return str(await self.run(user_input))

	async def stop(self) -> str:
		return str(await self.run("Call MCP tool `stop` now. Return ONLY the tool result."))

	async def status(self) -> str:
		return str(await self.run("Call MCP tool `status` now. Return ONLY the tool result."))
