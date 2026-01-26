from __future__ import annotations

import os
from dataclasses import dataclass

from agent.src.base_agent import BaseAgentConfig, BaseWorkbenchChatAgent
from agent.src.config import MCPServerConfig, OpenAIClientConfig, load_yaml, resolve_repo_root


@dataclass(frozen=True)
class SafetyAgentSettings:
	name: str
	instructions: str
	openai_model: str
	openai_base_url: str | None
	safety_mcp_url: str
	env_file_path: str | None

	default_threshold_cm: float


class SafetyAgent(BaseWorkbenchChatAgent):
	"""Agent Framework wrapper for the `services/safety` MCP server."""

	settings: SafetyAgentSettings

	def __init__(self, settings: SafetyAgentSettings):
		self.settings = settings
		agent_cfg = BaseAgentConfig(
			name=settings.name,
			instructions=settings.instructions,
			openai=OpenAIClientConfig(
				model=settings.openai_model,
				base_url=settings.openai_base_url,
				env_file_path=settings.env_file_path,
			),
			mcp_servers=[MCPServerConfig(name="pi_rc_bot Safety MCP", url=settings.safety_mcp_url)],
		)
		super().__init__(agent_cfg)

	@classmethod
	def from_config_yaml(cls, path: str) -> "SafetyAgent":
		cfg = load_yaml(path)
		here = os.path.dirname(os.path.abspath(path))
		repo_root = resolve_repo_root(here)
		env_file_path = os.path.join(repo_root, ".env") if os.path.isfile(os.path.join(repo_root, ".env")) else None

		agent_cfg = cfg.get("agent", {}) if isinstance(cfg, dict) else {}
		openai_cfg = cfg.get("openai", {}) if isinstance(cfg, dict) else {}
		mcp_cfg = cfg.get("mcp", {}) if isinstance(cfg, dict) else {}

		name = str((agent_cfg or {}).get("name") or "SafetyAgent")
		instructions = str((agent_cfg or {}).get("instructions") or "")
		default_threshold_cm = float((agent_cfg or {}).get("default_threshold_cm") or 35.0)

		openai_model = str((openai_cfg or {}).get("model") or "gpt-4o-mini")
		base_url_raw = str((openai_cfg or {}).get("base_url") or "").strip()
		openai_base_url = base_url_raw or None

		safety_mcp_url = str((mcp_cfg or {}).get("safety_mcp_url") or "http://127.0.0.1:8609/mcp").strip()

		if not instructions.strip():
			instructions = (
				"You are the robot's safety controller. You MUST use MCP tools and you MUST prioritize safety.\n\n"
				"Tools:\n"
				"- check {threshold_cm?:number}\n"
				"- estop_on {}\n"
				"- estop_off {}\n"
				"- guarded_drive {speed:int, steer_deg?:int, duration_s?:number, threshold_cm?:number}\n"
				"- stop {}\n"
				"- status {}\n\n"
				"If `check.safe_to_drive` is false, do NOT call guarded_drive."
			)

		return cls(
			SafetyAgentSettings(
				name=name,
				instructions=instructions,
				openai_model=openai_model,
				openai_base_url=openai_base_url,
				safety_mcp_url=safety_mcp_url,
				env_file_path=env_file_path,
				default_threshold_cm=default_threshold_cm,
			)
		)

	async def check(self, *, threshold_cm: float | None = None) -> str:
		user_input = (
			"Call MCP tool `check` now. Provide threshold_cm if given. Return ONLY the tool result.\n\n"
			f"threshold_cm={threshold_cm if threshold_cm is not None else ''}"
		)
		return str(await self.run(user_input))

	async def estop_on(self) -> str:
		return str(await self.run("Call MCP tool `estop_on` now. Return ONLY the tool result."))

	async def estop_off(self) -> str:
		return str(await self.run("Call MCP tool `estop_off` now. Return ONLY the tool result."))

	async def guarded_drive(self, *, speed: int, steer_deg: int = 0, duration_s: float | None = None, threshold_cm: float | None = None) -> str:
		user_input = (
			"Call MCP tool `guarded_drive` now with speed and optional steer_deg, duration_s, threshold_cm. "
			"Return ONLY the tool result.\n\n"
			f"speed={int(speed)}\n"
			f"steer_deg={int(steer_deg)}\n"
			f"duration_s={duration_s if duration_s is not None else ''}\n"
			f"threshold_cm={threshold_cm if threshold_cm is not None else ''}"
		)
		return str(await self.run(user_input))

	async def stop(self) -> str:
		return str(await self.run("Call MCP tool `stop` now. Return ONLY the tool result."))

	async def status(self) -> str:
		return str(await self.run("Call MCP tool `status` now. Return ONLY the tool result."))
