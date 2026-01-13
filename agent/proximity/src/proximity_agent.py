from __future__ import annotations

import os
from dataclasses import dataclass

from agent.src.base_agent import BaseAgentConfig, BaseWorkbenchChatAgent
from agent.src.config import MCPServerConfig, OpenAIClientConfig, load_yaml, resolve_repo_root


@dataclass(frozen=True)
class ProximityAgentSettings:
	name: str
	instructions: str
	openai_model: str
	openai_base_url: str | None
	proximity_mcp_url: str
	env_file_path: str | None

	default_threshold_cm: float


class ProximityAgent(BaseWorkbenchChatAgent):
	"""Agent Framework wrapper for the `services/proximity` MCP server."""

	settings: ProximityAgentSettings

	def __init__(self, settings: ProximityAgentSettings):
		self.settings = settings
		agent_cfg = BaseAgentConfig(
			name=settings.name,
			instructions=settings.instructions,
			openai=OpenAIClientConfig(
				model=settings.openai_model,
				base_url=settings.openai_base_url,
				env_file_path=settings.env_file_path,
			),
			mcp_servers=[MCPServerConfig(name="pi_rc_bot Proximity MCP", url=settings.proximity_mcp_url)],
		)
		super().__init__(agent_cfg)

	@classmethod
	def from_config_yaml(cls, path: str) -> "ProximityAgent":
		cfg = load_yaml(path)
		here = os.path.dirname(os.path.abspath(path))
		repo_root = resolve_repo_root(here)
		env_file_path = os.path.join(repo_root, ".env") if os.path.isfile(os.path.join(repo_root, ".env")) else None

		agent_cfg = cfg.get("agent", {}) if isinstance(cfg, dict) else {}
		openai_cfg = cfg.get("openai", {}) if isinstance(cfg, dict) else {}
		mcp_cfg = cfg.get("mcp", {}) if isinstance(cfg, dict) else {}

		name = str((agent_cfg or {}).get("name") or "ProximityAgent")
		instructions = str((agent_cfg or {}).get("instructions") or "")
		default_threshold_cm = float((agent_cfg or {}).get("default_threshold_cm") or 35.0)

		openai_model = str((openai_cfg or {}).get("model") or "gpt-4o-mini")
		base_url_raw = str((openai_cfg or {}).get("base_url") or "").strip()
		openai_base_url = base_url_raw or None

		proximity_mcp_url = str((mcp_cfg or {}).get("proximity_mcp_url") or "http://127.0.0.1:8607/mcp").strip()

		if not instructions.strip():
			instructions = (
				"You are the robot's proximity sensor reader. You MUST use MCP tools. "
				"Never fabricate sensor values.\n\n"
				"Tools:\n"
				"- distance_cm {}\n"
				"- is_obstacle {threshold_cm?:number}\n"
				"- status {}\n\n"
				"If a tool fails, retry ONCE. Then return ONLY the tool result."
			)

		return cls(
			ProximityAgentSettings(
				name=name,
				instructions=instructions,
				openai_model=openai_model,
				openai_base_url=openai_base_url,
				proximity_mcp_url=proximity_mcp_url,
				env_file_path=env_file_path,
				default_threshold_cm=default_threshold_cm,
			)
		)

	async def distance_cm(self) -> str:
		return str(await self.run("Call MCP tool `distance_cm` now. Return ONLY the tool result."))

	async def is_obstacle(self, *, threshold_cm: float | None = None) -> str:
		user_input = (
			"Call MCP tool `is_obstacle` now. Provide threshold_cm if given. Return ONLY the tool result.\n\n"
			f"threshold_cm={threshold_cm if threshold_cm is not None else ''}"
		)
		return str(await self.run(user_input))

	async def status(self) -> str:
		return str(await self.run("Call MCP tool `status` now. Return ONLY the tool result."))
