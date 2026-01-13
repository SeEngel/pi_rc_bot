from __future__ import annotations

import os
from dataclasses import dataclass

from agent.src.base_agent import BaseAgentConfig, BaseWorkbenchChatAgent
from agent.src.config import MCPServerConfig, OpenAIClientConfig, load_yaml, resolve_repo_root


@dataclass(frozen=True)
class PerceptionAgentSettings:
	name: str
	instructions: str
	openai_model: str
	openai_base_url: str | None
	perception_mcp_url: str
	env_file_path: str | None


class PerceptionAgent(BaseWorkbenchChatAgent):
	"""Agent Framework wrapper for the `services/perception` MCP server."""

	settings: PerceptionAgentSettings

	def __init__(self, settings: PerceptionAgentSettings):
		self.settings = settings
		agent_cfg = BaseAgentConfig(
			name=settings.name,
			instructions=settings.instructions,
			openai=OpenAIClientConfig(
				model=settings.openai_model,
				base_url=settings.openai_base_url,
				env_file_path=settings.env_file_path,
			),
			mcp_servers=[MCPServerConfig(name="pi_rc_bot Perception MCP", url=settings.perception_mcp_url)],
		)
		super().__init__(agent_cfg)

	@classmethod
	def from_config_yaml(cls, path: str) -> "PerceptionAgent":
		cfg = load_yaml(path)
		here = os.path.dirname(os.path.abspath(path))
		repo_root = resolve_repo_root(here)
		env_file_path = os.path.join(repo_root, ".env") if os.path.isfile(os.path.join(repo_root, ".env")) else None

		agent_cfg = cfg.get("agent", {}) if isinstance(cfg, dict) else {}
		openai_cfg = cfg.get("openai", {}) if isinstance(cfg, dict) else {}
		mcp_cfg = cfg.get("mcp", {}) if isinstance(cfg, dict) else {}

		name = str((agent_cfg or {}).get("name") or "PerceptionAgent")
		instructions = str((agent_cfg or {}).get("instructions") or "")

		openai_model = str((openai_cfg or {}).get("model") or "gpt-4o-mini")
		base_url_raw = str((openai_cfg or {}).get("base_url") or "").strip()
		openai_base_url = base_url_raw or None

		perception_mcp_url = str((mcp_cfg or {}).get("perception_mcp_url") or "http://127.0.0.1:8608/mcp").strip()

		if not instructions.strip():
			instructions = (
				"You are the robot's perception assistant. You MUST use the MCP tool `detect` to get detections. "
				"Never fabricate detections.\n\n"
				"Tools:\n- detect {}\n- status {}\n\n"
				"If a tool fails, retry ONCE. Then return ONLY the tool result."
			)

		return cls(
			PerceptionAgentSettings(
				name=name,
				instructions=instructions,
				openai_model=openai_model,
				openai_base_url=openai_base_url,
				perception_mcp_url=perception_mcp_url,
				env_file_path=env_file_path,
			)
		)

	async def detect(self) -> str:
		return str(await self.run("Call MCP tool `detect` now. Return ONLY the tool result."))

	async def status(self) -> str:
		return str(await self.run("Call MCP tool `status` now. Return ONLY the tool result."))
