from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any

from agent.src.base_agent import BaseAgentConfig, BaseWorkbenchChatAgent
from agent.src.config import MCPServerConfig, OpenAIClientConfig, load_yaml, resolve_repo_root


@dataclass(frozen=True)
class ObserverAgentSettings:
	name: str
	instructions: str
	openai_model: str
	openai_base_url: str | None
	observe_mcp_url: str
	env_file_path: str | None

	default_question: str


class ObserverAgent(BaseWorkbenchChatAgent):
	"""An Agent Framework-based observer agent.

	It is specialized for using the `services/observe` MCP server.
	"""

	settings: ObserverAgentSettings

	def __init__(self, settings: ObserverAgentSettings):
		self.settings = settings
		agent_cfg = BaseAgentConfig(
			name=settings.name,
			instructions=settings.instructions,
			openai=OpenAIClientConfig(
				model=settings.openai_model,
				base_url=settings.openai_base_url,
				env_file_path=settings.env_file_path,
			),
			mcp_servers=[MCPServerConfig(name="pi_rc_bot Observe MCP", url=settings.observe_mcp_url)],
		)
		super().__init__(agent_cfg)

	@classmethod
	def from_config_yaml(cls, path: str) -> "ObserverAgent":
		cfg = load_yaml(path)
		here = os.path.dirname(os.path.abspath(path))
		repo_root = resolve_repo_root(here)
		env_file_path = os.path.join(repo_root, ".env") if os.path.isfile(os.path.join(repo_root, ".env")) else None

		agent_cfg = cfg.get("agent", {}) if isinstance(cfg, dict) else {}
		openai_cfg = cfg.get("openai", {}) if isinstance(cfg, dict) else {}
		mcp_cfg = cfg.get("mcp", {}) if isinstance(cfg, dict) else {}

		name = str((agent_cfg or {}).get("name") or "ObserverAgent")
		instructions = str((agent_cfg or {}).get("instructions") or "")
		default_question = str((agent_cfg or {}).get("default_question") or "What do you see?")

		openai_model = str((openai_cfg or {}).get("model") or "gpt-4o-mini")
		base_url_raw = str((openai_cfg or {}).get("base_url") or "").strip()
		openai_base_url = base_url_raw or None

		observe_mcp_url = str((mcp_cfg or {}).get("observe_mcp_url") or "http://127.0.0.1:8603/mcp").strip()

		if not instructions.strip():
			instructions = (
				"You are the robot's observer. You MUST use the provided MCP tools to see the world. "
				"Never fabricate what you see.\n\n"
				"When the user asks to describe the scene, call tool `observe` with {question: <string>} and return the tool's `text`.\n"
				"When the user asks for a movement suggestion, call tool `observe_direction` with {question: <string>} and summarize: action, why, fit." 
			)

		return cls(
			ObserverAgentSettings(
				name=name,
				instructions=instructions,
				openai_model=openai_model,
				openai_base_url=openai_base_url,
				observe_mcp_url=observe_mcp_url,
				env_file_path=env_file_path,
				default_question=default_question,
			)
		)

	async def describe(self, question: str | None = None) -> str:
		q = (question or self.settings.default_question or "").strip() or "Describe the scene."
		prompt = (
			"Use the MCP tool `observe` now. "
			"Pass the user's question as the tool input field `question`. "
			"Return ONLY the tool result text, with no extra commentary.\n\n"
			f"User question: {q}"
		)
		res = await self.run(prompt)
		return str(res)

	async def suggest_direction(self, question: str | None = None) -> str:
		q = (question or "").strip() or "Where should the robot move next to approach the most interesting object?"
		prompt = (
			"Use the MCP tool `observe_direction` now. "
			"Pass the user's goal as the tool input field `question`. "
			"Then reply in 3 lines exactly:\n"
			"action: <action>\n"
			"why: <why>\n"
			"fit: <fit>\n\n"
			f"User goal: {q}"
		)
		res = await self.run(prompt)
		return str(res)
