from __future__ import annotations

import os
from dataclasses import dataclass

from agent.src.base_agent import BaseAgentConfig, BaseWorkbenchChatAgent
from agent.src.config import MCPServerConfig, OpenAIClientConfig, load_yaml, resolve_repo_root


@dataclass(frozen=True)
class ListenerAgentSettings:
	name: str
	instructions: str
	openai_model: str
	openai_base_url: str | None
	listen_mcp_url: str
	env_file_path: str | None

	default_prompt: str


class ListenerAgent(BaseWorkbenchChatAgent):
	"""An Agent Framework-based listener agent.

	It is specialized for using the `services/listening` MCP server.
	"""

	settings: ListenerAgentSettings

	def __init__(self, settings: ListenerAgentSettings):
		self.settings = settings
		agent_cfg = BaseAgentConfig(
			name=settings.name,
			instructions=settings.instructions,
			openai=OpenAIClientConfig(
				model=settings.openai_model,
				base_url=settings.openai_base_url,
				env_file_path=settings.env_file_path,
			),
			mcp_servers=[MCPServerConfig(name="pi_rc_bot Listening MCP", url=settings.listen_mcp_url)],
		)
		super().__init__(agent_cfg)

	@classmethod
	def from_config_yaml(cls, path: str) -> "ListenerAgent":
		cfg = load_yaml(path)
		here = os.path.dirname(os.path.abspath(path))
		repo_root = resolve_repo_root(here)
		env_file_path = os.path.join(repo_root, ".env") if os.path.isfile(os.path.join(repo_root, ".env")) else None

		agent_cfg = cfg.get("agent", {}) if isinstance(cfg, dict) else {}
		openai_cfg = cfg.get("openai", {}) if isinstance(cfg, dict) else {}
		mcp_cfg = cfg.get("mcp", {}) if isinstance(cfg, dict) else {}

		name = str((agent_cfg or {}).get("name") or "ListenerAgent")
		instructions = str((agent_cfg or {}).get("instructions") or "")
		default_prompt = str((agent_cfg or {}).get("default_prompt") or "Listen once and return only the transcript.")

		openai_model = str((openai_cfg or {}).get("model") or "gpt-4o-mini")
		base_url_raw = str((openai_cfg or {}).get("base_url") or "").strip()
		openai_base_url = base_url_raw or None

		listen_mcp_url = str((mcp_cfg or {}).get("listen_mcp_url") or "http://127.0.0.1:8602/mcp").strip()

		if not instructions.strip():
			instructions = (
				"You are the robot's speech-to-text assistant. You MUST use the provided MCP tool to listen. "
				"Never fabricate a transcript.\n\n"
				"Tool reliability policy (important):\n"
				"- If a tool call fails, returns an error, returns empty text, or returns unusable output, you MUST retry ONCE.\n"
				"- On retry, prefer calling `listen` with an EMPTY input. If you include options, use only: stream, speech_pause_seconds.\n"
				"- If the second attempt still fails, return a concise error in the format: 'error: ...' then 'hint: ...'.\n\n"
				"To record and transcribe, call tool `listen` (tool input may be empty, or include stream/speech_pause_seconds). "
				"Then return ONLY the tool's returned `text`."
			)

		return cls(
			ListenerAgentSettings(
				name=name,
				instructions=instructions,
				openai_model=openai_model,
				openai_base_url=openai_base_url,
				listen_mcp_url=listen_mcp_url,
				env_file_path=env_file_path,
				default_prompt=default_prompt,
			)
		)

	async def listen(self, prompt: str | None = None) -> str:
		"""Ask the LLM to call the listening MCP tool and return the transcript."""
		p = (prompt or self.settings.default_prompt or "").strip() or "Listen once and return only the transcript."
		user_input = (
			"Use the MCP tool `listen` now. "
			"Tool input may be empty. "
			"Return ONLY the tool's returned `text`, with no extra commentary.\n\n"
			f"Prompt: {p}"
		)
		res = await self.run(user_input)
		return str(res)
