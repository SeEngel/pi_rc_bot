from __future__ import annotations

import os
from dataclasses import dataclass

from agent.src.base_agent import BaseAgentConfig, BaseWorkbenchChatAgent
from agent.src.config import MCPServerConfig, OpenAIClientConfig, load_yaml, resolve_repo_root


@dataclass(frozen=True)
class MemorizerAgentSettings:
	name: str
	instructions: str
	openai_model: str
	openai_base_url: str | None
	memory_mcp_url: str
	env_file_path: str | None

	default_top_n: int


class MemorizerAgent(BaseWorkbenchChatAgent):
	"""An Agent Framework-based memorizer agent.

	It is specialized for using the `services/memory` MCP server.
	"""

	settings: MemorizerAgentSettings

	def __init__(self, settings: MemorizerAgentSettings):
		self.settings = settings
		agent_cfg = BaseAgentConfig(
			name=settings.name,
			instructions=settings.instructions,
			openai=OpenAIClientConfig(
				model=settings.openai_model,
				base_url=settings.openai_base_url,
				env_file_path=settings.env_file_path,
			),
			mcp_servers=[MCPServerConfig(name="pi_rc_bot Memory MCP", url=settings.memory_mcp_url)],
		)
		super().__init__(agent_cfg)

	@classmethod
	def from_config_yaml(cls, path: str) -> "MemorizerAgent":
		cfg = load_yaml(path)
		here = os.path.dirname(os.path.abspath(path))
		repo_root = resolve_repo_root(here)
		env_file_path = os.path.join(repo_root, ".env") if os.path.isfile(os.path.join(repo_root, ".env")) else None

		agent_cfg = cfg.get("agent", {}) if isinstance(cfg, dict) else {}
		openai_cfg = cfg.get("openai", {}) if isinstance(cfg, dict) else {}
		mcp_cfg = cfg.get("mcp", {}) if isinstance(cfg, dict) else {}

		name = str((agent_cfg or {}).get("name") or "MemorizerAgent")
		instructions = str((agent_cfg or {}).get("instructions") or "")
		default_top_n = int((agent_cfg or {}).get("default_top_n") or 3)

		openai_model = str((openai_cfg or {}).get("model") or "gpt-4o-mini")
		base_url_raw = str((openai_cfg or {}).get("base_url") or "").strip()
		openai_base_url = base_url_raw or None

		memory_mcp_url = str((mcp_cfg or {}).get("memory_mcp_url") or "http://127.0.0.1:8604/mcp").strip()

		if not instructions.strip():
			# Keep a reasonable fallback in case config.yaml is missing.
			instructions = (
				"You are the robot's memory manager. You MUST use the provided MCP tools to store and retrieve memories. "
				"Never claim memory actions without tool calls.\n\n"
				"To store: call tool `store_memory` with {content: <string>, tags: <list[string]>}.\n"
				"To retrieve: call tool `get_top_n_memory` with {content: <string>, top_n: <int 1..10>}.\n"
				"Be selective: store stable, useful facts; avoid sensitive secrets.\n"
			)

		return cls(
			MemorizerAgentSettings(
				name=name,
				instructions=instructions,
				openai_model=openai_model,
				openai_base_url=openai_base_url,
				memory_mcp_url=memory_mcp_url,
				env_file_path=env_file_path,
				default_top_n=default_top_n,
			)
		)

	async def ingest(self, info: str, *, force_store: bool = False) -> str:
		"""Decide whether to store `info` (and store if appropriate)."""
		text = (info or "").strip()
		if not text:
			return "error: empty input\nhint: pass --text or provide a non-empty info string"

		policy = "You MAY store only if it is genuinely useful long-term." if not force_store else "You MUST store something useful derived from the input."
		prompt = (
			"Decide whether the following information should be stored in memory. "
			"If you store it, you MUST call the MCP tool `store_memory`. "
			"If you skip storing, do NOT call any tool.\n\n"
			f"{policy}\n\n"
			"When storing:\n"
			"- Create ONE compact atomic memory statement (prefer <= 280 characters).\n"
			"- Optionally add 0..8 tags.\n\n"
			"Reply format (no extra text):\n"
			"- If stored:\n"
			"  stored: <memory content you stored>\n"
			"  tags: <comma-separated tags or '-'>\n"
			"- If skipped:\n"
			"  skipped: <short reason>\n\n"
			f"Input: {text}"
		)
		res = await self.run(prompt)
		return str(res)

	async def recall(self, query: str, *, top_n: int | None = None) -> str:
		"""Retrieve relevant memories and answer the query."""
		q = (query or "").strip()
		if not q:
			return "error: empty query\nhint: pass --query or provide a non-empty query string"

		n = int(top_n or self.settings.default_top_n or 3)
		n = max(1, min(10, n))
		prompt = (
			"Use the MCP tool `get_top_n_memory` to retrieve relevant memories. "
			"Pass the user's query as `content` and use the provided top_n. "
			"Then answer the user based ONLY on the returned memories.\n\n"
			"Reply format (no extra preamble):\n"
			"answer: <your best concise answer>\n"
			"evidence:\n"
			"- <short memory 1>\n"
			"- <short memory 2>\n\n"
			f"top_n: {n}\n"
			f"User query: {q}"
		)
		res = await self.run(prompt)
		return str(res)

	async def compact(self, topic: str, *, top_n: int = 5) -> str:
		"""Create and store a compact summary memory for a topic.

		This does NOT delete old memories; it creates a new summary memory item.
		"""
		q = (topic or "").strip()
		if not q:
			return "error: empty topic\nhint: pass --topic or provide a non-empty topic string"

		n = max(1, min(10, int(top_n or 5)))
		prompt = (
			"First, call the MCP tool `get_top_n_memory` to retrieve relevant memories for the topic. "
			"Then create ONE compact summary memory statement that captures the most useful enduring information. "
			"Finally, store it using the MCP tool `store_memory` with tag 'summary' plus any other helpful tags.\n\n"
			"Constraints for the stored summary:\n"
			"- <= 280 characters preferred\n"
			"- self-contained, not referencing 'above'/'previous'\n\n"
			"Reply format (no extra text):\n"
			"stored: <summary content you stored>\n"
			"tags: <comma-separated tags>\n\n"
			f"top_n: {n}\n"
			f"Topic: {q}"
		)
		res = await self.run(prompt)
		return str(res)
