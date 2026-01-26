from __future__ import annotations

from contextlib import AsyncExitStack
from dataclasses import dataclass
from typing import Any, Sequence

from .config import MCPServerConfig, OpenAIClientConfig
from .workbench import MCPWorkbench


@dataclass(frozen=True)
class BaseAgentConfig:
	name: str
	instructions: str
	openai: OpenAIClientConfig
	mcp_servers: list[MCPServerConfig]


class BaseWorkbenchChatAgent:
	"""Reusable base agent built on Microsoft Agent Framework.

	- Uses OpenAI via Agent Framework's `OpenAIChatClient`
	- Can attach one or more MCP servers as a "workbench" (tooling surface)
	
	Subclasses typically only:
	- provide instructions
	- provide MCP servers
	- add small convenience methods
	"""

	def __init__(self, cfg: BaseAgentConfig):
		self.cfg = cfg
		self._stack: AsyncExitStack | None = None
		self._agent: Any | None = None
		self._workbench: MCPWorkbench | None = None

	async def __aenter__(self) -> "BaseWorkbenchChatAgent":
		from agent_framework.openai import OpenAIChatClient  # type: ignore

		self._stack = AsyncExitStack()

		# MCP workbench first, so its tools can be passed to the agent on creation.
		self._workbench = MCPWorkbench(self.cfg.mcp_servers)
		await self._stack.enter_async_context(self._workbench)

		client = OpenAIChatClient(
			model_id=self.cfg.openai.model,
			api_key=None,  # Load from env / env_file_path
			base_url=self.cfg.openai.base_url,
			env_file_path=self.cfg.openai.env_file_path,
			env_file_encoding="utf-8",
		)

		# Create the agent with tools on agent-level.
		agent_cm = client.create_agent(
			name=self.cfg.name,
			instructions=self.cfg.instructions,
			tools=self._workbench.tools,
		)
		self._agent = await self._stack.enter_async_context(agent_cm)
		return self

	async def __aexit__(self, exc_type, exc, tb) -> None:
		if self._stack is not None:
			await self._stack.aclose()
		self._stack = None
		self._agent = None
		self._workbench = None

	async def run(self, user_input: Any, **kwargs: Any) -> Any:
		if self._agent is None:
			raise RuntimeError("Agent not initialized (did you forget 'async with'?)")
		return await self._agent.run(user_input, **kwargs)

	async def run_stream(self, user_input: Any, **kwargs: Any) -> Any:
		"""Stream the agent's response, yielding chunks as they arrive."""
		if self._agent is None:
			raise RuntimeError("Agent not initialized (did you forget 'async with'?)")
		async for update in self._agent.run_stream(user_input, **kwargs):
			yield update

	def get_new_thread(self) -> Any:
		if self._agent is None:
			raise RuntimeError("Agent not initialized (did you forget 'async with'?)")
		return self._agent.get_new_thread()
