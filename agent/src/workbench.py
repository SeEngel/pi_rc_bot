from __future__ import annotations

from contextlib import AsyncExitStack
from dataclasses import dataclass
from typing import Iterable

from .config import MCPServerConfig


@dataclass
class MCPWorkbench:
	"""Manages one or more MCP servers as a tool workbench."""

	servers: list[MCPServerConfig]

	_stack: AsyncExitStack | None = None
	_tools: list[object] | None = None

	async def __aenter__(self) -> "MCPWorkbench":
		# Lazy import so this repo can be imported even without agent-framework installed.
		from agent_framework import MCPStreamableHTTPTool  # type: ignore

		self._stack = AsyncExitStack()
		self._tools = []
		try:
			for server in self.servers:
				tool_cm = MCPStreamableHTTPTool(name=server.name, url=server.url)
				tool = await self._stack.enter_async_context(tool_cm)
				self._tools.append(tool)
			return self
		except Exception:
			await self.__aexit__(None, None, None)
			raise

	async def __aexit__(self, exc_type, exc, tb) -> None:
		if self._stack is not None:
			await self._stack.aclose()
		self._stack = None
		self._tools = None

	@property
	def tools(self) -> list[object]:
		if self._tools is None:
			return []
		return list(self._tools)

	@staticmethod
	def single(name: str, url: str) -> "MCPWorkbench":
		return MCPWorkbench(servers=[MCPServerConfig(name=name, url=url)])
