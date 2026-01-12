from __future__ import annotations

import argparse
import asyncio
import os
import sys


async def _mcp_smoke_test(mcp_url: str, *, timeout_seconds: float = 20.0) -> str:
	"""Validate the observe MCP server by actually invoking its tools (no curl).

	This avoids false negatives from hitting /mcp with GET and proves the endpoint
	works with the Agent Framework MCP client.
	"""
	from agent_framework import MCPStreamableHTTPTool  # type: ignore

	async with MCPStreamableHTTPTool(name="observe", url=mcp_url, timeout=timeout_seconds) as mcp:
		# 1) Basic observe call
		res = await mcp.call_tool("observe", question="Smoke test: briefly describe what you see.")
		text = None
		for item in res:
			if getattr(item, "type", None) == "text" and hasattr(item, "text"):
				text = str(getattr(item, "text"))
				break
		if not text:
			raise RuntimeError(f"MCP smoke test failed: unexpected tool result: {res!r}")

		# 2) Direction tool (optional but useful)
		res2 = await mcp.call_tool(
			"observe_direction",
			question="Smoke test: pick a movement direction toward the most interesting object.",
		)
		text2 = None
		for item in res2:
			if getattr(item, "type", None) == "text" and hasattr(item, "text"):
				text2 = str(getattr(item, "text"))
				break
		if not text2:
			raise RuntimeError(f"MCP smoke test failed (observe_direction): unexpected tool result: {res2!r}")

	return "\n".join(
		[
			"MCP smoke test OK.",
			f"observe => {text}",
			f"observe_direction => {text2}",
		]
	)
async def _amain() -> int:
	parser = argparse.ArgumentParser(description="Run the ObserverAgent (Agent Framework + MCP)")
	parser.add_argument(
		"--config",
		default=os.path.join(os.path.dirname(__file__), "config.yaml"),
		help="Path to agent/observer/config.yaml",
	)
	parser.add_argument("--question", default=None, help="Question/goal for the observer")
	parser.add_argument(
		"--mode",
		choices=("describe", "direction"),
		default="describe",
		help="What to ask the observer to do",
	)
	parser.add_argument(
		"--skip-health-check",
		action="store_true",
		help="Skip checking that services/observe is running",
	)
	parser.add_argument(
		"--mcp-smoke-test",
		action="store_true",
		help="Validate the MCP endpoint by calling its tools (no OpenAI required)",
	)
	args = parser.parse_args()

	# Ensure repo root is importable.
	here = os.path.dirname(os.path.abspath(__file__))
	repo_root = os.path.abspath(os.path.join(here, "..", ".."))
	if repo_root not in sys.path:
		sys.path.insert(0, repo_root)

	from agent.observer.src.observer_agent import ObserverAgent

	agent = ObserverAgent.from_config_yaml(args.config)
	if args.mcp_smoke_test:
		print(await _mcp_smoke_test(agent.settings.observe_mcp_url))
		return 0
	if not args.skip_health_check:
		# We intentionally do *not* use curl/GET for MCP; prove connectivity via tool invocation.
		await _mcp_smoke_test(agent.settings.observe_mcp_url)

	async with agent:
		if args.mode == "direction":
			out = await agent.suggest_direction(args.question)
		else:
			out = await agent.describe(args.question)
		print(out)

	return 0


def main() -> int:
	try:
		return asyncio.run(_amain())
	except KeyboardInterrupt:
		return 130


if __name__ == "__main__":
	raise SystemExit(main())
