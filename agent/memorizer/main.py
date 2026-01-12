from __future__ import annotations

import argparse
import asyncio
import os
import sys
from typing import Any


async def _mcp_smoke_test(mcp_url: str, *, timeout_seconds: float = 20.0) -> str:
	"""Validate the memory MCP server by invoking its health tool.

	This avoids false negatives from hitting /mcp with GET and proves the endpoint
	works with the Agent Framework MCP client.
	"""
	from agent_framework import MCPStreamableHTTPTool  # type: ignore

	async with MCPStreamableHTTPTool(name="memory", url=mcp_url, timeout=timeout_seconds) as mcp:
		res = await mcp.call_tool("healthz_healthz_get")
		text = None
		for item in res:
			if getattr(item, "type", None) == "text" and hasattr(item, "text"):
				text = str(getattr(item, "text"))
				break
		if not text:
			raise RuntimeError(f"MCP smoke test failed: unexpected tool result: {res!r}")
		return text


async def _direct_call(mcp_url: str, tool: str, **kwargs: Any) -> str:
	from agent_framework import MCPStreamableHTTPTool  # type: ignore

	async with MCPStreamableHTTPTool(name="memory", url=mcp_url, timeout=60.0) as mcp:
		res = await mcp.call_tool(tool, **kwargs)
		for item in res:
			if getattr(item, "type", None) == "text" and hasattr(item, "text"):
				return str(getattr(item, "text"))
		return str(res)


async def _amain() -> int:
	parser = argparse.ArgumentParser(description="Run the MemorizerAgent (Agent Framework + MCP)")
	parser.add_argument(
		"--config",
		default=os.path.join(os.path.dirname(__file__), "config.yaml"),
		help="Path to agent/memorizer/config.yaml",
	)
	parser.add_argument(
		"--mode",
		choices=("ingest", "recall", "compact"),
		default="ingest",
		help="What the memorizer should do",
	)
	parser.add_argument("--text", default=None, help="(ingest) Information to consider storing")
	parser.add_argument("--query", default=None, help="(recall) Query for memory retrieval")
	parser.add_argument("--topic", default=None, help="(compact) Topic to summarize/compact")
	parser.add_argument(
		"--top-n",
		type=int,
		default=None,
		help="(recall/compact) Top-N per tier to retrieve (1..10)",
	)
	parser.add_argument(
		"--force-store",
		action="store_true",
		help="(ingest) Force storing some derived memory (use carefully)",
	)
	parser.add_argument(
		"--direct",
		action="store_true",
		help="Call the memory MCP tools directly (no OpenAI/LLM)",
	)
	parser.add_argument(
		"--tags",
		default=None,
		help="(direct store) Comma-separated tags",
	)
	parser.add_argument(
		"--skip-health-check",
		action="store_true",
		help="Skip checking that services/memory MCP is running",
	)
	parser.add_argument(
		"--mcp-smoke-test",
		action="store_true",
		help="Validate the MCP endpoint by calling its health tool (no OpenAI required)",
	)
	args = parser.parse_args()

	# Ensure repo root is importable.
	here = os.path.dirname(os.path.abspath(__file__))
	repo_root = os.path.abspath(os.path.join(here, "..", ".."))
	if repo_root not in sys.path:
		sys.path.insert(0, repo_root)

	from agent.memorizer.src.memorizer_agent import MemorizerAgent

	agent = MemorizerAgent.from_config_yaml(args.config)

	if args.mcp_smoke_test:
		print(await _mcp_smoke_test(agent.settings.memory_mcp_url))
		return 0

	if not args.skip_health_check:
		# Prove connectivity via tool invocation.
		await _mcp_smoke_test(agent.settings.memory_mcp_url)

	if args.direct:
		if args.mode == "ingest":
			content = (args.text or "").strip()
			if not content:
				print("error: missing --text\nhint: provide --text for direct store")
				return 2
			tags = []
			if args.tags:
				tags = [t.strip() for t in str(args.tags).split(",") if t.strip()]
			out = await _direct_call(agent.settings.memory_mcp_url, "store_memory", content=content, tags=tags)
			print(out)
			return 0

		if args.mode == "recall":
			query = (args.query or "").strip()
			if not query:
				print("error: missing --query\nhint: provide --query for direct recall")
				return 2
			n = int(args.top_n or agent.settings.default_top_n or 3)
			out = await _direct_call(agent.settings.memory_mcp_url, "get_top_n_memory", content=query, top_n=n)
			print(out)
			return 0

		if args.mode == "compact":
			print("error: direct compact is not supported\nhint: run without --direct to let the LLM summarize")
			return 2

		print("error: unknown mode")
		return 2

	# LLM-driven modes
	async with agent:
		if args.mode == "recall":
			query = args.query or ""
			out = await agent.recall(query, top_n=args.top_n)
			print(out)
			return 0
		if args.mode == "compact":
			topic = args.topic or ""
			out = await agent.compact(topic, top_n=int(args.top_n or 5))
			print(out)
			return 0

		text = args.text or ""
		out = await agent.ingest(text, force_store=bool(args.force_store))
		print(out)
		return 0


def main() -> int:
	return asyncio.run(_amain())


if __name__ == "__main__":
	raise SystemExit(main())
