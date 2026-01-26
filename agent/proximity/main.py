from __future__ import annotations

import argparse
import asyncio
import os
import sys


async def _mcp_health_check(mcp_url: str, *, timeout_seconds: float = 20.0) -> str:
	from agent_framework import MCPStreamableHTTPTool  # type: ignore

	async with MCPStreamableHTTPTool(name="proximity", url=mcp_url, timeout=timeout_seconds) as mcp:
		res = await mcp.call_tool("healthz_healthz_get")
		text = None
		for item in res:
			if getattr(item, "type", None) == "text" and hasattr(item, "text"):
				text = str(getattr(item, "text"))
				break
		if not text:
			raise RuntimeError(f"MCP health check failed: unexpected tool result: {res!r}")
		return "\n".join(["MCP health check OK.", f"healthz => {text}"])


async def _direct_distance(mcp_url: str) -> str:
	from agent_framework import MCPStreamableHTTPTool  # type: ignore

	async with MCPStreamableHTTPTool(name="proximity", url=mcp_url, timeout=10.0) as mcp:
		res = await mcp.call_tool("distance_cm")
		text = None
		for item in res:
			if getattr(item, "type", None) == "text" and hasattr(item, "text"):
				text = str(getattr(item, "text"))
				break
		if not text:
			raise RuntimeError(f"Direct distance failed: unexpected tool result: {res!r}")
		return text


async def _direct_obstacle(mcp_url: str, *, threshold_cm: float) -> str:
	from agent_framework import MCPStreamableHTTPTool  # type: ignore

	async with MCPStreamableHTTPTool(name="proximity", url=mcp_url, timeout=10.0) as mcp:
		res = await mcp.call_tool("is_obstacle", threshold_cm=float(threshold_cm))
		text = None
		for item in res:
			if getattr(item, "type", None) == "text" and hasattr(item, "text"):
				text = str(getattr(item, "text"))
				break
		if not text:
			raise RuntimeError(f"Direct is_obstacle failed: unexpected tool result: {res!r}")
		return text


async def _amain() -> int:
	parser = argparse.ArgumentParser(description="Run the ProximityAgent (Agent Framework + MCP)")
	parser.add_argument(
		"--config",
		default=os.path.join(os.path.dirname(__file__), "config.yaml"),
		help="Path to agent/proximity/config.yaml",
	)
	parser.add_argument("--threshold-cm", type=float, default=None)
	parser.add_argument("--obstacle", action="store_true")
	parser.add_argument("--direct", action="store_true")
	parser.add_argument("--mcp-smoke-test", action="store_true")
	parser.add_argument("--skip-health-check", action="store_true")
	args = parser.parse_args()

	here = os.path.dirname(os.path.abspath(__file__))
	repo_root = os.path.abspath(os.path.join(here, "..", ".."))
	if repo_root not in sys.path:
		sys.path.insert(0, repo_root)

	from agent.proximity.src.proximity_agent import ProximityAgent

	agent = ProximityAgent.from_config_yaml(args.config)

	if args.mcp_smoke_test:
		print(await _mcp_health_check(agent.settings.proximity_mcp_url))
		return 0

	if not args.skip_health_check:
		print(await _mcp_health_check(agent.settings.proximity_mcp_url))

	threshold = float(args.threshold_cm if args.threshold_cm is not None else agent.settings.default_threshold_cm)

	if args.direct:
		if args.obstacle:
			print(await _direct_obstacle(agent.settings.proximity_mcp_url, threshold_cm=threshold))
		else:
			print(await _direct_distance(agent.settings.proximity_mcp_url))
		return 0

	async with agent:
		if args.obstacle:
			print(await agent.is_obstacle(threshold_cm=threshold))
		else:
			print(await agent.distance_cm())
		return 0


def main() -> int:
	try:
		return asyncio.run(_amain())
	except KeyboardInterrupt:
		return 130


if __name__ == "__main__":
	raise SystemExit(main())
