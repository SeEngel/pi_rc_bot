from __future__ import annotations

import argparse
import asyncio
import os
import sys


async def _mcp_health_check(mcp_url: str, *, timeout_seconds: float = 20.0) -> str:
	from agent_framework import MCPStreamableHTTPTool  # type: ignore

	async with MCPStreamableHTTPTool(name="head", url=mcp_url, timeout=timeout_seconds) as mcp:
		res = await mcp.call_tool("healthz_healthz_get")
		text = None
		for item in res:
			if getattr(item, "type", None) == "text" and hasattr(item, "text"):
				text = str(getattr(item, "text"))
				break
		if not text:
			raise RuntimeError(f"MCP health check failed: unexpected tool result: {res!r}")
		return "\n".join(["MCP health check OK.", f"healthz => {text}"])


async def _direct_set_angles(mcp_url: str, *, pan_deg: int | None, tilt_deg: int | None) -> str:
	from agent_framework import MCPStreamableHTTPTool  # type: ignore

	kwargs = {}
	if pan_deg is not None:
		kwargs["pan_deg"] = int(pan_deg)
	if tilt_deg is not None:
		kwargs["tilt_deg"] = int(tilt_deg)

	async with MCPStreamableHTTPTool(name="head", url=mcp_url, timeout=30.0) as mcp:
		res = await mcp.call_tool("set_angles", **kwargs)
		text = None
		for item in res:
			if getattr(item, "type", None) == "text" and hasattr(item, "text"):
				text = str(getattr(item, "text"))
				break
		if not text:
			raise RuntimeError(f"Direct set_angles failed: unexpected tool result: {res!r}")
		return text


async def _direct_center(mcp_url: str) -> str:
	from agent_framework import MCPStreamableHTTPTool  # type: ignore

	async with MCPStreamableHTTPTool(name="head", url=mcp_url, timeout=20.0) as mcp:
		res = await mcp.call_tool("center")
		text = None
		for item in res:
			if getattr(item, "type", None) == "text" and hasattr(item, "text"):
				text = str(getattr(item, "text"))
				break
		if not text:
			raise RuntimeError(f"Direct center failed: unexpected tool result: {res!r}")
		return text


async def _amain() -> int:
	parser = argparse.ArgumentParser(description="Run the HeadAgent (Agent Framework + MCP)")
	parser.add_argument(
		"--config",
		default=os.path.join(os.path.dirname(__file__), "config.yaml"),
		help="Path to agent/head/config.yaml",
	)
	parser.add_argument("--pan-deg", type=int, default=None)
	parser.add_argument("--tilt-deg", type=int, default=None)
	parser.add_argument("--center", action="store_true")
	parser.add_argument("--direct", action="store_true")
	parser.add_argument("--mcp-smoke-test", action="store_true")
	parser.add_argument("--skip-health-check", action="store_true")
	args = parser.parse_args()

	here = os.path.dirname(os.path.abspath(__file__))
	repo_root = os.path.abspath(os.path.join(here, "..", ".."))
	if repo_root not in sys.path:
		sys.path.insert(0, repo_root)

	from agent.head.src.head_agent import HeadAgent

	agent = HeadAgent.from_config_yaml(args.config)

	if args.mcp_smoke_test:
		print(await _mcp_health_check(agent.settings.head_mcp_url))
		return 0

	if not args.skip_health_check:
		print(await _mcp_health_check(agent.settings.head_mcp_url))

	if args.center:
		if args.direct:
			print(await _direct_center(agent.settings.head_mcp_url))
			return 0
		async with agent:
			print(await agent.center())
			return 0

	pan = args.pan_deg if args.pan_deg is not None else agent.settings.default_pan_deg
	tilt = args.tilt_deg if args.tilt_deg is not None else agent.settings.default_tilt_deg

	if args.direct:
		print(await _direct_set_angles(agent.settings.head_mcp_url, pan_deg=pan, tilt_deg=tilt))
		return 0

	async with agent:
		print(await agent.set_angles(pan_deg=pan, tilt_deg=tilt))
		return 0


def main() -> int:
	try:
		return asyncio.run(_amain())
	except KeyboardInterrupt:
		return 130


if __name__ == "__main__":
	raise SystemExit(main())
