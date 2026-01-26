from __future__ import annotations

import argparse
import asyncio
import os
import sys


async def _mcp_health_check(mcp_url: str, *, timeout_seconds: float = 20.0) -> str:
	from agent_framework import MCPStreamableHTTPTool  # type: ignore

	async with MCPStreamableHTTPTool(name="safety", url=mcp_url, timeout=timeout_seconds) as mcp:
		res = await mcp.call_tool("healthz_healthz_get")
		text = None
		for item in res:
			if getattr(item, "type", None) == "text" and hasattr(item, "text"):
				text = str(getattr(item, "text"))
				break
		if not text:
			raise RuntimeError(f"MCP health check failed: unexpected tool result: {res!r}")
		return "\n".join(["MCP health check OK.", f"healthz => {text}"])


async def _direct_call(mcp_url: str, tool: str, **kwargs) -> str:
	from agent_framework import MCPStreamableHTTPTool  # type: ignore

	async with MCPStreamableHTTPTool(name="safety", url=mcp_url, timeout=60.0) as mcp:
		res = await mcp.call_tool(tool, **kwargs)
		text = None
		for item in res:
			if getattr(item, "type", None) == "text" and hasattr(item, "text"):
				text = str(getattr(item, "text"))
				break
		if not text:
			raise RuntimeError(f"Direct {tool} failed: unexpected tool result: {res!r}")
		return text


async def _amain() -> int:
	parser = argparse.ArgumentParser(description="Run the SafetyAgent (Agent Framework + MCP)")
	parser.add_argument(
		"--config",
		default=os.path.join(os.path.dirname(__file__), "config.yaml"),
		help="Path to agent/safety/config.yaml",
	)
	parser.add_argument("--direct", action="store_true")
	parser.add_argument("--mcp-smoke-test", action="store_true")
	parser.add_argument("--skip-health-check", action="store_true")

	parser.add_argument("--check", action="store_true")
	parser.add_argument("--threshold-cm", type=float, default=None)
	parser.add_argument("--estop-on", action="store_true")
	parser.add_argument("--estop-off", action="store_true")
	parser.add_argument("--stop", action="store_true")

	parser.add_argument("--speed", type=int, default=None)
	parser.add_argument("--steer-deg", type=int, default=0)
	parser.add_argument("--duration-s", type=float, default=None)

	args = parser.parse_args()

	here = os.path.dirname(os.path.abspath(__file__))
	repo_root = os.path.abspath(os.path.join(here, "..", ".."))
	if repo_root not in sys.path:
		sys.path.insert(0, repo_root)

	from agent.safety.src.safety_agent import SafetyAgent

	agent = SafetyAgent.from_config_yaml(args.config)

	if args.mcp_smoke_test:
		print(await _mcp_health_check(agent.settings.safety_mcp_url))
		return 0

	if not args.skip_health_check:
		print(await _mcp_health_check(agent.settings.safety_mcp_url))

	threshold = float(args.threshold_cm if args.threshold_cm is not None else agent.settings.default_threshold_cm)

	if args.direct:
		if args.estop_on:
			print(await _direct_call(agent.settings.safety_mcp_url, "estop_on"))
			return 0
		if args.estop_off:
			print(await _direct_call(agent.settings.safety_mcp_url, "estop_off"))
			return 0
		if args.stop:
			print(await _direct_call(agent.settings.safety_mcp_url, "stop"))
			return 0
		if args.check:
			print(await _direct_call(agent.settings.safety_mcp_url, "check", threshold_cm=threshold))
			return 0
		if args.speed is not None:
			kwargs = {"speed": int(args.speed), "steer_deg": int(args.steer_deg), "threshold_cm": threshold}
			if args.duration_s is not None:
				kwargs["duration_s"] = float(args.duration_s)
			print(await _direct_call(agent.settings.safety_mcp_url, "guarded_drive", **kwargs))
			return 0
		raise SystemExit("No action specified")

	async with agent:
		if args.estop_on:
			print(await agent.estop_on())
			return 0
		if args.estop_off:
			print(await agent.estop_off())
			return 0
		if args.stop:
			print(await agent.stop())
			return 0
		if args.check:
			print(await agent.check(threshold_cm=threshold))
			return 0
		if args.speed is not None:
			print(await agent.guarded_drive(speed=int(args.speed), steer_deg=int(args.steer_deg), duration_s=args.duration_s, threshold_cm=threshold))
			return 0
		raise SystemExit("No action specified")


def main() -> int:
	try:
		return asyncio.run(_amain())
	except KeyboardInterrupt:
		return 130


if __name__ == "__main__":
	raise SystemExit(main())
