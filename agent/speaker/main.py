from __future__ import annotations

import argparse
import asyncio
import os
import sys


async def _mcp_health_check(mcp_url: str, *, timeout_seconds: float = 20.0) -> str:
	"""Validate the speak MCP server by invoking its health tool."""
	from agent_framework import MCPStreamableHTTPTool  # type: ignore

	async with MCPStreamableHTTPTool(name="speak", url=mcp_url, timeout=timeout_seconds) as mcp:
		res = await mcp.call_tool("healthz_healthz_get")
		text = None
		for item in res:
			if getattr(item, "type", None) == "text" and hasattr(item, "text"):
				text = str(getattr(item, "text"))
				break
		if not text:
			raise RuntimeError(f"MCP health check failed: unexpected tool result: {res!r}")
		return "\n".join(["MCP health check OK.", f"healthz => {text}"])


async def _direct_speak(mcp_url: str, *, text: str, timeout_seconds: float = 60.0) -> str:
	"""Call the MCP `speak` tool directly (no LLM required)."""
	from agent_framework import MCPStreamableHTTPTool  # type: ignore

	async with MCPStreamableHTTPTool(name="speak", url=mcp_url, timeout=timeout_seconds) as mcp:
		res = await mcp.call_tool("speak", text=str(text))
		out = None
		for item in res:
			if getattr(item, "type", None) == "text" and hasattr(item, "text"):
				out = str(getattr(item, "text"))
				break
		if not out:
			raise RuntimeError(f"Direct speak failed: unexpected tool result: {res!r}")
		return out


async def _amain() -> int:
	parser = argparse.ArgumentParser(description="Run the SpeakerAgent (Agent Framework + MCP)")
	parser.add_argument(
		"--config",
		default=os.path.join(os.path.dirname(__file__), "config.yaml"),
		help="Path to agent/speaker/config.yaml",
	)
	parser.add_argument("--text", default=None, help="Text to speak")
	parser.add_argument(
		"--direct",
		action="store_true",
		help="Call the speak MCP tool directly (no OpenAI/LLM required)",
	)
	parser.add_argument(
		"--mcp-smoke-test",
		action="store_true",
		help="Validate the MCP endpoint by calling its health tool",
	)
	parser.add_argument(
		"--skip-health-check",
		action="store_true",
		help="Skip checking that services/speak MCP is running",
	)
	args = parser.parse_args()

	# Ensure repo root is importable.
	here = os.path.dirname(os.path.abspath(__file__))
	repo_root = os.path.abspath(os.path.join(here, "..", ".."))
	if repo_root not in sys.path:
		sys.path.insert(0, repo_root)

	from agent.speaker.src.speaker_agent import SpeakerAgent

	agent = SpeakerAgent.from_config_yaml(args.config)

	if args.mcp_smoke_test:
		print(await _mcp_health_check(agent.settings.speak_mcp_url))
		return 0

	if not args.skip_health_check:
		print(await _mcp_health_check(agent.settings.speak_mcp_url))

	text = (args.text or agent.settings.default_text or "").strip()
	if not text:
		raise SystemExit("Missing --text (or set agent.default_text in config)")

	if args.direct:
		print(await _direct_speak(agent.settings.speak_mcp_url, text=text))
		return 0

	async with agent:
		out = await agent.speak(text)
		print(out)
		return 0


def main() -> int:
	try:
		return asyncio.run(_amain())
	except KeyboardInterrupt:
		return 130


if __name__ == "__main__":
	raise SystemExit(main())
