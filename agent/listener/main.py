from __future__ import annotations

import argparse
import asyncio
import os
import sys


async def _mcp_health_check(mcp_url: str, *, timeout_seconds: float = 20.0) -> str:
	"""Validate the listening MCP server by invoking its health tool."""
	from agent_framework import MCPStreamableHTTPTool  # type: ignore

	async with MCPStreamableHTTPTool(name="listening", url=mcp_url, timeout=timeout_seconds) as mcp:
		# FastAPI's default operation id for GET /healthz becomes `healthz_healthz_get`.
		res = await mcp.call_tool("healthz_healthz_get")
		text = None
		for item in res:
			if getattr(item, "type", None) == "text" and hasattr(item, "text"):
				text = str(getattr(item, "text"))
				break
		if not text:
			raise RuntimeError(f"MCP health check failed: unexpected tool result: {res!r}")
		return "\n".join(["MCP health check OK.", f"healthz => {text}"])


async def _direct_listen(
	mcp_url: str,
	*,
	stream: bool = False,
	speech_pause_seconds: float | None = None,
	timeout_seconds: float = 120.0,
) -> str:
	"""Call the MCP `listen` tool directly (no LLM required)."""
	from agent_framework import MCPStreamableHTTPTool  # type: ignore

	payload: dict[str, object] = {}
	if stream:
		payload["stream"] = True
	if speech_pause_seconds is not None:
		payload["speech_pause_seconds"] = float(speech_pause_seconds)

	async with MCPStreamableHTTPTool(name="listening", url=mcp_url, timeout=timeout_seconds) as mcp:
		res = await mcp.call_tool("listen", **payload)
		text = None
		for item in res:
			if getattr(item, "type", None) == "text" and hasattr(item, "text"):
				text = str(getattr(item, "text"))
				break
		if not text:
			raise RuntimeError(f"Direct listen failed: unexpected tool result: {res!r}")
		return text


async def _amain() -> int:
	parser = argparse.ArgumentParser(description="Run the ListenerAgent (Agent Framework + MCP)")
	parser.add_argument(
		"--config",
		default=os.path.join(os.path.dirname(__file__), "config.yaml"),
		help="Path to agent/listener/config.yaml",
	)
	parser.add_argument("--prompt", default=None, help="Optional instruction for what to listen for")
	parser.add_argument(
		"--direct",
		action="store_true",
		help="Call the listening MCP tool directly (no OpenAI/LLM required)",
	)
	parser.add_argument(
		"--stream",
		action="store_true",
		help="(direct) Request stream mode (Vosk only; if supported by your stack)",
	)
	parser.add_argument(
		"--speech-pause-seconds",
		type=float,
		default=None,
		help="(direct) OpenAI-only: stop after this much continuous silence after speech starts",
	)
	parser.add_argument(
		"--mcp-smoke-test",
		action="store_true",
		help="Validate the MCP endpoint by calling its health tool",
	)
	parser.add_argument(
		"--skip-health-check",
		action="store_true",
		help="Skip checking that services/listening MCP is running",
	)
	args = parser.parse_args()

	# Ensure repo root is importable.
	here = os.path.dirname(os.path.abspath(__file__))
	repo_root = os.path.abspath(os.path.join(here, "..", ".."))
	if repo_root not in sys.path:
		sys.path.insert(0, repo_root)

	from agent.listener.src.listener_agent import ListenerAgent

	agent = ListenerAgent.from_config_yaml(args.config)

	if args.mcp_smoke_test:
		print(await _mcp_health_check(agent.settings.listen_mcp_url))
		return 0

	if not args.skip_health_check:
		print(await _mcp_health_check(agent.settings.listen_mcp_url))

	if args.direct:
		out = await _direct_listen(
			agent.settings.listen_mcp_url,
			stream=bool(args.stream),
			speech_pause_seconds=args.speech_pause_seconds,
		)
		print(out)
		return 0

	async with agent:
		out = await agent.listen(args.prompt)
		print(out)
		return 0


def main() -> int:
	try:
		return asyncio.run(_amain())
	except KeyboardInterrupt:
		return 130


if __name__ == "__main__":
	raise SystemExit(main())
