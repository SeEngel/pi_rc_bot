from __future__ import annotations

import os
import sys


def main() -> int:
	"""Offline smoke test (no OpenAI, no MCP).

	- Confirms repo import paths work
	- Confirms YAML config loads and settings object is created
	"""
	here = os.path.dirname(os.path.abspath(__file__))
	repo_root = os.path.abspath(os.path.join(here, "..", ".."))
	if repo_root not in sys.path:
		sys.path.insert(0, repo_root)

	from agent.memorizer.src.memorizer_agent import MemorizerAgent

	agent = MemorizerAgent.from_config_yaml(os.path.join(here, "config.yaml"))
	print("ok: true")
	print(f"name: {agent.settings.name}")
	print(f"openai_model: {agent.settings.openai_model}")
	print(f"memory_mcp_url: {agent.settings.memory_mcp_url}")
	print(f"default_top_n: {agent.settings.default_top_n}")
	return 0


if __name__ == "__main__":
	raise SystemExit(main())
