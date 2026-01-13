from __future__ import annotations

import argparse
import asyncio
import os
import sys


async def _amain() -> int:
	parser = argparse.ArgumentParser(description="Run the AdvisorAgent (long-running orchestrator)")
	parser.add_argument(
		"--config",
		default=os.path.join(os.path.dirname(__file__), "config.yaml"),
		help="Path to agent/advisor/config.yaml",
	)
	parser.add_argument(
		"--max-iterations",
		type=int,
		default=None,
		help="Stop after N polling iterations (testing)",
	)
	parser.add_argument(
		"--dry-run",
		action="store_true",
		help="Do not call MCP services or OpenAI; prints actions instead (testing)",
	)
	args = parser.parse_args()

	# Ensure repo root is importable.
	here = os.path.dirname(os.path.abspath(__file__))
	repo_root = os.path.abspath(os.path.join(here, "..", ".."))
	if repo_root not in sys.path:
		sys.path.insert(0, repo_root)

	from agent.advisor.src.advisor_agent import AdvisorAgent
	settings = AdvisorAgent.settings_from_config_yaml(args.config)
	agent = AdvisorAgent(settings, dry_run=bool(args.dry_run))
	await agent.run_forever(max_iterations=args.max_iterations)
	return 0


def main() -> int:
	try:
		return asyncio.run(_amain())
	except KeyboardInterrupt:
		return 130


if __name__ == "__main__":
	raise SystemExit(main())
