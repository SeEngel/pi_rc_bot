from __future__ import annotations

import argparse
import os
import sys


def main(argv: list[str] | None = None) -> int:
	ap = argparse.ArgumentParser(description="Local TodoAgent (no MCP/LLM)")
	ap.add_argument(
		"--config",
		default=os.path.join(os.path.dirname(__file__), "config.yaml"),
		help="Path to agent/todo/config.yaml",
	)
	ap.add_argument("--status", action="store_true", help="Print todo status")
	ap.add_argument("--next", dest="show_next", action="store_true", help="Print next task")
	ap.add_argument("--done", action="store_true", help="Mark current/next task as done")
	ap.add_argument("--add", default=None, help="Add a task title")
	ap.add_argument("--clear", action="store_true", help="Clear mission and tasks")
	ap.add_argument("--set-mission", default=None, help="Set mission text")
	ap.add_argument("--tasks", nargs="*", default=None, help="Task titles (when using --set-mission)")
	ap.add_argument("--from-text", default=None, help="Parse mission+tasks from freeform text")
	args = ap.parse_args(argv)

	here = os.path.dirname(os.path.abspath(__file__))
	repo_root = os.path.abspath(os.path.join(here, "..", ".."))
	if repo_root not in sys.path:
		sys.path.insert(0, repo_root)

	from agent.todo.src.todo_agent import TodoAgent

	agent = TodoAgent.from_config_yaml(args.config)
	agent.load()

	if args.clear:
		agent.clear()

	if args.from_text:
		agent.set_from_freeform_text(args.from_text)

	if args.set_mission is not None:
		tasks = args.tasks if isinstance(args.tasks, list) else None
		agent.set_mission(args.set_mission, tasks=tasks)

	if args.add:
		agent.add_task(args.add)

	if args.done:
		t = agent.complete_current_or_next(note=None)
		if t is None:
			print("(no open tasks)")
		else:
			print(f"done: {t['id']}: {t['title']}")

	if args.show_next:
		t = agent.next_task()
		if t is None:
			print("(no open tasks)")
		else:
			print(f"next: {t['id']}: {t['title']}")

	if args.status or (not (args.show_next or args.done or args.add or args.clear or args.set_mission or args.from_text)):
		print(agent.status_text())

	return 0


if __name__ == "__main__":
	raise SystemExit(main())
