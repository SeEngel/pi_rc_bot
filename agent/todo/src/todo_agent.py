from __future__ import annotations

import json
import os
import re
import time
from dataclasses import dataclass
from typing import Any

from agent.src.config import load_yaml, resolve_repo_root


@dataclass(frozen=True)
class TodoAgentSettings:
	enabled: bool
	state_path: str  # absolute
	autosave: bool
	max_status_tasks: int = 6


def _now_ts() -> float:
	return time.time()


def _clamp_int(v: Any, default: int, lo: int, hi: int) -> int:
	try:
		i = int(v)
	except Exception:
		return default
	return max(lo, min(hi, i))


class TodoAgent:
	"""Local-only todo manager.

	- No MCP
	- No LLM
	- Optional persistence to a JSON file

	State model:
	{
	  "mission": "...",
	  "tasks": [{"id": 1, "title": "...", "status": "open|done|blocked|skipped", ...}],
	  "version": 1,
	  "updated_ts": 123.45
	}
	"""

	def __init__(self, settings: TodoAgentSettings):
		self.settings = settings
		self._state: dict[str, Any] = {
			"mission": None,
			"tasks": [],
			"version": 0,
			"updated_ts": None,
		}

	@classmethod
	def from_config_yaml(cls, path: str) -> "TodoAgent":
		cfg = load_yaml(path)
		here = os.path.dirname(os.path.abspath(path))
		repo_root = resolve_repo_root(here)
		todo_cfg = cfg.get("todo", {}) if isinstance(cfg, dict) else {}

		enabled = bool((todo_cfg or {}).get("enabled") if "enabled" in (todo_cfg or {}) else True)
		state_path_raw = str((todo_cfg or {}).get("state_path") or "memory/todo/state.json").strip()
		if not os.path.isabs(state_path_raw):
			state_path_raw = os.path.join(repo_root, state_path_raw)
		state_path = os.path.abspath(state_path_raw)

		autosave = bool((todo_cfg or {}).get("autosave") if "autosave" in (todo_cfg or {}) else True)
		max_status_tasks = _clamp_int((todo_cfg or {}).get("max_status_tasks"), 6, 1, 50)

		return cls(TodoAgentSettings(enabled=enabled, state_path=state_path, autosave=autosave, max_status_tasks=max_status_tasks))

	# --- persistence ---

	def load(self) -> None:
		"""Load from disk if present."""
		if not self.settings.enabled:
			return
		path = self.settings.state_path
		if not os.path.exists(path):
			return
		try:
			with open(path, "r", encoding="utf-8") as f:
				obj = json.load(f)
		except Exception:
			return
		if not isinstance(obj, dict):
			return
		# minimal validation
		mission = obj.get("mission")
		tasks = obj.get("tasks")
		if tasks is None:
			tasks = []
		if not isinstance(tasks, list):
			tasks = []
		clean_tasks: list[dict[str, Any]] = []
		for t in tasks:
			if isinstance(t, dict):
				clean_tasks.append(dict(t))
		self._state = {
			"mission": (str(mission) if mission is not None and str(mission).strip() else None),
			"tasks": clean_tasks,
			"version": int(obj.get("version") or 0),
			"updated_ts": obj.get("updated_ts"),
		}
		self._reindex_ids_inplace()

	def save(self) -> None:
		if not self.settings.enabled:
			return
		path = self.settings.state_path
		try:
			os.makedirs(os.path.dirname(path), exist_ok=True)
			with open(path, "w", encoding="utf-8") as f:
				json.dump(self._state, f, ensure_ascii=False, indent=2)
		except Exception:
			return

	def _touch(self) -> None:
		self._state["version"] = int(self._state.get("version") or 0) + 1
		self._state["updated_ts"] = _now_ts()
		if self.settings.autosave:
			self.save()

	# --- core operations ---

	def clear(self) -> None:
		self._state["mission"] = None
		self._state["tasks"] = []
		self._touch()

	def set_mission(self, mission: str, *, tasks: list[str] | None = None) -> None:
		m = (mission or "").strip()
		self._state["mission"] = m or None
		if tasks is not None:
			new_tasks: list[dict[str, Any]] = []
			for title in tasks:
				t = (title or "").strip()
				if not t:
					continue
				new_tasks.append({"id": len(new_tasks) + 1, "title": t, "status": "open", "notes": None})
			self._state["tasks"] = new_tasks
			self._touch()
		else:
			# keep tasks unchanged, but still bump version
			self._touch()

	def add_task(self, title: str, *, notes: str | None = None) -> dict[str, Any] | None:
		t = (title or "").strip()
		if not t:
			return None
		tasks = self._tasks()
		next_id = 1
		if tasks:
			try:
				next_id = max(int(x.get("id") or 0) for x in tasks) + 1
			except Exception:
				next_id = len(tasks) + 1
		obj = {"id": int(next_id), "title": t, "status": "open", "notes": (notes.strip() if isinstance(notes, str) and notes.strip() else None)}
		tasks.append(obj)
		self._touch()
		return obj

	def complete_task(self, task_id: int, *, note: str | None = None) -> dict[str, Any] | None:
		tasks = self._tasks()
		for t in tasks:
			if int(t.get("id") or 0) == int(task_id):
				t["status"] = "done"
				if note and str(note).strip():
					t["notes"] = str(note).strip()
				self._touch()
				return t
		return None

	def complete_current_or_next(self, *, note: str | None) -> dict[str, Any] | None:
		nxt = self.next_task()
		if nxt is None:
			return None
		return self.complete_task(int(nxt["id"]), note=note)

	def next_task(self) -> dict[str, Any] | None:
		for t in self._tasks():
			if str(t.get("status") or "open") == "open":
				return t
		return None

	def open_tasks(self) -> list[dict[str, Any]]:
		return [t for t in self._tasks() if str(t.get("status") or "open") == "open"]

	def has_open_tasks(self) -> bool:
		return self.next_task() is not None

	def mission(self) -> str | None:
		m = self._state.get("mission")
		return str(m).strip() if m is not None and str(m).strip() else None

	def status_text(self) -> str:
		"""Human-readable status (for prompt injection or debugging)."""
		m = self.mission() or "(no mission)"
		open_tasks = self.open_tasks()
		done = len([t for t in self._tasks() if str(t.get("status")) == "done"])
		total = len(self._tasks())
		lines = [f"mission: {m}", f"tasks: {done}/{total} done"]
		if not open_tasks:
			lines.append("open: (none)")
			return "\n".join(lines)

		lines.append("open:")
		max_n = max(1, int(self.settings.max_status_tasks or 6))
		for t in open_tasks[:max_n]:
			lines.append(f"- [{t.get('id')}] {t.get('title')}")
		if len(open_tasks) > max_n:
			lines.append(f"- ... (+{len(open_tasks) - max_n} more)")
		return "\n".join(lines)

	# --- parsing / replanning helpers ---

	def set_from_freeform_text(self, text: str) -> None:
		"""Parse a mission + tasks from freeform text.

		Supported:
		- numbered lists: "1) ...", "1. ..."
		- bullets: "- ...", "* ..."
		If no list is found, creates a single task equal to the mission.
		"""
		raw = (text or "").strip()
		if not raw:
			return

		lines = [ln.strip() for ln in raw.splitlines() if ln.strip()]
		tasks: list[str] = []
		for ln in lines:
			m = re.match(r"^(?:\d+\s*[\.)]|[-*])\s+(.*)$", ln)
			if m:
				item = (m.group(1) or "").strip()
				if item:
					tasks.append(item)

		if tasks:
			# Mission: first non-list line if available, else whole text.
			# Heuristic: mission is the first line that is NOT a list item.
			mission = None
			for ln in lines:
				if not re.match(r"^(?:\d+\s*[\.)]|[-*])\s+", ln):
					mission = ln
					break
			if mission is None:
				mission = raw
			self.set_mission(mission, tasks=tasks)
			return

		# No tasks found -> treat whole text as mission + one task
		self.set_mission(raw, tasks=[raw])

	# --- internals ---

	def _tasks(self) -> list[dict[str, Any]]:
		tasks = self._state.get("tasks")
		if not isinstance(tasks, list):
			tasks = []
			self._state["tasks"] = tasks
		return tasks  # type: ignore[return-value]

	def _reindex_ids_inplace(self) -> None:
		tasks = self._tasks()
		# If ids are missing/invalid/duplicate, reassign sequentially.
		seen: set[int] = set()
		need_fix = False
		for t in tasks:
			try:
				tid = int(t.get("id") or 0)
			except Exception:
				tid = 0
			if tid <= 0 or tid in seen:
				need_fix = True
				break
			seen.add(tid)
		if not need_fix:
			return
		for i, t in enumerate(tasks):
			t["id"] = i + 1
