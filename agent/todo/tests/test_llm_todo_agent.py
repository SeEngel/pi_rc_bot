"""Tests for the LLM-based TodoAgent."""

import asyncio
import unittest

from agent.todo.src.llm_todo_agent import LLMTodoAgent, LLMTodoAgentSettings


def _make_settings(*, enabled: bool) -> LLMTodoAgentSettings:
    # These tests validate local behavior only; they must not require network/API keys.
    return LLMTodoAgentSettings(
        enabled=enabled,
        model="gpt-4o-mini",
        base_url="https://api.openai.com/v1",
        env_file_path="",
        language="de",
    )


class TestLLMTodoAgent(unittest.TestCase):
    """Unit tests for LLMTodoAgent (no network calls)."""

    def test_init_disabled(self):
        """Test that disabled agent doesn't crash."""
        agent = LLMTodoAgent(_make_settings(enabled=False))
        self.assertFalse(agent.is_enabled())
        self.assertFalse(agent.has_active_mission())
        self.assertFalse(agent.has_pending_tasks())

    def test_status_text_empty(self):
        """Test status text when no mission is active."""
        agent = LLMTodoAgent(_make_settings(enabled=True))
        status = agent.get_status_text()
        self.assertIn("Keine aktive Aufgabenliste", status)

    def test_clear(self):
        """Test clearing the todo list."""
        agent = LLMTodoAgent(_make_settings(enabled=True))
        # Manually set some state
        agent._mission = "Test mission"
        agent._tasks = [{"id": 1, "title": "Test", "status": "pending"}]
        
        agent.clear()
        
        self.assertIsNone(agent._mission)
        self.assertEqual(agent._tasks, [])
        self.assertFalse(agent.has_active_mission())

    def test_parse_tasks_json_valid(self):
        """Test parsing valid JSON tasks."""
        agent = LLMTodoAgent(_make_settings(enabled=True))
        
        content = """[
            {"id": 1, "title": "Look up", "action": {"type": "head_set_angles", "pan_deg": 0, "tilt_deg": 30}},
            {"id": 2, "title": "Describe", "action": {"type": "observe"}}
        ]"""
        
        tasks = agent._parse_tasks_json(content)

        self.assertEqual(len(tasks), 2)
        self.assertEqual(tasks[0]["title"], "Look up")
        self.assertEqual(tasks[0]["status"], "pending")
        self.assertEqual(tasks[1]["title"], "Describe")

    def test_parse_tasks_json_with_code_fence(self):
        """Test parsing JSON wrapped in markdown code fence."""
        agent = LLMTodoAgent(_make_settings(enabled=True))
        
        content = """```json
[
    {"id": 1, "title": "Task 1", "action": {}}
]
```"""
        
        tasks = agent._parse_tasks_json(content)

        self.assertEqual(len(tasks), 1)
        self.assertEqual(tasks[0]["title"], "Task 1")

    def test_parse_tasks_json_invalid(self):
        """Test parsing invalid JSON returns empty list."""
        agent = LLMTodoAgent(_make_settings(enabled=True))
        
        tasks = agent._parse_tasks_json("not valid json")
        self.assertEqual(tasks, [])

    def test_get_next_task(self):
        """Test getting next pending task."""
        agent = LLMTodoAgent(_make_settings(enabled=True))
        agent._tasks = [
            {"id": 1, "title": "Done task", "status": "done"},
            {"id": 2, "title": "Pending task", "status": "pending"},
        ]
        
        next_task = asyncio.run(agent.get_next_task())

        self.assertIsNotNone(next_task)
        assert next_task is not None
        self.assertEqual(next_task["id"], 2)
        self.assertEqual(next_task["title"], "Pending task")

    def test_get_next_task_none_pending(self):
        """Test getting next task when none pending."""
        agent = LLMTodoAgent(_make_settings(enabled=True))
        agent._tasks = [
            {"id": 1, "title": "Done task", "status": "done"},
        ]
        
        next_task = asyncio.run(agent.get_next_task())

        self.assertIsNone(next_task)

    def test_mark_task_done(self):
        """Test marking a task as done."""
        agent = LLMTodoAgent(_make_settings(enabled=True))
        agent._tasks = [
            {"id": 1, "title": "Task 1", "status": "pending"},
        ]
        
        result = asyncio.run(agent.mark_task_done(1, "Completed successfully"))

        self.assertIs(result, True)
        self.assertEqual(agent._tasks[0]["status"], "done")
        self.assertEqual(agent._tasks[0]["result"], "Completed successfully")

    def test_mark_task_done_not_found(self):
        """Test marking non-existent task."""
        agent = LLMTodoAgent(_make_settings(enabled=True))
        agent._tasks = []
        
        result = asyncio.run(agent.mark_task_done(999, "Result"))

        self.assertIs(result, False)
