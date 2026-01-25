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
        review_after_planning=True,
        review_after_completion=True,
        autonomous_mode_enabled=True,
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

    def test_get_tasks_summary(self):
        """Test getting a text summary of tasks."""
        agent = LLMTodoAgent(_make_settings(enabled=True))
        agent._tasks = [
            {"id": 1, "title": "Task 1", "status": "done", "result": "Completed OK"},
            {"id": 2, "title": "Task 2", "status": "pending"},
        ]
        
        summary = agent._get_tasks_summary()
        self.assertIn("[1] (done) Task 1", summary)
        self.assertIn("[2] (pending) Task 2", summary)

        summary_with_results = agent._get_tasks_summary(include_results=True)
        self.assertIn("Completed OK", summary_with_results)

    def test_add_tasks(self):
        """Test adding multiple tasks at once."""
        agent = LLMTodoAgent(_make_settings(enabled=True))
        agent._tasks = [
            {"id": 1, "title": "Existing task", "status": "done"},
        ]
        
        new_tasks = [
            {"title": "New task 1", "action": {"type": "observe"}},
            {"title": "New task 2", "action": {"type": "speak", "text": "hello"}},
        ]
        
        asyncio.run(agent.add_tasks(new_tasks))
        
        self.assertEqual(len(agent._tasks), 3)
        self.assertEqual(agent._tasks[1]["id"], 2)
        self.assertEqual(agent._tasks[2]["id"], 3)
        self.assertEqual(agent._tasks[1]["status"], "pending")

    def test_parse_review_response_approved(self):
        """Test parsing an approved review response."""
        agent = LLMTodoAgent(_make_settings(enabled=True))
        
        # JSON format - note: content is lowercased during parsing
        result = agent._parse_review_response('{"approved": true, "reason": "Plan ist gut"}')
        self.assertTrue(result["approved"])
        self.assertEqual(result["reason"], "plan ist gut")
        
        # Keyword-based
        result = agent._parse_review_response("Ja, der Plan passt so.")
        self.assertTrue(result["approved"])

    def test_parse_review_response_not_approved(self):
        """Test parsing a not-approved review response."""
        agent = LLMTodoAgent(_make_settings(enabled=True))
        
        # JSON format - note: content is lowercased during parsing
        result = agent._parse_review_response('{"approved": false, "suggestion": "Füge X hinzu"}')
        self.assertFalse(result["approved"])
        self.assertEqual(result["suggestion"], "füge x hinzu")
        
        # Keyword-based
        result = agent._parse_review_response("Nein, es fehlt noch ein Schritt.")
        self.assertFalse(result["approved"])

    def test_parse_completion_review_complete(self):
        """Test parsing a completion review response."""
        agent = LLMTodoAgent(_make_settings(enabled=True))
        
        result = agent._parse_completion_review_response('{"complete": true, "reason": "Alles erledigt"}')
        self.assertTrue(result["complete"])
        self.assertIsNone(result["additional_tasks"])

    def test_parse_completion_review_with_additional_tasks(self):
        """Test parsing a completion review with additional tasks."""
        agent = LLMTodoAgent(_make_settings(enabled=True))
        
        content = '''{"complete": false, "additional_tasks": [{"id": 3, "title": "Extra task"}], "reason": "Mehr nötig"}'''
        result = agent._parse_completion_review_response(content)
        
        self.assertFalse(result["complete"])
        self.assertEqual(result["reason"], "Mehr nötig")

    def test_clear_resets_all_state(self):
        """Test that clear() resets all internal state."""
        agent = LLMTodoAgent(_make_settings(enabled=True))
        agent._mission = "Test"
        agent._tasks = [{"id": 1, "title": "Task", "status": "done"}]
        agent._review_status = "approved"
        agent._completed_results = ["Result 1"]
        
        agent.clear()
        
        self.assertIsNone(agent._mission)
        self.assertEqual(agent._tasks, [])
        self.assertEqual(agent._review_status, "none")
        self.assertEqual(agent._completed_results, [])
