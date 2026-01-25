"""Tests for the LLM-based TodoAgent."""

import asyncio
import pytest
from agent.todo.src.llm_todo_agent import LLMTodoAgent, LLMTodoAgentSettings


class TestLLMTodoAgent:
    """Tests for LLMTodoAgent."""

    def test_init_disabled(self):
        """Test that disabled agent doesn't crash."""
        settings = LLMTodoAgentSettings(
            enabled=False,
            model="gpt-4o-mini",
            base_url="https://api.openai.com/v1",
            env_file_path="",
        )
        agent = LLMTodoAgent(settings)
        assert not agent.is_enabled()
        assert not agent.has_active_mission()
        assert not agent.has_pending_tasks()

    def test_status_text_empty(self):
        """Test status text when no mission is active."""
        settings = LLMTodoAgentSettings(
            enabled=True,
            model="gpt-4o-mini",
            base_url="https://api.openai.com/v1",
            env_file_path="",
        )
        agent = LLMTodoAgent(settings)
        status = agent.get_status_text()
        assert "Keine aktive Aufgabenliste" in status

    def test_clear(self):
        """Test clearing the todo list."""
        settings = LLMTodoAgentSettings(
            enabled=True,
            model="gpt-4o-mini",
            base_url="https://api.openai.com/v1",
            env_file_path="",
        )
        agent = LLMTodoAgent(settings)
        # Manually set some state
        agent._mission = "Test mission"
        agent._tasks = [{"id": 1, "title": "Test", "status": "pending"}]
        
        agent.clear()
        
        assert agent._mission is None
        assert agent._tasks == []
        assert not agent.has_active_mission()

    def test_parse_tasks_json_valid(self):
        """Test parsing valid JSON tasks."""
        settings = LLMTodoAgentSettings(
            enabled=True,
            model="gpt-4o-mini",
            base_url="https://api.openai.com/v1",
            env_file_path="",
        )
        agent = LLMTodoAgent(settings)
        
        content = """[
            {"id": 1, "title": "Look up", "action": {"type": "head_set_angles", "pan_deg": 0, "tilt_deg": 30}},
            {"id": 2, "title": "Describe", "action": {"type": "observe"}}
        ]"""
        
        tasks = agent._parse_tasks_json(content)
        
        assert len(tasks) == 2
        assert tasks[0]["title"] == "Look up"
        assert tasks[0]["status"] == "pending"
        assert tasks[1]["title"] == "Describe"

    def test_parse_tasks_json_with_code_fence(self):
        """Test parsing JSON wrapped in markdown code fence."""
        settings = LLMTodoAgentSettings(
            enabled=True,
            model="gpt-4o-mini",
            base_url="https://api.openai.com/v1",
            env_file_path="",
        )
        agent = LLMTodoAgent(settings)
        
        content = """```json
[
    {"id": 1, "title": "Task 1", "action": {}}
]
```"""
        
        tasks = agent._parse_tasks_json(content)
        
        assert len(tasks) == 1
        assert tasks[0]["title"] == "Task 1"

    def test_parse_tasks_json_invalid(self):
        """Test parsing invalid JSON returns empty list."""
        settings = LLMTodoAgentSettings(
            enabled=True,
            model="gpt-4o-mini",
            base_url="https://api.openai.com/v1",
            env_file_path="",
        )
        agent = LLMTodoAgent(settings)
        
        tasks = agent._parse_tasks_json("not valid json")
        
        assert tasks == []

    def test_get_next_task(self):
        """Test getting next pending task."""
        settings = LLMTodoAgentSettings(
            enabled=True,
            model="gpt-4o-mini",
            base_url="https://api.openai.com/v1",
            env_file_path="",
        )
        agent = LLMTodoAgent(settings)
        agent._tasks = [
            {"id": 1, "title": "Done task", "status": "done"},
            {"id": 2, "title": "Pending task", "status": "pending"},
        ]
        
        next_task = asyncio.run(agent.get_next_task())
        
        assert next_task is not None
        assert next_task["id"] == 2
        assert next_task["title"] == "Pending task"

    def test_get_next_task_none_pending(self):
        """Test getting next task when none pending."""
        settings = LLMTodoAgentSettings(
            enabled=True,
            model="gpt-4o-mini",
            base_url="https://api.openai.com/v1",
            env_file_path="",
        )
        agent = LLMTodoAgent(settings)
        agent._tasks = [
            {"id": 1, "title": "Done task", "status": "done"},
        ]
        
        next_task = asyncio.run(agent.get_next_task())
        
        assert next_task is None

    def test_mark_task_done(self):
        """Test marking a task as done."""
        settings = LLMTodoAgentSettings(
            enabled=True,
            model="gpt-4o-mini",
            base_url="https://api.openai.com/v1",
            env_file_path="",
        )
        agent = LLMTodoAgent(settings)
        agent._tasks = [
            {"id": 1, "title": "Task 1", "status": "pending"},
        ]
        
        result = asyncio.run(agent.mark_task_done(1, "Completed successfully"))
        
        assert result is True
        assert agent._tasks[0]["status"] == "done"
        assert agent._tasks[0]["result"] == "Completed successfully"

    def test_mark_task_done_not_found(self):
        """Test marking non-existent task."""
        settings = LLMTodoAgentSettings(
            enabled=True,
            model="gpt-4o-mini",
            base_url="https://api.openai.com/v1",
            env_file_path="",
        )
        agent = LLMTodoAgent(settings)
        agent._tasks = []
        
        result = asyncio.run(agent.mark_task_done(999, "Result"))
        
        assert result is False
