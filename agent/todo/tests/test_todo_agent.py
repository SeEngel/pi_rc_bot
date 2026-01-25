import json
import os
import tempfile
import unittest


from agent.todo.src.todo_agent import TodoAgent, TodoAgentSettings


class TestTodoAgent(unittest.TestCase):
	def test_add_complete_next(self):
		with tempfile.TemporaryDirectory() as d:
			state_path = os.path.join(d, "state.json")
			agent = TodoAgent(TodoAgentSettings(enabled=True, state_path=state_path, autosave=True))
			agent.set_mission("mission", tasks=["a", "b"])
			self.assertEqual(agent.next_task()["title"], "a")  # type: ignore[index]
			agent.complete_current_or_next(note=None)
			self.assertEqual(agent.next_task()["title"], "b")  # type: ignore[index]
			agent.complete_current_or_next(note=None)
			self.assertIsNone(agent.next_task())

			# persisted
			self.assertTrue(os.path.isfile(state_path))
			with open(state_path, "r", encoding="utf-8") as f:
				obj = json.load(f)
			self.assertEqual(len(obj.get("tasks") or []), 2)

	def test_set_from_freeform_text_parses_list(self):
		agent = TodoAgent(TodoAgentSettings(enabled=True, state_path="/dev/null", autosave=False))
		agent.set_from_freeform_text(
			"Treasure hunt\n1) find red cube\n2) find blue ball\n- ignore this?"
		)
		self.assertEqual(agent.mission(), "Treasure hunt")
		open_titles = [t.get("title") for t in agent.open_tasks()]
		self.assertIn("find red cube", open_titles)
		self.assertIn("find blue ball", open_titles)

	def test_load_is_resilient(self):
		with tempfile.TemporaryDirectory() as d:
			state_path = os.path.join(d, "state.json")
			with open(state_path, "w", encoding="utf-8") as f:
				f.write("not json")
			agent = TodoAgent(TodoAgentSettings(enabled=True, state_path=state_path, autosave=False))
			# should not raise
			agent.load()
			self.assertIsNone(agent.mission())

	def test_remove_task(self):
		"""Test removing a task by ID."""
		agent = TodoAgent(TodoAgentSettings(enabled=True, state_path="/dev/null", autosave=False))
		agent.set_mission("Test", tasks=["a", "b", "c"])
		
		# Remove the second task
		result = agent.remove_task(2)
		self.assertTrue(result)
		open_titles = [t.get("title") for t in agent.open_tasks()]
		self.assertEqual(open_titles, ["a", "c"])
		
		# Try to remove non-existent task
		result = agent.remove_task(999)
		self.assertFalse(result)

	def test_modify_task(self):
		"""Test modifying an existing task."""
		agent = TodoAgent(TodoAgentSettings(enabled=True, state_path="/dev/null", autosave=False))
		agent.set_mission("Test", tasks=["original title"])
		
		# Modify the task
		modified = agent.modify_task(1, title="new title", notes="some notes")
		self.assertIsNotNone(modified)
		self.assertEqual(modified["title"], "new title")
		self.assertEqual(modified["notes"], "some notes")
		
		# Modify status
		modified = agent.modify_task(1, status="blocked")
		self.assertEqual(modified["status"], "blocked")
		
		# Modify non-existent task
		result = agent.modify_task(999, title="foo")
		self.assertIsNone(result)

	def test_insert_task_after(self):
		"""Test inserting a task after a specific task."""
		agent = TodoAgent(TodoAgentSettings(enabled=True, state_path="/dev/null", autosave=False))
		agent.set_mission("Test", tasks=["first", "second", "third"])
		
		# Insert after the first task
		new_task = agent.insert_task_after(1, "inserted")
		self.assertIsNotNone(new_task)
		self.assertEqual(new_task["title"], "inserted")
		
		# Check order
		open_titles = [t.get("title") for t in agent.open_tasks()]
		self.assertEqual(open_titles, ["first", "inserted", "second", "third"])
		
		# Insert after non-existent task
		result = agent.insert_task_after(999, "should fail")
		self.assertIsNone(result)

	def test_extend_tasks(self):
		"""Test adding multiple tasks at once."""
		agent = TodoAgent(TodoAgentSettings(enabled=True, state_path="/dev/null", autosave=False))
		agent.set_mission("Test", tasks=["existing"])
		
		added = agent.extend_tasks(["new1", "new2", "new3"])
		self.assertEqual(len(added), 3)
		
		open_titles = [t.get("title") for t in agent.open_tasks()]
		self.assertEqual(open_titles, ["existing", "new1", "new2", "new3"])

	def test_reorder_tasks(self):
		"""Test reordering tasks."""
		agent = TodoAgent(TodoAgentSettings(enabled=True, state_path="/dev/null", autosave=False))
		agent.set_mission("Test", tasks=["a", "b", "c"])
		
		# Reverse the order
		agent.reorder_tasks([3, 2, 1])
		open_titles = [t.get("title") for t in agent.open_tasks()]
		self.assertEqual(open_titles, ["c", "b", "a"])


if __name__ == "__main__":
	unittest.main()