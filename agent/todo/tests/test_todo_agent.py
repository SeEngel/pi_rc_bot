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


if __name__ == "__main__":
	unittest.main()