import unittest

from agent.advisor.src.advisor_agent import (
	AdvisorMemoryConfig,
	AdvisorMemorizerConfig,
	AdvisorSettings,
	AdvisorTodoConfig,
)

from agent.advisor_split_brain_move.src.advisor_agent import (
	SplitBrainAdvisorAgent,
	SplitBrainAdvisorSettings,
)


class TestSplitBrainSettings(unittest.TestCase):
	def test_settings_from_config_yaml_has_move_advisor_url(self):
		settings = SplitBrainAdvisorAgent.settings_from_config_yaml(
			"agent/advisor_split_brain_move/config.yaml"
		)
		self.assertTrue(bool(getattr(settings, "move_advisor_mcp_url", "")))
		self.assertIn("/mcp", settings.move_advisor_mcp_url)


class TestMoveAdvisorDispatch(unittest.IsolatedAsyncioTestCase):
	async def test_proximity_distance_delegates_to_move_advisor(self):
		base = AdvisorSettings(
			name="advisor",
			persona_instructions="Be helpful.",
			debug=False,
			debug_log_path=None,
			response_language="de",
			openai_model="dummy",
			openai_base_url=None,
			env_file_path=None,
			listen_mcp_url="http://127.0.0.1:8602/mcp",
			speak_mcp_url="http://127.0.0.1:8601/mcp",
			observe_mcp_url="http://127.0.0.1:8603/mcp",
			move_mcp_url="http://127.0.0.1:8605/mcp",
			head_mcp_url="http://127.0.0.1:8606/mcp",
			proximity_mcp_url="http://127.0.0.1:8607/mcp",
			perception_mcp_url="http://127.0.0.1:8608/mcp",
			safety_mcp_url="http://127.0.0.1:8609/mcp",
			min_transcript_chars=1,
			max_listen_attempts=1,
			reprompt_text="Bitte wiederholen.",
			listen_stream=False,
			interrupt_speech_on_sound=False,
			wait_for_speech_finish=False,
			speech_status_poll_interval_seconds=0.1,
			speech_max_wait_seconds=1.0,
			suppress_alone_mode_while_speaking=False,
			stop_speech_on_sound_while_waiting=False,
			post_speech_interaction_grace_seconds=0.0,
			brief_listen_timeout_seconds=3.0,
			stop_words=["stop"],
			think_interval_seconds=999.0,
			observation_question="Describe.",
			max_thought_chars=200,
			sound_enabled=False,
			sound_threshold_rms=0,
			sound_active_windows_required=1,
			sound_sample_rate_hz=16000,
			sound_window_seconds=0.1,
			sound_poll_interval_seconds=0.1,
			sound_arecord_device=None,
			sound_fallback_to_interaction_on_error=False,
			memory=AdvisorMemoryConfig(
				max_tokens=100,
				avg_chars_per_token=4,
				summary_dir=".tmp_test_summaries",
				summary_max_chars=200,
			),
			memorizer=AdvisorMemorizerConfig(
				enabled=False,
				config_path="/dev/null",
				ingest_user_utterances=False,
				recall_for_questions=False,
				recall_top_n=1,
				ingest_timeout_seconds=0.1,
				recall_timeout_seconds=0.1,
			),
			todo=AdvisorTodoConfig(
				enabled=False,
				config_path="/dev/null",
				include_in_prompt=False,
				mention_next_in_response=False,
			),
		)
		settings = SplitBrainAdvisorSettings(
			base=base,
			move_advisor_mcp_url="http://127.0.0.1:8611/mcp",
			move_advisor_preflight=False,
		)
		agent = SplitBrainAdvisorAgent(settings, dry_run=False)

		calls = []

		async def _fake_call_mcp_tool_json(*, url: str, tool_name: str, timeout_seconds: float = 60.0, **kwargs):
			calls.append({"url": url, "tool_name": tool_name, "timeout_seconds": timeout_seconds, "kwargs": kwargs})
			# move_advisor wraps downstream result in {ok, result}
			return {"ok": True, "result": {"ok": True, "distance_cm": 42.0}}

		import agent.advisor_split_brain_move.src.advisor_agent as mod

		orig = mod.call_mcp_tool_json
		mod.call_mcp_tool_json = _fake_call_mcp_tool_json  # type: ignore[assignment]
		try:
			d = await agent._proximity_distance_cm()
			self.assertEqual(d, 42.0)
			self.assertEqual(len(calls), 1)
			self.assertEqual(calls[0]["url"], "http://127.0.0.1:8611/mcp")
			self.assertEqual(calls[0]["tool_name"], "execute_action")
			action = calls[0]["kwargs"].get("action")
			self.assertIsInstance(action, dict)
			self.assertEqual(action.get("type"), "proximity_distance")
		finally:
			mod.call_mcp_tool_json = orig  # type: ignore[assignment]
			await agent.close()

	async def test_guarded_drive_delegates_to_move_advisor(self):
		base = AdvisorSettings(
			name="advisor",
			persona_instructions="Be helpful.",
			debug=False,
			debug_log_path=None,
			response_language="de",
			openai_model="dummy",
			openai_base_url=None,
			env_file_path=None,
			listen_mcp_url="http://127.0.0.1:8602/mcp",
			speak_mcp_url="http://127.0.0.1:8601/mcp",
			observe_mcp_url="http://127.0.0.1:8603/mcp",
			move_mcp_url="http://127.0.0.1:8605/mcp",
			head_mcp_url="http://127.0.0.1:8606/mcp",
			proximity_mcp_url="http://127.0.0.1:8607/mcp",
			perception_mcp_url="http://127.0.0.1:8608/mcp",
			safety_mcp_url="http://127.0.0.1:8609/mcp",
			min_transcript_chars=1,
			max_listen_attempts=1,
			reprompt_text="Bitte wiederholen.",
			listen_stream=False,
			interrupt_speech_on_sound=False,
			wait_for_speech_finish=False,
			speech_status_poll_interval_seconds=0.1,
			speech_max_wait_seconds=1.0,
			suppress_alone_mode_while_speaking=False,
			stop_speech_on_sound_while_waiting=False,
			post_speech_interaction_grace_seconds=0.0,
			brief_listen_timeout_seconds=3.0,
			stop_words=["stop"],
			think_interval_seconds=999.0,
			observation_question="Describe.",
			max_thought_chars=200,
			sound_enabled=False,
			sound_threshold_rms=0,
			sound_active_windows_required=1,
			sound_sample_rate_hz=16000,
			sound_window_seconds=0.1,
			sound_poll_interval_seconds=0.1,
			sound_arecord_device=None,
			sound_fallback_to_interaction_on_error=False,
			memory=AdvisorMemoryConfig(
				max_tokens=100,
				avg_chars_per_token=4,
				summary_dir=".tmp_test_summaries",
				summary_max_chars=200,
			),
			memorizer=AdvisorMemorizerConfig(
				enabled=False,
				config_path="/dev/null",
				ingest_user_utterances=False,
				recall_for_questions=False,
				recall_top_n=1,
				ingest_timeout_seconds=0.1,
				recall_timeout_seconds=0.1,
			),
			todo=AdvisorTodoConfig(
				enabled=False,
				config_path="/dev/null",
				include_in_prompt=False,
				mention_next_in_response=False,
			),
		)
		settings = SplitBrainAdvisorSettings(
			base=base,
			move_advisor_mcp_url="http://127.0.0.1:8611/mcp",
			move_advisor_preflight=False,
		)
		agent = SplitBrainAdvisorAgent(settings, dry_run=False)

		calls = []

		async def _fake_call_mcp_tool_json(*, url: str, tool_name: str, timeout_seconds: float = 60.0, **kwargs):
			calls.append({"url": url, "tool_name": tool_name, "timeout_seconds": timeout_seconds, "kwargs": kwargs})
			return {"ok": True, "result": {"ok": True, "blocked": False}}

		import agent.advisor_split_brain_move.src.advisor_agent as mod

		orig = mod.call_mcp_tool_json
		mod.call_mcp_tool_json = _fake_call_mcp_tool_json  # type: ignore[assignment]
		try:
			res = await agent._safety_guarded_drive(speed=30, steer_deg=10, duration_s=0.5, threshold_cm=40.0)
			self.assertIsInstance(res, dict)
			assert isinstance(res, dict)
			self.assertTrue(bool(res.get("ok")))
			self.assertEqual(len(calls), 1)
			self.assertEqual(calls[0]["url"], "http://127.0.0.1:8611/mcp")
			self.assertEqual(calls[0]["tool_name"], "execute_action")
			action = calls[0]["kwargs"].get("action")
			self.assertIsInstance(action, dict)
			self.assertEqual(action.get("type"), "guarded_drive")
			self.assertEqual(action.get("speed"), 30)
			self.assertEqual(action.get("steer_deg"), 10)
			self.assertEqual(action.get("duration_s"), 0.5)
			self.assertEqual(action.get("threshold_cm"), 40.0)
		finally:
			mod.call_mcp_tool_json = orig  # type: ignore[assignment]
			await agent.close()

	async def test_unreachable_move_advisor_does_not_crash(self):
		base = AdvisorSettings(
			name="advisor",
			persona_instructions="Be helpful.",
			debug=False,
			debug_log_path=None,
			response_language="de",
			openai_model="dummy",
			openai_base_url=None,
			env_file_path=None,
			listen_mcp_url="http://127.0.0.1:8602/mcp",
			speak_mcp_url="http://127.0.0.1:8601/mcp",
			observe_mcp_url="http://127.0.0.1:8603/mcp",
			move_mcp_url="http://127.0.0.1:8605/mcp",
			head_mcp_url="http://127.0.0.1:8606/mcp",
			proximity_mcp_url="http://127.0.0.1:8607/mcp",
			perception_mcp_url="http://127.0.0.1:8608/mcp",
			safety_mcp_url="http://127.0.0.1:8609/mcp",
			min_transcript_chars=1,
			max_listen_attempts=1,
			reprompt_text="Bitte wiederholen.",
			listen_stream=False,
			interrupt_speech_on_sound=False,
			wait_for_speech_finish=False,
			speech_status_poll_interval_seconds=0.1,
			speech_max_wait_seconds=1.0,
			suppress_alone_mode_while_speaking=False,
			stop_speech_on_sound_while_waiting=False,
			post_speech_interaction_grace_seconds=0.0,
			brief_listen_timeout_seconds=3.0,
			stop_words=["stop"],
			think_interval_seconds=999.0,
			observation_question="Describe.",
			max_thought_chars=200,
			sound_enabled=False,
			sound_threshold_rms=0,
			sound_active_windows_required=1,
			sound_sample_rate_hz=16000,
			sound_window_seconds=0.1,
			sound_poll_interval_seconds=0.1,
			sound_arecord_device=None,
			sound_fallback_to_interaction_on_error=False,
			memory=AdvisorMemoryConfig(
				max_tokens=100,
				avg_chars_per_token=4,
				summary_dir=".tmp_test_summaries",
				summary_max_chars=200,
			),
			memorizer=AdvisorMemorizerConfig(
				enabled=False,
				config_path="/dev/null",
				ingest_user_utterances=False,
				recall_for_questions=False,
				recall_top_n=1,
				ingest_timeout_seconds=0.1,
				recall_timeout_seconds=0.1,
			),
			todo=AdvisorTodoConfig(
				enabled=False,
				config_path="/dev/null",
				include_in_prompt=False,
				mention_next_in_response=False,
			),
		)
		settings = SplitBrainAdvisorSettings(
			base=base,
			move_advisor_mcp_url="http://127.0.0.1:8611/mcp",
			move_advisor_preflight=False,
		)
		agent = SplitBrainAdvisorAgent(settings, dry_run=False)

		async def _fake_call_mcp_tool_json(*, url: str, tool_name: str, timeout_seconds: float = 60.0, **kwargs):
			raise RuntimeError("connect failed")

		import agent.advisor_split_brain_move.src.advisor_agent as mod

		orig = mod.call_mcp_tool_json
		mod.call_mcp_tool_json = _fake_call_mcp_tool_json  # type: ignore[assignment]
		try:
			# Should not raise even if move_advisor is unreachable.
			res = await agent._safety_guarded_drive(speed=25, steer_deg=0, duration_s=0.5, threshold_cm=35.0)
			self.assertIsNone(res)

			d = await agent._proximity_distance_cm()
			self.assertIsNone(d)
		finally:
			mod.call_mcp_tool_json = orig  # type: ignore[assignment]
			await agent.close()


if __name__ == "__main__":
	unittest.main()
