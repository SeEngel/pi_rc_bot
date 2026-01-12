import unittest

from agent.advisor.src.advisor_agent import AdvisorAgent, AdvisorMemoryConfig, AdvisorMemorizerConfig, AdvisorSettings


class TestDecisionParsing(unittest.TestCase):
	def test_parse_plain_json(self):
		raw = '{"response_text": "Hallo!", "need_observe": false}'
		resp, need = AdvisorAgent._parse_decision_json(object(), raw)
		self.assertEqual(resp, "Hallo!")
		self.assertIs(need, False)

	def test_parse_code_fenced_json(self):
		raw = """```json\n{\n  \"response_text\": \"Ich bin hier.\",\n  \"need_observe\": false\n}\n```"""
		resp, need = AdvisorAgent._parse_decision_json(object(), raw)
		self.assertEqual(resp, "Ich bin hier.")
		self.assertIs(need, False)

	def test_parse_json_with_prefix_suffix(self):
		raw = "Sure! Here you go:\n```json\n{\"response_text\": \"Okay\", \"need_observe\": true}\n```\n"  # stray text
		resp, need = AdvisorAgent._parse_decision_json(object(), raw)
		self.assertEqual(resp, "Okay")
		self.assertIs(need, True)


class TestDecideAndRespondIntegration(unittest.IsolatedAsyncioTestCase):
	async def test_decide_and_respond_uses_response_text_only(self):
		# Create a minimal AdvisorAgent instance with a mocked brain.
		settings = AdvisorSettings(
			name="advisor",
			persona_instructions="Be helpful.",
			debug=False,
			debug_log_path=None,
			response_language="de",
			openai_model="dummy",
			openai_base_url=None,
			env_file_path=None,
			listen_mcp_url="http://127.0.0.1:8600/mcp",
			speak_mcp_url="http://127.0.0.1:8601/mcp",
			observe_mcp_url="http://127.0.0.1:8602/mcp",
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
			stop_words=["stop"],
			think_interval_seconds=999.0,
			observation_question="Describe.",
			max_thought_chars=200,
			sound_enabled=False,
			sound_threshold_rms=0,
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
		)
		agent = AdvisorAgent(settings, dry_run=False)

		class MockBrain:
			async def __aenter__(self):
				return self

			async def __aexit__(self, exc_type, exc, tb):
				return None

			async def run(self, prompt: str) -> str:
				return """```json\n{\n  \"response_text\": \"Ich bin hier. Wie kann ich weiterhelfen?\",\n  \"need_observe\": false\n}\n```"""

		agent._brain = MockBrain()  # type: ignore[assignment]

		resp, need_observe = await agent._decide_and_respond(human_text="Hi", observation=None, memory_hint=None)
		self.assertEqual(resp, "Ich bin hier. Wie kann ich weiterhelfen?")
		self.assertIs(need_observe, False)

		await agent.close()


if __name__ == "__main__":
	unittest.main()
