"""LLM-based task planner for decomposing multi-step requests."""

from __future__ import annotations

import os
import re
from dataclasses import dataclass
from typing import Any

from agent_framework.openai import OpenAIChatClient
from dotenv import load_dotenv


@dataclass(frozen=True)
class TaskPlannerSettings:
    """Settings for the LLM-based task planner."""

    enabled: bool
    model: str
    base_url: str
    env_file_path: str
    min_words_for_planning: int = 6


class TaskPlanner:
    """LLM-based task planner for decomposing multi-step user requests into tasks."""

    def __init__(self, settings: TaskPlannerSettings) -> None:
        self.settings = settings
        self._client: OpenAIChatClient | None = None

        # Load API key from env file if provided
        if settings.env_file_path and os.path.exists(settings.env_file_path):
            load_dotenv(settings.env_file_path)

        self._api_key = os.environ.get("OPENAI_API_KEY", "")

    def _get_client(self) -> OpenAIChatClient:
        if self._client is None:
            base_url = self.settings.base_url or "https://api.openai.com/v1"
            self._client = OpenAIChatClient(
                model=self.settings.model,
                api_key=self._api_key,
                base_url=base_url,
            )
        return self._client

    async def close(self) -> None:
        """Close the client (no-op for agent-framework client)."""
        self._client = None

    def is_multi_step_request(self, text: str) -> bool:
        """Check if a request likely needs multi-step task planning.

        Simple heuristic: requests with enough words might be multi-step.
        """
        if not self.settings.enabled:
            return False

        words = (text or "").strip().split()
        if len(words) < self.settings.min_words_for_planning:
            return False

        # Additional heuristics: look for conjunctions or sequences
        lower = text.lower()
        multi_step_indicators = [
            " and ",
            " then ",
            " after ",
            " before ",
            " first ",
            " next ",
            " finally ",
            ", ",
        ]
        for indicator in multi_step_indicators:
            if indicator in lower:
                return True

        return False

    async def plan_tasks(
        self,
        user_request: str,
        language: str = "en",
    ) -> list[dict[str, Any]]:
        """Decompose a user request into a list of tasks using an LLM.

        Returns a list of dicts with at least "title" key, plus "action_hint" object.
        Each task that involves looking + describing should be ONE task with observe_after=true.
        """
        if not self.settings.enabled or not self._api_key:
            return []

        client = self._get_client()

        system_prompt = f"""You are a task planner for a robot assistant. Break down user requests into simple, sequential tasks.

The robot can:
- Move its head: pan_deg (-90 to 90, negative=left, positive=right), tilt_deg (-35 to 35, negative=down, positive=up)
- Drive: speed (-100 to 100), steer_deg (-45 to 45), duration_s (seconds)
- Observe: take a picture and describe what it sees
- Speak: say something

CRITICAL: When the user asks to "look somewhere AND describe what you see", combine them into ONE task with observe_after=true.
Do NOT create separate tasks for looking and describing the same direction.

Rules:
1. Each task should be ONE atomic action
2. Keep task titles short (3-7 words)
3. Order tasks logically
4. Return JSON array with objects containing "title" and "action_hint"
5. action_hint must be an object with "type" and parameters

Available action_hint types:
- {{"type": "head_set_angles", "pan_deg": <number>, "tilt_deg": <number>, "observe_after": true/false}}
- {{"type": "guarded_drive", "speed": <number>, "steer_deg": <number>, "duration_s": <number>}}
- {{"type": "observe", "question": "<what to describe>"}}
- {{"type": "speak", "text": "<what to say>"}}

Respond in {language} language for task titles.

Example for "look up and describe what you see, then look left and describe":
[
  {{"title": "Nach oben schauen und beschreiben", "action_hint": {{"type": "head_set_angles", "pan_deg": 0, "tilt_deg": 30, "observe_after": true}}}},
  {{"title": "Nach links schauen und beschreiben", "action_hint": {{"type": "head_set_angles", "pan_deg": -60, "tilt_deg": 0, "observe_after": true}}}}
]

Example for "drive forward 2 seconds then turn right":
[
  {{"title": "2 Sekunden vorwärts fahren", "action_hint": {{"type": "guarded_drive", "speed": 30, "steer_deg": 0, "duration_s": 2}}}},
  {{"title": "Nach rechts drehen", "action_hint": {{"type": "guarded_drive", "speed": 20, "steer_deg": 30, "duration_s": 1}}}}
]

Only output the JSON array, nothing else."""

        try:
            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_request},
            ]

            response = await client.complete(
                messages=messages,
                temperature=0.3,
                max_completion_tokens=500,
            )

            content = response.content if hasattr(response, "content") else str(response)
            return self._parse_tasks(content)

        except Exception:
            return []

    def _parse_tasks(self, content: str) -> list[dict[str, Any]]:
        """Parse LLM response into a list of task dicts."""
        import json

        content = (content or "").strip()

        # Try to extract JSON array from the content
        # Handle markdown code blocks
        if "```" in content:
            match = re.search(r"```(?:json)?\s*([\s\S]*?)\s*```", content)
            if match:
                content = match.group(1).strip()

        try:
            tasks = json.loads(content)
            if isinstance(tasks, list):
                # Validate each task has at least a title
                valid_tasks = []
                for t in tasks:
                    if isinstance(t, dict) and t.get("title"):
                        valid_tasks.append(t)
                return valid_tasks
        except json.JSONDecodeError:
            pass

        return []
