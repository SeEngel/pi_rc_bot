"""LLM-based TodoAgent for multi-step task planning and execution."""

from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass
from typing import Any

from agent.src.base_agent import BaseAgentConfig, BaseWorkbenchChatAgent
from agent.src.config import OpenAIClientConfig


@dataclass(frozen=True)
class LLMTodoAgentSettings:
    """Settings for the LLM-based TodoAgent."""

    enabled: bool
    model: str
    base_url: str
    env_file_path: str
    language: str = "de"
    # Review settings
    review_after_planning: bool = True  # Ask LLM if the plan is sufficient
    review_after_completion: bool = True  # Ask LLM if more todos are needed
    autonomous_mode_enabled: bool = True  # Generate random todos in alone mode


class LLMTodoAgent:
    """LLM-based TodoAgent that handles task planning, tracking, and completion.

    The advisor asks this agent:
    1. plan_tasks(user_request) -> creates todo list
    2. review_plan() -> asks LLM if the plan is sufficient or needs changes
    3. modify_plan(instruction) -> modifies the plan based on feedback
    4. get_next_task() -> returns next task with action instructions
    5. mark_task_done(task_id, result) -> marks task complete
    6. review_completion() -> asks LLM if more tasks are needed
    7. generate_autonomous_tasks() -> generates random tasks for alone mode
    8. get_status() -> returns current status

    Uses agent-framework's OpenAIChatClient for LLM calls.
    """

    def __init__(self, settings: LLMTodoAgentSettings) -> None:
        self.settings = settings
        self._planner_agent: _TodoPlannerAgent | None = None
        self._reviewer_agent: _TodoReviewerAgent | None = None
        self._autonomous_agent: _AutonomousTodoAgent | None = None

        # Internal state
        self._mission: str | None = None
        self._tasks: list[dict[str, Any]] = []
        self._current_task_index: int = 0
        self._review_status: str = "none"  # "none", "pending", "approved", "needs_changes"
        self._completed_results: list[str] = []  # Store results of completed tasks

    async def _get_planner(self) -> "_TodoPlannerAgent":
        """Lazily initialize the planner agent."""
        if self._planner_agent is None:
            self._planner_agent = _TodoPlannerAgent(
                model=self.settings.model,
                base_url=self.settings.base_url or None,
                env_file_path=self.settings.env_file_path or None,
                language=self.settings.language,
            )
            await self._planner_agent.__aenter__()
        return self._planner_agent

    async def _get_reviewer(self) -> "_TodoReviewerAgent":
        """Lazily initialize the reviewer agent."""
        if self._reviewer_agent is None:
            self._reviewer_agent = _TodoReviewerAgent(
                model=self.settings.model,
                base_url=self.settings.base_url or None,
                env_file_path=self.settings.env_file_path or None,
                language=self.settings.language,
            )
            await self._reviewer_agent.__aenter__()
        return self._reviewer_agent

    async def _get_autonomous_agent(self) -> "_AutonomousTodoAgent":
        """Lazily initialize the autonomous todo agent."""
        if self._autonomous_agent is None:
            self._autonomous_agent = _AutonomousTodoAgent(
                model=self.settings.model,
                base_url=self.settings.base_url or None,
                env_file_path=self.settings.env_file_path or None,
                language=self.settings.language,
            )
            await self._autonomous_agent.__aenter__()
        return self._autonomous_agent

    async def close(self) -> None:
        """Close all agents."""
        if self._planner_agent is not None:
            await self._planner_agent.__aexit__(None, None, None)
            self._planner_agent = None
        if self._reviewer_agent is not None:
            await self._reviewer_agent.__aexit__(None, None, None)
            self._reviewer_agent = None
        if self._autonomous_agent is not None:
            await self._autonomous_agent.__aexit__(None, None, None)
            self._autonomous_agent = None

    def is_enabled(self) -> bool:
        return self.settings.enabled

    def has_active_mission(self) -> bool:
        return self._mission is not None and len(self._tasks) > 0

    def has_pending_tasks(self) -> bool:
        return any(t.get("status") == "pending" for t in self._tasks)

    def get_status_text(self) -> str:
        """Get human-readable status of the todo list."""
        if not self._mission:
            return "Keine aktive Aufgabenliste."

        total = len(self._tasks)
        done = len([t for t in self._tasks if t.get("status") == "done"])
        pending = total - done

        lines = [f"Mission: {self._mission}"]
        lines.append(f"Fortschritt: {done}/{total} erledigt")

        if pending > 0:
            lines.append("Offene Aufgaben:")
            for t in self._tasks:
                if t.get("status") == "pending":
                    lines.append(f"  - [{t.get('id')}] {t.get('title')}")

        return "\n".join(lines)

    async def plan_tasks(self, user_request: str) -> list[dict[str, Any]]:
        """Use LLM to decompose user request into tasks with action instructions."""
        if not self.is_enabled():
            print(f"[LLMTodoAgent] Not enabled")
            return []

        print(f"[LLMTodoAgent] Planning tasks with model={self.settings.model}")
        print(f"[LLMTodoAgent] User request: {user_request[:100]}...")

        try:
            planner = await self._get_planner()
            content = await planner.plan(user_request)
            print(f"[LLMTodoAgent] LLM response content: {content[:500]}...")
            tasks = self._parse_tasks_json(content)
            print(f"[LLMTodoAgent] Parsed {len(tasks)} tasks")

            if tasks:
                self._mission = user_request[:200]
                self._tasks = tasks
                self._current_task_index = 0

            return tasks

        except Exception as e:
            print(f"[LLMTodoAgent] plan_tasks error: {type(e).__name__}: {e}")
            return []

    def _parse_tasks_json(self, content: str) -> list[dict[str, Any]]:
        """Parse LLM response into list of task dicts."""
        content = (content or "").strip()

        # Handle markdown code blocks
        if "```" in content:
            match = re.search(r"```(?:json)?\s*([\s\S]*?)\s*```", content)
            if match:
                content = match.group(1).strip()

        try:
            tasks = json.loads(content)
            if isinstance(tasks, list):
                valid_tasks = []
                for i, t in enumerate(tasks):
                    if isinstance(t, dict) and t.get("title"):
                        # Ensure required fields
                        task = {
                            "id": t.get("id", i + 1),
                            "title": t.get("title", ""),
                            "status": "pending",
                            "action": t.get("action", {}),
                        }
                        valid_tasks.append(task)
                return valid_tasks
        except json.JSONDecodeError:
            pass

        return []

    async def get_next_task(self) -> dict[str, Any] | None:
        """Get the next pending task to execute."""
        for task in self._tasks:
            if task.get("status") == "pending":
                return task
        return None

    async def mark_task_done(self, task_id: int, result: str | None = None) -> bool:
        """Mark a task as completed."""
        for task in self._tasks:
            if task.get("id") == task_id:
                task["status"] = "done"
                if result:
                    task["result"] = result
                    self._completed_results.append(result)
                return True
        return False

    def clear(self) -> None:
        """Clear the todo list."""
        self._mission = None
        self._tasks = []
        self._current_task_index = 0
        self._review_status = "none"
        self._completed_results = []

    # --- Review and Modification Methods ---

    async def review_plan(self) -> dict[str, Any]:
        """Ask LLM if the current plan is sufficient or needs changes.

        Returns:
            {
                "approved": bool,
                "suggestion": str | None,  # What to change if not approved
                "reason": str              # Why approved or not
            }
        """
        if not self.is_enabled() or not self.settings.review_after_planning:
            return {"approved": True, "suggestion": None, "reason": "review_disabled"}

        if not self._tasks:
            return {"approved": False, "suggestion": "Keine Aufgaben vorhanden", "reason": "empty_list"}

        try:
            reviewer = await self._get_reviewer()
            tasks_summary = self._get_tasks_summary()
            content = await reviewer.review_plan(self._mission or "", tasks_summary)
            print(f"[LLMTodoAgent] Review response: {content[:300]}...")
            return self._parse_review_response(content)
        except Exception as e:
            print(f"[LLMTodoAgent] review_plan error: {type(e).__name__}: {e}")
            return {"approved": True, "suggestion": None, "reason": f"error: {e}"}

    async def review_completion(self) -> dict[str, Any]:
        """Ask LLM if the mission is truly complete or if more tasks are needed.

        Returns:
            {
                "complete": bool,
                "additional_tasks": list[dict] | None,  # New tasks if not complete
                "reason": str
            }
        """
        if not self.is_enabled() or not self.settings.review_after_completion:
            return {"complete": True, "additional_tasks": None, "reason": "review_disabled"}

        if self.has_pending_tasks():
            return {"complete": False, "additional_tasks": None, "reason": "pending_tasks_exist"}

        try:
            reviewer = await self._get_reviewer()
            tasks_summary = self._get_tasks_summary(include_results=True)
            results_summary = "\n".join(self._completed_results[-5:]) if self._completed_results else "Keine Ergebnisse"
            content = await reviewer.review_completion(self._mission or "", tasks_summary, results_summary)
            print(f"[LLMTodoAgent] Completion review response: {content[:300]}...")
            return self._parse_completion_review_response(content)
        except Exception as e:
            print(f"[LLMTodoAgent] review_completion error: {type(e).__name__}: {e}")
            return {"complete": True, "additional_tasks": None, "reason": f"error: {e}"}

    async def modify_plan(self, instruction: str) -> list[dict[str, Any]]:
        """Modify the current plan based on user or LLM feedback.

        Args:
            instruction: What to change (e.g., "füge einen Schritt hinzu um links zu schauen")

        Returns:
            Updated list of tasks
        """
        if not self.is_enabled():
            return self._tasks

        try:
            planner = await self._get_planner()
            tasks_summary = self._get_tasks_summary()
            prompt = f"""Aktuelle Mission: {self._mission}

Aktuelle Aufgabenliste:
{tasks_summary}

Änderungsanweisung: {instruction}

Gib die KOMPLETTE aktualisierte Aufgabenliste als JSON-Array zurück.
Behalte bereits erledigte Aufgaben (status: "done") bei.
Passe die IDs entsprechend an."""

            content = await planner.plan(prompt)
            new_tasks = self._parse_tasks_json(content)

            if new_tasks:
                # Preserve done status from old tasks
                old_done_ids = {t["id"] for t in self._tasks if t.get("status") == "done"}
                for task in new_tasks:
                    if task["id"] in old_done_ids:
                        task["status"] = "done"
                self._tasks = new_tasks
                print(f"[LLMTodoAgent] Plan modified, now {len(new_tasks)} tasks")

            return self._tasks

        except Exception as e:
            print(f"[LLMTodoAgent] modify_plan error: {type(e).__name__}: {e}")
            return self._tasks

    async def add_tasks(self, new_tasks: list[dict[str, Any]]) -> None:
        """Add new tasks to the current plan."""
        if not new_tasks:
            return

        # Get next available ID
        max_id = max((t.get("id", 0) for t in self._tasks), default=0)

        for i, task in enumerate(new_tasks):
            task["id"] = max_id + i + 1
            if "status" not in task:
                task["status"] = "pending"
            self._tasks.append(task)

        print(f"[LLMTodoAgent] Added {len(new_tasks)} tasks, total now {len(self._tasks)}")

    async def generate_autonomous_tasks(self, context: str | None = None) -> list[dict[str, Any]]:
        """Generate random/autonomous tasks for alone mode.

        The robot should come up with things to do on its own, like:
        - Look around (left, right, up, down)
        - Drive forward/backward
        - Compose a poem
        - Tell a joke
        - Describe what it sees

        Args:
            context: Optional context like recent observations

        Returns:
            List of generated tasks
        """
        if not self.is_enabled() or not self.settings.autonomous_mode_enabled:
            print(f"[LLMTodoAgent] Autonomous mode disabled")
            return []

        try:
            autonomous = await self._get_autonomous_agent()
            content = await autonomous.generate(context)
            print(f"[LLMTodoAgent] Autonomous tasks response: {content[:300]}...")
            tasks = self._parse_tasks_json(content)

            if tasks:
                self._mission = "Autonome Erkundung"
                self._tasks = tasks
                self._current_task_index = 0
                self._review_status = "none"
                self._completed_results = []

            return tasks

        except Exception as e:
            print(f"[LLMTodoAgent] generate_autonomous_tasks error: {type(e).__name__}: {e}")
            return []

    def _get_tasks_summary(self, include_results: bool = False) -> str:
        """Get a text summary of current tasks."""
        lines = []
        for t in self._tasks:
            status = t.get("status", "pending")
            title = t.get("title", "")
            line = f"[{t.get('id')}] ({status}) {title}"
            if include_results and status == "done" and t.get("result"):
                line += f" -> {t.get('result')[:50]}"
            lines.append(line)
        return "\n".join(lines) if lines else "Keine Aufgaben"

    def _parse_review_response(self, content: str) -> dict[str, Any]:
        """Parse the plan review response from LLM."""
        content = (content or "").strip().lower()

        # Handle markdown code blocks
        if "```" in content:
            match = re.search(r"```(?:json)?\s*([\s\S]*?)\s*```", content)
            if match:
                content = match.group(1).strip()

        # Try JSON parsing first
        try:
            data = json.loads(content)
            if isinstance(data, dict):
                return {
                    "approved": bool(data.get("approved", data.get("ok", True))),
                    "suggestion": data.get("suggestion", data.get("änderung")),
                    "reason": data.get("reason", data.get("grund", ""))
                }
        except json.JSONDecodeError:
            pass

        # Fallback: look for keywords
        approved = any(kw in content for kw in ["ja", "ok", "gut", "passt", "approved", "yes"])
        not_approved = any(kw in content for kw in ["nein", "fehlt", "änder", "ergänz", "mehr", "no", "missing"])

        if not_approved and not approved:
            return {"approved": False, "suggestion": content, "reason": "needs_changes"}

        return {"approved": True, "suggestion": None, "reason": "approved"}

    def _parse_completion_review_response(self, content: str) -> dict[str, Any]:
        """Parse the completion review response from LLM."""
        content = (content or "").strip()

        # Handle markdown code blocks
        if "```" in content:
            match = re.search(r"```(?:json)?\s*([\s\S]*?)\s*```", content)
            if match:
                content = match.group(1).strip()

        # Try JSON parsing first
        try:
            data = json.loads(content)
            if isinstance(data, dict):
                additional = data.get("additional_tasks", data.get("weitere_aufgaben"))
                if isinstance(additional, list):
                    additional = self._parse_tasks_json(json.dumps(additional))
                return {
                    "complete": bool(data.get("complete", data.get("fertig", True))),
                    "additional_tasks": additional,
                    "reason": data.get("reason", data.get("grund", ""))
                }
        except json.JSONDecodeError:
            pass

        # Fallback: assume complete
        return {"complete": True, "additional_tasks": None, "reason": "assumed_complete"}


class _TodoPlannerAgent(BaseWorkbenchChatAgent):
    """Internal agent for task planning using agent-framework."""

    SYSTEM_PROMPT_TEMPLATE = """You are a task planner for a robot assistant. Decompose user requests into simple, sequential tasks.

The robot can:
- Move head: pan_deg (-90 to 90, negative=left, positive=right), tilt_deg (-35 to 35, negative=down, positive=up)
- Drive: speed (-100 to 100), steer_deg (-45 to 45), duration_s (seconds)
- Observe: Take a photo and describe what it sees
- Speak: Say something
- Memory recall: Search the memory database for information (people, places, facts, previous conversations)
- Memory store: Save new information to the memory database

IMPORTANT for Memory:
- When user asks about information that might be stored (people, family, preferences, previous conversations), ALWAYS use memory_recall first!
- For indirect requests like "I want my favorite food" or "tell me about X" → use memory_recall
- For "analyze X in your database" → use memory_recall
- Store important new information with memory_store

IMPORTANT: When the user says "look at X AND describe/tell", that is ONE task with observe_after=true.
Do NOT create separate tasks for looking and describing the same direction.

Rules:
1. Each task is ONE atomic action
2. Keep task titles short (3-7 words)
3. Order tasks logically
4. Return JSON array with objects
5. For knowledge questions ALWAYS use memory_recall first!

Available action types:
- {{"type": "head_set_angles", "pan_deg": <number>, "tilt_deg": <number>, "observe_after": true/false, "observe_question": "<what to describe>"}}
- {{"type": "guarded_drive", "speed": <number>, "steer_deg": <number>, "duration_s": <number>}}
- {{"type": "observe", "question": "<what to describe>"}}
- {{"type": "speak", "text": "<what to say>"}}
- {{"type": "memory_recall", "query": "<what to search for>"}}
- {{"type": "memory_store", "content": "<what to store>", "tags": ["tag1", "tag2"]}}

Example for "what do you know about Sebastian Engel and then tell me about it":
[
  {{"id": 1, "title": "Memory: Search Sebastian Engel", "status": "pending", "action": {{"type": "memory_recall", "query": "Sebastian Engel"}}}},
  {{"id": 2, "title": "Read out result", "status": "pending", "action": {{"type": "speak", "text": "{{{{MEMORY_RESULT}}}}"}}}}
]

Example for "look up and tell what you see, then left and tell":
[
  {{"id": 1, "title": "Look up and describe", "status": "pending", "action": {{"type": "head_set_angles", "pan_deg": 0, "tilt_deg": 30, "observe_after": true, "observe_question": "What do you see above? Keep it brief."}}}},
  {{"id": 2, "title": "Look left and describe", "status": "pending", "action": {{"type": "head_set_angles", "pan_deg": -60, "tilt_deg": 0, "observe_after": true, "observe_question": "What do you see to the left? Keep it brief."}}}}
]

IMPORTANT: The language for task titles and spoken text MUST be: {language}

Return ONLY the JSON array, nothing else."""

    def __init__(
        self,
        *,
        model: str,
        base_url: str | None,
        env_file_path: str | None,
        language: str = "en",
    ):
        # Format the system prompt with the target language for output
        instructions = self.SYSTEM_PROMPT_TEMPLATE.format(language=language)
        cfg = BaseAgentConfig(
            name="TodoPlanner",
            instructions=instructions,
            openai=OpenAIClientConfig(
                model=model,
                base_url=base_url,
                env_file_path=env_file_path,
            ),
            mcp_servers=[],
        )
        super().__init__(cfg)
        self._language = language

    async def plan(self, user_request: str) -> str:
        """Plan tasks for the given user request, returns raw LLM response."""
        if self._agent is None:
            raise RuntimeError("Agent not initialized")
        
        thread = self.get_new_thread()
        response = await self._agent.run(user_request, thread=thread)
        
        # Extract text from response
        if hasattr(response, "text"):
            return str(response.text)
        elif hasattr(response, "content"):
            return str(response.content)
        else:
            return str(response)


class _TodoReviewerAgent(BaseWorkbenchChatAgent):
    """Internal agent for reviewing and suggesting changes to task plans."""

    REVIEW_PLAN_PROMPT = """You are a task reviewer for a robot assistant.

Your task: Check if the planned tasks will fulfill the mission.

Respond ONLY with JSON in this format:
{
  "approved": true/false,
  "suggestion": "What is missing or should be changed" or null,
  "reason": "Brief explanation"
}

Check:
1. Are all steps present to fulfill the mission?
2. Is the order logical?
3. Are important steps missing?
4. Are there unnecessary steps?

If everything is good: {"approved": true, "suggestion": null, "reason": "Plan is complete"}
If something is missing: {"approved": false, "suggestion": "Add X", "reason": "Missing Y"}"""

    REVIEW_COMPLETION_PROMPT = """You are a task reviewer for a robot assistant.

Your task: Check if the mission is truly complete or if more should be done.

Respond ONLY with JSON in this format:
{
  "complete": true/false,
  "additional_tasks": null or [{"title": "...", "action": {...}}],
  "reason": "Brief explanation"
}

Check:
1. Was the original mission fully completed?
2. Are there sensible follow-up actions?
3. Did the user possibly expect more?

If everything is done: {"complete": true, "additional_tasks": null, "reason": "Mission complete"}
If more is needed: {"complete": false, "additional_tasks": [...], "reason": "Should also do X"}"""

    def __init__(
        self,
        *,
        model: str,
        base_url: str | None,
        env_file_path: str | None,
        language: str = "en",
    ):
        cfg = BaseAgentConfig(
            name="TodoReviewer",
            instructions="You are a helpful task reviewer.",
            openai=OpenAIClientConfig(
                model=model,
                base_url=base_url,
                env_file_path=env_file_path,
            ),
            mcp_servers=[],
        )
        super().__init__(cfg)
        self._language = language

    async def review_plan(self, mission: str, tasks_summary: str) -> str:
        """Review the current plan."""
        if self._agent is None:
            raise RuntimeError("Agent not initialized")

        prompt = f"""{self.REVIEW_PLAN_PROMPT}

Mission: {mission}

Planned tasks:
{tasks_summary}

Is this plan sufficient?"""

        thread = self.get_new_thread()
        response = await self._agent.run(prompt, thread=thread)

        if hasattr(response, "text"):
            return str(response.text)
        elif hasattr(response, "content"):
            return str(response.content)
        return str(response)

    async def review_completion(self, mission: str, tasks_summary: str, results_summary: str) -> str:
        """Review if the mission is complete."""
        if self._agent is None:
            raise RuntimeError("Agent not initialized")

        prompt = f"""{self.REVIEW_COMPLETION_PROMPT}

Original mission: {mission}

Completed tasks:
{tasks_summary}

Results:
{results_summary}

Is the mission fully complete or should more be done?"""

        thread = self.get_new_thread()
        response = await self._agent.run(prompt, thread=thread)

        if hasattr(response, "text"):
            return str(response.text)
        elif hasattr(response, "content"):
            return str(response.content)
        return str(response)


class _AutonomousTodoAgent(BaseWorkbenchChatAgent):
    """Internal agent for generating autonomous/random tasks in alone mode."""

    SYSTEM_PROMPT_TEMPLATE = """You are a robot assistant that is bored and wants to do something on its own.

Generate 1-3 random, interesting tasks that a robot can do autonomously.
Be creative and varied!

Possible actions:
- Move head and look (left, right, up, down)
- Drive a bit (forward, backward, turn)
- Observe and describe what you see
- Tell a short poem or joke
- Philosophize about the life of a robot
- Tell a short story
- Introduce yourself
- Comment on the environment

Return ONLY a JSON array:
[
  {{"id": 1, "title": "Short description", "action": {{"type": "...", ...}}}},
  ...
]

Available action types:
- {{"type": "head_set_angles", "pan_deg": <-90 to 90>, "tilt_deg": <-35 to 35>, "observe_after": true/false, "observe_question": "<what to describe>"}}
- {{"type": "guarded_drive", "speed": <-100 to 100>, "steer_deg": <-45 to 45>, "duration_s": <seconds>}}
- {{"type": "observe", "question": "<what to describe>"}}
- {{"type": "speak", "text": "<what to say>"}}

Examples:
[{{"id": 1, "title": "Look left", "action": {{"type": "head_set_angles", "pan_deg": -60, "tilt_deg": 0, "observe_after": true, "observe_question": "What is to my left?"}}}}]
[{{"id": 1, "title": "Tell a joke", "action": {{"type": "speak", "text": "Why can't robots lie? Because they're made of hardware, not soft-where!"}}}}]
[{{"id": 1, "title": "Drive forward briefly", "action": {{"type": "guarded_drive", "speed": 20, "steer_deg": 0, "duration_s": 0.5}}}}]

IMPORTANT: The language for task titles and spoken text MUST be: {language}

Choose RANDOMLY and be CREATIVE! Don't always do the same thing."""

    def __init__(
        self,
        *,
        model: str,
        base_url: str | None,
        env_file_path: str | None,
        language: str = "en",
    ):
        # Format the system prompt with the target language for output
        instructions = self.SYSTEM_PROMPT_TEMPLATE.format(language=language)
        cfg = BaseAgentConfig(
            name="AutonomousTodo",
            instructions=instructions,
            openai=OpenAIClientConfig(
                model=model,
                base_url=base_url,
                env_file_path=env_file_path,
            ),
            mcp_servers=[],
        )
        super().__init__(cfg)
        self._language = language

    async def generate(self, context: str | None = None) -> str:
        """Generate autonomous tasks."""
        if self._agent is None:
            raise RuntimeError("Agent not initialized")

        prompt = "What could I do now that's interesting?"
        if context:
            prompt = f"Context: {context}\n\n{prompt}"

        thread = self.get_new_thread()
        response = await self._agent.run(prompt, thread=thread)

        if hasattr(response, "text"):
            return str(response.text)
        elif hasattr(response, "content"):
            return str(response.content)
        return str(response)
