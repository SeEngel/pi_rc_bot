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


class LLMTodoAgent:
    """LLM-based TodoAgent that handles task planning, tracking, and completion.

    The advisor asks this agent:
    1. plan_tasks(user_request) -> creates todo list
    2. get_next_task() -> returns next task with action instructions
    3. mark_task_done(task_id, result) -> marks task complete
    4. get_status() -> returns current status

    Uses agent-framework's OpenAIChatClient for LLM calls.
    """

    def __init__(self, settings: LLMTodoAgentSettings) -> None:
        self.settings = settings
        self._planner_agent: _TodoPlannerAgent | None = None

        # Internal state
        self._mission: str | None = None
        self._tasks: list[dict[str, Any]] = []
        self._current_task_index: int = 0

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

    async def close(self) -> None:
        """Close the planner agent."""
        if self._planner_agent is not None:
            await self._planner_agent.__aexit__(None, None, None)
            self._planner_agent = None

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
                return True
        return False

    def clear(self) -> None:
        """Clear the todo list."""
        self._mission = None
        self._tasks = []
        self._current_task_index = 0


class _TodoPlannerAgent(BaseWorkbenchChatAgent):
    """Internal agent for task planning using agent-framework."""

    SYSTEM_PROMPT = """Du bist ein Aufgabenplaner für einen Roboter-Assistenten. Zerlege Benutzeranfragen in einfache, sequentielle Aufgaben.

Der Roboter kann:
- Kopf bewegen: pan_deg (-90 bis 90, negativ=links, positiv=rechts), tilt_deg (-35 bis 35, negativ=unten, positiv=oben)
- Fahren: speed (-100 bis 100), steer_deg (-45 bis 45), duration_s (Sekunden)
- Beobachten: Ein Foto machen und beschreiben was er sieht
- Sprechen: Etwas sagen
- Memory abrufen: In der Erinnerungs-Datenbank nach Informationen suchen (Personen, Orte, Fakten, frühere Gespräche)
- Memory speichern: Neue Informationen in der Erinnerungs-Datenbank speichern

WICHTIG für Memory:
- Wenn der Benutzer nach Informationen fragt die gespeichert sein könnten (Personen, Familie, Vorlieben, frühere Gespräche), IMMER zuerst memory_recall nutzen!
- Bei indirekten Anfragen wie "ich will mein Lieblingsessen" oder "erzähl mir über X" → memory_recall nutzen
- Bei "analysiere X in deiner Datenbank" → memory_recall nutzen
- Speichere wichtige neue Informationen mit memory_store

WICHTIG: Wenn der Benutzer "schau nach X UND beschreibe/erzähle" sagt, ist das EINE Aufgabe mit observe_after=true.
Erstelle KEINE separaten Aufgaben für Schauen und Beschreiben der gleichen Richtung.

Regeln:
1. Jede Aufgabe ist EINE atomare Aktion
2. Aufgabentitel kurz halten (3-7 Wörter)
3. Aufgaben logisch ordnen
4. JSON-Array mit Objekten zurückgeben
5. Bei Wissensfragen IMMER zuerst memory_recall!

Verfügbare Aktionstypen:
- {"type": "head_set_angles", "pan_deg": <zahl>, "tilt_deg": <zahl>, "observe_after": true/false, "observe_question": "<was beschreiben>"}
- {"type": "guarded_drive", "speed": <zahl>, "steer_deg": <zahl>, "duration_s": <zahl>}
- {"type": "observe", "question": "<was beschreiben>"}
- {"type": "speak", "text": "<was sagen>"}
- {"type": "memory_recall", "query": "<wonach suchen>"}
- {"type": "memory_store", "content": "<was speichern>", "tags": ["tag1", "tag2"]}

Beispiel für "was weißt du über Sebastian Engel und dann erzähl mir davon":
[
  {"id": 1, "title": "Memory: Sebastian Engel suchen", "status": "pending", "action": {"type": "memory_recall", "query": "Sebastian Engel"}},
  {"id": 2, "title": "Ergebnis vorlesen", "status": "pending", "action": {"type": "speak", "text": "{{MEMORY_RESULT}}"}}
]

Beispiel für "schau nach oben und erzähl was du siehst, dann nach links und erzähl":
[
  {"id": 1, "title": "Nach oben schauen und beschreiben", "status": "pending", "action": {"type": "head_set_angles", "pan_deg": 0, "tilt_deg": 30, "observe_after": true, "observe_question": "Was siehst du oben? Halte dich kurz."}},
  {"id": 2, "title": "Nach links schauen und beschreiben", "status": "pending", "action": {"type": "head_set_angles", "pan_deg": -60, "tilt_deg": 0, "observe_after": true, "observe_question": "Was siehst du links? Halte dich kurz."}}
]

Gib NUR das JSON-Array zurück, nichts anderes."""

    def __init__(
        self,
        *,
        model: str,
        base_url: str | None,
        env_file_path: str | None,
        language: str = "de",
    ):
        cfg = BaseAgentConfig(
            name="TodoPlanner",
            instructions=self.SYSTEM_PROMPT,
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
