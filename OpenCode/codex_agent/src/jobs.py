"""Job tracker — queuing, state management, and progress logging for build/repair jobs."""

from __future__ import annotations

import logging
import threading
import time
import uuid
from dataclasses import dataclass, field as dc_field
from enum import Enum
from typing import Any

LOG = logging.getLogger("codex-agent")


class JobKind(str, Enum):
    BUILD = "build"
    REPAIR = "repair"


class JobState(str, Enum):
    QUEUED = "queued"
    RUNNING = "running"
    DONE = "done"
    FAILED = "failed"


@dataclass
class Job:
    job_id: str
    kind: JobKind
    tool_name: str
    description: str = ""
    state: JobState = JobState.QUEUED
    phase: str = "queued"
    progress: list[str] = dc_field(default_factory=list)
    result: dict[str, Any] = dc_field(default_factory=dict)
    created_at: float = dc_field(default_factory=time.time)
    finished_at: float | None = None
    # internal context
    session_id: str | None = None
    turns_completed: int = 0
    source_hash_before: str = ""

    def log(self, msg: str) -> None:
        ts = time.strftime("%H:%M:%S")
        entry = f"[{ts}] {msg}"
        self.progress.append(entry)
        LOG.info("Job %s (%s/%s): %s",
                 self.job_id[:8], self.kind.value, self.tool_name, msg)

    def elapsed(self) -> float:
        end = self.finished_at or time.time()
        return round(end - self.created_at, 1)


class JobTracker:
    """Thread-safe registry of build/repair jobs."""

    def __init__(self):
        self._jobs: dict[str, Job] = {}
        self._lock = threading.Lock()

    def create(self, kind: JobKind, tool_name: str, description: str = "") -> Job:
        jid = uuid.uuid4().hex[:12]
        job = Job(job_id=jid, kind=kind, tool_name=tool_name,
                  description=description)
        with self._lock:
            self._jobs[jid] = job
        return job

    def get(self, job_id: str) -> Job | None:
        with self._lock:
            return self._jobs.get(job_id)

    def has_running_for(self, tool_name: str) -> Job | None:
        """Return a running/queued job for this tool, or None."""
        with self._lock:
            for j in self._jobs.values():
                if (j.tool_name == tool_name
                        and j.state in (JobState.QUEUED, JobState.RUNNING)):
                    return j
        return None

    def all_jobs(self, limit: int = 30) -> list[Job]:
        with self._lock:
            items = sorted(self._jobs.values(),
                           key=lambda j: j.created_at, reverse=True)
        return items[:limit]
