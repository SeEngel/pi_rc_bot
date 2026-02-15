"""FastAPI application and MCP endpoints for the codex agent."""

from __future__ import annotations

import re
import threading
from pathlib import Path
from typing import Any

from fastapi import FastAPI
from pydantic import BaseModel, Field

from .jobs import JobKind, JobTracker
from .tools import ToolInventory
from .workers import BuildWorker, RepairWorker

app = FastAPI(title="robot_codex", version="1.0.0")

# These are set at startup by main_codex.py
_tracker: JobTracker | None = None
_build_worker: BuildWorker | None = None
_repair_worker: RepairWorker | None = None
_my_tools_dir: Path | None = None


def init_app(
    tracker: JobTracker,
    build_worker: BuildWorker,
    repair_worker: RepairWorker,
    my_tools_dir: Path,
) -> None:
    """Wire dependencies into the FastAPI app (called once at startup)."""
    global _tracker, _build_worker, _repair_worker, _my_tools_dir
    _tracker = tracker
    _build_worker = build_worker
    _repair_worker = repair_worker
    _my_tools_dir = my_tools_dir


# ── GET /healthz ────────────────────────────────────────────────

@app.get("/healthz")
async def healthz():
    return {"ok": True, "service": "robot_codex"}


# ── POST /build_tool ────────────────────────────────────────────

class BuildToolRequest(BaseModel):
    description: str = Field(
        description=(
            "Plain-text description of what you need. "
            "Be specific about the desired functionality."
        )
    )
    suggested_name: str = Field(
        default="",
        description="Optional: suggested name for the tool (lowercase, underscores).",
    )


class BuildToolResponse(BaseModel):
    ok: bool = True
    job_id: str = ""
    tool_name: str = ""
    summary: str = ""


@app.post("/build_tool", response_model=BuildToolResponse)
async def build_tool(req: BuildToolRequest):
    """Describe a new tool in plain text. The codex agent builds it as MCP in my_tools/."""
    assert _tracker is not None and _build_worker is not None

    if req.suggested_name:
        name = re.sub(r"[^a-z0-9_]", "_", req.suggested_name.lower().strip())
    else:
        words = re.sub(r"[^a-z0-9 ]", "", req.description.lower()).split()[:3]
        name = "_".join(words) if words else "new_tool"
    if not name:
        name = "new_tool"

    existing = _tracker.has_running_for(name)
    if existing:
        return BuildToolResponse(
            ok=True, job_id=existing.job_id, tool_name=name,
            summary=f"A job is already running for '{name}' — job_id={existing.job_id}",
        )

    job = _tracker.create(JobKind.BUILD, name, req.description)
    job.log(f"Build queued: {req.description[:200]}")

    thread = threading.Thread(target=_build_worker.run, args=(job,), daemon=True)
    thread.start()

    return BuildToolResponse(
        ok=True, job_id=job.job_id, tool_name=name,
        summary=f"Build started — job_id={job.job_id}. Use job_detail to track progress.",
    )


# ── POST /repair_tool ──────────────────────────────────────────

class RepairToolRequest(BaseModel):
    description: str = Field(
        description="Plain-text description of what is broken.",
    )


class RepairToolResponse(BaseModel):
    ok: bool = True
    job_id: str = ""
    tool_name: str = ""
    summary: str = ""


@app.post("/repair_tool", response_model=RepairToolResponse)
async def repair_tool(req: RepairToolRequest):
    """Describe what is broken in plain text. The codex agent figures out what to fix."""
    assert _tracker is not None and _repair_worker is not None

    tool_hint = "unknown"
    if _my_tools_dir and _my_tools_dir.exists():
        for d in sorted(_my_tools_dir.iterdir()):
            if d.is_dir() and d.name.lower() in req.description.lower():
                tool_hint = d.name
                break

    existing = _tracker.has_running_for(tool_hint)
    if existing and tool_hint != "unknown":
        return RepairToolResponse(
            ok=True, job_id=existing.job_id, tool_name=tool_hint,
            summary=f"A repair job is already running for '{tool_hint}' — job_id={existing.job_id}",
        )

    job = _tracker.create(JobKind.REPAIR, tool_hint, req.description)
    job.log(f"Repair queued: {req.description[:200]}")

    thread = threading.Thread(target=_repair_worker.run, args=(job,), daemon=True)
    thread.start()

    return RepairToolResponse(
        ok=True, job_id=job.job_id, tool_name=tool_hint,
        summary=f"Repair started — job_id={job.job_id}. Use job_detail to track progress.",
    )


# ── POST /list_jobs ─────────────────────────────────────────────

class JobSummary(BaseModel):
    job_id: str = Field(description="Unique job identifier")
    kind: str = Field(description="build or repair")
    tool_name: str = Field(description="Tool being built or repaired")
    state: str = Field(description="queued / running / done / failed")
    phase: str = Field(description="Current step")
    elapsed_seconds: float = Field(description="Time since job was created")


class ListJobsResponse(BaseModel):
    ok: bool = True
    total: int = 0
    jobs: list[JobSummary] = []


@app.post("/list_jobs", response_model=ListJobsResponse)
async def list_jobs():
    """List all build and repair jobs with high-level info."""
    assert _tracker is not None
    items = _tracker.all_jobs()
    return ListJobsResponse(
        total=len(items),
        jobs=[
            JobSummary(
                job_id=j.job_id,
                kind=j.kind.value,
                tool_name=j.tool_name,
                state=j.state.value,
                phase=j.phase,
                elapsed_seconds=j.elapsed(),
            )
            for j in items
        ],
    )


# ── POST /job_detail ────────────────────────────────────────────

class JobDetailRequest(BaseModel):
    job_id: str = Field(description="The job_id from list_jobs")


class JobDetailResponse(BaseModel):
    ok: bool = True
    job_id: str = ""
    kind: str = ""
    tool_name: str = ""
    description: str = Field(default="", description="Original request text")
    state: str = ""
    phase: str = ""
    elapsed_seconds: float = 0.0
    turns_completed: int = 0
    progress: list[str] = Field(default_factory=list, description="Step-by-step log")
    result: dict[str, Any] = Field(default_factory=dict, description="Final result")


@app.post("/job_detail", response_model=JobDetailResponse)
async def job_detail(req: JobDetailRequest):
    """Get detailed status of a specific job by job_id."""
    assert _tracker is not None
    job = _tracker.get(req.job_id)
    if job is None:
        return JobDetailResponse(
            ok=False, job_id=req.job_id, state="not_found",
            phase=f"Job '{req.job_id}' not found",
        )
    return JobDetailResponse(
        ok=True,
        job_id=job.job_id,
        kind=job.kind.value,
        tool_name=job.tool_name,
        description=job.description,
        state=job.state.value,
        phase=job.phase,
        elapsed_seconds=job.elapsed(),
        turns_completed=job.turns_completed,
        progress=list(job.progress),
        result=dict(job.result) if job.result else {},
    )
