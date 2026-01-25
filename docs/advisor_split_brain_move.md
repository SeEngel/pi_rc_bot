# advisor_split_brain_move (two-cluster architecture)

## Goal
Run two **parallel** clusters of processes:

- **Brain cluster** (interactive): listening, speaking, observing, memory, todo, dialog.
- **Move cluster** (actuation): safety, proximity, perception, low-level drive, and a *move advisor* that executes movement intents.

This reduces end-to-end latency in the advisor loop (the brain no longer sequentially calls many low-level tools) and enables true parallelism on CPython by using **separate OS processes**.

## High-level topology

```mermaid
flowchart LR
  subgraph BrainCluster[Brain cluster]
    A[AdvisorAgent\n(agent/advisor)]
    L[listen service\n(services/listening)]
    S[speak service\n(services/speak)]
    O[observe service\n(services/observe)]
    M[memorizer agent\n(agent/memorizer)]
    T[todo agent\n(agent/todo)]
  end

  subgraph MoveCluster[Move cluster]
    MA[move_advisor service\n(services/move_advisor)]
    SA[safety service\n(services/safety)]
    PX[proximity service\n(services/proximity)]
    PC[perception service\n(services/perception)]
    MV[move service\n(services/move)]
    HW[(Robot HW)]
  end

  A <-- MCP --> L
  A <-- MCP --> S
  A <-- MCP --> O

  A <-- MCP --> MA

  MA <-- MCP --> SA
  MA <-- MCP --> PX
  MA <-- MCP --> PC
  MA <-- MCP --> MV

  SA --> HW
  MV --> HW
  PX --> HW
  O --> HW
```

## Responsibilities

### Brain cluster
**AdvisorAgent** stays the only component that:

- decides *what* to do (intent-level)
- talks to humans (STT/TTS)
- chooses when to observe
- writes memory + manages todo

It should **not** directly drive the robot (or call perception/proximity/safety) in the split-brain setup.

### Move cluster
**move_advisor** is the only component that:

- decides *how* to safely execute a requested movement
- performs safety checks and obstacle gating
- uses proximity/perception as needed
- executes motion via `safety.guarded_drive` (preferred) or `move.drive` (explicit backdoor)
- supports background jobs and cancellation

## Brain ↔ Move IPC
We reuse the project’s existing transport: **MCP over HTTP**.

### Minimal interface (implemented)
`services/move_advisor` exposes MCP tools:

- `execute_action(action: dict, background: bool = false, request_id: str | null = null)`
- `job_status(job_id: str)`
- `job_cancel(job_id: str)`
- `healthz_healthz_get()`

The brain sends the same action dict it already uses internally, e.g.

- `{ "type": "guarded_drive", "speed": 25, "steer_deg": 0, "duration_s": 0.7, "threshold_cm": 35 }`
- `{ "type": "stop" }`
- `{ "type": "perception_detect" }`

### Why job-based calls?
For anything longer than a short pulse (e.g., navigation loops), `background=true` allows the brain to keep listening/speaking while the move cluster continues execution.

## Failure & safety model
- Move cluster must treat *unknown* actions as `ok=false`.
- `job_cancel` always triggers a best-effort `stop` via the safety service.
- Brain cluster should time out on move_advisor calls and fall back to speaking an error + optionally engaging estop.

## Ports
Default move_advisor ports:

- HTTP: `8011`
- MCP: `8611` (HTTP port + 600, consistent with other services)

## Configuration
Add to `agent/advisor/config.yaml`:

- `mcp.move_advisor_mcp_url: "http://127.0.0.1:8611/mcp"`

When this is set, the advisor can delegate motion/perception/proximity/safety actions to move_advisor.

## Next steps (recommended)
- Teach the brain to output **one** high-level `{"type": "move_task", ...}` action instead of many micro-actions.
- Expand move_advisor with a small planner loop:
  - preflight: `safety.check`
  - perceive: `perception.detect` and/or `observe_direction`
  - act: repeated `guarded_drive` pulses
  - report: consolidated execution summary
