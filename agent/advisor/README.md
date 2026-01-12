# Advisor agent

The **advisor** is an always-on orchestrator that alternates between:

1. **Interaction mode**: detect a loud sound → listen → think → speak
2. **Alone mode**: when quiet → observe → think out loud

It uses the already-running MCP services:

- `services/listening` (MCP tool `listen`)
- `services/speak` (MCP tool `speak`)
- `services/observe` (MCP tool `observe`)

## Run

```bash
cd /home/engelbot/Desktop/pi_rc_bot
uv run python agent/advisor/main.py
```

For a short test run:

```bash
uv run python agent/advisor/main.py --max-iterations 5
```

## Memory

The advisor keeps a lightweight event log. When it grows beyond the configured token limit,
it writes a summary to `memory/advisor/` and restarts the advisor LLM context seeded with that summary.

## Debug protocol (live)

If `advisor.debug: true` in `agent/advisor/config.yaml`, the advisor prints a **streaming JSONL protocol** to stdout.
Each line is a single JSON object describing the advisor's state/mode and what component it is using (brain vs MCP tools).

- Optional file output: set `advisor.debug_log_path` (e.g. `memory/advisor/protocol.jsonl`).
- Tip: pretty-print in another terminal with `jq`:

```bash
uv run python agent/advisor/main.py | jq
```

### Tuning sound trigger

If the advisor doesn't react when you speak, it's usually because it never enters interaction mode.
Watch the `"event":"sound"` lines and adjust `sound_activity.threshold_rms`:

- If `rms` stays near 0: audio capture is broken or the wrong device is selected (set `sound_activity.arecord_device`).
- If `rms` is non-zero but `active` stays false: lower `threshold_rms`.
- To force always-listen behavior: set `sound_activity.enabled: false`.
