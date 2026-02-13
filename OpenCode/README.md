# OpenCode Robot Brain

> Replace all Python sub-agents with a single **OpenCode** instance that talks to the robot's existing MCP services.

## Architecture

```
┌─────────────────────────────────────────────────────────┐
│  main.py  (Python supervisor — runs forever)            │
│                                                         │
│  ┌──────────────┐   sound?   ┌───────────────────┐     │
│  │ Sound detect  │──────────▶│ OpenCode server    │     │
│  │ (sounddevice/ │  prompt   │ (opencode serve)   │     │
│  │  arecord)     │◀──────────│                    │     │
│  └──────────────┘  response  │  Uses MCP tools:   │     │
│                              │  speak, listen,    │     │
│                              │  observe, memory,  │     │
│                              │  move, head, …     │     │
│                              └───────────────────┘     │
└─────────────────────────────────────────────────────────┘
                          │
          ┌───────────────┼───────────────┐
          ▼               ▼               ▼
    MCP services    MCP services    MCP services
   (speak:8601)   (observe:8603)  (memory:8604)  …
```

## Prerequisites

| Requirement | Version | Install |
|---|---|---|
| **Node.js** | ≥ 22 | `curl -fsSL https://deb.nodesource.com/setup_22.x \| sudo -E bash - && sudo apt install -y nodejs` |
| **OpenCode** | latest | `curl -fsSL https://opencode.ai/install \| bash` |
| **Python 3** | ≥ 3.12 | Usually pre-installed on Raspberry Pi OS |
| **uv** | latest | `curl -LsSf https://astral.sh/uv/install.sh \| sh` (already in this project) |
| **Python deps** | — | `uv add requests sounddevice numpy` (already added to `pyproject.toml`) |

> On a headless Raspberry Pi without `sounddevice`, the supervisor automatically falls back to `arecord` (ALSA). The `sounddevice` + `numpy` deps are still declared in `pyproject.toml` but unused at runtime in that case.

## Installation

```bash
# 1. Install Node.js 22+ (skip if already installed)
node --version   # should be v22+
# If not:
curl -fsSL https://deb.nodesource.com/setup_22.x | sudo -E bash -
sudo apt install -y nodejs

# 2. Install OpenCode
curl -fsSL https://opencode.ai/install | bash
opencode --version

# 3. Install Python dependencies via uv (from project root)
cd /home/engelbot/Desktop/pi_rc_bot
uv sync   # installs all deps from pyproject.toml (requests, sounddevice, numpy, …)

# 4. Configure your LLM provider API key
#    Option A: via OpenCode TUI
cd /home/engelbot/Desktop/pi_rc_bot/OpenCode
opencode          # then run /connect inside the TUI

#    Option B: via environment variable
export ANTHROPIC_API_KEY="sk-ant-..."
#    or
export OPENAI_API_KEY="sk-..."
```

## Running

### Quick start (3 terminals)

```bash
# Terminal 1: Start all MCP services
cd /home/engelbot/Desktop/pi_rc_bot
bash services/main.sh

# Terminal 2: Start OpenCode headless server
cd /home/engelbot/Desktop/pi_rc_bot/OpenCode
opencode serve --port 4096

# Terminal 3: Start the supervisor
cd /home/engelbot/Desktop/pi_rc_bot
uv run python OpenCode/main.py
```

### Dry-run (test config + sound detection without connecting to OpenCode)

```bash
cd /home/engelbot/Desktop/pi_rc_bot
uv run python OpenCode/main.py --dry-run
```

### Using `opencode run` fallback (no server needed)

If you don't want to run `opencode serve`, the supervisor automatically falls back to calling `opencode run` for each prompt. This is slower (cold-starts MCP connections each time) but simpler:

```bash
# Just start services + supervisor (no opencode serve needed)
bash services/main.sh
uv run python OpenCode/main.py
```

## Configuration

Edit `OpenCode/config.yaml` to tune behaviour:

| Setting | Default | Description |
|---|---|---|
| `opencode.port` | `4096` | OpenCode server port |
| `sound.threshold_rms` | `1200` | RMS threshold for voice detection |
| `sound.active_windows_required` | `2` | Consecutive active windows needed |
| `alone.think_interval_seconds` | `20.0` | Seconds between autonomous think cycles |
| `interaction.min_transcript_chars` | `3` | Minimum transcript length to process |
| `interaction.stop_words` | `[stop, stopp, halt, genug]` | Words that end interaction |

### Environment variable overrides

| Variable | Overrides |
|---|---|
| `OPENCODE_HOST` | `opencode.host` |
| `OPENCODE_PORT` | `opencode.port` |
| `OPENCODE_MODEL` | LLM model (e.g. `anthropic/claude-sonnet-4-20250514`) |
| `OPENCODE_AGENT` | Agent name (default: `robot`) |
| `OPENCODE_LOG_LEVEL` | `log_level` |

## Files

| File | Purpose |
|---|---|
| `opencode.json` | OpenCode project config — registers all 10 MCP servers, sets permissions, defines "robot" agent |
| `AGENTS.md` | System prompt for the robot agent — describes two workstreams, tools, safety rules, memory requirements |
| `config.yaml` | Supervisor config — sound detection, timing, MCP URLs |
| `main.py` | Python supervisor — forever loop, sound detection, sends prompts to OpenCode |
| `README.md` | This file |

## How it works

1. **Sound detection loop** — The supervisor polls the microphone (via `sounddevice` or `arecord`) to measure ambient RMS volume.

2. **Alone mode** (no speech detected) — Every `think_interval_seconds`, the supervisor sends an `[ALONE]` prompt to OpenCode. OpenCode then uses its MCP tools to observe, think, optionally act, and store a memory.

3. **Interaction mode** (speech detected) — The supervisor calls the listening MCP service (`POST /listen`) to get a transcript, then sends an `[INTERACTION]` prompt with the transcript to OpenCode. OpenCode recalls memories, responds via the speak MCP, and stores a memory.

4. **Session rotation** — After every 50 turns the supervisor creates a fresh OpenCode session to prevent context overflow.

5. **Graceful shutdown** — `SIGTERM` or `SIGINT` sets a flag that cleanly exits the loop.

## Systemd service (optional)

To run the supervisor as a systemd user service:

```bash
mkdir -p ~/.config/systemd/user

cat > ~/.config/systemd/user/opencode-serve.service << 'EOF'
[Unit]
Description=OpenCode headless server
After=network.target

[Service]
Type=simple
WorkingDirectory=/home/engelbot/Desktop/pi_rc_bot/OpenCode
ExecStart=/usr/local/bin/opencode serve --port 4096
Restart=always
RestartSec=5
Environment=ANTHROPIC_API_KEY=sk-ant-YOUR-KEY-HERE

[Install]
WantedBy=default.target
EOF

cat > ~/.config/systemd/user/opencode-supervisor.service << 'EOF'
[Unit]
Description=OpenCode Robot Supervisor
After=opencode-serve.service
Requires=opencode-serve.service

[Service]
Type=simple
WorkingDirectory=/home/engelbot/Desktop/pi_rc_bot
ExecStart=/home/engelbot/.local/bin/uv run python OpenCode/main.py
Restart=always
RestartSec=5

[Install]
WantedBy=default.target
EOF

systemctl --user daemon-reload
systemctl --user enable opencode-serve opencode-supervisor
systemctl --user start opencode-serve opencode-supervisor
```

## Troubleshooting

| Problem | Fix |
|---|---|
| "OpenCode server not reachable" | Make sure `opencode serve --port 4096` is running in the `OpenCode/` directory |
| Sound detection always returns 0 | Check microphone: `arecord -d 1 -f S16_LE -r 16000 /tmp/test.wav && aplay /tmp/test.wav` |
| MCP tools fail | Ensure services are running: `curl http://127.0.0.1:8001/healthz` |
| "No module named requests" | `uv sync` (from project root) or `uv add requests` |
| OpenCode can't find `opencode.json` | Run `opencode serve` from inside the `OpenCode/` directory |
