# OpenCode Robot Brain

> A single **OpenCode** instance replaces all Python sub-agents.  A Python supervisor (`main.py`) manages the lifecycle, detects sound, and feeds prompts to OpenCode — which talks to the robot's MCP services, builds its own tools, and repairs them autonomously.

---

## Architecture Overview

```
┌──────────────────────────────────────────────────────────────────────┐
│                       main.py  (Python supervisor)                   │
│                                                                      │
│  ┌──────────────┐  sound?  ┌──────────────────────────────────────┐  │
│  │ Sound detect  │────────▶│         OpenCodeClient               │  │
│  │ (sounddevice/ │ prompt  │  POST /session/:id/message           │  │
│  │  arecord)     │◀────────│  (builds prompt, sends to OC #1)    │  │
│  └──────────────┘ response └──────────┬───────────────────────────┘  │
│                                       │ HTTP                         │
│  Lifecycle managed by main.py:        ▼                              │
│  • opencode serve :4096        ┌─────────────────────────────┐      │
│  • codex agent   :8012/8612   │   OpenCode #1  (Go binary)  │      │
│  • my_tools/*     :9100+       │   port 4096                 │      │
│                                │                             │      │
│                                │   opencode.json config:     │      │
│                                │   • model: gpt-4.1-mini     │      │
│                                │   • 7 MCP servers           │      │
│                                │   • agent: "robot"          │      │
│                                │   • AGENTS.md system prompt │      │
│                                └──────────┬──────────────────┘      │
└───────────────────────────────────────────┼──────────────────────────┘
                                            │  MCP (SSE)
          ┌──────────────┬──────────────┬───┴───┬──────────────┐
          ▼              ▼              ▼       ▼              ▼
   ┌────────────┐ ┌────────────┐ ┌──────────┐ ┌────────────┐ ┌──────────┐
   │speak :8601 │ │head  :8606 │ │prox :8607│ │percep :8608│ │safety    │
   └────────────┘ └────────────┘ └──────────┘ └────────────┘ │    :8609 │
                                                             └──────────┘
   ┌──────────────┐  ┌──────────────────────────────────────────────┐
   │move_adv :8611│  │ robot_codex :8612  (FastAPI + FastMCP)      │
   └──────────────┘  │   ↕ internally uses OpenCode #2 (:4097)     │
                     │   tools: diagnose / repair / scan_all        │
   ┌─────────────┐   └──────────────────────────────────────────────┘
   │my_tools/*   │
   │ :9100+      │   (agent-created custom MCP tools)
   └─────────────┘
```

---

## What is OpenCode?

**OpenCode** is a third-party headless AI coding agent (a Go binary).  When you run `opencode serve --port 4096`, it starts a **local HTTP REST server** — a stateful LLM gateway with:

- **Session management** — create sessions, send messages, get responses
- **MCP tool routing** — connects to MCP servers and lets the LLM call them
- **File editing + bash** — the LLM can edit code and run shell commands
- **`.env` loading** — automatically reads `OPENAI_API_KEY` from the working directory

Your Python code **never calls OpenAI directly** — it always goes through OpenCode.

### OpenCode HTTP API (used by main.py)

| Endpoint | Purpose |
|---|---|
| `GET /global/health` | Health check (`{ "healthy": true }`) |
| `POST /session` | Create a new conversation session |
| `POST /session/:id/message` | Send a prompt, receive LLM response |
| `POST /mcp` | Hot-register a new MCP server at runtime |

---

## Two OpenCode Instances

| Instance | Port | Config dir | Model | Purpose |
|---|---|---|---|---|
| **#1 (main brain)** | `4096` | `OpenCode/` | `gpt-4.1-mini` | Robot agent — talks to 7 MCP services, thinks, moves, speaks |
| **#2 (codex AI)** | `4097` | `OpenCode/codex_agent/` | `gpt-5.2` | Codex technician — only has `bash` + `edit` tools, fixes broken code |

OpenCode #1 is started by `main.py`.  OpenCode #2 is started internally by the codex agent (`main_codex.py`).

---

## Data Flow

### Main Agent (thinking / interacting)

```
1.  main.py detects sound (or timer fires for alone mode)
         │
2.  main.py prefetches observe + memory via direct HTTP (:8003, :8004)
         │  (parallel, using ThreadPoolExecutor)
         │
3.  main.py builds prompt string (scene + memory + instructions)
         │
4.  POST http://127.0.0.1:4096/session/{sid}/message
         │  { "parts": [{"type":"text","text": prompt}], "agent": "robot" }
         │
5.  OpenCode #1 receives it → forwards to OpenAI gpt-4.1-mini
         │                       with AGENTS.md as system prompt
         │
6.  LLM responds, possibly with MCP tool calls:
         │  e.g. "call robot_move_advisor → get_direction"
         │
7.  OpenCode #1 routes tool call → MCP SSE to the service port
         │
8.  MCP service responds → OpenCode forwards back to LLM
         │
9.  LLM final text response → OpenCode returns to main.py
         │
10. main.py logs it, waits for next cycle
```

### Codex Agent (when main agent calls `robot_codex → repair`)

```
1.  OpenCode #1 gets LLM tool-call: robot_codex → repair(tool_name="my_weather")
         │
2.  OpenCode #1 sends MCP request to :8612 (main_codex.py)
         │
3.  main_codex.py reads my_weather/server.py source + server.log
         │
4.  Builds repair prompt with source code + error logs
         │
5.  POST http://127.0.0.1:4097/session/{sid}/message
         │  { "parts": [...], "agent": "repair" }
         │
6.  OpenCode #2 receives it → forwards to OpenAI gpt-5.2
         │                       with repair AGENTS.md (bash + edit only)
         │
7.  LLM uses bash/edit tools to fix the file → responds "FIXED: ..."
         │
8.  main_codex.py restarts the tool process, re-registers with OpenCode #1
         │
9.  Returns result back through MCP → OpenCode #1 → LLM → main.py
```

### Dynamic Tool Creation (my_tools)

```
1.  LLM decides it needs a new capability (e.g. web search)
         │
2.  Uses bash tool to create my_tools/my_web_search/server.py
         │  (from _template/server.py scaffold)
         │
3.  Writes port.txt, starts with: uv run server.py
         │
4.  Hot-registers with OpenCode #1:
         │  POST http://127.0.0.1:4096/mcp
         │  { "name": "my_web_search", "type": "remote",
         │    "url": "http://127.0.0.1:9700/mcp" }
         │
5.  Tool is immediately available for subsequent LLM turns
```

---

## Boot Sequence

When you run `uv run python OpenCode/main.py`, this happens in order:

```
1.  Load config.yaml
2.  Start OpenCode #1  (opencode serve --port 4096)
3.  Wait for /global/health → healthy
4.  Start my_tools/* servers (from my_tools/ subdirectories)
5.  Register my_tools with OpenCode #1 (POST /mcp for each)
6.  Start codex agent (python codex_agent/main_codex.py --port 8012)
     └─ codex agent internally starts OpenCode #2 on port 4097
7.  Wait for codex agent /healthz → ok
8.  Register robot_codex with OpenCode #1 (POST /mcp)
9.  Enter main loop:
     ├─ Sound detected? → Interaction mode (listen → prompt → respond)
     └─ No sound?       → Alone mode (every 12s: observe → think → act)
10. Session rotation every 10 turns (fresh context)
11. SIGTERM/SIGINT → graceful shutdown of all child processes
```

---

## Prerequisites

| Requirement | Version | Install |
|---|---|---|
| **Node.js** | ≥ 22 | `curl -fsSL https://deb.nodesource.com/setup_22.x \| sudo -E bash - && sudo apt install -y nodejs` |
| **OpenCode** | latest | `curl -fsSL https://opencode.ai/install \| bash` |
| **Python** | ≥ 3.12 | Usually pre-installed on Raspberry Pi OS |
| **uv** | latest | `curl -LsSf https://astral.sh/uv/install.sh \| sh` |
| **Python deps** | — | Managed via `pyproject.toml` — `uv sync` installs everything |

> On a headless Raspberry Pi without `sounddevice`, the supervisor automatically falls back to `arecord` (ALSA).

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

# 3. Install Python dependencies
cd /home/engelbot/Desktop/pi_rc_bot
uv sync

# 4. Configure your API key (one of):
#    Option A: .env file (recommended — auto-loaded by OpenCode)
echo 'OPENAI_API_KEY=sk-...' > OpenCode/.env

#    Option B: OpenCode TUI
cd OpenCode && opencode   # then /connect inside the TUI

#    Option C: Environment variable
export OPENAI_API_KEY="sk-..."
```

## Running

### One command (recommended)

`main.py` auto-starts everything — OpenCode, codex agent, custom tools:

```bash
# Terminal 1: Start all robot MCP services
cd /home/engelbot/Desktop/pi_rc_bot
bash services/main.sh

# Terminal 2: Start the supervisor (starts OpenCode + codex agent automatically)
cd /home/engelbot/Desktop/pi_rc_bot
uv run python OpenCode/main.py
```

That's it. No need to manually run `opencode serve`.

### Dry-run (test config + sound detection without sending prompts)

```bash
uv run python OpenCode/main.py --dry-run
```

---

## Configuration

### `OpenCode/config.yaml` — Supervisor settings

| Setting | Default | Description |
|---|---|---|
| `opencode.port` | `4096` | Main OpenCode server port |
| `opencode.request_timeout` | `60` | Seconds to wait for LLM response |
| `sound.threshold_rms` | `1000` | RMS threshold for voice detection |
| `sound.active_windows_required` | `1` | Consecutive active windows before interaction mode |
| `alone.think_interval_seconds` | `12.0` | Seconds between autonomous think cycles |
| `interaction.max_listen_rounds` | `6` | Max consecutive listen rounds per interaction |
| `interaction.stop_words` | `[stop, stopp, halt, genug]` | Words that end an interaction |

### `OpenCode/opencode.json` — OpenCode project config

Defines which MCP servers OpenCode #1 connects to, model selection, agent persona, and tool permissions.

### `OpenCode/codex_agent/config.yaml` — Codex agent settings

| Setting | Default | Description |
|---|---|---|
| `opencode.port` | `4097` | Codex agent's own OpenCode port |
| `server.api_port` | `8012` | Codex agent HTTP REST port |
| `server.mcp_port` | `8612` | Codex agent MCP protocol port |
| `main_opencode_port` | `4096` | Main OpenCode port (for re-registration) |
| `log_tail_lines` | `80` | Lines of server.log to read for diagnosis |

### Environment variable overrides

| Variable | Overrides |
|---|---|
| `OPENCODE_HOST` | `opencode.host` |
| `OPENCODE_PORT` | `opencode.port` |
| `OPENCODE_MODEL` | LLM model override |
| `OPENCODE_AGENT` | Agent name (default: `robot`) |
| `OPENCODE_LOG_LEVEL` | `log_level` |

---

## Port Map

| Port | Protocol | Service |
|---|---|---|
| `4096` | HTTP | OpenCode #1 (main brain) |
| `4097` | HTTP | OpenCode #2 (codex AI) |
| `8001` | HTTP | speak (REST) |
| `8002` | HTTP | listen (REST) |
| `8003` | HTTP | observe (REST) |
| `8004` | HTTP | memory (REST) |
| `8005` | HTTP | move (REST) |
| `8006` | HTTP | head (REST) |
| `8007` | HTTP | proximity (REST) |
| `8008` | HTTP | perception (REST) |
| `8009` | HTTP | safety (REST) |
| `8011` | HTTP | move_advisor (REST) |
| `8012` | HTTP | codex agent (REST) |
| `8601` | MCP/SSE | speak |
| `8606` | MCP/SSE | head |
| `8607` | MCP/SSE | proximity |
| `8608` | MCP/SSE | perception |
| `8609` | MCP/SSE | safety |
| `8611` | MCP/SSE | move_advisor |
| `8612` | MCP/SSE | robot_codex |
| `9100+` | HTTP+MCP | my_tools/* (agent-created) |

> Robot services expose **both** HTTP REST (80xx) and MCP/SSE (86xx) on separate ports.  The supervisor uses direct HTTP for prefetching (faster); OpenCode uses MCP for tool-calling.

---

## Files

| File | Purpose |
|---|---|
| `main.py` | Python supervisor — lifecycle management, sound detection, prompt building, sends to OpenCode |
| `opencode.json` | OpenCode #1 project config — 7 MCP servers, `robot` agent, tool permissions |
| `AGENTS.md` | System prompt — robot personality, two workstreams, tool-building instructions, repair tools |
| `config.yaml` | Supervisor config — sound detection, timing, MCP URLs for direct HTTP |
| `.env` | API key (`OPENAI_API_KEY=...`) — auto-loaded by OpenCode |
| `codex_agent/main_codex.py` | Repair MCP server (FastAPI + FastMCP) — `diagnose`, `repair`, `scan_all` tools |
| `codex_agent/opencode.json` | OpenCode #2 config — `repair` agent, bash + edit only, no MCP servers |
| `codex_agent/AGENTS.md` | Codex technician system prompt |
| `codex_agent/config.yaml` | Codex agent config — ports, timeouts, log settings |
| `my_tools/` | Directory for agent-created custom MCP tools |
| `my_tools/_template/server.py` | FastAPI + FastMCP scaffold for new tools |
| `my_tools/manage.py` | CLI to create/list/delete custom tools |

---

## How It Works

### Alone Mode (no speech detected)

Every `12` seconds, the supervisor:
1. Pre-fetches scene observation and memory in parallel (direct HTTP to `:8003` and `:8004`)
2. Builds an `[ALONE]` prompt with the pre-fetched context
3. Sends it to OpenCode #1 → LLM thinks → calls MCP tools (move, speak, head, etc.)
4. Logs the response

### Interaction Mode (speech detected)

When the microphone picks up sound (RMS > threshold):
1. Calls the listen MCP service (up to 6 rounds of continuous listening)
2. Concatenates the full transcript
3. Pre-fetches memory in parallel
4. Builds an `[INTERACTION]` prompt with the transcript
5. Sends it to OpenCode #1 → LLM responds → calls speak MCP → stores memory

### Self-Repair

If a custom tool in `my_tools/` goes down, `check_my_tools_health()` detects it and injects a broken-tools warning into the next prompt.  The LLM then calls `robot_codex → repair(tool_name="...")` which triggers the full AI-powered repair cycle.

### Self-Extending (my_tools)

The agent can create new MCP tools at runtime:
1. Writes a new `my_tools/<name>/server.py` from the template
2. Starts it (`uv run server.py`)
3. Hot-registers it with OpenCode via `POST /mcp`
4. The tool is immediately available for subsequent turns

### Session Rotation

After every 10 turns, the supervisor creates a fresh OpenCode session to prevent context window overflow.

### Graceful Shutdown

`SIGTERM` or `SIGINT` → stops the main loop → terminates all child processes (OpenCode #1, codex agent, my_tools).

---

## Systemd Service (optional)

Since `main.py` auto-starts OpenCode, you only need **one** systemd unit:

```bash
mkdir -p ~/.config/systemd/user

cat > ~/.config/systemd/user/robot-brain.service << 'EOF'
[Unit]
Description=OpenCode Robot Brain (supervisor + OpenCode + codex agent)
After=network.target

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
systemctl --user enable robot-brain
systemctl --user start robot-brain
```

> Make sure robot MCP services (`services/main.sh`) are running before starting the brain, or set up a separate systemd unit for them with `Before=robot-brain.service`.

---

## Troubleshooting

| Problem | Fix |
|---|---|
| "OpenCode server not reachable" | Check logs: `cat OpenCode/opencode_serve.log` |
| Sound detection always returns 0 | Test mic: `arecord -d 1 -f S16_LE -r 16000 /tmp/test.wav && aplay /tmp/test.wav` |
| MCP tools fail | Check services: `curl http://127.0.0.1:8001/healthz` |
| Codex agent not starting | Check logs: `cat OpenCode/codex_agent/codex_supervisor.log` |
| "No module named requests" | `cd /home/engelbot/Desktop/pi_rc_bot && uv sync` |
| OpenCode can't find config | Verify `opencode.json` exists in working dir |
| API key not found | Ensure `OpenCode/.env` contains `OPENAI_API_KEY=sk-...` |
| Custom tool broken | Call `curl http://127.0.0.1:8012/scan_all` to diagnose |
