# advisor_split_brain_move

This agent is a **drop-in alternative** to `agent/advisor`, implementing the split-brain design:

- Brain cluster (this agent): listen/speak/observe + dialog + memory + todo
- Move cluster: `services/move_advisor` executes motion/proximity/perception/safety actions

Legacy `agent/advisor` is left unchanged.

## Config
This agent uses a config file compatible with `agent/advisor/config.yaml`, plus one extra field:

- `mcp.move_advisor_mcp_url` (default: `http://127.0.0.1:8611/mcp`)

## Run (manual)

```bash
cd ~/Desktop/pi_rc_bot

# Start services (choose workflow in services/config.yaml)
./services/main.sh

# Run the split-brain advisor
cd agent/advisor_split_brain_move
uv run main.py
```
