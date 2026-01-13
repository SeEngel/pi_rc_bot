# Todo agent

A tiny **local-only** sub-agent that manages a mission + mutable todo list.

- No MCP tools
- No OpenAI calls
- Optional persistence to a JSON file

This is meant to be used by `AdvisorAgent` as deterministic “temporal memory”.

## Run (CLI demo)

From repo root:

```bash
python agent/todo/main.py --help
```

Examples:

```bash
python agent/todo/main.py --set-mission "Go around and comment on five objects" --tasks "Explore safely" "Describe object 1/5" "Describe object 2/5"
python agent/todo/main.py --status
python agent/todo/main.py --next
python agent/todo/main.py --done
python agent/todo/main.py --add "Return to the user"
```
