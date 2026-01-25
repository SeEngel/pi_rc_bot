#!/usr/bin/env bash

set -euo pipefail

SERVICES_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CONFIG_PATH="$SERVICES_DIR/config.yaml"

# Parse workflow_mode from services/config.yaml (defaults to legacy).
workflow_mode="legacy"
if [[ -f "$CONFIG_PATH" ]]; then
	val="$(grep -E '^[[:space:]]*workflow_mode[[:space:]]*:' "$CONFIG_PATH" | head -n 1 | sed -E 's/^[[:space:]]*workflow_mode[[:space:]]*:[[:space:]]*//')"
	val="${val%%#*}"
	val="$(echo "$val" | tr -d '"\r' | xargs)"
	if [[ -n "$val" ]]; then
		workflow_mode="$val"
	fi
fi

# Use global python3 by default.
# Override with: PI_RC_BOT_PYTHON=/path/to/python
PYTHON="${PI_RC_BOT_PYTHON:-/usr/bin/python3}"
if [[ ! -x "$PYTHON" ]]; then
	PYTHON="$(command -v python3)"
fi

USE_SETSID=0
if command -v setsid >/dev/null 2>&1; then
	USE_SETSID=1
fi

declare -a PIDS=()

cleanup() {
	local code=$?

	if ((${#PIDS[@]} > 0)); then
		if [[ "$USE_SETSID" -eq 1 ]]; then
			for pid in "${PIDS[@]}"; do
				kill -TERM -- "-$pid" 2>/dev/null || true
			done
		else
			for pid in "${PIDS[@]}"; do
				kill -TERM -- "$pid" 2>/dev/null || true
			done
		fi

		sleep 0.5

		if [[ "$USE_SETSID" -eq 1 ]]; then
			for pid in "${PIDS[@]}"; do
				kill -KILL -- "-$pid" 2>/dev/null || true
			done
		else
			for pid in "${PIDS[@]}"; do
				kill -KILL -- "$pid" 2>/dev/null || true
			done
		fi

		wait 2>/dev/null || true
	fi

	exit "$code"
}

trap cleanup INT TERM EXIT

start_service() {
	local dir="$1"
	local name
	name="$(basename "$dir")"
	local log_file="$SERVICES_DIR/log_${name}.out"

	: >"$log_file"
	{
		echo "[$(date -Is)] starting service '${name}'"
		echo "[$(date -Is)] cwd: ${dir}"
		echo "[$(date -Is)] python: ${PYTHON}"
		echo
	} >>"$log_file"

	if [[ "$USE_SETSID" -eq 1 ]]; then
		(
			cd "$dir"
			exec setsid "$PYTHON" main.py
		) >>"$log_file" 2>&1 &
	else
		(
			cd "$dir"
			exec "$PYTHON" main.py
		) >>"$log_file" 2>&1 &
	fi

	PIDS+=("$!")
	echo "started ${name} (pid ${PIDS[-1]}) -> ${log_file}"
}

# Move-cluster services (actuation + safety).
# move_advisor is only included in split_brain_move mode.
SERVICES=(
	"robot"
	"head"
	"move"
	"proximity"
	"perception"
	"safety"
)

# Add move_advisor only in split_brain_move mode.
if [[ "$workflow_mode" == "split_brain_move" ]]; then
	SERVICES+=("move_advisor")
	echo "workflow_mode=split_brain_move: including move_advisor service"
else
	echo "workflow_mode=$workflow_mode: skipping move_advisor service"
fi

found_any=0
for name in "${SERVICES[@]}"; do
	dir="$SERVICES_DIR/$name"
	if [[ -f "$dir/main.py" ]]; then
		found_any=1
		start_service "$dir"
	else
		echo "warn: missing service folder or main.py: $dir" >&2
	fi

done

if [[ "$found_any" -eq 0 ]]; then
	echo "no move services found under: $SERVICES_DIR" >&2
	exit 1
fi

wait
