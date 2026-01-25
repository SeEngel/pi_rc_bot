#!/usr/bin/env bash

set -euo pipefail

SERVICES_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

CONFIG_PATH="$SERVICES_DIR/config.yaml"

workflow_mode="legacy"
if [[ -f "$CONFIG_PATH" ]]; then
	# Parse a single top-level key: workflow_mode: legacy|split_brain_move
	# (Keep parsing deliberately simple and dependency-free.)
	val="$(grep -E '^[[:space:]]*workflow_mode[[:space:]]*:' "$CONFIG_PATH" | head -n 1 | sed -E 's/^[[:space:]]*workflow_mode[[:space:]]*:[[:space:]]*//')"
	val="${val%%#*}"
	val="$(echo "$val" | tr -d '"\r' | xargs)"
	if [[ -n "$val" ]]; then
		workflow_mode="$val"
	fi
fi

# Use global python3 by default (robot services are typically installed system-wide).
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

should_start_service() {
	local name="$1"
	# Always skip helper/internal folders.
	if [[ "$name" == "systemd" ]]; then
		return 1
	fi

	case "$workflow_mode" in
		legacy)
			# Legacy mode: do NOT start the split-brain move_advisor.
			if [[ "$name" == "move_advisor" ]]; then
				return 1
			fi
			return 0
			;;
		split_brain_move)
			# Split-brain mode: start everything (including move_advisor).
			return 0
			;;
		*)
			echo "unknown workflow_mode in services/config.yaml: '$workflow_mode' (expected legacy|split_brain_move)" >&2
			exit 2
			;;
	esac
}

# Helper: check if a service depends on 'robot' being available first.
needs_robot_service() {
	local name="$1"
	case "$name" in
		move|safety|proximity|perception|head)
			return 0
			;;
		*)
			return 1
			;;
	esac
}

found_any=0

# Start robot service FIRST (other services depend on it).
for main_py in "$SERVICES_DIR"/*/main.py; do
	[[ -f "$main_py" ]] || continue
	name="$(basename "$(dirname "$main_py")")"
	if [[ "$name" == "robot" ]]; then
		if should_start_service "$name"; then
			found_any=1
			start_service "$(dirname "$main_py")"
		fi
		break
	fi
done

# Give robot service time to start before dependent services.
if [[ "$found_any" -eq 1 ]]; then
	echo "waiting 2s for robot service to initialize..."
	sleep 2
fi

# Start remaining services.
for main_py in "$SERVICES_DIR"/*/main.py; do
	[[ -f "$main_py" ]] || continue
	name="$(basename "$(dirname "$main_py")")"
	# Skip robot (already started above).
	if [[ "$name" == "robot" ]]; then
		continue
	fi
	if ! should_start_service "$name"; then
		continue
	fi
	found_any=1
	start_service "$(dirname "$main_py")"
done

if [[ "$found_any" -eq 0 ]]; then
	echo "no services found under: $SERVICES_DIR"
	exit 1
fi

wait
