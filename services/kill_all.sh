#!/usr/bin/env bash

set -euo pipefail

SERVICES_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Ports used by services/* (HTTP 8001-8011) and MCP (8601-8611)
PORTS=(
	8001 8002 8003 8004 8005 8006 8007 8008 8009 8010 8011
	8601 8602 8603 8604 8605 8606 8607 8608 8609 8610 8611
)

log() { echo "[kill_all] $*"; }

log "Killing known service processes (best-effort)…"

# 1) Kill any python processes that are running one of our service entrypoints.
# This is the cleanest target because services are started as: python main.py
# NOTE: Only matches processes that include the repo path.
REPO_ROOT="$(cd "$SERVICES_DIR/.." && pwd)"

pkill -TERM -f "${REPO_ROOT}/services/[^ ]+/main\.py" 2>/dev/null || true

# 2) Free ports (in case something else is bound). Prefer fuser if available.
if command -v fuser >/dev/null 2>&1; then
	for p in "${PORTS[@]}"; do
		fuser -k "${p}/tcp" >/dev/null 2>&1 || true
	done
fi

# 3) Give processes a moment to exit.
sleep 0.4

# 4) Escalate to SIGKILL for any remaining matching service processes.
pkill -KILL -f "${REPO_ROOT}/services/[^ ]+/main\.py" 2>/dev/null || true

# 5) Final check: report still-listening ports.
if command -v ss >/dev/null 2>&1; then
	still="$(ss -ltnp 2>/dev/null | grep -E ':(800[1-9]|8010|8011|860[1-9]|8610|8611)\b' || true)"
	if [[ -n "$still" ]]; then
		log "Some expected ports are still in use:" >&2
		echo "$still" >&2
		exit 1
	fi
fi

log "Done. Services should be fully stopped."
