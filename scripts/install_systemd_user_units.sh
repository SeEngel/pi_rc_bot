#!/usr/bin/env bash

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
UNIT_SRC_DIR="${ROOT_DIR}/services/systemd"
UNIT_DST_DIR="${XDG_CONFIG_HOME:-$HOME/.config}/systemd/user"
CONFIG_PATH="${ROOT_DIR}/services/config.yaml"

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

mkdir -p "${UNIT_DST_DIR}"

# Always install the base services unit.
cp -f "${UNIT_SRC_DIR}/pi_rc_services.service" "${UNIT_DST_DIR}/pi_rc_services.service"

# Install the appropriate advisor based on workflow mode.
case "${workflow_mode}" in
	legacy)
		echo "workflow_mode=legacy: installing pi_rc_advisor.service"
		cp -f "${UNIT_SRC_DIR}/pi_rc_advisor.service" "${UNIT_DST_DIR}/pi_rc_advisor.service"
		rm -f "${UNIT_DST_DIR}/pi_rc_advisor_split_brain.service" 2>/dev/null || true
		;;
	split_brain_move)
		echo "workflow_mode=split_brain_move: installing pi_rc_advisor_split_brain.service"
		cp -f "${UNIT_SRC_DIR}/pi_rc_advisor_split_brain.service" "${UNIT_DST_DIR}/pi_rc_advisor_split_brain.service"
		rm -f "${UNIT_DST_DIR}/pi_rc_advisor.service" 2>/dev/null || true
		;;
	*)
		echo "ERROR: Unknown workflow_mode in services/config.yaml: '${workflow_mode}'" >&2
		exit 1
		;;
esac

systemctl --user daemon-reload
systemctl --user enable --now pi_rc_services.service

case "${workflow_mode}" in
	legacy)
		systemctl --user enable --now pi_rc_advisor.service
		systemctl --user disable --now pi_rc_advisor_split_brain.service 2>/dev/null || true
		;;
	split_brain_move)
		systemctl --user enable --now pi_rc_advisor_split_brain.service
		systemctl --user disable --now pi_rc_advisor.service 2>/dev/null || true
		;;
esac

echo

echo "Installed and started systemd *user* services:"
echo "  - pi_rc_services.service"
case "${workflow_mode}" in
	legacy)
		echo "  - pi_rc_advisor.service"
		;;
	split_brain_move)
		echo "  - pi_rc_advisor_split_brain.service"
		;;
esac
echo

echo "To start at boot even without GUI/login, enable lingering (requires sudo):"
echo "  sudo loginctl enable-linger ${USER}"
