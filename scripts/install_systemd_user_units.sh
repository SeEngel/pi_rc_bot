#!/usr/bin/env bash

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
UNIT_SRC_DIR="${ROOT_DIR}/services/systemd"
UNIT_DST_DIR="${XDG_CONFIG_HOME:-$HOME/.config}/systemd/user"
CONFIG_PATH="${ROOT_DIR}/services/config.yaml"

render_unit_file() {
	local src="$1"
	local dst="$2"
	local unit_repo_prefix="$3"

	# Replace the default repo path used in templates with the actually installed repo path.
	# The templates use %h/Desktop/pi_rc_bot.
	sed "s|%h/Desktop/pi_rc_bot|${unit_repo_prefix}|g" "${src}" >"${dst}"
}

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

# Ensure service scripts are executable for systemd ExecStart.
chmod +x "${ROOT_DIR}/scripts/wait_for_network.sh" 2>/dev/null || true
chmod +x "${ROOT_DIR}/services/main.sh" 2>/dev/null || true
chmod +x "${ROOT_DIR}/services/brain.sh" 2>/dev/null || true
chmod +x "${ROOT_DIR}/services/move_cluster.sh" 2>/dev/null || true

# Allow overriding install location.
INSTALL_DIR="${INSTALL_DIR:-${ROOT_DIR}}"
INSTALL_DIR="$(cd "${INSTALL_DIR}" && pwd)"

# Prefer %h/... when under HOME to keep units user-portable.
if [[ "${INSTALL_DIR}" == "${HOME}"* ]]; then
	UNIT_REPO_PREFIX="%h/${INSTALL_DIR#"${HOME}/"}"
else
	UNIT_REPO_PREFIX="${INSTALL_DIR}"
fi

# Install units based on workflow mode.
case "${workflow_mode}" in
	legacy)
		echo "workflow_mode=legacy: installing pi_rc_services.service + pi_rc_advisor.service"
		render_unit_file "${UNIT_SRC_DIR}/pi_rc_services.service" "${UNIT_DST_DIR}/pi_rc_services.service" "${UNIT_REPO_PREFIX}"
		render_unit_file "${UNIT_SRC_DIR}/pi_rc_advisor.service" "${UNIT_DST_DIR}/pi_rc_advisor.service" "${UNIT_REPO_PREFIX}"
		rm -f "${UNIT_DST_DIR}/pi_rc_advisor_split_brain.service" 2>/dev/null || true
		rm -f "${UNIT_DST_DIR}/pi_rc_brain_services.service" 2>/dev/null || true
		rm -f "${UNIT_DST_DIR}/pi_rc_move_services.service" 2>/dev/null || true
		;;
	split_brain_move)
		echo "workflow_mode=split_brain_move: installing pi_rc_brain_services.service + pi_rc_move_services.service + pi_rc_advisor_split_brain.service"
		render_unit_file "${UNIT_SRC_DIR}/pi_rc_brain_services.service" "${UNIT_DST_DIR}/pi_rc_brain_services.service" "${UNIT_REPO_PREFIX}"
		render_unit_file "${UNIT_SRC_DIR}/pi_rc_move_services.service" "${UNIT_DST_DIR}/pi_rc_move_services.service" "${UNIT_REPO_PREFIX}"
		render_unit_file "${UNIT_SRC_DIR}/pi_rc_advisor_split_brain.service" "${UNIT_DST_DIR}/pi_rc_advisor_split_brain.service" "${UNIT_REPO_PREFIX}"
		rm -f "${UNIT_DST_DIR}/pi_rc_advisor.service" 2>/dev/null || true
		rm -f "${UNIT_DST_DIR}/pi_rc_services.service" 2>/dev/null || true
		;;
	*)
		echo "ERROR: Unknown workflow_mode in services/config.yaml: '${workflow_mode}'" >&2
		exit 1
		;;
esac

systemctl --user daemon-reload

case "${workflow_mode}" in
	legacy)
		systemctl --user enable --now pi_rc_services.service
		systemctl --user enable --now pi_rc_advisor.service
		systemctl --user disable --now pi_rc_advisor_split_brain.service 2>/dev/null || true
		systemctl --user disable --now pi_rc_brain_services.service 2>/dev/null || true
		systemctl --user disable --now pi_rc_move_services.service 2>/dev/null || true
		;;
	split_brain_move)
		systemctl --user enable --now pi_rc_brain_services.service
		systemctl --user enable --now pi_rc_move_services.service
		systemctl --user enable --now pi_rc_advisor_split_brain.service
		systemctl --user disable --now pi_rc_advisor.service 2>/dev/null || true
		systemctl --user disable --now pi_rc_services.service 2>/dev/null || true
		;;
esac

echo

echo "Installed and started systemd *user* services:"
case "${workflow_mode}" in
	legacy)
		echo "  - pi_rc_services.service"
		echo "  - pi_rc_advisor.service"
		;;
	split_brain_move)
		echo "  - pi_rc_brain_services.service"
		echo "  - pi_rc_move_services.service"
		echo "  - pi_rc_advisor_split_brain.service"
		;;
esac
echo

echo "To start at boot even without GUI/login, enable lingering (requires sudo):"
echo "  sudo loginctl enable-linger ${USER}"
