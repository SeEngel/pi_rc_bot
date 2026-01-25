#!/usr/bin/env bash

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
UNIT_SRC_DIR="${ROOT_DIR}/services/systemd"
UNIT_DST_DIR="${XDG_CONFIG_HOME:-$HOME/.config}/systemd/user"
CONFIG_PATH="${ROOT_DIR}/services/config.yaml"

log() {
	echo "[install.sh] $*"
}

die() {
	echo "[install.sh] ERROR: $*" >&2
	exit 1
}

need_cmd() {
	command -v "$1" >/dev/null 2>&1 || die "Missing required command: $1"
}

# Parse workflow_mode from services/config.yaml (defaults to legacy).
get_workflow_mode() {
	local mode="legacy"
	if [[ -f "${CONFIG_PATH}" ]]; then
		local val
		val="$(grep -E '^[[:space:]]*workflow_mode[[:space:]]*:' "${CONFIG_PATH}" | head -n 1 | sed -E 's/^[[:space:]]*workflow_mode[[:space:]]*:[[:space:]]*//')"
		val="${val%%#*}"
		val="$(echo "$val" | tr -d '"\r' | xargs)"
		if [[ -n "$val" ]]; then
			mode="$val"
		fi
	fi
	echo "$mode"
}

render_unit_file() {
	local src="$1"
	local dst="$2"
	local unit_repo_prefix="$3"

	[[ -f "${src}" ]] || die "Unit template not found: ${src}"

	# Replace the default repo path used in templates with the actually installed repo path.
	# The templates use %h/Desktop/pi_rc_bot.
	sed "s|%h/Desktop/pi_rc_bot|${unit_repo_prefix}|g" "${src}" >"${dst}"
}

# --- Preconditions
need_cmd systemctl
need_cmd sed

# --- Ensure scripts are executable
chmod +x "${ROOT_DIR}/scripts/wait_for_network.sh" || true
chmod +x "${ROOT_DIR}/services/main.sh" || true
chmod +x "${ROOT_DIR}/services/brain.sh" || true
chmod +x "${ROOT_DIR}/services/move_cluster.sh" || true
chmod +x "${ROOT_DIR}/scripts/install_systemd_user_units.sh" || true

# --- Ensure .env exists (OpenAI services require OPENAI_API_KEY)
if [[ ! -f "${ROOT_DIR}/.env" ]]; then
	log "Creating ${ROOT_DIR}/.env (placeholder)."
	cat >"${ROOT_DIR}/.env" <<'EOF'
# Copy/paste your OpenAI key here
OPENAI_API_KEY=

# Optional overrides
# OPENAI_BASE_URL=https://your-endpoint/v1
EOF
else
	log "Found existing ${ROOT_DIR}/.env (leaving unchanged)."
fi

# --- Optionally sync Python deps (best-effort)
if [[ "${SKIP_UV_SYNC:-0}" != "1" ]]; then
	if command -v uv >/dev/null 2>&1; then
		log "Running 'uv sync' (set SKIP_UV_SYNC=1 to skip)."
		(
			cd "${ROOT_DIR}"
			uv sync
		)
	else
		log "uv not found; skipping dependency sync. Install uv: https://docs.astral.sh/uv/getting-started/installation/"
	fi
fi

# --- Install systemd user units
mkdir -p "${UNIT_DST_DIR}"

# Allow overriding install location.
INSTALL_DIR="${INSTALL_DIR:-${ROOT_DIR}}"
INSTALL_DIR="$(cd "${INSTALL_DIR}" && pwd)"

# Prefer %h/... when under HOME to keep units user-portable.
if [[ "${INSTALL_DIR}" == "${HOME}"* ]]; then
	UNIT_REPO_PREFIX="%h/${INSTALL_DIR#"${HOME}/"}"
else
	UNIT_REPO_PREFIX="${INSTALL_DIR}"
fi

log "Installing systemd user units to: ${UNIT_DST_DIR}"
log "Repo path in unit files will be: ${UNIT_REPO_PREFIX}"

# Determine workflow mode and install appropriate services.
WORKFLOW_MODE="$(get_workflow_mode)"
log "Workflow mode: ${WORKFLOW_MODE}"

# Install units based on workflow mode.
case "${WORKFLOW_MODE}" in
	legacy)
		log "Installing legacy mode units (monolithic services + legacy advisor)."
		render_unit_file "${UNIT_SRC_DIR}/pi_rc_services.service" "${UNIT_DST_DIR}/pi_rc_services.service" "${UNIT_REPO_PREFIX}"
		render_unit_file "${UNIT_SRC_DIR}/pi_rc_advisor.service" "${UNIT_DST_DIR}/pi_rc_advisor.service" "${UNIT_REPO_PREFIX}"
		# Remove split-brain units if present from a previous install.
		rm -f "${UNIT_DST_DIR}/pi_rc_advisor_split_brain.service" 2>/dev/null || true
		rm -f "${UNIT_DST_DIR}/pi_rc_brain_services.service" 2>/dev/null || true
		rm -f "${UNIT_DST_DIR}/pi_rc_move_services.service" 2>/dev/null || true
		;;
	split_brain_move)
		log "Installing split_brain_move units (brain services + move cluster + split-brain advisor)."
		render_unit_file "${UNIT_SRC_DIR}/pi_rc_brain_services.service" "${UNIT_DST_DIR}/pi_rc_brain_services.service" "${UNIT_REPO_PREFIX}"
		render_unit_file "${UNIT_SRC_DIR}/pi_rc_move_services.service" "${UNIT_DST_DIR}/pi_rc_move_services.service" "${UNIT_REPO_PREFIX}"
		render_unit_file "${UNIT_SRC_DIR}/pi_rc_advisor_split_brain.service" "${UNIT_DST_DIR}/pi_rc_advisor_split_brain.service" "${UNIT_REPO_PREFIX}"
		# Remove legacy/monolithic units if present from a previous install.
		rm -f "${UNIT_DST_DIR}/pi_rc_advisor.service" 2>/dev/null || true
		rm -f "${UNIT_DST_DIR}/pi_rc_services.service" 2>/dev/null || true
		;;
	*)
		die "Unknown workflow_mode in services/config.yaml: '${WORKFLOW_MODE}' (expected legacy|split_brain_move)"
		;;
esac

# Reload, enable and start
systemctl --user daemon-reload

# Clear any previous start-limit hits (e.g., after a failed install run).
systemctl --user reset-failed pi_rc_services.service 2>/dev/null || true
systemctl --user reset-failed pi_rc_brain_services.service 2>/dev/null || true
systemctl --user reset-failed pi_rc_move_services.service 2>/dev/null || true
systemctl --user reset-failed pi_rc_advisor.service 2>/dev/null || true
systemctl --user reset-failed pi_rc_advisor_split_brain.service 2>/dev/null || true

case "${WORKFLOW_MODE}" in
	legacy)
		systemctl --user enable --now pi_rc_services.service
		systemctl --user enable --now pi_rc_advisor.service
		# Ensure split-brain units are stopped/disabled if they were running.
		systemctl --user disable --now pi_rc_brain_services.service 2>/dev/null || true
		systemctl --user disable --now pi_rc_move_services.service 2>/dev/null || true
		systemctl --user disable --now pi_rc_advisor_split_brain.service 2>/dev/null || true
		log "Installed + started: pi_rc_services.service, pi_rc_advisor.service"
		;;
	split_brain_move)
		systemctl --user enable --now pi_rc_brain_services.service
		systemctl --user enable --now pi_rc_move_services.service
		systemctl --user enable --now pi_rc_advisor_split_brain.service
		# Ensure legacy/monolithic units are stopped/disabled if they were running.
		systemctl --user disable --now pi_rc_advisor.service 2>/dev/null || true
		systemctl --user disable --now pi_rc_services.service 2>/dev/null || true
		log "Installed + started: pi_rc_brain_services.service, pi_rc_move_services.service, pi_rc_advisor_split_brain.service"
		;;
esac

# --- Optional: enable lingering so user services start at boot without GUI/login
linger_mode="${ENABLE_LINGER:-auto}"
if [[ "${linger_mode}" == "1" || "${linger_mode}" == "true" || "${linger_mode}" == "yes" ]]; then
	linger_mode="force"
fi

if [[ "${linger_mode}" == "force" || "${linger_mode}" == "auto" ]]; then
	if command -v loginctl >/dev/null 2>&1 && command -v sudo >/dev/null 2>&1; then
		if [[ "${linger_mode}" == "force" ]]; then
			log "Enabling lingering for user ${USER} (requires sudo)."
			sudo loginctl enable-linger "${USER}"
		else
			# Auto: only do it if sudo won't prompt.
			if sudo -n true >/dev/null 2>&1; then
				log "Enabling lingering for user ${USER} (passwordless sudo detected)."
				sudo loginctl enable-linger "${USER}"
			else
				log "To start at boot without GUI/login, enable lingering (requires sudo):"
				log "  sudo loginctl enable-linger ${USER}"
				log "Or re-run with: ENABLE_LINGER=1 ./scripts/install.sh"
			fi
		fi
	else
		log "Cannot enable lingering automatically (missing loginctl or sudo)."
		log "To start at boot without GUI/login: sudo loginctl enable-linger ${USER}"
	fi
else
	log "Lingering not enabled (ENABLE_LINGER=${ENABLE_LINGER:-auto})."
fi

# Final status message based on workflow mode.
case "${WORKFLOW_MODE}" in
	legacy)
		log "Done. Check status with: systemctl --user status pi_rc_services.service pi_rc_advisor.service"
		;;
	split_brain_move)
		log "Done. Check status with: systemctl --user status pi_rc_brain_services.service pi_rc_move_services.service pi_rc_advisor_split_brain.service"
		;;
esac 
