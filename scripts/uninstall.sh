#!/usr/bin/env bash

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
UNIT_DST_DIR="${XDG_CONFIG_HOME:-$HOME/.config}/systemd/user"

log() {
	echo "[uninstall.sh] $*"
}

die() {
	echo "[uninstall.sh] ERROR: $*" >&2
	exit 1
}

need_cmd() {
	command -v "$1" >/dev/null 2>&1 || die "Missing required command: $1"
}

need_cmd systemctl

log "Stopping and disabling systemd user units (ignore errors if not installed)."
# Stop and disable both legacy and split-brain advisor services.
systemctl --user disable --now pi_rc_advisor.service >/dev/null 2>&1 || true
systemctl --user disable --now pi_rc_advisor_split_brain.service >/dev/null 2>&1 || true
systemctl --user disable --now pi_rc_services.service >/dev/null 2>&1 || true

log "Removing installed unit files from: ${UNIT_DST_DIR}"
rm -f "${UNIT_DST_DIR}/pi_rc_advisor.service" \
      "${UNIT_DST_DIR}/pi_rc_advisor_split_brain.service" \
      "${UNIT_DST_DIR}/pi_rc_services.service" || true

systemctl --user daemon-reload

# Optional: disable lingering
linger_mode="${DISABLE_LINGER:-auto}"
if [[ "${linger_mode}" == "1" || "${linger_mode}" == "true" || "${linger_mode}" == "yes" ]]; then
	linger_mode="force"
fi

if [[ "${linger_mode}" == "force" || "${linger_mode}" == "auto" ]]; then
	if command -v loginctl >/dev/null 2>&1 && command -v sudo >/dev/null 2>&1; then
		if [[ "${linger_mode}" == "force" ]]; then
			log "Disabling lingering for user ${USER} (requires sudo)."
			sudo loginctl disable-linger "${USER}"
		else
			# Auto: only do it if sudo won't prompt.
			if sudo -n true >/dev/null 2>&1; then
				log "Disabling lingering for user ${USER} (passwordless sudo detected)."
				sudo loginctl disable-linger "${USER}"
			else
				log "If you previously enabled lingering and want to undo it (requires sudo):"
				log "  sudo loginctl disable-linger ${USER}"
				log "Or re-run with: DISABLE_LINGER=1 ./scripts/uninstall.sh"
			fi
		fi
	else
		log "Cannot disable lingering automatically (missing loginctl or sudo)."
		log "If you previously enabled lingering and want to undo it: sudo loginctl disable-linger ${USER}"
	fi
else
	log "Lingering left unchanged (DISABLE_LINGER=${DISABLE_LINGER:-auto})."
fi

log "Done. Units removed; they will not start on next boot." 
