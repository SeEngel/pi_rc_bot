#!/usr/bin/env bash

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
UNIT_SRC_DIR="${ROOT_DIR}/services/systemd"
UNIT_DST_DIR="${XDG_CONFIG_HOME:-$HOME/.config}/systemd/user"

mkdir -p "${UNIT_DST_DIR}"

cp -f "${UNIT_SRC_DIR}/pi_rc_services.service" "${UNIT_DST_DIR}/pi_rc_services.service"
cp -f "${UNIT_SRC_DIR}/pi_rc_advisor.service" "${UNIT_DST_DIR}/pi_rc_advisor.service"

systemctl --user daemon-reload
systemctl --user enable --now pi_rc_services.service
systemctl --user enable --now pi_rc_advisor.service

echo

echo "Installed and started systemd *user* services:"
echo "  - pi_rc_services.service"
echo "  - pi_rc_advisor.service"
echo

echo "To start at boot even without GUI/login, enable lingering (requires sudo):"
echo "  sudo loginctl enable-linger ${USER}"
