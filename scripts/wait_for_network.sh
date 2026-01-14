#!/usr/bin/env bash

set -euo pipefail

timeout="${NETWORK_TIMEOUT_SECONDS:-120}"
require_internet="${REQUIRE_INTERNET:-0}"
ping_host="${PING_HOST:-1.1.1.1}"

start_ts="$(date +%s)"

log() {
	echo "[$(date -Is)] wait_for_network: $*"
}

is_wifi_connected() {
	# Prefer SSID check when available.
	if command -v iwgetid >/dev/null 2>&1; then
		local ssid
		ssid="$(iwgetid -r 2>/dev/null || true)"
		[[ -n "${ssid}" ]] && return 0
	fi

	# NetworkManager check (if installed).
	if command -v nmcli >/dev/null 2>&1; then
		nmcli -t -f TYPE,STATE dev status 2>/dev/null | grep -q '^wifi:connected$' && return 0
	fi

	return 1
}

has_default_route() {
	ip route show default 2>/dev/null | grep -q '^default '
}

can_ping() {
	ping -c 1 -W 1 "${ping_host}" >/dev/null 2>&1
}

# If NetworkManager is present, prefer its readiness check.
if command -v nm-online >/dev/null 2>&1; then
	log "Waiting for NetworkManager to report online (timeout ${timeout}s)…"
	while true; do
		if nm-online -q --timeout=1; then
			log "NetworkManager reports online."
			exit 0
		fi

		now_ts="$(date +%s)"
		if (( now_ts - start_ts >= timeout )); then
			log "Timed out waiting for NetworkManager online."
			exit 1
		fi

		sleep 1
	done
fi

log "Waiting for Wi‑Fi/network (timeout ${timeout}s, require_internet=${require_internet})…"

while true; do
	if has_default_route; then
		if [[ "${require_internet}" == "1" ]]; then
			if can_ping; then
				log "Default route + ping ok (${ping_host})."
				exit 0
			fi
		else
			if is_wifi_connected; then
				log "Wi‑Fi connected + default route present."
				exit 0
			fi
			log "Default route present (Wi‑Fi check unavailable or not connected). Proceeding anyway."
			exit 0
		fi
	fi

	now_ts="$(date +%s)"
	if (( now_ts - start_ts >= timeout )); then
		log "Timed out waiting for default route / connectivity."
		exit 1
	fi

	sleep 1

done
