https://docs.sunfounder.com/projects/picar-x-v20/en/latest/

hostname rsp5: engelbot
user: engelbot
pw: lmu

raspberry pi connect: https://www.raspberrypi.com/software/connect/
--> Desktop via browser into pi

1. Install all modules https://docs.sunfounder.com/projects/picar-x-v20/en/latest/python/python_start/install_all_modules.html
2. Install "sudo pip3 install -r requirements.txt --break-system-packages" (toDo: need to ecapsulate step 1 and 2 to docker because sudo installation is not a good style)
3. Install uv: https://docs.astral.sh/uv/getting-started/installation/
4. Set up venv with "uv venv .venv --python 3.12"
5. activate venv: "source .venv/bin/activate"
6. sync: "uv sync"

## Autostart on Linux boot (systemd)

Goal:

1. Wait until the OS is up and Wi‑Fi/network is online
2. Start `~/Desktop/pi_rc_bot/services/main.sh`
3. Sleep 5 seconds
4. Start `~/Desktop/pi_rc_bot/agent/advisor` via `uv run main.py`

The unit files live in the repo under `services/systemd/`.

### Install (user service)

```bash
cd ~/Desktop/pi_rc_bot
./scripts/install_systemd_user_units.sh
```

To start at boot even without GUI/login (requires sudo):

```bash
sudo loginctl enable-linger $USER
```

Status/logs:

```bash
systemctl --user status pi_rc_services.service pi_rc_advisor.service
journalctl --user -u pi_rc_services.service -u pi_rc_advisor.service -f
```

### What exactly was added/changed?

New/updated files in this repo:

- `scripts/wait_for_network.sh`
	- Waits until the network is “online” before starting anything.
	- Prefers `nm-online` (NetworkManager). If not available, it checks for a default route (`ip route`) and optionally ping.
	- Configurable via environment variables:
		- `NETWORK_TIMEOUT_SECONDS` (default: `120`)
		- `REQUIRE_INTERNET` (`0`/`1`, default: `0`)
		- `PING_HOST` (default: `1.1.1.1`)

- `services/systemd/pi_rc_services.service`
	- systemd *user* unit to start the MCP services via `~/Desktop/pi_rc_bot/services/main.sh`.
	- `Wants=network-online.target` + `After=network-online.target`.
	- Calls `scripts/wait_for_network.sh` as an `ExecStartPre` step.

- `services/systemd/pi_rc_advisor.service`
	- systemd *user* unit to start the advisor.
	- Dependency: `Requires=pi_rc_services.service` and `After=... pi_rc_services.service`.
	- Also waits for network (`ExecStartPre=...wait_for_network.sh`) and additionally `sleep 5`.
	- Then starts in `~/Desktop/pi_rc_bot/agent/advisor` via `uv run main.py`.

- `scripts/install_systemd_user_units.sh`
	- Copies the unit files to `~/.config/systemd/user/`
	- Runs `systemctl --user daemon-reload`
	- Enables and starts both services: `pi_rc_services.service` and `pi_rc_advisor.service`

Important: this does NOT write to `/etc/systemd/system/` (so it’s not a system-wide service). It uses *user* services.

### Where are the active unit files located?

After running the install script, the “active” copies are here:

- `~/.config/systemd/user/pi_rc_services.service`
- `~/.config/systemd/user/pi_rc_advisor.service`

If you change anything in `services/systemd/*.service`, run this again:

```bash
cd ~/Desktop/pi_rc_bot
./scripts/install_systemd_user_units.sh
```

### Enable/disable

Stop (immediately):

```bash
systemctl --user stop pi_rc_advisor.service
systemctl --user stop pi_rc_services.service
```

Disable (no longer autostart):

```bash
systemctl --user disable --now pi_rc_advisor.service
systemctl --user disable --now pi_rc_services.service
```

Enable again:

```bash
systemctl --user enable --now pi_rc_services.service
systemctl --user enable --now pi_rc_advisor.service
```

### Remove completely (undo)

```bash
systemctl --user disable --now pi_rc_advisor.service pi_rc_services.service
rm -f ~/.config/systemd/user/pi_rc_advisor.service ~/.config/systemd/user/pi_rc_services.service
systemctl --user daemon-reload
```

Optional (if you enabled “linger” and want to undo it):

```bash
sudo loginctl disable-linger $USER
```

### What to edit if you want to change behavior


- Order / delay:
	- The 5 seconds are in `~/.config/systemd/user/pi_rc_advisor.service` as `ExecStartPre=/usr/bin/sleep 5`.
	- Change it (e.g. to 10s) and then:
		- `systemctl --user daemon-reload`
		- `systemctl --user restart pi_rc_advisor.service`


- Make network waiting stricter (require real internet):
	- In `~/.config/systemd/user/pi_rc_services.service` or `pi_rc_advisor.service` add for example:
		- `Environment=REQUIRE_INTERNET=1`
		- `Environment=PING_HOST=8.8.8.8`


- The actual start commands:
	- Services: `ExecStart=%h/Desktop/pi_rc_bot/services/main.sh`
	- Advisor: `ExecStart=/usr/bin/env uv run main.py`

### Logs

- systemd journal:
	- `journalctl --user -u pi_rc_services.service -f`
	- `journalctl --user -u pi_rc_advisor.service -f`


- Per-service log files (written by `services/main.sh`):
	- `~/Desktop/pi_rc_bot/services/log_listening.out`
	- `~/Desktop/pi_rc_bot/services/log_speak.out`
	- etc.
