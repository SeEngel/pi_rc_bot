#!/usr/bin/env python3
"""
OpenCode Robot Supervisor
=========================
Runs forever in two workstreams, sending prompts to a headless OpenCode server:
  • Alone mode  – robot thinks autonomously (observe → think → act → remember)
  • Interaction – robot talks to a human   (listen → respond → remember)

Usage:
  1.  Start the MCP services   (services/main.sh  or  systemd units)
  2.  Run this supervisor:      uv run python OpenCode/main.py

Environment variables (optional overrides):
  OPENCODE_HOST        – default 127.0.0.1
  OPENCODE_PORT        – default 4096
  OPENCODE_MODEL       – e.g.  anthropic/claude-sonnet-4-20250514
  OPENCODE_AGENT       – default "robot"
  OPENCODE_LOG_LEVEL   – DEBUG / INFO / WARNING / ERROR
"""

from __future__ import annotations

import argparse
import logging
import signal
from pathlib import Path

import yaml

from src.config import deep_get, load_config
from src.sensor import detect_sound_activity
from src.supervisor import Supervisor

BASE_DIR = Path(__file__).resolve().parent
CONFIG_PATH = BASE_DIR / "config.yaml"

LOG = logging.getLogger("supervisor")


def main() -> None:
    global CONFIG_PATH

    parser = argparse.ArgumentParser(description="OpenCode Robot Supervisor")
    parser.add_argument("--config", type=str, default=str(CONFIG_PATH),
                        help="Path to config.yaml")
    parser.add_argument("--dry-run", action="store_true",
                        help="Print config and exit without running")
    parser.add_argument("--thorough-codex", action="store_true",
                        help="Enable thorough mode for the codex agent")
    args = parser.parse_args()

    CONFIG_PATH = Path(args.config)
    cfg = load_config(CONFIG_PATH)

    # ── Logging setup ───────────────────────────────────────────
    log_level = cfg.get("log_level", "INFO").upper()

    log_file = BASE_DIR / "log.out"
    file_handler = logging.FileHandler(log_file, mode="w", encoding="utf-8")
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(logging.Formatter(
        "%(asctime)s [%(levelname)-7s] %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    ))

    console_handler = logging.StreamHandler()
    console_handler.setLevel(getattr(logging, log_level, logging.INFO))
    console_handler.setFormatter(logging.Formatter(
        "%(asctime)s [%(levelname)-7s] %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    ))

    root_logger = logging.getLogger()
    root_logger.setLevel(logging.DEBUG)
    root_logger.addHandler(file_handler)
    root_logger.addHandler(console_handler)

    LOG.info("Logging to %s", log_file)

    # ── Dry-run mode ────────────────────────────────────────────
    if args.dry_run:
        LOG.info("── Dry-run mode ──")
        LOG.info("Config:\n%s", yaml.dump(cfg, default_flow_style=False))
        LOG.info("Sound test:")
        result = detect_sound_activity(
            threshold_rms=deep_get(cfg, "sound", "threshold_rms", default=1200),
            sample_rate_hz=deep_get(cfg, "sound", "sample_rate", default=16000),
            window_seconds=deep_get(cfg, "sound", "window_seconds", default=0.15),
        )
        LOG.info("  %s", result)
        return

    # ── Run ─────────────────────────────────────────────────────
    supervisor = Supervisor(cfg, BASE_DIR)

    if args.thorough_codex:
        supervisor._codex.enable_thorough_mode()

    signal.signal(signal.SIGTERM, lambda *_: supervisor.shutdown())
    signal.signal(signal.SIGINT, lambda *_: supervisor.shutdown())

    supervisor.run()


if __name__ == "__main__":
    main()
