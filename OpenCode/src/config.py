"""Configuration loading and helpers."""

from __future__ import annotations

import os
from pathlib import Path

import yaml


def deep_get(d: dict, *keys, default=None):
    """Safely traverse nested dicts.

    >>> deep_get({"a": {"b": 1}}, "a", "b")
    1
    """
    for k in keys:
        if not isinstance(d, dict):
            return default
        d = d.get(k, default)
    return d


def load_config(config_path: Path) -> dict:
    """Load *config_path* (YAML) and apply ``OPENCODE_*`` env-var overrides."""
    cfg: dict = {}
    if config_path.exists():
        with open(config_path) as f:
            cfg = yaml.safe_load(f) or {}

    # env overrides
    if v := os.environ.get("OPENCODE_HOST"):
        cfg.setdefault("opencode", {})["host"] = v
    if v := os.environ.get("OPENCODE_PORT"):
        cfg.setdefault("opencode", {})["port"] = int(v)
    if v := os.environ.get("OPENCODE_LOG_LEVEL"):
        cfg["log_level"] = v

    return cfg
