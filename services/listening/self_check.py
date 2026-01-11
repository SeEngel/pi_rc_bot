from __future__ import annotations

import os
from typing import Any

import yaml


def _load_yaml(path: str) -> dict[str, Any]:
	with open(path, "r", encoding="utf-8") as f:
		data = yaml.safe_load(f) or {}
	if not isinstance(data, dict):
		raise ValueError("config.yaml must contain a YAML mapping at the root")
	return data


def main() -> int:
	here = os.path.dirname(os.path.abspath(__file__))
	cfg = _load_yaml(os.path.join(here, "config.yaml"))

	# Avoid microphone access during the self-check.
	stt_cfg = cfg.get("stt", {}) if isinstance(cfg, dict) else {}
	if not isinstance(stt_cfg, dict):
		stt_cfg = {}
	stt_cfg["dry_run"] = True
	cfg["stt"] = stt_cfg

	# Allow `from src import Listener` when running from repo root.
	if here not in os.sys.path:
		os.sys.path.insert(0, here)

	from src import Listener

	li = Listener.from_config_dict(cfg)
	print(
		{
			"ok": True,
			"engine": li.settings.engine,
			"language": li.settings.language,
			"stt_available": li.is_available,
			"stt_unavailable_reason": li.unavailable_reason,
		}
	)
	return 0


if __name__ == "__main__":
	raise SystemExit(main())
