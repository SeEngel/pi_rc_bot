from __future__ import annotations

import argparse
import os
import sys
from typing import Any


def _load_yaml(path: str) -> dict[str, Any]:
	try:
		import yaml  # type: ignore
	except Exception as exc:
		raise RuntimeError("PyYAML is required to load config.yaml") from exc

	with open(path, "r", encoding="utf-8") as f:
		data = yaml.safe_load(f) or {}
	if not isinstance(data, dict):
		raise ValueError("config.yaml must contain a YAML mapping at the root")
	return data


def main(argv: list[str] | None = None) -> int:
	ap = argparse.ArgumentParser(description="Run one TTS playback job")
	ap.add_argument("--config", required=True, help="Path to services/speak/config.yaml")
	ap.add_argument("--text", required=True, help="Text to speak")
	args = ap.parse_args(argv)

	cfg_path = os.path.abspath(args.config)
	text = (args.text or "").strip()
	if not text:
		return 0

	# Ensure `import src` works (src/ is a package under services/speak).
	here = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
	if here not in sys.path:
		sys.path.insert(0, here)

	from src import Speaker  # pylint: disable=import-error

	cfg = _load_yaml(cfg_path)
	sp = Speaker.from_config_dict(cfg)
	if not sp.is_available:
		# Nothing to play.
		return 0

	sp.say(text)
	return 0


if __name__ == "__main__":
	raise SystemExit(main())
