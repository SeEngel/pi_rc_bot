from __future__ import annotations

import argparse
import os
import sys
import time
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
	ap = argparse.ArgumentParser(description="Run one timed move job")
	ap.add_argument("--config", required=True)
	ap.add_argument("--speed", required=True, type=int, help="Signed speed (-100..100)")
	ap.add_argument("--steer-deg", required=True, type=int)
	ap.add_argument("--duration-s", required=True, type=float)
	args = ap.parse_args(argv)

	cfg_path = os.path.abspath(args.config)
	speed = int(args.speed)
	steer = int(args.steer_deg)
	dur = max(0.0, float(args.duration_s))

	# Ensure `import src` works (src/ is a package under services/move).
	here = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
	if here not in sys.path:
		sys.path.insert(0, here)

	from src import MoveController  # pylint: disable=import-error

	cfg = _load_yaml(cfg_path)
	mc = MoveController.from_config_dict(cfg)
	if not mc.is_available:
		return 0

	mc.drive(speed=speed, steer_deg=steer)
	if dur > 0:
		time.sleep(dur)
	mc.stop()
	return 0


if __name__ == "__main__":
	raise SystemExit(main())
