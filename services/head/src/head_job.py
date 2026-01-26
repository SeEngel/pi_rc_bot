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
	ap = argparse.ArgumentParser(description="Run one head scan job")
	ap.add_argument("--config", required=True)
	ap.add_argument("--pattern", default="sweep", choices=["sweep", "nod"]) 
	ap.add_argument("--duration-s", type=float, default=3.0)
	ap.add_argument("--step-deg", type=int, default=None)
	ap.add_argument("--interval-s", type=float, default=None)
	args = ap.parse_args(argv)

	cfg_path = os.path.abspath(args.config)

	here = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
	if here not in sys.path:
		sys.path.insert(0, here)

	from src import HeadController  # pylint: disable=import-error

	cfg = _load_yaml(cfg_path)
	hc = HeadController.from_config_dict(cfg)
	if not hc.is_available:
		return 0

	pattern = str(args.pattern)
	duration = max(0.0, float(args.duration_s))
	step = args.step_deg
	interval = args.interval_s

	start = time.time()
	if pattern == "nod":
		# tilt up/down
		hc.center()
		t = 0
		while time.time() - start < duration:
			angle = 20 if (t % 2 == 0) else -20
			hc.set_angles(pan_deg=0, tilt_deg=angle)
			t += 1
			time.sleep(interval or 0.2)
		hc.center()
		return 0

	# sweep
	step_v = int(step) if step is not None else int(hc.settings.scan_step_deg)
	interval_v = float(interval) if interval is not None else float(hc.settings.scan_interval_s)
	step_v = max(1, min(20, step_v))
	interval_v = max(0.02, min(1.0, interval_v))

	pan = -hc.settings.max_pan_deg
	dirn = 1
	hc.set_angles(pan_deg=pan, tilt_deg=0)
	while time.time() - start < duration:
		pan += dirn * step_v
		if pan >= hc.settings.max_pan_deg:
			pan = hc.settings.max_pan_deg
			dirn = -1
		elif pan <= -hc.settings.max_pan_deg:
			pan = -hc.settings.max_pan_deg
			dirn = 1
		hc.set_angles(pan_deg=pan, tilt_deg=0)
		time.sleep(interval_v)
	hc.center()
	return 0


if __name__ == "__main__":
	raise SystemExit(main())
