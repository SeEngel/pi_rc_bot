from __future__ import annotations

import os
import math
import shutil
import subprocess
import struct
from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class SoundActivityResult:
	active: bool
	rms: int
	backend: str
	reason: str | None = None


def _rms_s16le(pcm: bytes) -> int:
	"""Compute RMS for signed 16-bit little-endian mono PCM."""
	if not pcm:
		return 0
	# Ensure even number of bytes
	pcm = pcm[: (len(pcm) // 2) * 2]
	n = len(pcm) // 2
	if n <= 0:
		return 0
	total = 0
	for (x,) in struct.iter_unpack("<h", pcm):
		total += int(x) * int(x)
	return int(math.sqrt(total / n))


def detect_sound_activity(
	*,
	threshold_rms: int,
	sample_rate_hz: int,
	window_seconds: float,
	arecord_device: str | None = None,
) -> SoundActivityResult:
	"""Return whether the current sound level exceeds `threshold_rms`.

	Backends (in order):
	- sounddevice (if available)
	- arecord (ALSA)

	If nothing works, returns inactive.
	"""
	threshold = max(1, int(threshold_rms))
	sr = max(8000, int(sample_rate_hz))
	win = float(window_seconds)
	win = max(0.05, min(1.0, win))

	# 1) sounddevice backend (optional)
	try:
		import numpy as np  # type: ignore
		import sounddevice as sd  # type: ignore

		frames = int(sr * win)
		if frames <= 0:
			raise RuntimeError("invalid window")

		# Use default input device.
		data = sd.rec(frames, samplerate=sr, channels=1, dtype="int16")
		sd.wait()
		arr = np.asarray(data, dtype=np.int16).reshape(-1)
		b = arr.tobytes()
		rms = _rms_s16le(b)
		return SoundActivityResult(active=(rms >= threshold), rms=rms, backend="sounddevice")
	except Exception as exc:
		_sd_err = str(exc)

	# 2) arecord backend
	arecord = shutil.which("arecord")
	if arecord is not None:
		try:
			# Record raw PCM to stdout.
			# NOTE: `arecord -d` expects whole seconds on many systems; passing a fractional
			# value like "0.15" can result in 0 seconds recorded -> empty output -> rms=0.
			# To support sub-second windows, we start arecord without a duration, read the
			# needed bytes, then terminate.
			# -q: quiet
			# -t raw: raw
			# -f S16_LE: 16-bit
			# -c 1: mono
			# -r: sample rate
			cmd = [arecord, "-q", "-t", "raw", "-f", "S16_LE", "-c", "1", "-r", str(sr)]
			if arecord_device:
				cmd.extend(["-D", str(arecord_device)])

			frames = int(sr * win)
			bytes_needed = max(0, frames) * 2  # 16-bit mono
			if bytes_needed <= 0:
				return SoundActivityResult(active=False, rms=0, backend="arecord", reason="invalid window")

			proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL)
			try:
				pcm = b""
				if proc.stdout is not None:
					pcm = proc.stdout.read(bytes_needed) or b""
			finally:
				try:
					proc.terminate()
				except Exception:
					pass
				try:
					proc.wait(timeout=1.0)
				except Exception:
					try:
						proc.kill()
					except Exception:
						pass
			rms = _rms_s16le(pcm)
			return SoundActivityResult(active=(rms >= threshold), rms=rms, backend="arecord")
		except Exception as exc:
			return SoundActivityResult(active=False, rms=0, backend="arecord", reason=str(exc))

	# No backend available
	return SoundActivityResult(active=False, rms=0, backend="none", reason=_sd_err if "_sd_err" in locals() else "no backend")
