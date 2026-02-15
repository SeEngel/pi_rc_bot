"""Sensor inputs: sound detection, listening (STT), and scene observation."""

from __future__ import annotations

import logging
import math
import shutil
import struct
import subprocess
import time
from dataclasses import dataclass

import requests

from .config import deep_get

LOG = logging.getLogger("supervisor")


# ═══════════════════════════════════════════════════════════════
#  Sound-activity detection
# ═══════════════════════════════════════════════════════════════

@dataclass(frozen=True)
class SoundActivityResult:
    active: bool
    rms: int
    backend: str
    reason: str | None = None


def _rms_s16le(pcm: bytes) -> int:
    """RMS for signed-16-bit-LE mono PCM bytes."""
    if not pcm:
        return 0
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
    threshold_rms: int = 1200,
    sample_rate_hz: int = 16000,
    window_seconds: float = 0.15,
) -> SoundActivityResult:
    """Return whether the ambient sound level exceeds *threshold_rms*."""
    threshold = max(1, int(threshold_rms))
    sr = max(8000, int(sample_rate_hz))
    win = max(0.05, min(1.0, float(window_seconds)))

    # 1) sounddevice
    try:
        import numpy as np
        import sounddevice as sd

        frames = int(sr * win)
        data = sd.rec(frames, samplerate=sr, channels=1, dtype="int16")
        sd.wait()
        pcm = data.astype(np.int16).reshape(-1).tobytes()
        rms = _rms_s16le(pcm)
        return SoundActivityResult(active=(rms >= threshold), rms=rms, backend="sounddevice")
    except Exception as exc:
        _sd_err = str(exc)

    # 2) arecord fallback
    arecord = shutil.which("arecord")
    if arecord is not None:
        try:
            frames = int(sr * win)
            bytes_needed = max(frames, 1) * 2
            cmd = [arecord, "-q", "-t", "raw", "-f", "S16_LE", "-c", "1", "-r", str(sr)]
            proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL)
            try:
                pcm = proc.stdout.read(bytes_needed) if proc.stdout else b""
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

    return SoundActivityResult(
        active=False, rms=0, backend="none",
        reason=_sd_err if "_sd_err" in dir() else "no backend",
    )


def is_sound_active(cfg: dict) -> bool:
    """Poll multiple windows and return True if enough consecutive windows are active."""
    sound_cfg = cfg.get("sound", {})
    threshold = sound_cfg.get("threshold_rms", 1200)
    windows_required = sound_cfg.get("active_windows_required", 2)
    window_sec = sound_cfg.get("window_seconds", 0.15)
    poll_interval = sound_cfg.get("poll_interval_seconds", 0.25)
    sr = sound_cfg.get("sample_rate", 16000)

    consecutive_active = 0
    for _ in range(windows_required + 1):
        result = detect_sound_activity(
            threshold_rms=threshold,
            sample_rate_hz=sr,
            window_seconds=window_sec,
        )
        if result.active:
            consecutive_active += 1
            if consecutive_active >= windows_required:
                LOG.debug("Sound active  (rms=%d, backend=%s)", result.rms, result.backend)
                return True
        else:
            consecutive_active = 0
        time.sleep(poll_interval)
    return False


# ═══════════════════════════════════════════════════════════════
#  Listen (STT via MCP listen service)
# ═══════════════════════════════════════════════════════════════

def listen_once(
    listen_url: str,
    pause_seconds: float,
    timeout: float,
    max_wait_for_speech_seconds: float | None = None,
) -> str:
    """Single call to the listen service.  Returns transcript text (may be empty)."""
    try:
        payload: dict = {"speech_pause_seconds": pause_seconds}
        if max_wait_for_speech_seconds is not None:
            payload["max_wait_for_speech_seconds"] = max_wait_for_speech_seconds
        r = requests.post(f"{listen_url}/listen", json=payload, timeout=timeout)
        if r.ok:
            return r.json().get("text", "").strip()
    except Exception as exc:
        LOG.warning("Listen call failed: %s", exc)
    return ""


def listen_via_mcp(cfg: dict) -> str | None:
    """Listen continuously until the human stops talking.

    Returns the full transcript or ``None`` if nothing usable was captured.
    """
    listen_url = deep_get(cfg, "mcp", "listen", default="http://127.0.0.1:8002")
    interaction_cfg = cfg.get("interaction", {})
    min_chars = interaction_cfg.get("min_transcript_chars", 3)
    max_rounds = interaction_cfg.get("max_listen_rounds", 6)
    pause_first = interaction_cfg.get("listen_pause_first", 3.0)
    pause_continue = interaction_cfg.get("listen_pause_continue", 2.0)
    listen_timeout = interaction_cfg.get("listen_timeout", 30)

    chunks: list[str] = []

    for rnd in range(1, max_rounds + 1):
        pause = pause_first if rnd == 1 else pause_continue
        text = listen_once(listen_url, pause, listen_timeout)

        if text:
            chunks.append(text)
            LOG.info("Listen round %d: %s", rnd, text[:120])
        else:
            LOG.debug("Listen round %d: empty", rnd)

        if rnd >= 1 and chunks:
            if not is_sound_active(cfg):
                LOG.debug("Silence after round %d — done listening", rnd)
                break
            LOG.debug("Still hearing sound — continuing to listen")
        elif not text:
            break

    if not chunks:
        return None

    transcript = " ".join(chunks)
    if len(transcript) < min_chars:
        LOG.debug("Full transcript too short (%d chars)", len(transcript))
        return None

    LOG.info(
        "Full transcript (%d rounds, %d chars): %s",
        len(chunks), len(transcript), transcript[:200],
    )
    return transcript


# ═══════════════════════════════════════════════════════════════
#  Scene observation (pre-fetch via direct HTTP)
# ═══════════════════════════════════════════════════════════════

def prefetch_observe(cfg: dict) -> str:
    """Call the observe service directly and return the scene description."""
    observe_url = deep_get(cfg, "mcp", "observe", default="http://127.0.0.1:8003")
    try:
        r = requests.post(f"{observe_url}/observe", timeout=30)
        if r.ok:
            text = r.json().get("text", "").strip()
            if text:
                LOG.debug("Prefetch observe (%d chars): %s", len(text), text[:120])
                return text
    except Exception as exc:
        LOG.warning("Prefetch observe failed: %s", exc)
    return "(observe unavailable)"
