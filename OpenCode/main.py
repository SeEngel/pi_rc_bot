#!/usr/bin/env python3
"""
OpenCode Robot Supervisor
=========================
Runs forever in two workstreams, sending prompts to a headless OpenCode server:
  • Alone mode  – robot thinks autonomously (observe → think → act → remember)
  • Interaction – robot talks to a human   (listen → respond → remember)

Requires:
  uv sync   (from project root — installs pyyaml, requests, sounddevice, numpy)

Usage:
  1.  Start the MCP services              (services/main.sh  or  systemd units)
  2.  Run this supervisor:                 uv run python OpenCode/main.py
      (it auto-starts 'opencode serve' and kills it on exit)

Environment variables (optional overrides):
  OPENCODE_HOST        – default 127.0.0.1
  OPENCODE_PORT        – default 4096
  OPENCODE_MODEL       – e.g.  anthropic/claude-sonnet-4-20250514
  OPENCODE_AGENT       – default "robot"
  OPENCODE_LOG_LEVEL   – DEBUG / INFO / WARNING / ERROR
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import os
import shutil
import signal
import struct
import subprocess
import sys
import textwrap
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

import yaml

try:
    import requests
except ImportError:
    sys.exit(
        "ERROR: 'requests' is required.  Install it:  pip3 install requests"
    )

# ──────────────────────────── globals ────────────────────────────

LOG = logging.getLogger("supervisor")
RUNNING = True

BASE_DIR = Path(__file__).resolve().parent
CONFIG_PATH = BASE_DIR / "config.yaml"


# ──────────────────────────── config ─────────────────────────────

def _deep_get(d: dict, *keys, default=None):
    """Safely traverse nested dicts."""
    for k in keys:
        if not isinstance(d, dict):
            return default
        d = d.get(k, default)
    return d


def load_config() -> dict:
    """Load config.yaml and apply environment variable overrides."""
    cfg: dict = {}
    if CONFIG_PATH.exists():
        with open(CONFIG_PATH) as f:
            cfg = yaml.safe_load(f) or {}

    # env overrides
    if v := os.environ.get("OPENCODE_HOST"):
        cfg.setdefault("opencode", {})["host"] = v
    if v := os.environ.get("OPENCODE_PORT"):
        cfg.setdefault("opencode", {})["port"] = int(v)
    if v := os.environ.get("OPENCODE_LOG_LEVEL"):
        cfg["log_level"] = v

    return cfg


# ──────────────────────────── sound detection ────────────────────

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
    """Return whether the ambient sound level exceeds *threshold_rms*.

    Tries sounddevice first, falls back to arecord.
    """
    threshold = max(1, int(threshold_rms))
    sr = max(8000, int(sample_rate_hz))
    win = max(0.05, min(1.0, float(window_seconds)))

    # ── 1) sounddevice ──
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

    # ── 2) arecord ──
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
    for _ in range(windows_required + 1):  # allow one extra poll
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


# ──────────────────────────── OpenCode serve process ─────────────

_opencode_proc: subprocess.Popen | None = None


def start_opencode_serve(host: str, port: int) -> subprocess.Popen:
    """Launch `opencode serve` as a child process.

    The process inherits the current environment (so it picks up
    OPENAI_API_KEY etc.) and runs in the OpenCode project directory
    where opencode.json and AGENTS.md live.

    IMPORTANT: stdout/stderr are sent to a log file (opencode_serve.log)
    rather than captured via PIPE, because PIPE without continuous
    draining causes the child to block once the OS pipe buffer fills.
    """
    global _opencode_proc

    opencode_bin = shutil.which("opencode")
    if opencode_bin is None:
        # Try common install locations
        for candidate in [
            Path.home() / ".local" / "bin" / "opencode",
            Path.home() / ".opencode" / "bin" / "opencode",
            Path("/usr/local/bin/opencode"),
        ]:
            if candidate.exists():
                opencode_bin = str(candidate)
                break
    if opencode_bin is None:
        raise FileNotFoundError(
            "'opencode' not found in PATH. Install it: "
            "curl -fsSL https://opencode.ai/install | bash"
        )

    cmd = [
        opencode_bin, "serve",
        "--port", str(port),
        "--hostname", host,
    ]
    LOG.info("Starting OpenCode server: %s", " ".join(cmd))

    # Redirect child stdout/stderr to a log file to prevent pipe-buffer
    # blocking.  The file is opened in write mode (truncated on restart).
    serve_log_path = BASE_DIR / "opencode_serve.log"
    _serve_log_fh = open(serve_log_path, "w", encoding="utf-8")  # noqa: SIM115
    LOG.info("OpenCode server output → %s", serve_log_path)

    proc = subprocess.Popen(
        cmd,
        cwd=str(BASE_DIR),
        stdout=_serve_log_fh,
        stderr=subprocess.STDOUT,
    )
    # Keep the file handle alive on the process object so it isn't GC'd
    proc._serve_log_fh = _serve_log_fh  # type: ignore[attr-defined]
    _opencode_proc = proc
    LOG.info("OpenCode server started (pid %d)", proc.pid)
    return proc


def stop_opencode_serve() -> None:
    """Gracefully stop the managed opencode serve process."""
    global _opencode_proc
    if _opencode_proc is None:
        return
    proc = _opencode_proc
    _opencode_proc = None
    LOG.info("Stopping OpenCode server (pid %d)…", proc.pid)
    try:
        proc.terminate()
        try:
            proc.wait(timeout=10)
            LOG.info("OpenCode server stopped gracefully")
        except subprocess.TimeoutExpired:
            LOG.warning("OpenCode server did not stop in 10s — killing")
            proc.kill()
            proc.wait(timeout=5)
    except Exception as exc:
        LOG.error("Error stopping OpenCode server: %s", exc)
    finally:
        # Close the log-file handle opened in start_opencode_serve
        fh = getattr(proc, "_serve_log_fh", None)
        if fh is not None:
            try:
                fh.close()
            except Exception:
                pass


import atexit as _atexit
_atexit.register(stop_opencode_serve)


# ──────────────────────────── OpenCode HTTP client ───────────────

class OpenCodeClient:
    """Thin wrapper around the OpenCode headless HTTP server API."""

    def __init__(self, host: str = "127.0.0.1", port: int = 4096,
                 timeout: int = 120, agent: str = "robot",
                 model: str | None = None):
        self.base = f"http://{host}:{port}"
        self.timeout = timeout
        self.agent = agent
        self.model = model
        self._session_id: str | None = None

    # ── health ──

    def healthy(self) -> bool:
        try:
            r = requests.get(f"{self.base}/global/health", timeout=5)
            return r.ok and r.json().get("healthy", False)
        except Exception:
            return False

    def wait_for_server(self, retries: int = 30, delay: float = 2.0) -> None:
        """Block until the OpenCode server is reachable."""
        for i in range(retries):
            if self.healthy():
                LOG.info("OpenCode server is healthy at %s", self.base)
                return
            LOG.info("Waiting for OpenCode server (%d/%d)…", i + 1, retries)
            time.sleep(delay)
        raise RuntimeError(f"OpenCode server at {self.base} not reachable after {retries} attempts")

    # ── session management ──

    @property
    def session_id(self) -> str:
        if self._session_id is None:
            self._session_id = self._create_session()
        return self._session_id

    def _create_session(self) -> str:
        r = requests.post(
            f"{self.base}/session",
            json={"title": "robot-supervisor"},
            timeout=self.timeout,
        )
        r.raise_for_status()
        data = r.json()
        sid = data.get("id") or data.get("ID") or data.get("sessionID")
        if not sid:
            raise RuntimeError(f"No session ID in response: {data}")
        LOG.info("Created OpenCode session %s", sid)
        return sid

    def new_session(self) -> str:
        """Force a fresh session (e.g. after many turns to keep context small)."""
        self._session_id = self._create_session()
        return self._session_id

    # ── messaging ──

    def send(self, prompt: str) -> dict:
        """Send a prompt to the current session and wait for the assistant's
        reply.  Returns the raw JSON response body.
        """
        body: dict[str, Any] = {
            "parts": [{"type": "text", "text": prompt}],
        }
        if self.agent:
            body["agent"] = self.agent
        if self.model:
            body["model"] = self.model

        url = f"{self.base}/session/{self.session_id}/message"
        LOG.debug("POST %s  (prompt length %d)", url, len(prompt))
        r = requests.post(url, json=body, timeout=self.timeout)
        LOG.debug("Response status=%d, content-length=%s", r.status_code, r.headers.get("content-length", "?"))
        r.raise_for_status()

        # Handle empty or non-JSON responses
        raw = r.text.strip()
        if not raw:
            LOG.warning("Empty response body from OpenCode server")
            return {"parts": [], "info": {}}
        try:
            return r.json()
        except Exception as exc:
            LOG.error("Failed to parse response as JSON: %s — raw (first 500): %s", exc, raw[:500])
            return {"parts": [{"type": "text", "text": raw[:2000]}], "info": {}}

    def abort(self) -> None:
        """Abort a running request in the current session."""
        if self._session_id is None:
            return
        try:
            requests.post(
                f"{self.base}/session/{self._session_id}/abort",
                timeout=10,
            )
        except Exception:
            pass

    # ── helpers ──

    def extract_text(self, response: dict) -> str:
        """Pull readable text out of the server response."""
        parts = response.get("parts", [])
        texts: list[str] = []
        for p in parts:
            if isinstance(p, dict):
                if p.get("type") == "text":
                    texts.append(p.get("text", ""))
                elif "content" in p:
                    texts.append(str(p["content"]))
        if texts:
            return "\n".join(texts)
        # Fallback: look at info.metadata or stringify
        info = response.get("info", {})
        if isinstance(info, dict) and info.get("content"):
            return str(info["content"])
        return json.dumps(response, indent=2, ensure_ascii=False)[:2000]


# ──────────────────────────── prompt builders ────────────────────

def alone_prompt(observation: str, memories: str) -> str:
    return textwrap.dedent(f"""\
        [ALONE] You are on your own — no human is speaking. Time to EXPLORE!

        ## What you see right now
        {observation}

        ## Relevant memories
        {memories}

        ## Your task (up to 4 tool calls)
        1. Think about what you see + remember. Are you following a plan? Continue it!
        2. Say a SHORT thought out loud → robot_speak → speak (under 240 chars).
        3. **DRIVE somewhere interesting!** Use guarded_drive with speed 40-70, duration 2-5s.
           Chain multiple drives to cover real ground. Explore the room. Be bold!
        4. Do NOT call robot_observe or robot_memory — the data is above.

        The supervisor will store a memory for you after this turn.
        Be adventurous. Drive far. Don't just sit there!
    """)


def interaction_prompt(transcript: str, observation: str, memories: str) -> str:
    return textwrap.dedent(f"""\
        [INTERACTION] A human just spoke to you. Their words:

        \"\"\"{transcript}\"\"\"

        ## What you see right now
        {observation}

        ## Relevant memories
        {memories}

        ## Your task (up to 5 tool calls)
        1. Decide what to do based on what the human said.
        2. If they gave you a GOAL (go somewhere, explore, find something):
           - Make a plan. Start executing it NOW with multiple guarded_drive calls.
           - Use speed 40-70, duration 2-5s per drive. Chain 2-3 drives per turn.
           - Do NOT ask for permission. Just GO and report what happened.
        3. Reply out loud → robot_speak → speak.
        4. Do NOT call robot_observe or robot_memory — the data is above.

        The supervisor will store a memory for you after this turn.
        Respond in the language the human used (default: German / de-DE).
        When given a goal: ACT FIRST, talk second. Be bold!
    """)


# ──────────────────────────── listen via MCP ─────────────────────

def _listen_once(listen_url: str, pause_seconds: float, timeout: float) -> str:
    """Single call to the listen service.  Returns transcript text (may be empty)."""
    try:
        r = requests.post(
            f"{listen_url}/listen",
            json={"speech_pause_seconds": pause_seconds},
            timeout=timeout,
        )
        if r.ok:
            return r.json().get("text", "").strip()
    except Exception as exc:
        LOG.warning("Listen call failed: %s", exc)
    return ""


def listen_via_mcp(cfg: dict) -> str | None:
    """Listen continuously until the human stops talking.

    Strategy:
      1. First call uses a short silence-pause (speech_pause_seconds) to
         capture the initial utterance quickly.
      2. Then immediately check if there is still sound.  If yes, keep
         calling listen with a short pause to accumulate more chunks.
      3. Stop when either:
         - a listen call returns empty / too-short text AND sound is quiet, or
         - we hit max_listen_rounds.
    All chunks are joined into one transcript.
    """
    listen_url = _deep_get(cfg, "mcp", "listen", default="http://127.0.0.1:8002")
    interaction_cfg = cfg.get("interaction", {})
    min_chars = interaction_cfg.get("min_transcript_chars", 3)
    max_rounds = interaction_cfg.get("max_listen_rounds", 6)
    pause_first = interaction_cfg.get("listen_pause_first", 3.0)
    pause_continue = interaction_cfg.get("listen_pause_continue", 2.0)
    listen_timeout = interaction_cfg.get("listen_timeout", 30)

    chunks: list[str] = []

    for rnd in range(1, max_rounds + 1):
        pause = pause_first if rnd == 1 else pause_continue
        text = _listen_once(listen_url, pause, listen_timeout)

        if text:
            chunks.append(text)
            LOG.info("Listen round %d: %s", rnd, text[:120])
        else:
            LOG.debug("Listen round %d: empty", rnd)

        # After first chunk, keep going only if there's still sound
        if rnd >= 1 and chunks:
            if not is_sound_active(cfg):
                LOG.debug("Silence after round %d — done listening", rnd)
                break
            LOG.debug("Still hearing sound — continuing to listen")
        elif not text:
            # First call returned nothing useful
            break

    if not chunks:
        return None

    transcript = " ".join(chunks)
    if len(transcript) < min_chars:
        LOG.debug("Full transcript too short (%d chars)", len(transcript))
        return None

    LOG.info("Full transcript (%d rounds, %d chars): %s",
             len(chunks), len(transcript), transcript[:200])
    return transcript


# ──────────────────────────── pre-fetch helpers ──────────────

def prefetch_observe(cfg: dict) -> str:
    """Call the observe service directly and return the scene description.

    This avoids a round-trip through OpenCode → MCP, saving ~8-15 s.
    """
    observe_url = _deep_get(cfg, "mcp", "observe", default="http://127.0.0.1:8003")
    try:
        r = requests.post(f"{observe_url}/observe", timeout=15)
        if r.ok:
            data = r.json()
            text = data.get("text", "").strip()
            if text:
                LOG.debug("Prefetch observe (%d chars): %s", len(text), text[:120])
                return text
    except Exception as exc:
        LOG.warning("Prefetch observe failed: %s", exc)
    return "(observe unavailable)"


def prefetch_memory(cfg: dict, query: str, top_n: int = 3) -> str:
    """Call the memory service directly and return formatted memories.

    This avoids a round-trip through OpenCode → MCP, saving ~5-10 s.
    """
    memory_url = _deep_get(cfg, "mcp", "memory", default="http://127.0.0.1:8004")
    try:
        r = requests.post(
            f"{memory_url}/get_top_n_memory",
            json={"content": query, "top_n": top_n},
            timeout=10,
        )
        if r.ok:
            data = r.json()
            items = data.get("short_term_memory", []) or []
            if items:
                lines = []
                for it in items:
                    c = it.get("content", "")
                    tags = it.get("tags", [])
                    ts = it.get("timestamp", "")
                    lines.append(f"- [{ts}] {c}  (tags: {tags})")
                result = "\n".join(lines)
                LOG.debug("Prefetch memory (%d items): %s", len(items), result[:120])
                return result
    except Exception as exc:
        LOG.warning("Prefetch memory failed: %s", exc)
    return "(no memories found)"


def post_store_memory(cfg: dict, content: str, tags: list[str]) -> bool:
    """Store a memory entry directly via HTTP, saving a tool-call round-trip."""
    memory_url = _deep_get(cfg, "mcp", "memory", default="http://127.0.0.1:8004")
    try:
        r = requests.post(
            f"{memory_url}/store_memory",
            json={"content": content, "tags": tags},
            timeout=10,
        )
        if r.ok:
            LOG.debug("Stored memory (%d chars, tags=%s)", len(content), tags)
            return True
    except Exception as exc:
        LOG.warning("Post-store memory failed: %s", exc)
    return False


# ──────────────────────────── stop-word check ────────────────

def has_stop_word(text: str, cfg: dict) -> bool:
    stop_words = _deep_get(cfg, "interaction", "stop_words", default=[])
    lower = text.lower()
    return any(w.lower() in lower for w in stop_words)


# ──────────────────────────── main loop ──────────────────────────

def run_forever(cfg: dict) -> None:
    global RUNNING

    oc_cfg = cfg.get("opencode", {})
    host = oc_cfg.get("host", "127.0.0.1")
    port = oc_cfg.get("port", 4096)
    timeout = oc_cfg.get("request_timeout", 120)
    agent = os.environ.get("OPENCODE_AGENT", "robot")
    model = os.environ.get("OPENCODE_MODEL")

    think_interval = _deep_get(cfg, "alone", "think_interval_seconds", default=20.0)

    # Auto-start opencode serve if not already running
    client = OpenCodeClient(host=host, port=port, timeout=timeout, agent=agent, model=model)

    if not client.healthy():
        LOG.info("OpenCode server not running — starting it automatically…")
        start_opencode_serve(host, port)
        client.wait_for_server(retries=30, delay=2.0)
    else:
        LOG.info("OpenCode server already running at %s", client.base)

    def send_prompt(prompt: str) -> str:
        resp = client.send(prompt)
        return client.extract_text(resp)

    # Track turns for periodic session rotation
    turn_count = 0
    SESSION_ROTATE_EVERY = 10  # start fresh session every N turns

    LOG.info("═══ Robot supervisor started ═══")
    LOG.info("  think_interval = %.1fs", think_interval)
    LOG.info("  sound threshold = %d RMS", _deep_get(cfg, "sound", "threshold_rms", default=1200))

    while RUNNING:
        try:
            # ── Rotate session periodically to avoid context overflow ──
            if client and turn_count > 0 and turn_count % SESSION_ROTATE_EVERY == 0:
                LOG.info("Rotating session after %d turns", turn_count)
                client.new_session()

            # ── Check for sound ──
            sound_active = is_sound_active(cfg)

            if sound_active:
                # ════════ INTERACTION MODE ════════
                LOG.info("🎤 Sound detected — entering interaction mode")

                transcript = listen_via_mcp(cfg)
                if transcript is None:
                    LOG.info("No usable transcript — back to alone mode")
                    continue

                if has_stop_word(transcript, cfg):
                    LOG.info("Stop word detected in: %s", transcript[:60])
                    try:
                        send_prompt(
                            f'[INTERACTION] The human said: "{transcript}"\n'
                            "They want you to stop. Acknowledge briefly via robot_speak → speak."
                        )
                        post_store_memory(cfg, f"[INTERACTION] Human said stop: {transcript[:200]}", ["interaction", "stop"])
                    except Exception as exc:
                        LOG.error("Stop-word response failed: %s", exc)
                    turn_count += 1
                    continue

                # Pre-fetch observe + memory in parallel-ish (both fast HTTP)
                t0 = time.monotonic()
                obs = prefetch_observe(cfg)
                mem = prefetch_memory(cfg, transcript)
                LOG.debug("Prefetch took %.1fs", time.monotonic() - t0)

                try:
                    reply = send_prompt(interaction_prompt(transcript, obs, mem))
                    LOG.info("Interaction reply (first 200 chars): %s", reply[:200])
                    # Store memory on behalf of the agent
                    summary = f"[INTERACTION] Human: {transcript[:200]} | Scene: {obs[:150]} | Reply: {reply[:200]}"
                    post_store_memory(cfg, summary, ["interaction", "human"])
                except Exception as exc:
                    LOG.error("Interaction prompt failed: %s", exc)

                turn_count += 1

            else:
                # ════════ ALONE MODE ════════
                LOG.debug("😶 Quiet — alone mode")

                # Pre-fetch observe + memory (saves ~15-20s per cycle)
                t0 = time.monotonic()
                obs = prefetch_observe(cfg)
                mem = prefetch_memory(cfg, obs[:200])  # query memory with what we see
                LOG.debug("Prefetch took %.1fs", time.monotonic() - t0)

                try:
                    reply = send_prompt(alone_prompt(obs, mem))
                    LOG.info("Alone reply (first 200 chars): %s", reply[:200])
                    # Store memory on behalf of the agent
                    summary = f"[ALONE] Scene: {obs[:200]} | Thought: {reply[:200]}"
                    post_store_memory(cfg, summary, ["observation", "alone"])
                except Exception as exc:
                    LOG.error("Alone prompt failed: %s", exc)

                turn_count += 1

                # Wait before next think cycle, checking for sound frequently
                deadline = time.monotonic() + think_interval
                while RUNNING and time.monotonic() < deadline:
                    # is_sound_active already samples for ~0.3s, so no
                    # extra sleep needed — just loop continuously.
                    if is_sound_active(cfg):
                        LOG.debug("Sound interrupted alone wait")
                        break

        except KeyboardInterrupt:
            LOG.info("KeyboardInterrupt — shutting down…")
            RUNNING = False
        except Exception as exc:
            LOG.error("Unexpected error in main loop: %s", exc, exc_info=True)
            time.sleep(5)

    # cleanup
    LOG.info("═══ Robot supervisor stopping… ═══")
    client.abort()
    stop_opencode_serve()
    LOG.info("═══ Robot supervisor stopped ═══")


# ──────────────────────────── signal handlers ────────────────────

def _handle_signal(signum, _frame):
    global RUNNING
    LOG.info("Received signal %d — shutting down", signum)
    RUNNING = False


# ──────────────────────────── entry point ────────────────────────

def main() -> None:
    global CONFIG_PATH

    parser = argparse.ArgumentParser(description="OpenCode Robot Supervisor")
    parser.add_argument("--config", type=str, default=str(CONFIG_PATH),
                        help="Path to config.yaml")
    parser.add_argument("--dry-run", action="store_true",
                        help="Print config and exit without running")
    args = parser.parse_args()

    CONFIG_PATH = Path(args.config)

    cfg = load_config()

    log_level = cfg.get("log_level", "INFO").upper()

    # Log to both console and OpenCode/log.out
    log_file = BASE_DIR / "log.out"
    file_handler = logging.FileHandler(log_file, mode="w", encoding="utf-8")
    file_handler.setLevel(logging.DEBUG)  # always capture everything in file
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

    if args.dry_run:
        LOG.info("── Dry-run mode ──")
        LOG.info("Config:\n%s", yaml.dump(cfg, default_flow_style=False))
        LOG.info("Sound test:")
        result = detect_sound_activity(
            threshold_rms=_deep_get(cfg, "sound", "threshold_rms", default=1200),
            sample_rate_hz=_deep_get(cfg, "sound", "sample_rate", default=16000),
            window_seconds=_deep_get(cfg, "sound", "window_seconds", default=0.15),
        )
        LOG.info("  %s", result)
        return

    signal.signal(signal.SIGTERM, _handle_signal)
    signal.signal(signal.SIGINT, _handle_signal)

    run_forever(cfg)


if __name__ == "__main__":
    main()
