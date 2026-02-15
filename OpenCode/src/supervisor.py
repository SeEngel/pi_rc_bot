"""Robot Supervisor — orchestrates alone-mode and interaction-mode loops."""

from __future__ import annotations

import logging
import os
import time
from concurrent.futures import ThreadPoolExecutor

import requests

from .client import OpenCodeClient
from .config import deep_get
from .memory import prefetch_memory, store_memory
from .process_manager import (
    CodexAgentProcess,
    MyToolsManager,
    OpenCodeProcess,
)
from .prompts import alone_prompt, interaction_prompt
from .sensor import (
    is_sound_active,
    listen_once,
    listen_via_mcp,
    prefetch_observe,
)

LOG = logging.getLogger("supervisor")


class Supervisor:
    """The main run-forever loop that alternates between alone- and interaction-mode."""

    def __init__(self, cfg: dict, base_dir):
        self.cfg = cfg
        self.base_dir = base_dir
        self.running = True

        # Sub-systems
        self._oc_process = OpenCodeProcess(base_dir)
        self._tools = MyToolsManager(base_dir)
        self._codex = CodexAgentProcess(base_dir)

        # Speech-guard state
        self._last_speech_ended: float = 0.0

    # ════════════════════════════════════════════════════════════
    #  Public API
    # ════════════════════════════════════════════════════════════

    def shutdown(self) -> None:
        self.running = False

    def run(self) -> None:
        """Boot all child processes and enter the main loop."""
        cfg = self.cfg
        oc_cfg = cfg.get("opencode", {})
        host = oc_cfg.get("host", "127.0.0.1")
        port = oc_cfg.get("port", 4096)
        timeout = oc_cfg.get("request_timeout", 120)
        agent = os.environ.get("OPENCODE_AGENT", "robot")
        model = os.environ.get("OPENCODE_MODEL")
        think_interval = deep_get(cfg, "alone", "think_interval_seconds", default=20.0)

        # 1) OpenCode server
        client = OpenCodeClient(host=host, port=port, timeout=timeout,
                                agent=agent, model=model)
        if not client.healthy():
            LOG.info("OpenCode server not running — starting…")
            self._oc_process.start(host, port)
            client.wait_for_server(retries=30, delay=2.0)
        else:
            LOG.info("OpenCode server already running at %s", client.base)

        # 2) Codex agent
        self._codex.start()
        codex_ready = self._codex.wait_until_ready()
        if codex_ready:
            self._codex.register_with_opencode(cfg)

        # 3) Custom MCP tools — start what we can, repair broken ones in background
        self._tools.start_all(cfg)
        if codex_ready:
            self._codex.queue_repairs(cfg, self._tools)

        # Convenience closure
        def send_prompt(prompt: str) -> str:
            return client.extract_text(client.send(prompt))

        turn_count = 0
        session_rotate_every = 10

        LOG.info("═══ Robot supervisor started ═══")
        LOG.info("  think_interval = %.1fs", think_interval)
        LOG.info("  sound threshold = %d RMS",
                 deep_get(cfg, "sound", "threshold_rms", default=1200))

        # ── main loop ──────────────────────────────────────────

        while self.running:
            try:
                # Rotate session periodically
                if turn_count > 0 and turn_count % session_rotate_every == 0:
                    LOG.info("Rotating session after %d turns", turn_count)
                    client.new_session()
                    orphans = self._tools.cleanup_orphans(cfg)
                    if orphans:
                        LOG.info("Periodic cleanup removed %d orphaned tool(s)", len(orphans))

                # Wait for robot to finish speaking
                if self._is_robot_speaking() or self._in_speech_cooldown():
                    LOG.debug("🔇 Robot speaking or cooldown — skipping sound check")
                    self._wait_for_speech_done()

                # Check for sound
                sound_active = is_sound_active(cfg)

                if sound_active:
                    self._handle_interaction(cfg, send_prompt, client)
                else:
                    self._handle_alone(cfg, send_prompt, think_interval)

                turn_count += 1

            except KeyboardInterrupt:
                LOG.info("KeyboardInterrupt — shutting down…")
                self.running = False
            except Exception as exc:
                LOG.error("Unexpected error in main loop: %s", exc, exc_info=True)
                time.sleep(5)

        # cleanup
        LOG.info("═══ Robot supervisor stopping… ═══")
        client.abort()
        self._codex.stop()
        self._tools.stop_all()
        self._oc_process.stop()
        LOG.info("═══ Robot supervisor stopped ═══")

    # ════════════════════════════════════════════════════════════
    #  Interaction mode
    # ════════════════════════════════════════════════════════════

    def _handle_interaction(self, cfg, send_prompt, client) -> None:
        LOG.info("🎤 Sound detected — entering interaction mode")

        transcript = listen_via_mcp(cfg)
        if transcript is None:
            LOG.info("No usable transcript — back to alone mode")
            return

        if self._has_stop_word(transcript):
            self._respond_stop(send_prompt, transcript)
            return

        reply = self._do_interaction_turn(cfg, send_prompt, transcript)

        # Post-interaction listen window
        self._wait_for_speech_done()
        post_transcript = self._post_interaction_listen(cfg)

        if not post_transcript:
            return

        # Re-enter interaction with already-captured transcript
        if self._has_stop_word(post_transcript):
            self._respond_stop(send_prompt, post_transcript)
            return

        self._do_interaction_turn(cfg, send_prompt, post_transcript)

    def _do_interaction_turn(self, cfg, send_prompt, transcript: str) -> str:
        obs, mem = self._prefetch(cfg, transcript)

        broken = self._tools.check_health(cfg)
        prompt = interaction_prompt(transcript, obs, mem)
        if broken:
            prompt += self._broken_tools_addendum(broken)

        try:
            reply = send_prompt(prompt)
            LOG.info("Interaction reply (first 200 chars): %s", reply[:200])
            summary = (
                f"[INTERACTION] Human: {transcript[:200]} "
                f"| Scene: {obs[:150]} | Reply: {reply[:200]}"
            )
            store_memory(cfg, summary, ["interaction", "human"])
            return reply
        except Exception as exc:
            LOG.error("Interaction prompt failed: %s", exc)
            return ""

    def _post_interaction_listen(self, cfg) -> str | None:
        post_listen_sec = deep_get(
            cfg, "interaction", "post_reply_listen_seconds", default=6.0,
        )
        LOG.info("👂 Post-interaction — immediate listen (%.1fs window)…", post_listen_sec)

        listen_url = deep_get(cfg, "mcp", "listen", default="http://127.0.0.1:8002")
        min_chars = deep_get(cfg, "interaction", "min_transcript_chars", default=3)

        text = listen_once(
            listen_url,
            pause_seconds=2.0,
            timeout=post_listen_sec + 15,
            max_wait_for_speech_seconds=post_listen_sec,
        )
        if text and len(text) >= min_chars:
            LOG.info("🎤 Human continued: %s", text[:120])
            return text
        return None

    # ════════════════════════════════════════════════════════════
    #  Alone mode
    # ════════════════════════════════════════════════════════════

    def _handle_alone(self, cfg, send_prompt, think_interval: float) -> None:
        LOG.debug("😶 Quiet — alone mode")

        obs, mem = self._prefetch(cfg, "what is around me, environment, exploration")

        broken = self._tools.check_health(cfg)
        prompt = alone_prompt(obs, mem)
        if broken:
            prompt += (
                "\n\n## ⚠️ BROKEN TOOLS\n"
                f"The following custom tools have errors: {', '.join(broken)}\n"
                "Call robot_codex → repair_tool with a description of the problem to fix them.\n"
            )

        try:
            reply = send_prompt(prompt)
            LOG.info("Alone reply (first 200 chars): %s", reply[:200])
            summary = f"[ALONE] Scene: {obs[:200]} | Thought: {reply[:200]}"
            store_memory(cfg, summary, ["observation", "alone"])
        except Exception as exc:
            LOG.error("Alone prompt failed: %s", exc)

        self._wait_for_speech_done()

        # Wait before next think cycle, checking for sound frequently
        deadline = time.monotonic() + think_interval
        while self.running and time.monotonic() < deadline:
            if self._is_robot_speaking() or self._in_speech_cooldown():
                time.sleep(0.3)
                continue
            if is_sound_active(cfg):
                LOG.debug("Sound interrupted alone wait")
                break

    # ════════════════════════════════════════════════════════════
    #  Helpers
    # ════════════════════════════════════════════════════════════

    def _prefetch(self, cfg, query: str) -> tuple[str, str]:
        """Pre-fetch observe + memory in parallel.  Returns (obs, mem)."""
        t0 = time.monotonic()
        with ThreadPoolExecutor(max_workers=2) as pool:
            f_obs = pool.submit(prefetch_observe, cfg)
            f_mem = pool.submit(prefetch_memory, cfg, query)
            try:
                obs = f_obs.result(timeout=30)
            except Exception as exc:
                LOG.warning("Prefetch observe timed out / failed: %s", exc)
                obs = "(observe unavailable)"
            try:
                mem = f_mem.result(timeout=30)
            except Exception as exc:
                LOG.warning("Prefetch memory timed out / failed: %s", exc)
                mem = "(no memories found)"
        LOG.debug("Prefetch took %.1fs", time.monotonic() - t0)
        return obs, mem

    def _has_stop_word(self, text: str) -> bool:
        stop_words = deep_get(self.cfg, "interaction", "stop_words", default=[])
        lower = text.lower()
        return any(w.lower() in lower for w in stop_words)

    def _respond_stop(self, send_prompt, transcript: str) -> None:
        LOG.info("Stop word detected in: %s", transcript[:60])
        try:
            send_prompt(
                f'[INTERACTION] The human said: "{transcript}"\n'
                "They want you to stop. Acknowledge briefly via robot_speak → speak."
            )
            store_memory(
                self.cfg,
                f"[INTERACTION] Human said stop: {transcript[:200]}",
                ["interaction", "stop"],
            )
        except Exception as exc:
            LOG.error("Stop-word response failed: %s", exc)

    @staticmethod
    def _broken_tools_addendum(broken: list[str]) -> str:
        return (
            "\n\n## ⚠️ BROKEN TOOLS\n"
            f"The following custom tools have errors: {', '.join(broken)}\n"
            "Tell the human: 'Ein Tool hat einen Fehler, ich kümmere mich darum!' "
            "Then call robot_codex → repair_tool with a description of the problem. "
            "After the repair, retry the action.\n"
        )

    # ── speech guard ────────────────────────────────────────────

    def _is_robot_speaking(self) -> bool:
        speak_url = deep_get(self.cfg, "mcp", "speak", default="http://127.0.0.1:8001")
        try:
            r = requests.get(f"{speak_url}/status", timeout=3)
            if r.ok:
                return r.json().get("speaking", False)
        except Exception:
            pass
        return False

    def _in_speech_cooldown(self) -> bool:
        cooldown = deep_get(self.cfg, "sound", "speech_cooldown_seconds", default=0.4)
        return (time.monotonic() - self._last_speech_ended) < cooldown

    def _wait_for_speech_done(self, timeout: float = 30.0, grace: float = 2.0) -> None:
        cooldown = deep_get(self.cfg, "sound", "speech_cooldown_seconds", default=0.4)
        deadline = time.monotonic() + timeout

        # Phase 1: wait up to *grace* seconds for TTS to start
        grace_deadline = time.monotonic() + grace
        was_speaking = False
        while time.monotonic() < grace_deadline:
            if self._is_robot_speaking():
                was_speaking = True
                break
            time.sleep(0.1)

        # Phase 2: wait for TTS to finish
        if was_speaking:
            while time.monotonic() < deadline:
                if self._is_robot_speaking():
                    time.sleep(0.1)
                else:
                    break
            self._last_speech_ended = time.monotonic()
            LOG.debug("Robot finished speaking — cooldown %.1fs", cooldown)
            time.sleep(cooldown)
