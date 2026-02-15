"""OpenCode headless-server HTTP client."""

from __future__ import annotations

import json
import logging
import time
from typing import Any

import requests

LOG = logging.getLogger("supervisor")


class OpenCodeClient:
    """Thin wrapper around the OpenCode headless HTTP server API."""

    def __init__(
        self,
        host: str = "127.0.0.1",
        port: int = 4096,
        timeout: int = 180,
        agent: str = "robot",
        model: str | None = None,
    ):
        self.base = f"http://{host}:{port}"
        self.timeout = timeout
        self.agent = agent
        self.model = model
        self._session_id: str | None = None

    # ── health ──────────────────────────────────────────────────

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
        raise RuntimeError(
            f"OpenCode server at {self.base} not reachable after {retries} attempts"
        )

    # ── session management ──────────────────────────────────────

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
        """Force a fresh session (e.g. after many turns)."""
        self._session_id = self._create_session()
        return self._session_id

    # ── messaging ───────────────────────────────────────────────

    def send(self, prompt: str) -> dict:
        """Send a prompt and wait for the assistant reply (raw JSON)."""
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
        LOG.debug(
            "Response status=%d, content-length=%s",
            r.status_code,
            r.headers.get("content-length", "?"),
        )
        r.raise_for_status()

        raw = r.text.strip()
        if not raw:
            LOG.warning("Empty response body from OpenCode server")
            return {"parts": [], "info": {}}
        try:
            return r.json()
        except Exception as exc:
            LOG.error(
                "Failed to parse response as JSON: %s — raw (first 500): %s",
                exc,
                raw[:500],
            )
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

    # ── helpers ─────────────────────────────────────────────────

    @staticmethod
    def extract_text(response: dict) -> str:
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
        info = response.get("info", {})
        if isinstance(info, dict) and info.get("content"):
            return str(info["content"])
        return json.dumps(response, indent=2, ensure_ascii=False)[:2000]
