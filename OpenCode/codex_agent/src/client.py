"""Codex-agent OpenCode HTTP client."""

from __future__ import annotations

import logging
import re
import time

import requests

LOG = logging.getLogger("codex-agent")

_SECRET_RE = re.compile(r"sk-[A-Za-z0-9]{10,}")


def redact_secrets(text: str) -> str:
    """Best-effort redaction for logs (avoid leaking API keys)."""
    if not text:
        return text
    text = _SECRET_RE.sub("sk-***", text)
    text = re.sub(r"(OPENAI_API_KEY\s*=\s*)(\S+)", r"\1***", text)
    return text


class CodexClient:
    """Send prompts to the dedicated codex OpenCode instance."""

    def __init__(self, host: str = "127.0.0.1", port: int = 4097,
                 timeout: int = 600):
        self.base = f"http://{host}:{port}"
        self.timeout = timeout
        self._session_id: str | None = None
        self._last_tool_calls: list[str] = []
        self._last_text: str = ""

    # ── session management ──────────────────────────────────────

    @property
    def session_id(self) -> str:
        if self._session_id is None:
            r = requests.post(
                f"{self.base}/session",
                json={"title": "codex"},
                timeout=self.timeout,
            )
            r.raise_for_status()
            data = r.json()
            self._session_id = (
                data.get("id") or data.get("ID") or data.get("sessionID")
            )
        return self._session_id  # type: ignore[return-value]

    def new_session(self) -> None:
        self._session_id = None
        _ = self.session_id

    # ── send prompt ─────────────────────────────────────────────

    def send(self, prompt: str) -> str:
        """Send a prompt and return the extracted text reply."""
        url = f"{self.base}/session/{self.session_id}/message"
        r = requests.post(
            url,
            json={
                "parts": [{"type": "text", "text": prompt}],
                "agent": "codex",
            },
            timeout=self.timeout,
        )
        r.raise_for_status()
        raw = r.text.strip()
        if not raw:
            return ""
        try:
            data = r.json()
        except Exception:
            return raw[:2000]

        parts = data.get("parts", [])
        texts: list[str] = []
        tool_calls: list[str] = []

        for p in parts:
            if not isinstance(p, dict):
                continue
            if p.get("type") == "text":
                texts.append(p.get("text", ""))
                continue
            # Tool call
            t = str(p.get("type", ""))
            if t in {
                "tool-invocation", "tool_invocation",
                "tool-call", "tool_call", "tool",
            } or (
                ("toolName" in p or "tool" in p)
                and any(k in p for k in ("input", "args", "parameters"))
            ):
                tool_name = (
                    p.get("toolName") or p.get("tool") or p.get("name") or "unknown"
                )
                tool_input = (
                    p.get("input") or p.get("args") or p.get("parameters") or {}
                )
                if isinstance(tool_input, dict):
                    cmd = tool_input.get("command") or tool_input.get("cmd") or ""
                    file_path = tool_input.get("filePath") or tool_input.get("file") or ""
                    summary = f"🔧 {tool_name}: {cmd or file_path or tool_input}"
                else:
                    summary = f"🔧 {tool_name}: {tool_input}"
                summary = redact_secrets(str(summary))[:220]
                tool_calls.append(summary)
                LOG.info("AI tool call: %s", summary)
                continue
            if "content" in p:
                texts.append(str(p["content"]))

        self._last_tool_calls = tool_calls
        joined = "\n".join(texts) if texts else raw[:2000]
        self._last_text = joined
        return joined

    @property
    def last_tool_calls(self) -> list[str]:
        return self._last_tool_calls

    @property
    def last_text(self) -> str:
        return self._last_text


# ── wait for server ─────────────────────────────────────────────

def wait_for_opencode(host: str, port: int, retries: int = 30,
                      delay: float = 2.0) -> bool:
    base = f"http://{host}:{port}"
    for i in range(retries):
        try:
            r = requests.get(f"{base}/global/health", timeout=5)
            if r.ok and r.json().get("healthy", False):
                LOG.info("Codex OpenCode healthy at %s", base)
                return True
        except Exception:
            pass
        LOG.info("Waiting for codex OpenCode (%d/%d)...", i + 1, retries)
        time.sleep(delay)
    LOG.error("Codex OpenCode not reachable after %d attempts", retries)
    return False
