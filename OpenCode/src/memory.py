"""Memory client — read/write memories via the MCP memory service."""

from __future__ import annotations

import logging

import requests

from .config import deep_get

LOG = logging.getLogger("supervisor")


def prefetch_memory(cfg: dict, query: str, top_n: int = 3) -> str:
    """Call the memory service directly and return formatted memories."""
    memory_url = deep_get(cfg, "mcp", "memory", default="http://127.0.0.1:8004")
    try:
        r = requests.post(
            f"{memory_url}/get_top_n_memory",
            json={"content": query, "top_n": top_n},
            timeout=20,
        )
        if r.ok:
            items = r.json().get("short_term_memory", []) or []
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


def store_memory(cfg: dict, content: str, tags: list[str]) -> bool:
    """Store a memory entry directly via HTTP."""
    memory_url = deep_get(cfg, "mcp", "memory", default="http://127.0.0.1:8004")
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
