from __future__ import annotations

import json
from typing import Any


async def call_mcp_tool_json(
	*,
	url: str,
	tool_name: str,
	timeout_seconds: float = 60.0,
	**kwargs: Any,
) -> dict[str, Any]:
	"""Call an MCP tool and parse the first text block as JSON.

	Many of our MCP tools return JSON bodies; `agent_framework` exposes those as a text payload.
	This helper returns a dict even if parsing fails.
	"""
	from agent_framework import MCPStreamableHTTPTool  # type: ignore

	async with MCPStreamableHTTPTool(name="mcp", url=url, timeout=timeout_seconds) as mcp:
		res = await mcp.call_tool(tool_name, **kwargs)
		text = None
		for item in res:
			if getattr(item, "type", None) == "text" and hasattr(item, "text"):
				text = str(getattr(item, "text"))
				break

	if text is None:
		return {"ok": False, "error": f"No text payload from MCP tool {tool_name}", "raw": repr(res)}

	# Attempt JSON parsing.
	try:
		obj = json.loads(text)
		if isinstance(obj, dict):
			return obj
		return {"ok": True, "value": obj, "text": text}
	except Exception:
		# Fallback: return as text
		return {"ok": True, "text": text}
