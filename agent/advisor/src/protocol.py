from __future__ import annotations

import json
import os
import sys
import time
from dataclasses import dataclass
from typing import Any


def _now_iso() -> str:
	# ISO-ish without requiring datetime (keeps imports light)
	return time.strftime("%Y-%m-%dT%H:%M:%S", time.localtime())


def _truncate(value: Any, *, max_chars: int) -> Any:
	if value is None:
		return None
	if isinstance(value, (int, float, bool)):
		return value
	if isinstance(value, str):
		s = value
		if len(s) <= max_chars:
			return s
		return s[: max(0, max_chars - 12)] + "…(truncated)"
	if isinstance(value, dict):
		out: dict[str, Any] = {}
		for k, v in value.items():
			out[str(k)] = _truncate(v, max_chars=max_chars)
		return out
	if isinstance(value, list):
		return [_truncate(v, max_chars=max_chars) for v in value]
	# Fallback
	return _truncate(str(value), max_chars=max_chars)


@dataclass
class ProtocolLogger:
	enabled: bool
	log_path: str | None = None
	max_field_chars: int = 800

	_fh: Any = None

	def open(self) -> None:
		if not self.enabled:
			return
		if self.log_path:
			os.makedirs(os.path.dirname(os.path.abspath(self.log_path)), exist_ok=True)
			self._fh = open(self.log_path, "a", encoding="utf-8")

	def close(self) -> None:
		fh = self._fh
		self._fh = None
		if fh is not None:
			try:
				fh.flush()
			except Exception:
				pass
			try:
				fh.close()
			except Exception:
				pass

	def emit(self, event: str, **fields: Any) -> None:
		if not self.enabled:
			return

		payload: dict[str, Any] = {
			"ts": _now_iso(),
			"event": str(event),
		}
		for k, v in fields.items():
			payload[str(k)] = _truncate(v, max_chars=int(self.max_field_chars))

		line = json.dumps(payload, ensure_ascii=False, separators=(",", ":"))
		# Always stream to stdout for "live protocol".
		print(line, file=sys.stdout, flush=True)
		if self._fh is not None:
			try:
				self._fh.write(line + "\n")
				self._fh.flush()
			except Exception:
				# Don't break the advisor loop due to logging failures.
				pass
