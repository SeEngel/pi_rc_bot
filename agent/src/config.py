from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class OpenAIClientConfig:
	model: str
	base_url: str | None
	# Path to .env to load (Agent Framework supports env_file_path natively)
	env_file_path: str | None = None


@dataclass(frozen=True)
class MCPServerConfig:
	name: str
	url: str


def load_yaml(path: str) -> dict[str, Any]:
	try:
		import yaml  # type: ignore
	except Exception as exc:
		raise RuntimeError(
			"PyYAML is required to load config.yaml. Install it with: pip install pyyaml"
		) from exc

	with open(path, "r", encoding="utf-8") as f:
		data = yaml.safe_load(f) or {}
	if not isinstance(data, dict):
		raise ValueError(f"YAML root must be a mapping: {path}")
	return data


def resolve_repo_root(start: str) -> str:
	"""Return repo root by walking up until we find `.git` or `requirements.txt`."""
	here = os.path.abspath(start)
	cur = here
	for _ in range(8):
		if os.path.isdir(os.path.join(cur, ".git")):
			return cur
		if os.path.isfile(os.path.join(cur, "requirements.txt")) and os.path.isdir(os.path.join(cur, "services")):
			return cur
		parent = os.path.dirname(cur)
		if parent == cur:
			break
		cur = parent
	return os.path.abspath(start)
