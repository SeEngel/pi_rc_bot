from __future__ import annotations

import os
from dataclasses import dataclass

from agent.src.base_agent import BaseAgentConfig, BaseWorkbenchChatAgent
from agent.src.config import MCPServerConfig, OpenAIClientConfig, load_yaml, resolve_repo_root


@dataclass(frozen=True)
class HeadAgentSettings:
	name: str
	instructions: str
	openai_model: str
	openai_base_url: str | None
	head_mcp_url: str
	env_file_path: str | None

	default_pan_deg: int
	default_tilt_deg: int


class HeadAgent(BaseWorkbenchChatAgent):
	"""Agent Framework wrapper for the `services/head` MCP server."""

	settings: HeadAgentSettings

	def __init__(self, settings: HeadAgentSettings):
		self.settings = settings
		agent_cfg = BaseAgentConfig(
			name=settings.name,
			instructions=settings.instructions,
			openai=OpenAIClientConfig(
				model=settings.openai_model,
				base_url=settings.openai_base_url,
				env_file_path=settings.env_file_path,
			),
			mcp_servers=[MCPServerConfig(name="pi_rc_bot Head MCP", url=settings.head_mcp_url)],
		)
		super().__init__(agent_cfg)

	@classmethod
	def from_config_yaml(cls, path: str) -> "HeadAgent":
		cfg = load_yaml(path)
		here = os.path.dirname(os.path.abspath(path))
		repo_root = resolve_repo_root(here)
		env_file_path = os.path.join(repo_root, ".env") if os.path.isfile(os.path.join(repo_root, ".env")) else None

		agent_cfg = cfg.get("agent", {}) if isinstance(cfg, dict) else {}
		openai_cfg = cfg.get("openai", {}) if isinstance(cfg, dict) else {}
		mcp_cfg = cfg.get("mcp", {}) if isinstance(cfg, dict) else {}

		name = str((agent_cfg or {}).get("name") or "HeadAgent")
		instructions = str((agent_cfg or {}).get("instructions") or "")
		default_pan_deg = int((agent_cfg or {}).get("default_pan_deg") or 0)
		default_tilt_deg = int((agent_cfg or {}).get("default_tilt_deg") or 0)

		openai_model = str((openai_cfg or {}).get("model") or "gpt-4o-mini")
		base_url_raw = str((openai_cfg or {}).get("base_url") or "").strip()
		openai_base_url = base_url_raw or None

		head_mcp_url = str((mcp_cfg or {}).get("head_mcp_url") or "http://127.0.0.1:8606/mcp").strip()

		if not instructions.strip():
			instructions = (
				"You are the robot's head (pan/tilt) controller. You MUST use MCP tools. "
				"Never claim you moved the head unless you called the tool.\n\n"
				"Tools:\n"
				"- set_angles {pan_deg?:int, tilt_deg?:int}\n"
				"- center {}\n"
				"- scan {pattern?:string, duration_s?:number, step_deg?:int, step_delay_s?:number}\n"
				"- stop {}\n"
				"- status {}\n\n"
				"If a tool fails, retry ONCE with minimal inputs. Then return ONLY the tool result."
			)

		return cls(
			HeadAgentSettings(
				name=name,
				instructions=instructions,
				openai_model=openai_model,
				openai_base_url=openai_base_url,
				head_mcp_url=head_mcp_url,
				env_file_path=env_file_path,
				default_pan_deg=default_pan_deg,
				default_tilt_deg=default_tilt_deg,
			)
		)

	async def set_angles(self, *, pan_deg: int | None = None, tilt_deg: int | None = None) -> str:
		user_input = (
			"Call MCP tool `set_angles` now. Provide pan_deg and tilt_deg if given. Return ONLY the tool result.\n\n"
			f"pan_deg={pan_deg if pan_deg is not None else ''}\n"
			f"tilt_deg={tilt_deg if tilt_deg is not None else ''}"
		)
		return str(await self.run(user_input))

	async def center(self) -> str:
		return str(await self.run("Call MCP tool `center` now. Return ONLY the tool result."))

	async def scan(self, *, pattern: str | None = None, duration_s: float | None = None) -> str:
		user_input = (
			"Call MCP tool `scan` now. Provide pattern and duration_s if given. Return ONLY the tool result.\n\n"
			f"pattern={pattern or ''}\n"
			f"duration_s={duration_s if duration_s is not None else ''}"
		)
		return str(await self.run(user_input))

	async def stop(self) -> str:
		return str(await self.run("Call MCP tool `stop` now. Return ONLY the tool result."))

	async def status(self) -> str:
		return str(await self.run("Call MCP tool `status` now. Return ONLY the tool result."))
