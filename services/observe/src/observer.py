from __future__ import annotations

import base64
import json
import os
import tempfile
import time
from dataclasses import dataclass
from io import BytesIO
from typing import Any


class ObserverError(RuntimeError):
	pass


@dataclass(frozen=True)
class ObserverSettings:
	engine: str = "openai"  # openai
	camera_backend: str = "picamera2"  # picamera2
	camera_width: int = 1280
	camera_height: int = 720
	camera_warmup_seconds: float = 2.0
	camera_save_last_path: str | None = None

	dry_run: bool = False
	default_question: str = "Describe what you see in this image."
	instructions: str | None = "You are a helpful assistant that describes what is visible in the image."

	openai_model: str = "gpt-4o"
	openai_base_url: str | None = None
	openai_temperature: float = 0.2
	openai_max_tokens: int = 200

	# Debugging helpers (only used by /observe/direction)
	debug_return_grid_image: bool = False
	debug_save_grid_image_path: str | None = None


class Observer:
	def __init__(self, settings: ObserverSettings):
		self.settings = settings
		self._available_reason: str | None = None
		self._camera: Any | None = None
		self._openai_client: Any | None = None

		if self.settings.dry_run:
			return

		engine = (self.settings.engine or "").strip().lower()
		if engine != "openai":
			raise ObserverError(f"Unsupported vision engine: {self.settings.engine!r}")

		self._init_camera()
		self._init_openai()

	@property
	def is_available(self) -> bool:
		return self._available_reason is None

	@property
	def unavailable_reason(self) -> str | None:
		return self._available_reason

	def close(self) -> None:
		"""Best-effort resource cleanup (camera)."""
		cam = self._camera
		self._camera = None
		try:
			if cam is not None and hasattr(cam, "stop"):
				cam.stop()
		except Exception:
			pass

	def observe_once(self, *, question: str | None = None) -> dict[str, Any]:
		"""Capture one image and return a dict with text + metadata."""

		if self.settings.dry_run:
			q = (question or self.settings.default_question or "").strip() or "(no question)"
			return {"text": f"(dry_run) {q}", "raw": {}, "model": None}

		if not self.is_available:
			raise ObserverError(self._available_reason or "Vision unavailable")

		jpeg_bytes, saved_path = self._capture_jpeg_bytes()
		res = self._ask_openai_vision(jpeg_bytes=jpeg_bytes, question=question)
		# Attach capture info for debugging.
		res["capture"] = {
			"backend": self.settings.camera_backend,
			"width": int(self.settings.camera_width),
			"height": int(self.settings.camera_height),
			"saved_path": saved_path,
		}
		return res

	def observe_direction_once(self, *, question: str | None = None) -> dict[str, Any]:
		"""Capture one image, overlay a 2x3 grid, and ask for a navigation direction.

		The model is instructed to select a single grid cell (row, col):
		- rows: 0=top ("far"), 1=bottom ("near")
		- cols: 0=left, 1=center/forward, 2=right

		The returned dict includes:
		- cell: {row, col}
		- action: mapped movement string
		- why: model rationale
		- fit: model's commentary on whether the action mapping makes sense
		- llm_text: raw model response
		"""

		if self.settings.dry_run:
			q = (question or "").strip() or "(default)"
			row, col = 1, 1
			action = _grid_action(row=row, col=col)
			return {
				"cell": {"row": row, "col": col},
				"action": action,
				"why": f"(dry_run) Selected center because: {q}",
				"fit": "(dry_run) Mapping not evaluated.",
				"llm_text": f"(dry_run) {{\"row\":{row},\"col\":{col},\"why\":\"...\",\"fit\":\"...\"}}",
				"model": None,
				"raw": {},
			}

		if not self.is_available:
			raise ObserverError(self._available_reason or "Vision unavailable")

		jpeg_bytes, saved_path = self._capture_jpeg_bytes()
		try:
			jpeg_with_grid = _overlay_red_grid_2x3(jpeg_bytes)
		except Exception as exc:
			raise ObserverError(f"Failed to overlay grid: {exc}") from exc

		grid_saved_path: str | None = None
		if self.settings.debug_save_grid_image_path:
			try:
				with open(self.settings.debug_save_grid_image_path, "wb") as f:
					f.write(jpeg_with_grid)
				grid_saved_path = str(self.settings.debug_save_grid_image_path)
			except Exception:
				grid_saved_path = None

		prompt = _build_direction_prompt(question)
		res = self._ask_openai_vision(jpeg_bytes=jpeg_with_grid, question=prompt)

		parsed, parse_error = _parse_direction_json(res.get("text"))
		row = parsed.get("row")
		col = parsed.get("col")
		why = parsed.get("why")
		fit = parsed.get("fit")

		# Default to center-forward if parsing fails.
		if not isinstance(row, int) or row not in (0, 1):
			row = 1
		if not isinstance(col, int) or col not in (0, 1, 2):
			col = 1

		action = _grid_action(row=row, col=col)
		out: dict[str, Any] = {
			"cell": {"row": row, "col": col},
			"action": action,
			"why": str(why or "").strip(),
			"fit": str(fit or "").strip(),
			"llm": {
				"row": row,
				"col": col,
				"why": str(why or "").strip(),
				"fit": str(fit or "").strip(),
			},
			"llm_raw_text": str(res.get("text") or ""),
			"model": res.get("model"),
			"raw": res.get("raw"),
			"grid": {
				"rows": 2,
				"cols": 3,
				"mapping": _GRID_MAPPING,
			},
			"debug": {
				"grid_image_saved_path": grid_saved_path,
			},
			"capture": {
				"backend": self.settings.camera_backend,
				"width": int(self.settings.camera_width),
				"height": int(self.settings.camera_height),
				"saved_path": saved_path,
			},
		}
		if self.settings.debug_return_grid_image:
			try:
				b64 = base64.b64encode(jpeg_with_grid).decode("ascii")
				out["debug"]["grid_image_data_url"] = f"data:image/jpeg;base64,{b64}"
			except Exception:
				pass
		if parse_error:
			out["parse_error"] = parse_error
		return out

	@staticmethod
	def from_config_dict(cfg: dict[str, Any]) -> "Observer":
		camera_cfg = (cfg or {}).get("camera", {}) if isinstance(cfg, dict) else {}
		vision_cfg = (cfg or {}).get("vision", {}) if isinstance(cfg, dict) else {}
		openai_cfg = (vision_cfg or {}).get("openai", {}) if isinstance(vision_cfg, dict) else {}
		debug_cfg = (vision_cfg or {}).get("debug", {}) if isinstance(vision_cfg, dict) else {}

		def _get_str(d: dict[str, Any], key: str, default: str) -> str:
			val = d.get(key, default)
			return default if val is None else str(val)

		def _get_bool(d: dict[str, Any], key: str, default: bool) -> bool:
			val = d.get(key, default)
			return bool(val) if val is not None else default

		def _get_int(d: dict[str, Any], key: str, default: int) -> int:
			val = d.get(key, default)
			try:
				return int(val)
			except Exception:
				return default

		def _get_float(d: dict[str, Any], key: str, default: float) -> float:
			val = d.get(key, default)
			try:
				return float(val)
			except Exception:
				return default

		base_url = _get_str(openai_cfg if isinstance(openai_cfg, dict) else {}, "base_url", "").strip()
		if base_url == "":
			base_url_val: str | None = None
		else:
			base_url_val = base_url

		save_last_path = _get_str(camera_cfg if isinstance(camera_cfg, dict) else {}, "save_last_path", "").strip()
		if save_last_path == "":
			save_last_val: str | None = None
		else:
			save_last_val = save_last_path

		dry_run_cfg = _get_bool(vision_cfg if isinstance(vision_cfg, dict) else {}, "dry_run", False)
		# Convenience override for development/testing.
		# If set, we will not access camera or external APIs.
		if str(os.getenv("OBSERVE_DRY_RUN") or "").strip().lower() in {"1", "true", "yes", "y", "on"}:
			dry_run_cfg = True

		debug_return_grid = _get_bool(debug_cfg if isinstance(debug_cfg, dict) else {}, "return_grid_image", False)
		debug_save_grid_path = _get_str(debug_cfg if isinstance(debug_cfg, dict) else {}, "save_grid_image_path", "").strip()
		if debug_save_grid_path == "":
			debug_save_grid_val: str | None = None
		else:
			debug_save_grid_val = debug_save_grid_path

		settings = ObserverSettings(
			engine=_get_str(vision_cfg if isinstance(vision_cfg, dict) else {}, "engine", "openai"),
			camera_backend=_get_str(camera_cfg if isinstance(camera_cfg, dict) else {}, "backend", "picamera2"),
			camera_width=_get_int(camera_cfg if isinstance(camera_cfg, dict) else {}, "width", 1280),
			camera_height=_get_int(camera_cfg if isinstance(camera_cfg, dict) else {}, "height", 720),
			camera_warmup_seconds=_get_float(camera_cfg if isinstance(camera_cfg, dict) else {}, "warmup_seconds", 2.0),
			camera_save_last_path=save_last_val,
			dry_run=dry_run_cfg,
			default_question=_get_str(vision_cfg if isinstance(vision_cfg, dict) else {}, "default_question", "Describe what you see in this image."),
			instructions=_get_str(vision_cfg if isinstance(vision_cfg, dict) else {}, "instructions", "").strip() or None,
			openai_model=_get_str(openai_cfg if isinstance(openai_cfg, dict) else {}, "model", "gpt-4o"),
			openai_base_url=base_url_val,
			openai_temperature=_get_float(openai_cfg if isinstance(openai_cfg, dict) else {}, "temperature", 0.2),
			openai_max_tokens=_get_int(openai_cfg if isinstance(openai_cfg, dict) else {}, "max_tokens", 200),
			debug_return_grid_image=debug_return_grid,
			debug_save_grid_image_path=debug_save_grid_val,
		)
		return Observer(settings)

	def _init_camera(self) -> None:
		backend = (self.settings.camera_backend or "").strip().lower()
		if backend != "picamera2":
			self._available_reason = f"Unsupported camera backend: {self.settings.camera_backend!r}"
			return

		try:
			from picamera2 import Picamera2  # type: ignore
		except Exception as exc:
			self._available_reason = f"picamera2 not available: {exc}"
			return

		try:
			cam = Picamera2()
			config = cam.create_still_configuration(main={"size": (int(self.settings.camera_width), int(self.settings.camera_height))})
			cam.configure(config)
			cam.start()
			warm = float(self.settings.camera_warmup_seconds or 0.0)
			if warm > 0:
				time.sleep(warm)
			self._camera = cam
		except Exception as exc:
			self._available_reason = f"Camera init failed: {exc}"
			self._camera = None

	def _init_openai(self) -> None:
		api_key = os.getenv("OPENAI_API_KEY")
		if not api_key:
			self._available_reason = "Missing OPENAI_API_KEY"
			return

		try:
			from openai import OpenAI  # type: ignore
		except Exception as exc:
			self._available_reason = f"openai package not available: {exc}"
			return

		kwargs: dict[str, Any] = {"api_key": api_key}
		base_url = (self.settings.openai_base_url or "").strip()
		# If base_url is empty, use the OpenAI default base URL.
		# If set, it must point to an OpenAI-compatible API (often ending with /v1).
		if base_url:
			kwargs["base_url"] = base_url

		try:
			self._openai_client = OpenAI(**kwargs)
		except Exception as exc:
			self._available_reason = f"OpenAI client init failed: {exc}"
			self._openai_client = None

	def _capture_jpeg_bytes(self) -> tuple[bytes, str | None]:
		cam = self._camera
		if cam is None:
			raise ObserverError(self._available_reason or "Camera not initialized")

		# Capture to a temporary file (Picamera2 writes files efficiently).
		fd, tmp_path = tempfile.mkstemp(prefix="pi_rc_bot_observe_", suffix=".jpg")
		os.close(fd)
		saved_path: str | None = None
		try:
			cam.capture_file(tmp_path)
			with open(tmp_path, "rb") as f:
				jpeg_bytes = f.read()

			if self.settings.camera_save_last_path:
				try:
					# Best-effort copy for debugging.
					import shutil

					shutil.copyfile(tmp_path, self.settings.camera_save_last_path)
					saved_path = str(self.settings.camera_save_last_path)
				except Exception:
					saved_path = None
			return jpeg_bytes, saved_path
		finally:
			try:
				os.remove(tmp_path)
			except Exception:
				pass

	def _ask_openai_vision(self, *, jpeg_bytes: bytes, question: str | None) -> dict[str, Any]:
		client = self._openai_client
		if client is None:
			raise ObserverError(self._available_reason or "OpenAI client not initialized")

		q = (question or self.settings.default_question or "").strip()
		if not q:
			q = "Describe what you see in this image."

		b64 = base64.b64encode(jpeg_bytes).decode("ascii")
		data_url = f"data:image/jpeg;base64,{b64}"

		messages: list[dict[str, Any]] = []
		if self.settings.instructions:
			messages.append({"role": "system", "content": str(self.settings.instructions)})
		messages.append(
			{
				"role": "user",
				"content": [
					{"type": "text", "text": q},
					{"type": "image_url", "image_url": {"url": data_url}},
				],
			}
		)

		model = (self.settings.openai_model or "").strip() or "gpt-4o"
		try:
			resp = client.chat.completions.create(
				model=model,
				messages=messages,
				temperature=float(self.settings.openai_temperature),
				max_tokens=int(self.settings.openai_max_tokens),
			)
		except Exception as exc:
			raise ObserverError(f"Vision request failed: {exc}") from exc

		text = ""
		try:
			choice0 = resp.choices[0]
			msg = getattr(choice0, "message", None)
			text = str(getattr(msg, "content", "") or "")
		except Exception:
			text = ""

		raw: Any = {}
		try:
			raw = resp.model_dump()  # type: ignore[attr-defined]
		except Exception:
			try:
				raw = dict(resp)  # type: ignore[arg-type]
			except Exception:
				raw = {}

		return {"text": text, "model": model, "raw": raw}


_GRID_MAPPING: dict[str, str] = {
	"(0,0)": "go far left",
	"(0,1)": "go far forward",
	"(0,2)": "go far right",
	"(1,0)": "left",
	"(1,1)": "forward",
	"(1,2)": "right",
}


def _grid_action(*, row: int, col: int) -> str:
	key = f"({int(row)},{int(col)})"
	return _GRID_MAPPING.get(key, "forward")


def _build_direction_prompt(user_question: str | None) -> str:
	q = (user_question or "").strip()
	# Keep prompt explicit and parseable. The service will parse JSON out of the response.
	base = (
		"You will receive a photo with a red 2x3 grid overlay. "
		"Pick exactly ONE grid cell (row, col) that the robot should move toward.\n\n"
		"Grid indexing (normalized):\n"
		"- rows: 0 = top row (far), 1 = bottom row (near)\n"
		"- cols: 0 = left, 1 = center/forward, 2 = right\n\n"
		"Movement mapping (hard-coded by the service):\n"
		"- (0,0) -> go far left\n"
		"- (0,1) -> go far forward\n"
		"- (0,2) -> go far right\n"
		"- (1,0) -> left\n"
		"- (1,1) -> forward\n"
		"- (1,2) -> right\n\n"
		"Return ONLY a JSON object with exactly these keys:\n"
		"- row: integer (0 or 1)\n"
		"- col: integer (0, 1, or 2)\n"
		"- why: short explanation (1-3 sentences) referencing what you see\n"
		"- fit: say whether the mapped action string makes sense in context (or why not)\n"
	)
	if q:
		base += f"\nAdditional goal/question: {q}\n"
	return base


def _parse_direction_json(text: Any) -> tuple[dict[str, Any], str | None]:
	"""Parse the model response into {row, col, why, fit}.

	Returns (parsed_dict, parse_error). If parsing fails, parsed_dict may be empty.
	"""
	if text is None:
		return {}, "empty response"
	if not isinstance(text, str):
		text = str(text)
	s = text.strip()
	if not s:
		return {}, "empty response"

	def _try_load(candidate: str) -> tuple[dict[str, Any] | None, str | None]:
		try:
			obj = json.loads(candidate)
		except Exception as exc:
			return None, str(exc)
		if not isinstance(obj, dict):
			return None, "not a JSON object"
		return obj, None

	# 1) Whole-string JSON.
	obj, err = _try_load(s)
	if obj is None:
		# 2) Extract first {...} block.
		start = s.find("{")
		end = s.rfind("}")
		if start != -1 and end != -1 and end > start:
			obj, err = _try_load(s[start : end + 1])
		else:
			err = err or "no JSON object found"

	if obj is None:
		return {}, f"failed to parse JSON: {err}"

	# Normalize possible shapes.
	row = obj.get("row")
	col = obj.get("col")
	# Some models might return cell: [row, col] or cell: {row, col}
	cell = obj.get("cell")
	if (row is None or col is None) and cell is not None:
		try:
			if isinstance(cell, (list, tuple)) and len(cell) >= 2:
				row = cell[0]
				col = cell[1]
			elif isinstance(cell, dict):
				row = cell.get("row")
				col = cell.get("col")
		except Exception:
			pass

	def _to_int(v: Any) -> int | None:
		try:
			if isinstance(v, bool):
				return None
			return int(v)
		except Exception:
			return None

	parsed: dict[str, Any] = {
		"row": _to_int(row),
		"col": _to_int(col),
		"why": obj.get("why") or obj.get("reason") or obj.get("rationale"),
		"fit": obj.get("fit") or obj.get("mapping_fit") or obj.get("interpretation"),
	}
	return parsed, None


def _overlay_red_grid_2x3(jpeg_bytes: bytes) -> bytes:
	"""Overlay a red 2x3 grid (plus optional cell labels) onto a JPEG."""
	try:
		from PIL import Image, ImageDraw, ImageFont  # type: ignore
	except Exception as exc:
		raise RuntimeError(
			"Pillow is required to draw the grid overlay. "
			"Install it with: pip install -r services/observe/requirements.txt"
		) from exc

	img = Image.open(BytesIO(jpeg_bytes)).convert("RGB")
	w, h = img.size
	draw = ImageDraw.Draw(img)

	rows, cols = 2, 3
	red = (255, 0, 0)
	line_w = max(2, int(round(min(w, h) * 0.006)))

	# Outer border
	draw.rectangle([(0, 0), (w - 1, h - 1)], outline=red, width=line_w)
	# Vertical lines
	for c in range(1, cols):
		x = int(round(w * c / cols))
		draw.line([(x, 0), (x, h)], fill=red, width=line_w)
	# Horizontal lines
	for r in range(1, rows):
		y = int(round(h * r / rows))
		draw.line([(0, y), (w, y)], fill=red, width=line_w)

	# Cell labels to reduce ambiguity for the vision model.
	try:
		font = ImageFont.load_default()
		exc_font: Exception | None = None
	except Exception as exc:
		font = None  # type: ignore[assignment]
		exc_font = exc

	margin = line_w * 2
	for r in range(rows):
		for c in range(cols):
			x0 = int(round(w * c / cols)) + margin
			y0 = int(round(h * r / rows)) + margin
			label = f"({r},{c})"
			try:
				# stroke_* supported on modern Pillow; fallback if unsupported.
				draw.text(
					(x0, y0),
					label,
					fill=red,
					font=font,
					stroke_width=max(1, line_w // 2),
					stroke_fill=(0, 0, 0),
				)
			except TypeError:
				draw.text((x0, y0), label, fill=red, font=font)
			except Exception:
				# If font drawing fails, skip labels rather than failing the request.
				pass

	out = BytesIO()
	img.save(out, format="JPEG", quality=90, optimize=True)
	return out.getvalue()
