from __future__ import annotations

import os
import shutil
import subprocess
import sys
import tempfile
from dataclasses import dataclass
from typing import Any


class SpeakerError(RuntimeError):
    pass


@dataclass(frozen=True)
class SpeakerSettings:
    backend: str = "robot_hat"  # robot_hat | system (only relevant for pico2wave/espeak)
    engine: str = "openai"  # openai | piper | pico2wave | espeak
    language: str = "de-DE"  # e.g. en-US, en-GB, de-DE, es-ES, fr-FR, it-IT

    # Common behavior
    print_text: bool = True
    dry_run: bool = False

    # OS mixer volume (Linux only for now)
    # If set, attempts to set the system output volume to this percentage (0..100).
    # On Raspberry Pi OS this usually affects the actual loudness you hear.
    system_volume_percent: int | None = None

    # system/pico2wave
    player_cmd: str = "aplay"  # aplay | paplay

    # system/pico2wave loudness (optional)
    # - If `pico2wave_gain_db` != 0, we try to amplify the generated WAV using
    #   `sox` (preferred) or `ffmpeg` before playing it.
    # - `paplay_volume` is only used when player_cmd == "paplay".
    pico2wave_gain_db: float = 0.0
    paplay_volume: int | None = None  # 0..65536 (65536 = 100%)

    # system/espeak
    espeak_voice: str | None = None  # e.g. "de", "en", "en-us"
    espeak_speed: int = 160  # words per minute-ish
    espeak_amplitude: int = 120  # 0..200

    # robot_hat/OpenAI_TTS
    openai_model: str = "gpt-4o-mini-tts"
    openai_voice: str = "alloy"
    openai_instructions: str | None = None
    openai_stream: bool = True
    openai_gain: float = 1.5

    # robot_hat/Piper
    piper_model: str | None = None
    piper_stream: bool = False
    piper_gain: float = 1.0


class Speaker:
    """Simple Text-to-Speech wrapper.

    - Language is fixed at initialization (like `tts.set_lang('de-DE')` in the example).
    - Supports `robot_hat.tts` engines (via picar-x) and system binaries as fallback.
    """

    def __init__(self, settings: SpeakerSettings):
        self.settings = settings
        self._tts_obj: Any | None = None
        self._available_reason: str | None = None
        self._warned_volume_unsupported: bool = False

        backend = settings.backend.strip().lower()
        engine = settings.engine.strip().lower()

        if backend not in {"robot_hat", "system"}:
            raise SpeakerError(f"Unsupported backend: {settings.backend!r}")
        if engine not in {"pico2wave", "espeak", "piper", "openai"}:
            raise SpeakerError(f"Unsupported engine: {settings.engine!r}")

        # Piper/OpenAI_TTS are provided by robot_hat (and enable the speaker switch).
        if engine in {"piper", "openai"}:
            self._init_robot_hat(engine)
            self._apply_system_volume_if_configured()
            return

        if backend == "robot_hat":
            self._init_robot_hat(engine)
            self._apply_system_volume_if_configured()
            return

        self._init_system(engine)
        self._apply_system_volume_if_configured()

    @property
    def is_available(self) -> bool:
        return self._available_reason is None

    @property
    def unavailable_reason(self) -> str | None:
        return self._available_reason

    def say(self, text: str) -> bool:
        """Speak `text`. Returns True if it actually attempted audio output."""

        text = (text or "").strip()
        if not text:
            return False

        if self.settings.print_text:
            print(f"[speak] {text}")

        if self.settings.dry_run:
            return False

        if not self.is_available:
            # Keep runs demonstrable even on machines without audio/tts installed.
            return False

        engine = self.settings.engine.strip().lower()
        backend = self.settings.backend.strip().lower()

        if backend == "robot_hat" or engine in {"piper", "openai"}:
            try:
                if engine == "openai":
                    # robot_hat.tts.OpenAI_TTS supports instructions + stream.
                    self._tts_obj.say(
                        text,
                        instructions=self.settings.openai_instructions,
                        stream=bool(self.settings.openai_stream),
                    )
                    return True

                if engine == "piper":
                    # Piper supports streaming, but AudioPlayer gain is only configurable
                    # if we play ourselves.
                    gain = float(self.settings.piper_gain or 1.0)
                    if self.settings.piper_stream and gain == 1.0:
                        self._tts_obj.say(text, stream=True)
                        return True

                    # Non-streaming playback with optional gain.
                    from sunfounder_voice_assistant._audio_player import AudioPlayer  # type: ignore

                    file = "/tmp/tts_piper.wav"
                    self._tts_obj.tts(text, file)
                    with AudioPlayer(gain=gain) as player:
                        player.play_file(file)
                    return True

                # pico2wave/espeak via robot_hat
                self._tts_obj.say(text)
                return True
            except Exception as exc:  # pragma: no cover
                raise SpeakerError(f"robot_hat TTS failed: {exc}") from exc

        # system backend
        if engine == "pico2wave":
            return self._say_system_pico2wave(text)
        return self._say_system_espeak(text)

    @staticmethod
    def from_config_dict(cfg: dict[str, Any]) -> "Speaker":
        tts_cfg = (cfg or {}).get("tts", {}) if isinstance(cfg, dict) else {}

        def _get_str(key: str, default: str) -> str:
            val = tts_cfg.get(key, default)
            return default if val is None else str(val)

        def _get_bool(key: str, default: bool) -> bool:
            val = tts_cfg.get(key, default)
            return bool(val) if val is not None else default

        def _get_int(key: str, default: int) -> int:
            val = tts_cfg.get(key, default)
            try:
                return int(val)
            except Exception:
                return default

        def _get_float(key: str, default: float) -> float:
            val = tts_cfg.get(key, default)
            try:
                return float(val)
            except Exception:
                return default

        def _get_opt_int(key: str) -> int | None:
            if key not in tts_cfg:
                return None
            val = tts_cfg.get(key)
            if val is None or val == "":
                return None
            try:
                return int(val)
            except Exception:
                return None

        def _get_opt_int_clamped(key: str, min_val: int, max_val: int) -> int | None:
            val = _get_opt_int(key)
            if val is None:
                return None
            return max(min_val, min(max_val, val))

        espeak_cfg = tts_cfg.get("espeak", {}) if isinstance(tts_cfg, dict) else {}
        openai_cfg = tts_cfg.get("openai", {}) if isinstance(tts_cfg, dict) else {}
        piper_cfg = tts_cfg.get("piper", {}) if isinstance(tts_cfg, dict) else {}

        settings = SpeakerSettings(
            backend=_get_str("backend", "robot_hat"),
            engine=_get_str("engine", "openai"),
            language=_get_str("language", "de-DE"),
            print_text=_get_bool("print_text", True),
            dry_run=_get_bool("dry_run", False),
            system_volume_percent=_get_opt_int_clamped("system_volume_percent", 0, 100),
            player_cmd=_get_str("player_cmd", "aplay"),
            pico2wave_gain_db=_get_float("pico2wave_gain_db", 0.0),
            paplay_volume=_get_opt_int("paplay_volume"),
            espeak_voice=(str(espeak_cfg.get("voice")) if espeak_cfg.get("voice") is not None else None),
            espeak_speed=_get_int("espeak_speed", 160),
            espeak_amplitude=_get_int("espeak_amplitude", 120),
            openai_model=str(openai_cfg.get("model") or "gpt-4o-mini-tts"),
            openai_voice=str(openai_cfg.get("voice") or "alloy"),
            openai_instructions=(
                str(openai_cfg.get("instructions"))
                if openai_cfg.get("instructions") is not None and openai_cfg.get("instructions") != ""
                else None
            ),
            openai_stream=bool(openai_cfg.get("stream")) if openai_cfg.get("stream") is not None else True,
            openai_gain=(
                float(openai_cfg.get("gain"))
                if openai_cfg.get("gain") is not None and openai_cfg.get("gain") != ""
                else 1.5
            ),
            piper_model=(
                str(piper_cfg.get("model"))
                if piper_cfg.get("model") is not None and piper_cfg.get("model") != ""
                else None
            ),
            piper_stream=bool(piper_cfg.get("stream")) if piper_cfg.get("stream") is not None else False,
            piper_gain=(
                float(piper_cfg.get("gain"))
                if piper_cfg.get("gain") is not None and piper_cfg.get("gain") != ""
                else 1.0
            ),
        )
        return Speaker(settings)

    # --- system volume helpers ---

    def _apply_system_volume_if_configured(self) -> None:
        """Best-effort OS volume control.

        This is intentionally conservative: never fail the TTS pipeline just
        because the platform mixer can't be adjusted.
        """

        percent = self.settings.system_volume_percent
        if percent is None:
            return

        # Don't surprise in dry-run mode.
        if self.settings.dry_run:
            return

        if sys.platform != "linux":
            if self.settings.print_text and not self._warned_volume_unsupported:
                print(
                    f"[speak] Warning: tts.system_volume_percent is only implemented for Linux; "
                    f"current platform is {sys.platform!r}."
                )
                self._warned_volume_unsupported = True
            return

        percent = max(0, min(100, int(percent)))

        # Prefer PulseAudio/PipeWire controls when available.
        # - Raspberry Pi OS can be ALSA-only, PulseAudio, or PipeWire depending on image/version.
        if shutil.which("pactl") is not None:
            self._try_run(["pactl", "set-sink-mute", "@DEFAULT_SINK@", "0"])
            if self._try_run(["pactl", "set-sink-volume", "@DEFAULT_SINK@", f"{percent}%"]):
                return

        if shutil.which("wpctl") is not None:
            # wpctl uses 1.0 == 100%
            vol = max(0.0, min(1.0, percent / 100.0))
            self._try_run(["wpctl", "set-mute", "@DEFAULT_AUDIO_SINK@", "0"])
            if self._try_run(["wpctl", "set-volume", "@DEFAULT_AUDIO_SINK@", f"{vol:.2f}"]):
                return

        if shutil.which("amixer") is not None:
            # Try common mixer controls. Names vary between sound cards.
            for control in ("Master", "PCM", "Speaker", "Headphone"):
                if self._try_run(["amixer", "sset", control, f"{percent}%", "unmute"]):
                    return

        if self.settings.print_text:
            print(
                "[speak] Warning: couldn't set system volume. "
                "Install/configure one of: pactl (PulseAudio/PipeWire), wpctl (PipeWire), amixer (ALSA)."
            )

    def _try_run(self, cmd: list[str]) -> bool:
        try:
            subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            return True
        except Exception:
            return False

    # --- init helpers ---

    def _init_robot_hat(self, engine: str) -> None:
        try:
            # Prefer direct robot_hat import (picar-x uses it internally).
            from robot_hat import tts as robot_hat_tts  # type: ignore
        except Exception:
            # Fallback: picarx re-exports robot_hat.tts as `picarx.tts`.
            try:
                from picarx import tts as robot_hat_tts  # type: ignore
            except Exception as exc:
                self._available_reason = (
                    "robot_hat/picarx TTS not importable; install picar-x + robot-hat "
                    "or use backend=system"
                )
                self._tts_obj = None
                return

        if engine == "pico2wave":
            cls_name = "Pico2Wave"
        elif engine == "espeak":
            cls_name = "Espeak"
        elif engine == "piper":
            cls_name = "Piper"
        else:
            cls_name = "OpenAI_TTS"
        tts_cls = getattr(robot_hat_tts, cls_name, None)
        if tts_cls is None:
            self._available_reason = f"robot_hat TTS class not found: {cls_name}"
            self._tts_obj = None
            return

        # OpenAI requires an API key unless we're in dry-run.
        if engine == "openai":
            api_key = os.getenv("OPENAI_API_KEY")
            if not api_key:
                if self.settings.dry_run:
                    api_key = "DUMMY"
                else:
                    self._available_reason = "Missing OpenAI API key (set OPENAI_API_KEY)"
                    self._tts_obj = None
                    return

        if engine == "openai":
            # sunfounder_voice_assistant OpenAI_TTS requires api_key to be a str at init.
            obj = tts_cls(api_key=api_key)
        else:
            obj = tts_cls()

        # Match the example: `tts.set_lang('de-DE')` if supported.
        if hasattr(obj, "set_lang"):
            try:
                obj.set_lang(self.settings.language)
            except Exception:
                pass

        # Apply espeak tuning when supported by the backend.
        # (robot_hat.tts.Espeak exposes set_amp/set_speed/set_pitch/set_gap)
        if engine == "espeak":
            if hasattr(obj, "set_amp"):
                try:
                    obj.set_amp(int(self.settings.espeak_amplitude))
                except Exception:
                    pass
            if hasattr(obj, "set_speed"):
                try:
                    obj.set_speed(int(self.settings.espeak_speed))
                except Exception:
                    pass

        if engine == "piper":
            # Avoid downloading models during dry-run.
            if self.settings.dry_run:
                pass
            elif self.settings.piper_model:
                if hasattr(obj, "set_model"):
                    try:
                        obj.set_model(self.settings.piper_model)
                    except Exception as exc:
                        self._available_reason = f"Piper model error: {exc}"
                        self._tts_obj = None
                        return
            else:
                self._available_reason = "Piper model not set (set tts.piper.model)"

        if engine == "openai":
            try:
                if hasattr(obj, "set_api_key"):
                    obj.set_api_key(api_key)
                if hasattr(obj, "set_model"):
                    obj.set_model(self.settings.openai_model)
                if hasattr(obj, "set_voice"):
                    obj.set_voice(self.settings.openai_voice)
                if hasattr(obj, "set_gain"):
                    obj.set_gain(float(self.settings.openai_gain))
            except Exception as exc:
                self._available_reason = f"OpenAI TTS init error: {exc}"
                self._tts_obj = None
                return

        self._tts_obj = obj
        # If we made it here, it's usable (except the optional Piper-model warning above).
        if self._available_reason is None or engine != "piper":
            self._available_reason = None

    def _init_system(self, engine: str) -> None:
        if engine == "pico2wave":
            if shutil.which("pico2wave") is None:
                self._available_reason = "Missing binary: pico2wave"
                return
            if shutil.which(self.settings.player_cmd) is None:
                self._available_reason = f"Missing audio player: {self.settings.player_cmd}"
                return
            self._available_reason = None
            return

        # espeak
        if shutil.which("espeak") is None and shutil.which("espeak-ng") is None:
            self._available_reason = "Missing binary: espeak/espeak-ng"
            return
        self._available_reason = None

    # --- system backends ---

    def _say_system_pico2wave(self, text: str) -> bool:
        lang = self.settings.language

        with tempfile.TemporaryDirectory(prefix="pi_rc_bot_tts_") as td:
            wav_path = os.path.join(td, "tts.wav")
            cmd_tts = ["pico2wave", "-l", lang, "-w", wav_path, text]

            subprocess.run(cmd_tts, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

            play_wav_path = wav_path

            # Optional gain stage (useful when system volume is already maxed)
            gain_db = float(self.settings.pico2wave_gain_db or 0.0)
            if gain_db:
                amplified_wav_path = os.path.join(td, "tts.gain.wav")
                if shutil.which("sox") is not None:
                    subprocess.run(
                        ["sox", wav_path, amplified_wav_path, "gain", "-n", str(gain_db)],
                        check=True,
                        stdout=subprocess.DEVNULL,
                        stderr=subprocess.DEVNULL,
                    )
                    play_wav_path = amplified_wav_path
                elif shutil.which("ffmpeg") is not None:
                    subprocess.run(
                        [
                            "ffmpeg",
                            "-y",
                            "-i",
                            wav_path,
                            "-filter:a",
                            f"volume={gain_db}dB",
                            amplified_wav_path,
                        ],
                        check=True,
                        stdout=subprocess.DEVNULL,
                        stderr=subprocess.DEVNULL,
                    )
                    play_wav_path = amplified_wav_path
                else:
                    if self.settings.print_text:
                        print(
                            "[speak] pico2wave_gain_db set but neither 'sox' nor 'ffmpeg' is installed; "
                            "playing at normal volume"
                        )

            player = self.settings.player_cmd
            if player == "paplay":
                cmd_play = ["paplay"]
                if self.settings.paplay_volume is not None:
                    vol = int(self.settings.paplay_volume)
                    vol = max(0, min(65536, vol))
                    cmd_play.append(f"--volume={vol}")
                cmd_play.append(play_wav_path)
            else:
                cmd_play = [player, play_wav_path]

            subprocess.run(cmd_play, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            return True

    def _say_system_espeak(self, text: str) -> bool:
        voice = self.settings.espeak_voice

        exe = "espeak" if shutil.which("espeak") is not None else "espeak-ng"
        cmd = [
            exe,
            "-s",
            str(self.settings.espeak_speed),
            "-a",
            str(self.settings.espeak_amplitude),
        ]
        if voice:
            cmd += ["-v", voice]
        cmd.append(text)

        subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        return True
