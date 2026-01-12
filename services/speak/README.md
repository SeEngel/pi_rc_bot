# Speak service

A tiny text-to-speech (TTS) wrapper modeled after PiCar-X `example/14.voice_promt_car.py`.

## Files

- `src/speaker.py`: `Speaker` class (language is set at init)
- `config.yaml`: configuration (commented)
- `main.py`: demo runner

## Install (if needed)

```bash
pip install -r services/speak/requirements.txt
```

## Run

```bash
python services/speak/main.py
```

## Making it louder

- If you're on Raspberry Pi OS and the sound gets louder when you raise the *system* volume, set:
	- `tts.system_volume_percent: 100`
	- (Linux only; tries `pactl`, `wpctl`, then `amixer`)

- If you use `tts.engine: espeak`, increase `tts.espeak_amplitude` (max 200).
- If you use `tts.engine: pico2wave` and want a config-driven boost, switch to `tts.backend: system` and use:
	- `tts.pico2wave_gain_db` (requires `sox` or `ffmpeg`)
	- optionally `tts.player_cmd: paplay` + `tts.paplay_volume: 65536`

For `tts.engine: openai` and `tts.engine: piper`, use `tts.openai.gain` / `tts.piper.gain`.

## Troubleshooting crackling/noise (OpenAI TTS)

If you hear "good first sentence, then loud noise/crackling" on longer texts, the two most common causes are:

- **Streaming artifacts** in some `robot_hat` setups. Try:
	- `tts.openai.stream: false`
	- `tts.openai.chunking: true`
	- `tts.openai.max_chars: 400..800`

- **Clipping/distortion** from too much gain/volume. Try:
	- `tts.openai.gain: 1.0`
	- reduce OS volume a bit (e.g. `tts.system_volume_percent: 70..90`)

Chunking is implemented in `src/speaker.py` and speaks chunks sequentially (no parallelism, no audio-file concatenation needed).

If you're on `tts.backend: robot_hat`, loudness is primarily controlled by the OS mixer / amp, not by `config.yaml`.

If you have no audio device / no TTS binaries installed, set `tts.dry_run: true` in `services/speak/config.yaml`.
