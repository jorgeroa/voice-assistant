# Voice Conversation

A real-time voice conversation app powered by Whisper STT, OpenAI LLM, and Kokoro TTS. Talk to an AI assistant through your browser with natural voice interaction, barge-in support, and multilingual capabilities.

## Features

- **Voice Activity Detection (VAD)** — automatically detects when you start and stop speaking, or use push-to-talk mode
- **Barge-in** — interrupt the assistant mid-response by speaking
- **Multilingual** — English and Spanish with language-specific voices and system prompts
- **Multiple TTS backends** — local Kokoro ONNX (default), OpenAI TTS API, or Fish Audio API
- **Dual STT modes** — fast mode (faster-whisper) or quality mode (WhisperX with speaker diarization)
- **Streaming responses** — LLM output is synthesized sentence-by-sentence for low latency
- **Voice selection** — choose from multiple male/female voices per language

## Architecture

```
Browser (mic capture) → WebSocket → FastAPI server
                                       ├── Silero VAD (speech detection)
                                       ├── Whisper STT (transcription)
                                       ├── OpenAI GPT (response generation)
                                       └── Kokoro TTS (speech synthesis)
                                    WebSocket ← audio playback
```

## Prerequisites

- Python 3.11 – 3.12
- [uv](https://docs.astral.sh/uv/)
- ffmpeg
- espeak-ng

```bash
# macOS
brew install uv ffmpeg espeak-ng
```

## Setup

1. **Install dependencies:**

```bash
uv sync

# Optional: quality STT mode with speaker diarization
uv sync --extra quality

# Optional: Fish Audio TTS backend
uv sync --extra fishaudio
```

2. **Download models:**

```bash
uv run python scripts/download_models.py
```

This downloads the Kokoro TTS model (~325 MB) and pre-caches the Whisper base model.

3. **Configure environment:**

```bash
cp .env.example .env
```

Edit `.env` and set your `OPENAI_API_KEY`. For quality STT mode, also set `HF_TOKEN` (see [HuggingFace tokens](https://huggingface.co/settings/tokens)).

## Usage

```bash
uv run python main.py
```

Then open http://localhost:8000 in your browser.

### CLI options

```
--host            Server host (default: 0.0.0.0)
--port            Server port (default: 8000)
--language, -l    en or es
--stt-mode        fast or quality
--whisper-model   Whisper model name
--log-level       Logging level (default: info)
```

## Configuration

Key settings in `.env`:

| Variable | Default | Description |
|---|---|---|
| `OPENAI_API_KEY` | — | Required. OpenAI API key |
| `HF_TOKEN` | — | Required for quality STT mode |
| `LANGUAGE` | `en` | `en` or `es` |
| `STT_MODE` | `fast` | `fast` (faster-whisper) or `quality` (WhisperX) |
| `OPENAI_MODEL` | `gpt-4o-mini` | GPT model for responses |
| `TTS_SPEED` | `1.0` | Speech speed multiplier |
| `VAD_THRESHOLD` | `0.5` | Speech detection sensitivity (0–1) |
| `SILENCE_DURATION_MS` | `800` | Silence duration to end an utterance |

### Available Kokoro voices

**English** — af_heart (default), af_bella, af_nicole, af_sarah, af_sky, am_adam, am_michael, am_echo, am_eric

**Spanish** — ef_dora (default), em_alex, em_santa
