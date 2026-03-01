from __future__ import annotations

from typing import Literal

from pydantic_settings import BaseSettings, SettingsConfigDict


class AppSettings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8")

    # API keys
    openai_api_key: str = ""
    hf_token: str = ""

    # Language
    language: Literal["en", "es"] = "en"

    # STT
    stt_mode: Literal["fast", "quality"] = "fast"
    whisper_model: str = "small"
    whisper_fast_model: str = "base"
    whisper_device: str = "cpu"
    whisper_compute_type: str = "float32"
    whisper_batch_size: int = 8

    # LLM
    openai_model: str = "gpt-4o-mini"
    max_conversation_turns: int = 20
    system_prompt_en: str = (
        "You are a helpful voice assistant. Keep responses concise and conversational. "
        "Respond in 1-3 sentences unless the user asks for more detail."
    )
    system_prompt_es: str = (
        "Eres un asistente de voz útil. Mantén las respuestas concisas y conversacionales. "
        "Responde en 1-3 oraciones a menos que el usuario pida más detalle."
    )

    # TTS
    tts_voice_en: str = "af_heart"
    tts_voice_es: str = "ef_dora"
    tts_speed: float = 1.0
    kokoro_model_path: str = "models/kokoro-v1.0.onnx"
    kokoro_voices_path: str = "models/voices-v1.0.bin"

    # VAD
    vad_threshold: float = 0.5
    silence_duration_ms: int = 800
    audio_chunk_ms: int = 250

    # Audio
    sample_rate: int = 16000
    tts_sample_rate: int = 24000

    # Server
    host: str = "0.0.0.0"
    port: int = 8000

    @property
    def system_prompt(self) -> str:
        return self.system_prompt_es if self.language == "es" else self.system_prompt_en

    @property
    def tts_voice(self) -> str:
        return self.tts_voice_es if self.language == "es" else self.tts_voice_en
