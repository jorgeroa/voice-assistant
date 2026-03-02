from __future__ import annotations

import logging

import numpy as np

from core.tts_base import TTSBackend

logger = logging.getLogger(__name__)

OPENAI_VOICES = ["alloy", "echo", "fable", "onyx", "nova", "shimmer"]


class OpenAITTS(TTSBackend):
    """OpenAI TTS API backend."""

    name = "openai"

    def __init__(
        self,
        api_key: str,
        model: str = "tts-1",
        voice: str = "nova",
        speed: float = 1.0,
    ):
        self.api_key = api_key
        self.model = model
        self.voice = voice
        self.speed = speed
        self._client = None

    def _ensure_client(self):
        if self._client is not None:
            return
        import openai

        self._client = openai.OpenAI(api_key=self.api_key)
        logger.info("OpenAI TTS client initialized (model=%s)", self.model)

    def synthesize(self, text: str) -> tuple[np.ndarray, int]:
        if not text.strip():
            return np.array([], dtype=np.float32), 24000
        self._ensure_client()
        response = self._client.audio.speech.create(
            model=self.model,
            voice=self.voice,
            input=text,
            response_format="pcm",
            speed=self.speed,
        )
        # response_format="pcm" returns raw 24kHz 16-bit signed LE mono
        pcm_bytes = response.content
        samples = np.frombuffer(pcm_bytes, dtype=np.int16).astype(np.float32) / 32768.0
        return samples, 24000

    def set_voice(self, voice: str, language: str) -> None:
        if voice in OPENAI_VOICES:
            self.voice = voice

    def get_voices(self, language: str) -> list[dict[str, str]]:
        return [{"id": v, "label": v.title()} for v in OPENAI_VOICES]
