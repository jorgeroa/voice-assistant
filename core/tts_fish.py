from __future__ import annotations

import logging

import numpy as np

from core.tts_base import TTSBackend

logger = logging.getLogger(__name__)

# Popular Fish Audio pre-made voice model IDs
FISH_VOICES = [
    {"id": "", "label": "Default"},
]


class FishAudioTTS(TTSBackend):
    """Fish Audio TTS API backend with voice cloning support."""

    name = "fish"

    def __init__(
        self,
        api_key: str,
        model_id: str = "",
        speed: float = 1.0,
    ):
        self.api_key = api_key
        self.model_id = model_id
        self.voice = model_id or "default"
        self.speed = speed
        self._session = None

    def _ensure_session(self):
        if self._session is not None:
            return
        try:
            from fish_audio_sdk import Session
        except ImportError:
            raise ImportError(
                "Fish Audio SDK not installed. Run: pip install fish-audio-sdk"
            )
        self._session = Session(self.api_key)
        logger.info("Fish Audio session initialized")

    def synthesize(self, text: str) -> tuple[np.ndarray, int]:
        if not text.strip():
            return np.array([], dtype=np.float32), 44100
        self._ensure_session()
        from fish_audio_sdk import TTSRequest

        kwargs = {"text": text}
        if self.model_id:
            kwargs["reference_id"] = self.model_id

        # Collect streaming PCM chunks
        chunks = []
        for chunk in self._session.tts(TTSRequest(**kwargs)):
            chunks.append(chunk)

        if not chunks:
            return np.array([], dtype=np.float32), 44100

        pcm_bytes = b"".join(chunks)
        # Fish Audio returns 44.1kHz 16-bit signed LE mono by default
        samples = np.frombuffer(pcm_bytes, dtype=np.int16).astype(np.float32) / 32768.0
        return samples, 44100

    def set_voice(self, voice: str, language: str) -> None:
        self.model_id = voice if voice != "default" else ""
        self.voice = voice or "default"

    def get_voices(self, language: str) -> list[dict[str, str]]:
        return list(FISH_VOICES)
