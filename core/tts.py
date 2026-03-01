from __future__ import annotations

import logging
from pathlib import Path

import numpy as np

from config.voices import LANGUAGE_CODES

logger = logging.getLogger(__name__)


class TextToSpeech:
    """Kokoro ONNX text-to-speech wrapper with batch and streaming support."""

    def __init__(
        self,
        model_path: str = "models/kokoro-v1.0.onnx",
        voices_path: str = "models/voices-v1.0.bin",
        voice: str = "af_heart",
        speed: float = 1.0,
        lang: str = "en-us",
    ):
        self.model_path = model_path
        self.voices_path = voices_path
        self.voice = voice
        self.speed = speed
        self.lang = lang
        self._kokoro = None

    def _ensure_loaded(self):
        if self._kokoro is not None:
            return
        from kokoro_onnx import Kokoro

        model = Path(self.model_path)
        voices = Path(self.voices_path)
        if not model.exists() or not voices.exists():
            raise FileNotFoundError(
                f"Kokoro model files not found at {model} / {voices}. "
                "Run: python scripts/download_models.py"
            )
        logger.info("Loading Kokoro TTS model...")
        self._kokoro = Kokoro(str(model), str(voices))
        logger.info("Kokoro TTS model loaded.")

    def synthesize(self, text: str) -> tuple[np.ndarray, int]:
        """Synthesize text to audio. Returns (samples, sample_rate)."""
        self._ensure_loaded()
        if not text.strip():
            return np.array([], dtype=np.float32), 24000
        samples, sr = self._kokoro.create(
            text, voice=self.voice, speed=self.speed, lang=self.lang
        )
        return np.asarray(samples, dtype=np.float32), sr

    async def stream_synthesis(self, text: str):
        """Async streaming synthesis. Yields (samples, sample_rate) per sentence chunk."""
        self._ensure_loaded()
        if not text.strip():
            return
        stream = self._kokoro.create_stream(
            text, voice=self.voice, speed=self.speed, lang=self.lang
        )
        async for samples, sr in stream:
            yield np.asarray(samples, dtype=np.float32), sr

    def set_voice(self, voice: str, language: str):
        """Switch voice and language."""
        self.voice = voice
        self.lang = LANGUAGE_CODES.get(language, "en-us")
