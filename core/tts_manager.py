from __future__ import annotations

import logging

import numpy as np

from core.tts_base import TTSBackend

logger = logging.getLogger(__name__)


class TTSManager:
    """Manages multiple TTS backends. Drop-in replacement for TextToSpeech in session."""

    def __init__(self, backends: dict[str, TTSBackend], default: str = "kokoro"):
        self.backends = backends
        self._active_name = default if default in backends else next(iter(backends))
        self._active: TTSBackend = self.backends[self._active_name]

    @property
    def voice(self) -> str:
        return self._active.voice

    @property
    def active_backend(self) -> str:
        return self._active_name

    def synthesize(self, text: str) -> tuple[np.ndarray, int]:
        return self._active.synthesize(text)

    def set_voice(self, voice: str, language: str) -> None:
        self._active.set_voice(voice, language)

    def set_backend(self, name: str) -> bool:
        """Switch active backend. Returns True if changed."""
        if name not in self.backends:
            logger.warning("Unknown TTS backend: %s", name)
            return False
        if name == self._active_name:
            return False
        self._active_name = name
        self._active = self.backends[name]
        logger.info("TTS backend switched to: %s", name)
        return True

    def get_voices(self, language: str) -> list[dict[str, str]]:
        return self._active.get_voices(language)

    def get_backend_names(self) -> list[str]:
        return list(self.backends.keys())
