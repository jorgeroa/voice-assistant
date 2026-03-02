from __future__ import annotations

import abc

import numpy as np


class TTSBackend(abc.ABC):
    """Common interface for all TTS backends."""

    name: str  # "kokoro", "openai", "f5tts", "fish"
    voice: str  # current voice ID

    @abc.abstractmethod
    def synthesize(self, text: str) -> tuple[np.ndarray, int]:
        """Return (float32 samples in [-1,1], sample_rate)."""
        ...

    @abc.abstractmethod
    def set_voice(self, voice: str, language: str) -> None:
        ...

    @abc.abstractmethod
    def get_voices(self, language: str) -> list[dict[str, str]]:
        """Return [{"id": "...", "label": "..."}, ...] for given language."""
        ...
