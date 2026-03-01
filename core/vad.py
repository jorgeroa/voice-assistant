from __future__ import annotations

import logging
from enum import Enum

import numpy as np
import torch

logger = logging.getLogger(__name__)


class VADState(Enum):
    IDLE = "idle"
    SPEECH_DETECTED = "speech_detected"
    UTTERANCE_COMPLETE = "utterance_complete"


class VADResult:
    __slots__ = ("state", "probability", "is_speech")

    def __init__(self, state: VADState, probability: float, is_speech: bool):
        self.state = state
        self.probability = probability
        self.is_speech = is_speech


class VoiceActivityDetector:
    """Silero-VAD wrapper with state machine for utterance detection."""

    def __init__(
        self,
        threshold: float = 0.5,
        silence_duration_ms: int = 800,
        sample_rate: int = 16000,
        chunk_ms: int = 250,
    ):
        self.threshold = threshold
        self.sample_rate = sample_rate
        self.chunk_ms = chunk_ms
        self._silence_chunks_needed = max(1, silence_duration_ms // chunk_ms)
        self._consecutive_silence = 0
        self._speech_detected = False
        self._state = VADState.IDLE
        self._model = None

    def _ensure_loaded(self):
        if self._model is not None:
            return
        logger.info("Loading Silero VAD model...")
        self._model, _ = torch.hub.load(
            repo_or_dir="snakers4/silero-vad",
            model="silero_vad",
            trust_repo=True,
        )
        logger.info("Silero VAD model loaded.")

    def process_chunk(self, audio_chunk: np.ndarray) -> VADResult:
        """Process an audio chunk and return current VAD state.

        Args:
            audio_chunk: float32 numpy array, mono, at self.sample_rate
        """
        self._ensure_loaded()

        # Silero VAD expects float32 tensor
        if audio_chunk.dtype != np.float32:
            audio_chunk = audio_chunk.astype(np.float32)
        # Normalize int16 range to [-1, 1] if needed
        if np.abs(audio_chunk).max() > 1.0:
            audio_chunk = audio_chunk / 32768.0

        tensor = torch.from_numpy(audio_chunk)
        prob = self._model(tensor, self.sample_rate).item()
        is_speech = prob >= self.threshold

        if is_speech:
            self._speech_detected = True
            self._consecutive_silence = 0
            self._state = VADState.SPEECH_DETECTED
        elif self._speech_detected:
            self._consecutive_silence += 1
            if self._consecutive_silence >= self._silence_chunks_needed:
                self._state = VADState.UTTERANCE_COMPLETE
            else:
                self._state = VADState.SPEECH_DETECTED
        else:
            self._state = VADState.IDLE

        return VADResult(state=self._state, probability=prob, is_speech=is_speech)

    def reset(self):
        """Reset state for a new utterance."""
        self._consecutive_silence = 0
        self._speech_detected = False
        self._state = VADState.IDLE
        if self._model is not None:
            self._model.reset_states()
