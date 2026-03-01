from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field

import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class WordSegment:
    word: str
    start: float
    end: float
    speaker: str | None = None


@dataclass
class TranscriptionResult:
    text: str
    language: str
    segments: list[dict] = field(default_factory=list)
    words: list[WordSegment] = field(default_factory=list)
    duration: float = 0.0
    processing_time: float = 0.0


class FastSTT:
    """Fast mode: faster-whisper with no diarization or alignment."""

    def __init__(
        self,
        model_name: str = "base",
        device: str = "cpu",
        compute_type: str = "float32",
        language: str = "en",
    ):
        self.model_name = model_name
        self.device = device
        self.compute_type = compute_type
        self.language = language
        self._model = None

    def _ensure_loaded(self):
        if self._model is not None:
            return
        from faster_whisper import WhisperModel

        logger.info("Loading faster-whisper model '%s' (fast mode)...", self.model_name)
        self._model = WhisperModel(
            self.model_name,
            device=self.device,
            compute_type=self.compute_type,
        )
        logger.info("faster-whisper model loaded.")

    def transcribe(self, audio: np.ndarray, sample_rate: int = 16000) -> TranscriptionResult:
        self._ensure_loaded()
        start = time.time()

        segments_iter, info = self._model.transcribe(
            audio,
            language=self.language,
            beam_size=1,
            best_of=1,
            vad_filter=True,
        )

        text_parts = []
        segments = []
        for seg in segments_iter:
            text_parts.append(seg.text)
            segments.append({
                "start": seg.start,
                "end": seg.end,
                "text": seg.text,
            })

        full_text = " ".join(text_parts).strip()
        duration = len(audio) / sample_rate if sample_rate > 0 else 0.0

        return TranscriptionResult(
            text=full_text,
            language=info.language,
            segments=segments,
            words=[],
            duration=duration,
            processing_time=time.time() - start,
        )


class QualitySTT:
    """Quality mode: WhisperX with alignment and speaker diarization."""

    def __init__(
        self,
        model_name: str = "small",
        device: str = "cpu",
        compute_type: str = "float32",
        batch_size: int = 8,
        language: str = "en",
        hf_token: str = "",
    ):
        self.model_name = model_name
        self.device = device
        self.compute_type = compute_type
        self.batch_size = batch_size
        self.language = language
        self.hf_token = hf_token
        self._model = None
        self._align_model = None
        self._align_metadata = None
        self._diarize_pipeline = None

    def _ensure_loaded(self):
        if self._model is not None:
            return
        import whisperx

        logger.info("Loading WhisperX model '%s' (quality mode)...", self.model_name)
        self._model = whisperx.load_model(
            self.model_name,
            self.device,
            compute_type=self.compute_type,
            language=self.language,
        )

        logger.info("Loading alignment model for '%s'...", self.language)
        self._align_model, self._align_metadata = whisperx.load_align_model(
            language_code=self.language, device=self.device
        )

        if self.hf_token:
            logger.info("Loading diarization pipeline...")
            self._diarize_pipeline = whisperx.DiarizationPipeline(
                use_auth_token=self.hf_token, device=self.device
            )
            logger.info("Diarization pipeline loaded.")

        logger.info("WhisperX models loaded.")

    def transcribe(self, audio: np.ndarray, sample_rate: int = 16000) -> TranscriptionResult:
        self._ensure_loaded()
        import whisperx

        start = time.time()
        duration = len(audio) / sample_rate if sample_rate > 0 else 0.0

        # Step 1: Transcribe
        result = self._model.transcribe(audio, batch_size=self.batch_size)

        # Step 2: Align for word-level timestamps
        result = whisperx.align(
            result["segments"],
            self._align_model,
            self._align_metadata,
            audio,
            self.device,
            return_char_alignments=False,
        )

        # Step 3: Diarize (optional)
        if self._diarize_pipeline is not None:
            diarize_segments = self._diarize_pipeline(audio)
            result = whisperx.assign_word_speakers(diarize_segments, result)

        # Build result
        words = []
        text_parts = []
        for seg in result.get("segments", []):
            text_parts.append(seg.get("text", ""))
            for w in seg.get("words", []):
                words.append(WordSegment(
                    word=w.get("word", ""),
                    start=w.get("start", 0.0),
                    end=w.get("end", 0.0),
                    speaker=w.get("speaker"),
                ))

        return TranscriptionResult(
            text=" ".join(text_parts).strip(),
            language=result.get("language", self.language),
            segments=result.get("segments", []),
            words=words,
            duration=duration,
            processing_time=time.time() - start,
        )


class SpeechToText:
    """Dual-mode STT: switches between fast (faster-whisper) and quality (WhisperX)."""

    def __init__(
        self,
        mode: str = "fast",
        fast_model: str = "base",
        quality_model: str = "small",
        device: str = "cpu",
        compute_type: str = "float32",
        batch_size: int = 8,
        language: str = "en",
        hf_token: str = "",
    ):
        self.mode = mode
        self._fast = FastSTT(
            model_name=fast_model,
            device=device,
            compute_type=compute_type,
            language=language,
        )
        self._quality = QualitySTT(
            model_name=quality_model,
            device=device,
            compute_type=compute_type,
            batch_size=batch_size,
            language=language,
            hf_token=hf_token,
        )

    def transcribe(self, audio: np.ndarray, sample_rate: int = 16000) -> TranscriptionResult:
        if self.mode == "quality":
            return self._quality.transcribe(audio, sample_rate)
        return self._fast.transcribe(audio, sample_rate)

    def set_mode(self, mode: str):
        if mode in ("fast", "quality"):
            self.mode = mode

    def set_language(self, language: str):
        self._fast.language = language
        self._quality.language = language
