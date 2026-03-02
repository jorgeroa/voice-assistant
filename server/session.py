from __future__ import annotations

import asyncio
import base64
import json
import logging
import re
from collections.abc import Callable

import numpy as np

from config.settings import AppSettings
from config.voices import LANGUAGE_CODES, VOICES
from core.conversation import ConversationManager
from core.llm import LLMClient
from core.stt import SpeechToText
from core.tts import TextToSpeech
from core.vad import VADState, VoiceActivityDetector

logger = logging.getLogger(__name__)

# Sentence boundary pattern for TTS chunking
_SENTENCE_BOUNDARY = re.compile(r"[.!?;:\n]")


class ConversationSession:
    """Per-WebSocket-connection session managing the full voice pipeline."""

    def __init__(self, settings: AppSettings, send_fn: Callable):
        self.settings = settings
        self._send = send_fn  # async callable to send JSON to client
        self.language = settings.language
        self.input_mode = "ptt"  # "ptt" or "vad"
        self.stt_mode = settings.stt_mode

        # Audio buffer for accumulating chunks during recording
        self._audio_buffer: list[np.ndarray] = []
        self._is_recording = False

        # Components
        self.vad = VoiceActivityDetector(
            threshold=settings.vad_threshold,
            silence_duration_ms=settings.silence_duration_ms,
            sample_rate=settings.sample_rate,
            chunk_ms=settings.audio_chunk_ms,
        )
        self.stt = SpeechToText(
            mode=settings.stt_mode,
            fast_model=settings.whisper_fast_model,
            quality_model=settings.whisper_model,
            device=settings.whisper_device,
            compute_type=settings.whisper_compute_type,
            batch_size=settings.whisper_batch_size,
            language=settings.language,
            hf_token=settings.hf_token,
        )
        self.llm = LLMClient(
            api_key=settings.openai_api_key,
            model=settings.openai_model,
            system_prompt=settings.system_prompt,
            max_turns=settings.max_conversation_turns,
        )
        self.tts = TextToSpeech(
            model_path=settings.kokoro_model_path,
            voices_path=settings.kokoro_voices_path,
            voice=settings.tts_voice,
            speed=settings.tts_speed,
            lang=LANGUAGE_CODES.get(settings.language, "en-us"),
        )
        self.conversation = ConversationManager(max_turns=settings.max_conversation_turns)

    async def handle_message(self, data: dict):
        """Route incoming WebSocket messages."""
        msg_type = data.get("type", "")
        try:
            if msg_type == "audio_chunk":
                await self._handle_audio_chunk(data)
            elif msg_type == "start_recording":
                self._start_recording()
            elif msg_type == "stop_recording":
                await self._stop_recording()
            elif msg_type == "set_language":
                await self._set_language(data.get("language", "en"))
            elif msg_type == "set_mode":
                self.input_mode = data.get("mode", "ptt")
            elif msg_type == "set_stt_mode":
                mode = data.get("mode", "fast")
                self.stt_mode = mode
                self.stt.set_mode(mode)
            elif msg_type == "set_voice":
                voice = data.get("voice", "")
                if voice:
                    self.tts.set_voice(voice, self.language)
            else:
                logger.warning("Unknown message type: %s", msg_type)
        except Exception as e:
            logger.exception("Error handling message type '%s'", msg_type)
            await self._send({"type": "error", "message": str(e)})

    def _start_recording(self):
        """Begin accumulating audio (push-to-talk)."""
        self._audio_buffer.clear()
        self._is_recording = True
        self.vad.reset()

    async def _stop_recording(self):
        """Stop recording and process utterance (push-to-talk)."""
        self._is_recording = False
        if not self._audio_buffer:
            return
        audio = np.concatenate(self._audio_buffer)
        self._audio_buffer.clear()
        await self._process_utterance(audio)

    async def _handle_audio_chunk(self, data: dict):
        """Process an incoming audio chunk."""
        raw = base64.b64decode(data["data"])
        # Client sends 16-bit PCM at 16kHz mono
        chunk = np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32768.0

        if self.input_mode == "ptt":
            if self._is_recording:
                self._audio_buffer.append(chunk)
        else:
            # VAD mode: always accumulate and run VAD
            self._audio_buffer.append(chunk)
            vad_result = self.vad.process_chunk(chunk)
            await self._send({
                "type": "vad_status",
                "is_speech": vad_result.is_speech,
                "probability": round(vad_result.probability, 3),
            })
            if vad_result.state == VADState.UTTERANCE_COMPLETE:
                audio = np.concatenate(self._audio_buffer)
                self._audio_buffer.clear()
                self.vad.reset()
                await self._process_utterance(audio)

    async def _process_utterance(self, audio: np.ndarray):
        """Full pipeline: STT → LLM → TTS."""
        duration = len(audio) / self.settings.sample_rate
        logger.info("Processing utterance: %.2fs, %d samples", duration, len(audio))

        if duration < 0.3:
            logger.info("Audio too short (%.2fs), skipping STT", duration)
            await self._send({"type": "status", "message": "Recording too short — hold the mic button longer."})
            await self._send({"type": "turn_complete"})
            return

        await self._send({"type": "status", "message": "Transcribing..."})

        # Run STT in a thread to avoid blocking the event loop
        loop = asyncio.get_event_loop()
        result = await loop.run_in_executor(
            None, self.stt.transcribe, audio, self.settings.sample_rate
        )

        if not result.text.strip():
            await self._send({"type": "status", "message": "No speech detected."})
            await self._send({"type": "turn_complete"})
            return

        # Send transcript to client
        speaker = None
        word_data = []
        if result.words:
            speakers = {w.speaker for w in result.words if w.speaker}
            speaker = ", ".join(sorted(speakers)) if speakers else None
            word_data = [
                {"word": w.word, "start": w.start, "end": w.end, "speaker": w.speaker}
                for w in result.words
            ]

        await self._send({
            "type": "transcript",
            "text": result.text,
            "speaker": speaker,
            "words": word_data,
            "processing_time": round(result.processing_time, 2),
        })

        # Add to conversation
        self.conversation.add_user_turn(
            text=result.text,
            language=result.language,
            speaker_label=speaker,
            word_timestamps=word_data if word_data else None,
            audio_duration=result.duration,
        )

        # Generate LLM response with streaming TTS
        await self._send({"type": "status", "message": "Thinking..."})
        await self._generate_and_speak(loop)

    async def _generate_and_speak(self, loop: asyncio.AbstractEventLoop):
        """Stream LLM response, chunk by sentence, synthesize and send audio."""
        messages = self.conversation.get_openai_messages()
        full_response = ""
        sentence_buffer = ""

        def _stream_llm():
            return list(self.llm.stream_response(messages))

        # Get all LLM chunks (run in thread since OpenAI SDK is sync)
        # We process them as they come for text streaming, then TTS in batches
        chunks = await loop.run_in_executor(None, _stream_llm)

        for text_chunk in chunks:
            full_response += text_chunk
            sentence_buffer += text_chunk

            # Stream text to client
            await self._send({"type": "llm_chunk", "text": text_chunk})

            # Check for sentence boundary with minimum length
            if _SENTENCE_BOUNDARY.search(text_chunk) and len(sentence_buffer.strip()) > 10:
                await self._synthesize_and_send(loop, sentence_buffer.strip())
                sentence_buffer = ""

        # Flush remaining text
        if sentence_buffer.strip():
            await self._synthesize_and_send(loop, sentence_buffer.strip())

        self.conversation.add_assistant_turn(full_response, self.language)
        await self._send({"type": "turn_complete"})

    async def _synthesize_and_send(self, loop: asyncio.AbstractEventLoop, text: str):
        """Synthesize text and send audio to client."""

        def _synth():
            return self.tts.synthesize(text)

        samples, sr = await loop.run_in_executor(None, _synth)
        if len(samples) == 0:
            return

        # Convert float32 [-1,1] to int16 for transmission
        audio_int16 = (samples * 32767).clip(-32768, 32767).astype(np.int16)
        audio_b64 = base64.b64encode(audio_int16.tobytes()).decode("ascii")
        await self._send({
            "type": "audio_response",
            "data": audio_b64,
            "sample_rate": sr,
        })

    async def _set_language(self, language: str):
        """Switch language for all components."""
        if language not in ("en", "es"):
            return
        self.language = language
        self.settings.language = language
        self.stt.set_language(language)
        self.llm.set_system_prompt(self.settings.system_prompt)
        self.tts.set_voice(self.settings.tts_voice, language)
        await self._send_voices()
        await self._send({"type": "status", "message": f"Language set to {language}"})

    async def _send_voices(self):
        """Send available voices for the current language to the client."""
        lang_voices = VOICES.get(self.language, {})
        voice_list = []
        for gender in ("female", "male"):
            for vid in lang_voices.get(gender, []):
                label = vid.split("_", 1)[1].title() + f" ({gender[0].upper()})"
                voice_list.append({"id": vid, "label": label})
        await self._send({
            "type": "voices",
            "voices": voice_list,
            "current": self.tts.voice,
        })
