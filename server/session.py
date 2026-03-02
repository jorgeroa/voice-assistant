from __future__ import annotations

import asyncio
import base64
import logging
import re
from collections.abc import Callable

import numpy as np

from config.settings import AppSettings
from config.voices import LANGUAGE_CODES
from core.conversation import ConversationManager
from core.llm import LLMClient
from core.stt import SpeechToText
from core.tts import KokoroTTS
from core.tts_manager import TTSManager
from core.tts_openai import OpenAITTS
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

        # Generation state for barge-in support
        self._generation_task: asyncio.Task | None = None
        self._generation_id: int = 0

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
        self.tts = self._build_tts_manager(settings)
        self.conversation = ConversationManager(max_turns=settings.max_conversation_turns)

    @staticmethod
    def _build_tts_manager(settings: AppSettings) -> TTSManager:
        from core.tts_fish import FishAudioTTS

        backends = {}
        backends["kokoro"] = KokoroTTS(
            model_path=settings.kokoro_model_path,
            voices_path=settings.kokoro_voices_path,
            voice=settings.tts_voice,
            speed=settings.tts_speed,
            lang=LANGUAGE_CODES.get(settings.language, "en-us"),
        )
        backends["openai"] = OpenAITTS(
            api_key=settings.openai_api_key,
            model=settings.openai_tts_model,
            voice=settings.openai_tts_voice,
            speed=settings.tts_speed,
        )
        backends["fish"] = FishAudioTTS(
            api_key=settings.fish_audio_api_key,
            model_id=settings.fish_audio_model_id,
            speed=settings.tts_speed,
        )
        return TTSManager(backends=backends, default=settings.tts_backend)

    async def handle_message(self, data: dict):
        """Route incoming WebSocket messages."""
        msg_type = data.get("type", "")
        try:
            if msg_type == "audio_chunk":
                await self._handle_audio_chunk(data)
            elif msg_type == "interrupt":
                await self._interrupt()
            elif msg_type == "start_recording":
                if self._generation_task and not self._generation_task.done():
                    await self._interrupt()
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
            elif msg_type == "set_tts_backend":
                backend = data.get("backend", "")
                if backend and self.tts.set_backend(backend):
                    await self._send_voices()
                    await self._send({"type": "tts_backend_changed", "backend": backend})
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

    async def _interrupt(self):
        """Cancel active generation (barge-in)."""
        if self._generation_task and not self._generation_task.done():
            self._generation_task.cancel()
            try:
                await self._generation_task
            except asyncio.CancelledError:
                pass
            logger.info("Generation interrupted by user")
        self._generation_task = None
        await self._send({"type": "interrupt_ack"})

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

        # Generate LLM response with streaming TTS (as background task for barge-in)
        await self._send({"type": "status", "message": "Thinking..."})
        self._generation_id += 1
        gen_id = self._generation_id
        self._generation_task = asyncio.create_task(self._generate_and_speak(loop, gen_id))

    async def _generate_and_speak(self, loop: asyncio.AbstractEventLoop, gen_id: int):
        """Stream LLM response, chunk by sentence, synthesize and send audio."""
        messages = self.conversation.get_openai_messages()
        full_response = ""
        sentence_buffer = ""

        def _stream_llm():
            return list(self.llm.stream_response(messages))

        try:
            # Get all LLM chunks (run in thread since OpenAI SDK is sync)
            chunks = await loop.run_in_executor(None, _stream_llm)

            for text_chunk in chunks:
                if self._generation_id != gen_id:
                    break
                full_response += text_chunk
                sentence_buffer += text_chunk

                # Stream text to client
                await self._send({"type": "llm_chunk", "text": text_chunk})

                # Check for sentence boundary with minimum length
                if _SENTENCE_BOUNDARY.search(text_chunk) and len(sentence_buffer.strip()) > 10:
                    await self._synthesize_and_send(loop, sentence_buffer.strip(), gen_id)
                    sentence_buffer = ""

            # Flush remaining text
            if sentence_buffer.strip() and self._generation_id == gen_id:
                await self._synthesize_and_send(loop, sentence_buffer.strip(), gen_id)

            self.conversation.add_assistant_turn(full_response, self.language)
        except asyncio.CancelledError:
            # Interrupted — save partial response to conversation
            if full_response.strip():
                self.conversation.add_assistant_turn(full_response, self.language)
            logger.info("Generation task cancelled (gen_id=%d)", gen_id)
            return

        if self._generation_id == gen_id:
            await self._send({"type": "turn_complete"})

    async def _synthesize_and_send(self, loop: asyncio.AbstractEventLoop, text: str, gen_id: int):
        """Synthesize text and send audio to client."""
        if self._generation_id != gen_id:
            return

        def _synth():
            return self.tts.synthesize(text)

        samples, sr = await loop.run_in_executor(None, _synth)
        if len(samples) == 0 or self._generation_id != gen_id:
            return

        # Convert float32 [-1,1] to int16 for transmission
        audio_int16 = (samples * 32767).clip(-32768, 32767).astype(np.int16)
        audio_b64 = base64.b64encode(audio_int16.tobytes()).decode("ascii")
        await self._send({
            "type": "audio_response",
            "data": audio_b64,
            "sample_rate": sr,
            "gen_id": gen_id,
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
        """Send available voices for the current backend + language to the client."""
        voice_list = self.tts.get_voices(self.language)
        await self._send({
            "type": "voices",
            "voices": voice_list,
            "current": self.tts.voice,
        })

    async def _send_backends(self):
        """Send available TTS backends to client."""
        await self._send({
            "type": "tts_backends",
            "backends": self.tts.get_backend_names(),
            "current": self.tts.active_backend,
        })
