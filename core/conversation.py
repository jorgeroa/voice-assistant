from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Literal


@dataclass
class ConversationTurn:
    role: Literal["user", "assistant"]
    content: str
    timestamp: datetime
    language: str
    speaker_label: str | None = None
    word_timestamps: list[dict] | None = None
    audio_duration: float | None = None


class ConversationManager:
    """Manages conversation history with rich metadata."""

    def __init__(self, max_turns: int = 20):
        self.turns: list[ConversationTurn] = []
        self.max_turns = max_turns

    def add_user_turn(
        self,
        text: str,
        language: str,
        speaker_label: str | None = None,
        word_timestamps: list[dict] | None = None,
        audio_duration: float | None = None,
    ):
        self.turns.append(ConversationTurn(
            role="user",
            content=text,
            timestamp=datetime.now(),
            language=language,
            speaker_label=speaker_label,
            word_timestamps=word_timestamps,
            audio_duration=audio_duration,
        ))

    def add_assistant_turn(self, text: str, language: str):
        self.turns.append(ConversationTurn(
            role="assistant",
            content=text,
            timestamp=datetime.now(),
            language=language,
        ))

    def get_openai_messages(self) -> list[dict[str, str]]:
        """Return conversation in OpenAI message format, trimmed to max_turns."""
        recent = self.turns[-(self.max_turns * 2) :]
        return [{"role": t.role, "content": t.content} for t in recent]

    def get_transcript(self) -> list[dict]:
        """Return full transcript with metadata for UI display."""
        result = []
        for t in self.turns:
            entry = {
                "role": t.role,
                "content": t.content,
                "timestamp": t.timestamp.isoformat(),
                "language": t.language,
            }
            if t.speaker_label:
                entry["speaker"] = t.speaker_label
            if t.word_timestamps:
                entry["words"] = t.word_timestamps
            if t.audio_duration is not None:
                entry["audio_duration"] = t.audio_duration
            result.append(entry)
        return result

    def clear(self):
        self.turns.clear()
