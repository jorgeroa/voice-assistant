from __future__ import annotations

import logging
from collections.abc import Generator

import openai

logger = logging.getLogger(__name__)


class LLMClient:
    """OpenAI chat completions with streaming support."""

    def __init__(
        self,
        api_key: str,
        model: str = "gpt-4o-mini",
        system_prompt: str = "",
        max_turns: int = 20,
    ):
        self.client = openai.OpenAI(api_key=api_key)
        self.model = model
        self.system_prompt = system_prompt
        self.max_turns = max_turns

    def stream_response(
        self, conversation: list[dict[str, str]]
    ) -> Generator[str, None, None]:
        """Yield text chunks from OpenAI streaming response."""
        messages = [{"role": "system", "content": self.system_prompt}]
        # Keep only the most recent turns to stay within context limits
        recent = conversation[-(self.max_turns * 2) :]
        messages.extend(recent)

        stream = self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            stream=True,
            max_tokens=300,
            temperature=0.7,
        )
        for chunk in stream:
            delta = chunk.choices[0].delta
            if delta.content:
                yield delta.content

    def get_full_response(self, conversation: list[dict[str, str]]) -> str:
        """Non-streaming version."""
        return "".join(self.stream_response(conversation))

    def set_system_prompt(self, prompt: str):
        self.system_prompt = prompt
