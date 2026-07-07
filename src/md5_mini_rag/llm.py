from __future__ import annotations

from dataclasses import dataclass

from .config import Settings
from .prompting import SYSTEM_PROMPT


@dataclass(frozen=True)
class LlmAnswer:
    content: str
    model: str


class GroqChatClient:
    def __init__(self, settings: Settings) -> None:
        if not settings.groq_api_key:
            raise RuntimeError("GROQ_API_KEY is missing. Add it to .env before using rag ask.")

        from groq import Groq

        self.model = settings.groq_model
        self.client = Groq(api_key=settings.groq_api_key)

    def answer(self, user_prompt: str) -> LlmAnswer:
        completion = self.client.chat.completions.create(
            model=self.model,
            temperature=0.1,
            max_tokens=700,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_prompt},
            ],
        )
        content = completion.choices[0].message.content or ""
        return LlmAnswer(content=content.strip(), model=self.model)
