from __future__ import annotations

import json
from dataclasses import dataclass

from .config import Settings


@dataclass(frozen=True)
class ModerationResult:
    prompt_injection: bool
    raison: str


class Moderator:
    def __init__(self, settings: Settings) -> None:
        if not settings.groq_api_key:
            raise RuntimeError("GROQ_API_KEY is missing. Add it to .env before using moderation.")

        from groq import Groq

        self.model = settings.groq_model
        self.prompt_path = settings.moderator_prompt_path
        self.client = Groq(api_key=settings.groq_api_key)

    def moderate(self, question: str) -> ModerationResult:
        cleaned = question.strip()
        if not cleaned:
            return ModerationResult(prompt_injection=True, raison="La question est vide.")
        if len(cleaned) > 1200:
            return ModerationResult(prompt_injection=True, raison="La question est trop longue.")

        completion = self.client.chat.completions.create(
            messages=[
                {
                    "role": "system",
                    "content": self.prompt_path.read_text(encoding="utf-8"),
                },
                {
                    "role": "user",
                    "content": cleaned,
                },
            ],
            model=self.model,
            response_format={"type": "json_object"},
            temperature=0,
        )

        raw_content = completion.choices[0].message.content or "{}"
        payload = json.loads(raw_content)
        return ModerationResult(
            prompt_injection=bool(payload.get("prompt_injection", False)),
            raison=str(payload.get("raison", "")),
        )
