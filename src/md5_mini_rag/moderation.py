from __future__ import annotations

import re


BLOCKED_PATTERNS = [
    r"ignore\s+(les\s+)?(instructions|regles|consignes)",
    r"oublie\s+(les\s+)?(instructions|regles|consignes)",
    r"system\s+prompt",
    r"developer\s+message",
    r"revele\s+tes\s+instructions",
    r"affiche\s+tes\s+instructions",
]


def validate_question(question: str) -> tuple[bool, str | None]:
    cleaned = question.strip()
    if not cleaned:
        return False, "La question est vide."
    if len(cleaned) > 1200:
        return False, "La question est trop longue."

    lowered = cleaned.lower()
    for pattern in BLOCKED_PATTERNS:
        if re.search(pattern, lowered):
            return False, "Question bloquee : tentative d'injection de prompt detectee."
    return True, None
