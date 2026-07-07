from __future__ import annotations

from pathlib import Path

from .documents import SearchResult


CHUNKS_PLACEHOLDER = "{{Chunks}}"


def build_system_prompt(prompt_path: Path, results: list[SearchResult]) -> str:
    template = prompt_path.read_text(encoding="utf-8")
    context = "\n\n".join(_format_context_item(index, result) for index, result in enumerate(results, 1))
    return template.replace(CHUNKS_PLACEHOLDER, context if context else "Aucun chunk trouve.")


def build_user_prompt(question: str, results: list[SearchResult]) -> str:
    return f"""Question utilisateur :
{question}

Redige une reponse courte, precise et sourcee."""


def _format_context_item(index: int, result: SearchResult) -> str:
    source = format_source(result.metadata)
    return f"[S{index}] {source}\n{result.text}"


def format_source(metadata: dict[str, object]) -> str:
    parts: list[str] = []
    file_name = metadata.get("file_name") or metadata.get("source")
    if file_name:
        parts.append(str(file_name))
    if metadata.get("page"):
        parts.append(f"page {metadata['page']}")
    if metadata.get("article"):
        parts.append(f"article {metadata['article']}")
    if metadata.get("section"):
        parts.append(f"section {metadata['section']}")
    if metadata.get("row_id"):
        parts.append(f"id {metadata['row_id']}")
    if metadata.get("categorie"):
        parts.append(f"categorie {metadata['categorie']}")
    if metadata.get("chunk_index") is not None:
        parts.append(f"chunk {metadata['chunk_index']}")
    return " | ".join(parts) if parts else "source inconnue"
