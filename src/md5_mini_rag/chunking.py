from __future__ import annotations

import hashlib
import re

from .documents import Chunk, SourceDocument

ARTICLE_RE = re.compile(r"\bArticle\s+([LRD]\.?\s*\d[\w.-]*)", re.IGNORECASE)
HEADING_RE = re.compile(r"^(#{1,6}\s+.+|[A-Z][A-Z0-9 '\-]{8,})$", re.MULTILINE)


def chunk_documents(
    documents: list[SourceDocument],
    chunk_size: int = 1200,
    chunk_overlap: int = 180,
) -> list[Chunk]:
    chunks: list[Chunk] = []
    for document in documents:
        for index, text in enumerate(split_text(document.text, chunk_size, chunk_overlap)):
            metadata = dict(document.metadata)
            metadata["chunk_index"] = index
            article = _first_match(ARTICLE_RE, text)
            heading = _first_match(HEADING_RE, text)
            if article:
                metadata["article"] = article
            if heading:
                metadata["section"] = heading.strip("# ").strip()

            chunk_id = _chunk_id(metadata.get("source", "unknown"), index, text)
            chunks.append(Chunk(id=chunk_id, text=text, metadata=metadata))
    return chunks


def split_text(text: str, chunk_size: int, chunk_overlap: int) -> list[str]:
    cleaned = normalize_text(text)
    if not cleaned:
        return []
    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive")
    if chunk_overlap < 0 or chunk_overlap >= chunk_size:
        raise ValueError("chunk_overlap must be between 0 and chunk_size - 1")

    paragraphs = [p.strip() for p in re.split(r"\n\s*\n", cleaned) if p.strip()]
    chunks: list[str] = []
    current = ""

    for paragraph in paragraphs:
        units = _split_large_paragraph(paragraph, chunk_size)
        for unit in units:
            if not current:
                current = unit
            elif len(current) + 2 + len(unit) <= chunk_size:
                current = f"{current}\n\n{unit}"
            else:
                chunks.append(current.strip())
                overlap = _tail_overlap(current, chunk_overlap)
                current = f"{overlap}\n\n{unit}".strip() if overlap else unit

    if current:
        chunks.append(current.strip())

    return chunks


def normalize_text(text: str) -> str:
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def _split_large_paragraph(paragraph: str, chunk_size: int) -> list[str]:
    if len(paragraph) <= chunk_size:
        return [paragraph]

    sentences = re.split(r"(?<=[.!?])\s+", paragraph)
    units: list[str] = []
    current = ""
    for sentence in sentences:
        if len(sentence) > chunk_size:
            if current:
                units.append(current.strip())
                current = ""
            units.extend(_split_by_words(sentence, chunk_size))
        elif not current:
            current = sentence
        elif len(current) + 1 + len(sentence) <= chunk_size:
            current = f"{current} {sentence}"
        else:
            units.append(current.strip())
            current = sentence
    if current:
        units.append(current.strip())
    return units


def _split_by_words(text: str, chunk_size: int) -> list[str]:
    words = text.split()
    units: list[str] = []
    current: list[str] = []
    length = 0
    for word in words:
        added = len(word) + (1 if current else 0)
        if current and length + added > chunk_size:
            units.append(" ".join(current))
            current = [word]
            length = len(word)
        else:
            current.append(word)
            length += added
    if current:
        units.append(" ".join(current))
    return units


def _tail_overlap(text: str, max_chars: int) -> str:
    if max_chars == 0:
        return ""
    tail = text[-max_chars:]
    first_space = tail.find(" ")
    return tail[first_space + 1 :].strip() if first_space >= 0 else tail.strip()


def _first_match(pattern: re.Pattern[str], text: str) -> str | None:
    match = pattern.search(text)
    if not match:
        return None
    if match.lastindex:
        return match.group(1).strip()
    return match.group(0).strip()


def _chunk_id(source: object, index: int, text: str) -> str:
    digest = hashlib.sha1(f"{source}:{index}:{text[:120]}".encode("utf-8")).hexdigest()
    return digest[:20]
