from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class SourceDocument:
    text: str
    source_path: Path
    metadata: dict[str, Any]


@dataclass(frozen=True)
class Chunk:
    id: str
    text: str
    metadata: dict[str, Any]


@dataclass(frozen=True)
class SearchResult:
    id: str
    text: str
    metadata: dict[str, Any]
    distance: float | None
