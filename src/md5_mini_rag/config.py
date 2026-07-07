from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv


load_dotenv()


@dataclass(frozen=True)
class Settings:
    raw_dir: Path = Path(os.getenv("RAW_DIR", "data/raw"))
    chroma_dir: Path = Path(os.getenv("CHROMA_DIR", "data/chroma"))
    collection_name: str = os.getenv("COLLECTION_NAME", "mini_rag")
    embedding_model: str = os.getenv(
        "EMBEDDING_MODEL",
        "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
    )
    groq_api_key: str | None = os.getenv("GROQ_API_KEY") or None
    groq_model: str = os.getenv("GROQ_MODEL", "llama-3.1-8b-instant")
    chunk_size: int = int(os.getenv("CHUNK_SIZE", "1200"))
    chunk_overlap: int = int(os.getenv("CHUNK_OVERLAP", "180"))
    top_k: int = int(os.getenv("TOP_K", "4"))


def get_settings() -> Settings:
    return Settings()
