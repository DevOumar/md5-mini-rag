from __future__ import annotations

from dataclasses import dataclass

from .chunking import chunk_documents
from .config import Settings
from .embeddings import SentenceTransformerEmbedder
from .loaders import load_documents
from .vectorstore import ChromaStore


@dataclass(frozen=True)
class IndexReport:
    documents: int
    chunks: int
    indexed_chunks: int


def build_index(settings: Settings, reset: bool = False) -> IndexReport:
    documents = load_documents(settings.raw_dir)
    if not documents:
        raise RuntimeError(f"No supported documents found in {settings.raw_dir}")

    chunks = chunk_documents(
        documents,
        chunk_size=settings.chunk_size,
        chunk_overlap=settings.chunk_overlap,
    )
    embedder = SentenceTransformerEmbedder(settings.embedding_model)
    store = ChromaStore(settings.chroma_dir, settings.collection_name, embedder)
    if reset:
        store.reset()
    store.upsert_chunks(chunks)
    return IndexReport(
        documents=len(documents),
        chunks=len(chunks),
        indexed_chunks=store.count(),
    )
