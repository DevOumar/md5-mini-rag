from __future__ import annotations

from pathlib import Path
from typing import Any

from .documents import Chunk, SearchResult
from .embeddings import Embedder


class ChromaStore:
    def __init__(
        self,
        persist_dir: Path,
        collection_name: str,
        embedder: Embedder,
    ) -> None:
        import chromadb

        self.persist_dir = persist_dir
        self.collection_name = collection_name
        self.embedder = embedder
        self.persist_dir.mkdir(parents=True, exist_ok=True)
        self.client = chromadb.PersistentClient(path=str(self.persist_dir))
        self.collection = self.client.get_or_create_collection(
            name=collection_name,
            metadata={"embedding_model": embedder.model_name},
        )
        self._validate_embedding_model()

    def reset(self) -> None:
        try:
            self.client.delete_collection(self.collection_name)
        except ValueError:
            pass
        self.collection = self.client.get_or_create_collection(
            name=self.collection_name,
            metadata={"embedding_model": self.embedder.model_name},
        )

    def upsert_chunks(self, chunks: list[Chunk], batch_size: int = 64) -> None:
        for start in range(0, len(chunks), batch_size):
            batch = chunks[start : start + batch_size]
            embeddings = self.embedder.encode([chunk.text for chunk in batch])
            self.collection.upsert(
                ids=[chunk.id for chunk in batch],
                documents=[chunk.text for chunk in batch],
                embeddings=embeddings,
                metadatas=[_clean_metadata(chunk.metadata) for chunk in batch],
            )

    def search(self, query: str, top_k: int) -> list[SearchResult]:
        query_embedding = self.embedder.encode([query])[0]
        result = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=top_k,
            include=["documents", "metadatas", "distances"],
        )
        ids = result.get("ids", [[]])[0]
        documents = result.get("documents", [[]])[0]
        metadatas = result.get("metadatas", [[]])[0]
        distances = result.get("distances", [[]])[0]

        return [
            SearchResult(
                id=ids[index],
                text=documents[index],
                metadata=metadatas[index] or {},
                distance=distances[index] if index < len(distances) else None,
            )
            for index in range(len(ids))
        ]

    def count(self) -> int:
        return self.collection.count()

    def _validate_embedding_model(self) -> None:
        metadata = self.collection.metadata or {}
        indexed_model = metadata.get("embedding_model")
        if indexed_model and indexed_model != self.embedder.model_name:
            raise ValueError(
                "Embedding model mismatch. "
                f"Index uses {indexed_model!r}, current model is {self.embedder.model_name!r}. "
                "Reindex the corpus before querying."
            )


def _clean_metadata(metadata: dict[str, Any]) -> dict[str, str | int | float | bool]:
    cleaned: dict[str, str | int | float | bool] = {}
    for key, value in metadata.items():
        if isinstance(value, (str, int, float, bool)):
            cleaned[key] = value
        elif value is not None:
            cleaned[key] = str(value)
    return cleaned
