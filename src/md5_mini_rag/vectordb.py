from __future__ import annotations

import re
import unicodedata
from pathlib import Path
from typing import Any

from .documents import Chunk, SearchResult
from .embeddings import Embedder


class VectorDB:
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
        keyword_results = self._keyword_search(query, top_k)
        if keyword_results:
            return keyword_results

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

        vector_results = [
            SearchResult(
                id=ids[index],
                text=documents[index],
                metadata=metadatas[index] or {},
                distance=distances[index] if index < len(distances) else None,
            )
            for index in range(len(ids))
        ]
        return _merge_search_results([], vector_results, top_k)

    def count(self) -> int:
        return self.collection.count()

    def _keyword_search(self, query: str, top_k: int) -> list[SearchResult]:
        terms = _query_terms(query)
        if not terms:
            return []

        candidates: dict[str, tuple[int, SearchResult]] = {}
        for term in terms:
            for variant in _term_variants(term):
                response = self.collection.get(
                    where_document={"$contains": variant},
                    include=["documents", "metadatas"],
                    limit=max(top_k * 4, 12),
                )
                ids = response.get("ids", [])
                documents = response.get("documents", [])
                metadatas = response.get("metadatas", [])
                for index, item_id in enumerate(ids):
                    text = documents[index]
                    score = _document_score(text, terms)
                    if score < _minimum_keyword_score(terms):
                        continue
                    existing = candidates.get(item_id)
                    if existing and existing[0] >= score:
                        continue
                    candidates[item_id] = (
                        score,
                        SearchResult(
                            id=item_id,
                            text=text,
                            metadata=metadatas[index] or {},
                            distance=0.0,
                        ),
                    )
        ranked = sorted(
            candidates.values(),
            key=lambda item: (-item[0], str(item[1].metadata.get("row_id", item[1].id))),
        )
        return [result for _, result in ranked[:top_k]]

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


_STOPWORDS = {
    "appelle",
    "comment",
    "s'appelle",
    "avec",
    "dans",
    "de",
    "des",
    "dis",
    "donne",
    "est",
    "les",
    "moi",
    "parle",
    "pour",
    "que",
    "qui",
    "quoi",
    "sur",
    "une",
}


def _query_terms(query: str) -> list[str]:
    terms = []
    for term in re.findall(r"[\wÀ-ÿ'-]+", query, flags=re.UNICODE):
        cleaned = _normalize_token(term.strip("'-"))
        if len(cleaned) >= 3 and cleaned not in _STOPWORDS:
            terms.append(cleaned)
    return list(dict.fromkeys(terms))


def _term_variants(term: str) -> list[str]:
    return list(dict.fromkeys([term, term.capitalize(), term.upper()]))


def _document_score(text: str, terms: list[str]) -> int:
    tokens = set(_document_tokens(text))
    return sum(1 for term in terms if term in tokens)


def _document_tokens(text: str) -> list[str]:
    return [
        _normalize_token(token)
        for token in re.findall(r"[\wÀ-ÿ'-]+", text, flags=re.UNICODE)
        if token.strip("'-")
    ]


def _normalize_token(token: str) -> str:
    normalized = unicodedata.normalize("NFKD", token.lower())
    return "".join(char for char in normalized if not unicodedata.combining(char))


def _minimum_keyword_score(terms: list[str]) -> int:
    return 1 if len(terms) == 1 else 2


def _merge_search_results(
    keyword_results: list[SearchResult],
    vector_results: list[SearchResult],
    top_k: int,
) -> list[SearchResult]:
    merged: list[SearchResult] = []
    seen: set[str] = set()
    for result in [*keyword_results, *vector_results]:
        if result.id in seen:
            continue
        seen.add(result.id)
        merged.append(result)
        if len(merged) >= top_k:
            break
    return merged
