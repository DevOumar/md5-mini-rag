from __future__ import annotations

import hashlib
from typing import Protocol

import numpy as np


class Embedder(Protocol):
    model_name: str

    def encode(self, texts: list[str]) -> list[list[float]]:
        ...


class SentenceTransformerEmbedder:
    def __init__(self, model_name: str) -> None:
        from sentence_transformers import SentenceTransformer

        self.model_name = model_name
        self._model = SentenceTransformer(model_name)

    def encode(self, texts: list[str]) -> list[list[float]]:
        vectors = self._model.encode(
            texts,
            normalize_embeddings=True,
            show_progress_bar=False,
        )
        return vectors.astype(float).tolist()


class HashEmbedder:
    """Small deterministic embedder used for tests without external models."""

    def __init__(self, dimensions: int = 32) -> None:
        self.model_name = f"hash-{dimensions}"
        self.dimensions = dimensions

    def encode(self, texts: list[str]) -> list[list[float]]:
        vectors = []
        for text in texts:
            vector = np.zeros(self.dimensions, dtype=float)
            for token in text.lower().split():
                digest = hashlib.sha1(token.encode("utf-8")).digest()
                index = digest[0] % self.dimensions
                vector[index] += 1.0
            norm = np.linalg.norm(vector)
            if norm:
                vector = vector / norm
            vectors.append(vector.tolist())
        return vectors
