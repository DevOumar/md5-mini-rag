import math

from md5_mini_rag.embeddings import HashEmbedder


def test_hash_embedder_returns_normalized_vectors() -> None:
    embedder = HashEmbedder(dimensions=16)

    vector = embedder.encode(["chat bleu Henri"])[0]
    norm = math.sqrt(sum(value * value for value in vector))

    assert math.isclose(norm, 1.0)
