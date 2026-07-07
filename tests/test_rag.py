from md5_mini_rag.documents import SearchResult
from md5_mini_rag.rag import _best_chunk_is_too_far


def test_best_chunk_threshold_blocks_distant_result() -> None:
    results = [
        SearchResult(
            id="chunk_033",
            text="La capitale administrative de Villebrume-les-Cuilleres est un banc public.",
            metadata={"row_id": "chunk_033"},
            distance=1.5063,
        )
    ]

    assert _best_chunk_is_too_far(results, max_distance=1.2)


def test_best_chunk_threshold_accepts_close_result() -> None:
    results = [
        SearchResult(
            id="chunk_001",
            text="Le chat bleu de Bob s'appelle Henri.",
            metadata={"row_id": "chunk_001"},
            distance=0.4127,
        )
    ]

    assert not _best_chunk_is_too_far(results, max_distance=1.2)
