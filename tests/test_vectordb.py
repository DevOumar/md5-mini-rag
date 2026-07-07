from md5_mini_rag.documents import SearchResult
from md5_mini_rag.vectordb import _merge_search_results, _query_terms


def test_query_terms_keeps_named_entities() -> None:
    assert _query_terms("parle moi de Diego ?") == ["diego"]


def test_merge_search_results_prioritizes_keyword_results() -> None:
    keyword = SearchResult(
        id="chunk_152",
        text="Diego repare gratuitement le parapluie-radio de Nadia.",
        metadata={"row_id": "chunk_152"},
        distance=0.0,
    )
    vector = SearchResult(
        id="chunk_001",
        text="Le chat bleu de Bob s'appelle Henri.",
        metadata={"row_id": "chunk_001"},
        distance=1.7,
    )

    results = _merge_search_results([keyword], [vector], top_k=2)

    assert [result.id for result in results] == ["chunk_152", "chunk_001"]
