from md5_mini_rag.documents import SearchResult
from md5_mini_rag.vectordb import _document_score, _merge_search_results, _query_terms


def test_query_terms_keeps_named_entities() -> None:
    assert _query_terms("parle moi de Diego ?") == ["diego"]


def test_query_terms_ignores_generic_question_words() -> None:
    assert _query_terms("Comment s'appelle le chat bleu de Bob ?") == ["chat", "bleu", "bob"]


def test_document_score_uses_exact_tokens_not_substrings() -> None:
    terms = _query_terms("Qui repare les parapluies-radios ?")

    assert _document_score("Diego repare les parapluies-radios casses.", terms) == 2
    assert _document_score("La tarte se prepare sans four.", terms) == 0


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
