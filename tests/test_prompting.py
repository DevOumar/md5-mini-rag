from md5_mini_rag.documents import SearchResult
from md5_mini_rag.prompting import build_user_prompt, format_source


def test_format_source_includes_metadata() -> None:
    source = format_source(
        {
            "file_name": "code.md",
            "page": 2,
            "article": "R001",
            "row_id": "chunk_001",
            "categorie": "animaux",
            "chunk_index": 3,
        }
    )

    assert "code.md" in source
    assert "page 2" in source
    assert "article R001" in source
    assert "id chunk_001" in source
    assert "categorie animaux" in source
    assert "chunk 3" in source


def test_build_user_prompt_numbers_sources() -> None:
    result = SearchResult(
        id="1",
        text="Le chat bleu de Bob s'appelle Henri.",
        metadata={"file_name": "code.md", "article": "R001"},
        distance=0.1,
    )

    prompt = build_user_prompt("Comment s'appelle le chat bleu de Bob ?", [result])

    assert "[S1]" in prompt
    assert "R001" in prompt
    assert "Comment s'appelle le chat bleu de Bob ?" in prompt
