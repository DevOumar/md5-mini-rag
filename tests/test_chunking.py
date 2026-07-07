from md5_mini_rag.chunking import chunk_documents, split_text
from md5_mini_rag.documents import SourceDocument


def test_split_text_keeps_chunks_under_limit() -> None:
    text = "Phrase une. Phrase deux.\n\n" * 20

    chunks = split_text(text, chunk_size=80, chunk_overlap=10)

    assert chunks
    assert all(len(chunk) <= 100 for chunk in chunks)


def test_chunk_documents_adds_article_metadata() -> None:
    document = SourceDocument(
        text="Article R001\nLe chat bleu de Bob s'appelle Henri.",
        source_path="demo.md",  # type: ignore[arg-type]
        metadata={"source": "demo.md", "file_name": "demo.md"},
    )

    chunks = chunk_documents([document], chunk_size=200, chunk_overlap=20)

    assert len(chunks) == 1
    assert chunks[0].metadata["article"] == "R001"
    assert chunks[0].metadata["chunk_index"] == 0
