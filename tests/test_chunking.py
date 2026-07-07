from pathlib import Path

from md5_mini_rag.chunking import chunk_documents, split_text
from md5_mini_rag.documents import SourceDocument
from md5_mini_rag.loaders import load_documents


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


def test_load_documents_supports_teacher_csv(tmp_path: Path) -> None:
    csv_path = tmp_path / "corpus.csv"
    csv_path.write_text(
        '"id","text","source","categorie"\n'
        '"chunk_001","Henri dort sur le refrigerateur.","carnet_de_bob","animaux"\n',
        encoding="utf-8",
    )

    documents = load_documents(tmp_path)

    assert len(documents) == 1
    assert documents[0].text == "Henri dort sur le refrigerateur."
    assert documents[0].metadata["row_id"] == "chunk_001"
    assert documents[0].metadata["source"] == "carnet_de_bob"
    assert documents[0].metadata["categorie"] == "animaux"
