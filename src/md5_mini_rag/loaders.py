from __future__ import annotations

from pathlib import Path

from .documents import SourceDocument


SUPPORTED_EXTENSIONS = {".txt", ".md", ".pdf"}


def load_documents(raw_dir: Path) -> list[SourceDocument]:
    if not raw_dir.exists():
        raise FileNotFoundError(f"Corpus directory not found: {raw_dir}")

    documents: list[SourceDocument] = []
    for path in sorted(raw_dir.rglob("*")):
        if not path.is_file() or path.name.startswith("."):
            continue
        if path.suffix.lower() not in SUPPORTED_EXTENSIONS:
            continue

        if path.suffix.lower() == ".pdf":
            documents.extend(_load_pdf(path))
        else:
            documents.append(_load_text(path))

    return [doc for doc in documents if doc.text.strip()]


def _load_text(path: Path) -> SourceDocument:
    text = path.read_text(encoding="utf-8")
    return SourceDocument(
        text=text,
        source_path=path,
        metadata={
            "source": str(path),
            "file_name": path.name,
            "file_type": path.suffix.lower().lstrip("."),
        },
    )


def _load_pdf(path: Path) -> list[SourceDocument]:
    try:
        from pypdf import PdfReader
    except ImportError as exc:
        raise RuntimeError("pypdf is required to load PDF files") from exc

    reader = PdfReader(str(path))
    documents: list[SourceDocument] = []
    for index, page in enumerate(reader.pages, start=1):
        text = page.extract_text() or ""
        documents.append(
            SourceDocument(
                text=text,
                source_path=path,
                metadata={
                    "source": str(path),
                    "file_name": path.name,
                    "file_type": "pdf",
                    "page": index,
                },
            )
        )
    return documents
