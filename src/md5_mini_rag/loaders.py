from __future__ import annotations

import csv
from pathlib import Path

from .documents import SourceDocument


SUPPORTED_EXTENSIONS = {".txt", ".md", ".pdf", ".csv"}


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
        elif path.suffix.lower() == ".csv":
            documents.extend(_load_csv(path))
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


def _load_csv(path: Path) -> list[SourceDocument]:
    rows = _read_csv_rows(path)
    documents: list[SourceDocument] = []

    for index, row in enumerate(rows, start=1):
        text = (row.get("text") or "").strip()
        if not text:
            text = " ".join(value.strip() for value in row.values() if value and value.strip())
        if not text:
            continue

        metadata = {
            "source": row.get("source") or str(path),
            "file_name": path.name,
            "file_type": "csv",
            "row_number": index,
        }
        if row.get("id"):
            metadata["row_id"] = row["id"]
        if row.get("categorie"):
            metadata["categorie"] = row["categorie"]

        documents.append(SourceDocument(text=text, source_path=path, metadata=metadata))

    return documents


def _read_csv_rows(path: Path) -> list[dict[str, str]]:
    for encoding in ("utf-8-sig", "utf-8", "cp1252"):
        try:
            with path.open("r", encoding=encoding, newline="") as file:
                return [dict(row) for row in csv.DictReader(file)]
        except UnicodeDecodeError:
            continue
    raise UnicodeDecodeError("csv", b"", 0, 1, f"Unable to decode {path}")
