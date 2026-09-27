"""Document loaders for text, Markdown, and PDF files."""

from __future__ import annotations

from pathlib import Path

from pypdf import PdfReader

from .models import Document


SUPPORTED = {".txt", ".md", ".pdf"}


def load_path(path: str | Path) -> Document:
    file_path = Path(path)
    if file_path.suffix.lower() not in SUPPORTED:
        raise ValueError(f"Unsupported file type: {file_path.suffix}")

    if file_path.suffix.lower() == ".pdf":
        reader = PdfReader(str(file_path))
        text = "\n".join(page.extract_text() or "" for page in reader.pages)
    else:
        text = file_path.read_text(encoding="utf-8")

    text = text.strip()
    if not text:
        raise ValueError(f"No readable text found in {file_path}")
    return Document(source=file_path.name, text=text)


def load_directory(directory: str | Path) -> list[Document]:
    root = Path(directory)
    documents: list[Document] = []
    for path in sorted(root.rglob("*")):
        if path.is_file() and path.suffix.lower() in SUPPORTED:
            documents.append(load_path(path))
    return documents
