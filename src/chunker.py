"""Deterministic word-window chunking with overlap."""

from __future__ import annotations

import re

from .models import Chunk, Document

_WORD_RE = re.compile(r"\S+")


def chunk_document(document: Document, chunk_size: int = 120, overlap: int = 24) -> list[Chunk]:
    if chunk_size <= 0 or overlap < 0 or overlap >= chunk_size:
        raise ValueError(
            "chunk_size must be > 0 and overlap must satisfy 0 <= overlap < chunk_size"
        )

    words = _WORD_RE.findall(document.text)
    chunks: list[Chunk] = []
    step = chunk_size - overlap

    for start in range(0, len(words), step):
        current = words[start : start + chunk_size]
        if not current:
            break
        index = len(chunks)
        chunks.append(
            Chunk(
                chunk_id=f"{document.source}#chunk-{index:03d}",
                source=document.source,
                text=" ".join(current),
                metadata={"start_word": str(start), "end_word": str(start + len(current))},
            )
        )
        if start + chunk_size >= len(words):
            break
    return chunks
