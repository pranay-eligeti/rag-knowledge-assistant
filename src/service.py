"""High-level orchestration of ingestion, retrieval, and generation."""

from __future__ import annotations

from pathlib import Path

from .chunker import chunk_document
from .generator import Generator
from .loaders import load_directory
from .models import Answer, Chunk
from .retriever import TfidfRetriever


class RagService:
    def __init__(self, knowledge_dir: str | Path, generator: Generator | None = None) -> None:
        self.knowledge_dir = Path(knowledge_dir)
        self.generator = generator or Generator()
        self.chunks: list[Chunk] = []
        self.retriever: TfidfRetriever | None = None

    def ingest(self) -> int:
        documents = load_directory(self.knowledge_dir)
        self.chunks = [chunk for doc in documents for chunk in chunk_document(doc)]
        if not self.chunks:
            raise ValueError(f"No supported documents found in {self.knowledge_dir}")
        self.retriever = TfidfRetriever(self.chunks)
        return len(self.chunks)

    def query(self, question: str, top_k: int = 4) -> Answer:
        if self.retriever is None:
            self.ingest()
        assert self.retriever is not None
        results = self.retriever.search(question, top_k=top_k)
        return self.generator.answer(question, results)
