"""Retrieval implementations used by the assistant."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import normalize

from .models import Chunk, RetrievalResult


class TfidfRetriever:
    """Deterministic lexical/vector retrieval suitable for local demos and CI."""

    def __init__(self, chunks: list[Chunk]) -> None:
        if not chunks:
            raise ValueError("At least one chunk is required")
        self.chunks = chunks
        self.vectorizer = TfidfVectorizer(stop_words="english", ngram_range=(1, 2))
        self.matrix = normalize(self.vectorizer.fit_transform([chunk.text for chunk in chunks]))

    def search(self, question: str, top_k: int = 4) -> list[RetrievalResult]:
        if not question.strip():
            raise ValueError("question must not be empty")
        query = normalize(self.vectorizer.transform([question]))
        scores = (self.matrix @ query.T).toarray().ravel()
        order = np.argsort(-scores, kind="stable")[:top_k]
        return [RetrievalResult(chunk=self.chunks[i], score=float(scores[i])) for i in order]


@dataclass
class DenseRetriever:
    """Optional semantic retrieval backed by sentence-transformers.

    Kept optional so the core repo remains fast and deterministic in CI.
    """

    chunks: list[Chunk]
    model_name: str = "sentence-transformers/all-MiniLM-L6-v2"

    def __post_init__(self) -> None:
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError as exc:  # pragma: no cover - depends on optional extra
            raise RuntimeError(
                "Install the optional 'semantic' dependency to use DenseRetriever."
            ) from exc
        self.encoder = SentenceTransformer(self.model_name)
        self.matrix = self.encoder.encode(
            [chunk.text for chunk in self.chunks], normalize_embeddings=True
        )

    def search(self, question: str, top_k: int = 4) -> list[RetrievalResult]:
        query = self.encoder.encode([question], normalize_embeddings=True)[0]
        scores = np.asarray(self.matrix) @ np.asarray(query)
        order = np.argsort(-scores, kind="stable")[:top_k]
        return [RetrievalResult(chunk=self.chunks[i], score=float(scores[i])) for i in order]
