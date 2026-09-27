"""Core Pydantic/data models for the RAG pipeline."""

from __future__ import annotations

from pydantic import BaseModel, Field


class Document(BaseModel):
    source: str
    text: str


class Chunk(BaseModel):
    chunk_id: str
    source: str
    text: str
    metadata: dict[str, str] = Field(default_factory=dict)


class RetrievalResult(BaseModel):
    chunk: Chunk
    score: float


class Citation(BaseModel):
    source: str
    chunk_id: str
    score: float


class Answer(BaseModel):
    answer: str
    citations: list[Citation]
    mode: str


class QueryRequest(BaseModel):
    question: str = Field(min_length=3)
    top_k: int = Field(default=4, ge=1, le=10)


class QueryResponse(BaseModel):
    answer: str
    citations: list[Citation]
    mode: str
