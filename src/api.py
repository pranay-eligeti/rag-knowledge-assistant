"""FastAPI application exposing the RAG assistant."""

from __future__ import annotations

from fastapi import FastAPI, HTTPException

from .models import QueryRequest, QueryResponse
from .paths import DATA_DIR
from .service import RagService

service = RagService(DATA_DIR / "docs")
app = FastAPI(title="RAG Knowledge Assistant", version="0.1.0")


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/ingest")
def ingest() -> dict[str, int]:
    try:
        return {"chunks": service.ingest()}
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.post("/query", response_model=QueryResponse)
def query(request: QueryRequest) -> QueryResponse:
    try:
        answer = service.query(request.question, top_k=request.top_k)
        return QueryResponse(**answer.model_dump())
    except (ValueError, RuntimeError) as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
