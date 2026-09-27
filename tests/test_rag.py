import json
from pathlib import Path

from fastapi.testclient import TestClient

from src.api import app
from src.chunker import chunk_document
from src.generator import Generator
from src.models import Document
from src.retriever import TfidfRetriever
from src.service import RagService


def test_chunk_overlap_is_present():
    doc = Document(source="demo.md", text="one two three four five six seven eight")
    chunks = chunk_document(doc, chunk_size=5, overlap=2)
    assert chunks[0].text.endswith("five")
    assert chunks[1].text.startswith("four")


def test_retrieval_recall_at_one_on_eval_set():
    root = Path(__file__).resolve().parents[1]
    service = RagService(root / "data" / "docs", generator=Generator(mode="extractive"))
    service.ingest()
    rows = [json.loads(line) for line in (root / "data" / "eval" / "questions.jsonl").read_text().splitlines()]
    hits = 0
    for row in rows:
        result = service.retriever.search(row["question"], top_k=1)  # type: ignore[union-attr]
        if result and result[0].chunk.source == row["expected_source"]:
            hits += 1
    assert hits / len(rows) >= 2 / 3


def test_answer_contains_citations_when_context_matches():
    chunks = chunk_document(Document(source="source.md", text="Recall@K measures whether relevant context was retrieved."))
    answer = Generator(mode="extractive").answer(
        "What measures retrieval?", TfidfRetriever(chunks).search("What measures retrieval?", 2)
    )
    assert answer.citations
    assert answer.citations[0].source == "source.md"


def test_api_health_and_query():
    client = TestClient(app)
    assert client.get("/health").json() == {"status": "ok"}
    response = client.post("/query", json={"question": "What are the stages of a RAG pipeline?", "top_k": 2})
    assert response.status_code == 200
    payload = response.json()
    assert payload["answer"]
    assert payload["citations"]
