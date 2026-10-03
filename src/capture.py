"""Optional offline JSON export; no evaluation harness runtime dependency."""

import argparse
import json
from pathlib import Path
from typing import Any

from .generator import Generator
from .paths import DATA_DIR
from .service import RagService


def capture_run(
    knowledge_dir: Path,
    questions: list[dict[str, Any]],
    *,
    run_id: str,
    top_k: int = 3,
    revision: str | None = None,
) -> dict[str, Any]:
    """Capture the same ordered TF-IDF context used by extractive generation.

    Source filenames are metric IDs; chunk IDs remain provenance. The exporter
    deliberately fixes extractive mode even if live-generation env vars are set.
    """
    if not run_id.strip() or not questions or not 1 <= top_k <= 10:
        raise ValueError("Require run ID, questions, and top_k within [1, 10]")
    service = RagService(knowledge_dir, generator=Generator(mode="extractive"))
    service.ingest()
    assert service.retriever is not None
    cases = []
    seen = set()
    for index, row in enumerate(questions):
        case_id = row.get("case_id", f"rag-{index + 1}")
        question = row.get("question")
        if not isinstance(case_id, str) or not case_id.strip() or case_id in seen:
            raise ValueError("Missing or duplicate case ID")
        if not isinstance(question, str) or not question.strip():
            raise ValueError("Question must be nonblank text")
        seen.add(case_id)
        results = service.retriever.search(question, top_k=top_k)
        answer = service.generator.answer(question, results)
        expected = row.get("expected_source")
        if expected is not None and (not isinstance(expected, str) or not expected.strip()):
            raise ValueError("Expected source must be nonblank text")
        reference = row.get("reference_answer")
        if reference is not None and not isinstance(reference, str):
            raise ValueError("Reference answer must be text or null")
        cases.append(
            {
                "case_id": case_id,
                "question": question,
                "reference_answer": reference,
                "generated_answer": answer.answer,
                "retrieved_documents": [
                    {
                        "doc_id": result.chunk.source,
                        "text": result.chunk.text,
                        "score": result.score,
                        "metadata": {
                            "source_id": result.chunk.source,
                            "chunk_id": result.chunk.chunk_id,
                        },
                    }
                    for result in results
                ],
                "expected_document_ids": [expected] if expected is not None else None,
                "expected_citations": [expected] if expected is not None else None,
                "actual_citations": [citation.source for citation in answer.citations],
                "metadata": {"generation_mode": answer.mode},
            }
        )
    return {
        "schema_version": "capture-1",
        "run_id": run_id,
        "system": "rag-knowledge-assistant/tfidf-extractive",
        "revision": revision,
        "metadata": {"top_k": top_k, "metric_id_unit": "source"},
        "cases": cases,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Export an offline TF-IDF RAG capture")
    parser.add_argument("--knowledge-dir", type=Path, default=DATA_DIR / "docs")
    parser.add_argument("--questions", type=Path, default=DATA_DIR / "eval/questions.jsonl")
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--revision")
    parser.add_argument("--top-k", type=int, default=3)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    questions = [
        json.loads(line)
        for line in args.questions.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    capture = capture_run(
        args.knowledge_dir, questions, run_id=args.run_id, top_k=args.top_k, revision=args.revision
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(capture, sort_keys=True, indent=2, allow_nan=False) + "\n", encoding="utf-8"
    )


if __name__ == "__main__":
    main()
