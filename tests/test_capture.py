import json
import socket
from pathlib import Path

import pytest

from src.capture import capture_run

ROOT = Path(__file__).resolve().parents[1]


def test_capture_is_offline_and_repeatable(monkeypatch):
    def blocked(*args, **kwargs):
        raise AssertionError("Network forbidden")

    monkeypatch.setattr(socket.socket, "connect", blocked)
    monkeypatch.setenv("RAG_LLM_MODE", "openai")
    rows = [
        json.loads(line) for line in (ROOT / "data/eval/questions.jsonl").read_text().splitlines()
    ]
    first = capture_run(ROOT / "data/docs", rows, run_id="baseline")
    assert first == capture_run(ROOT / "data/docs", rows, run_id="baseline")
    assert len(first["cases"]) == 3
    assert first["cases"][0]["metadata"]["generation_mode"] == "extractive"
    assert first["cases"][0]["retrieved_documents"][0]["metadata"]["chunk_id"]


@pytest.mark.parametrize(
    "rows,k",
    [
        ([], 3),
        ([{"question": "q"}], 0),
        ([{"question": " "}], 3),
        ([{"question": "q", "expected_source": []}], 3),
        ([{"question": "q", "case_id": "a"}] * 2, 3),
    ],
)
def test_invalid_capture_inputs(rows, k):
    with pytest.raises(ValueError):
        capture_run(ROOT / "data/docs", rows, run_id="test", top_k=k)
