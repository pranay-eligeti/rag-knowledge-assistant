"""Optional generation boundaries use fake clients, never paid calls."""

import sys
from types import SimpleNamespace

import pytest

from src.generator import Generator
from src.models import Chunk, RetrievalResult


@pytest.mark.parametrize("fail", [False, True])
def test_openai_client_closed_and_errors_sanitized(monkeypatch, fail):
    closed = []

    class Client:
        def __enter__(self):
            return self

        def __exit__(self, *args):
            closed.append(True)

        @property
        def responses(self):
            return self

        def create(self, **kwargs):
            assert "SOURCE: public.md" in kwargs["input"]
            if fail:
                raise RuntimeError("private request body and dummy credential")
            return SimpleNamespace(output_text="Supported answer")

    monkeypatch.setitem(sys.modules, "openai", SimpleNamespace(OpenAI=Client))
    results = [
        RetrievalResult(chunk=Chunk(chunk_id="a", source="public.md", text="Context"), score=0.8)
    ]
    generator = Generator(mode="openai", model="offline-test")
    if fail:
        with pytest.raises(RuntimeError, match="OpenAI generation failed") as error:
            generator.answer("Question", results)
        assert "dummy credential" not in str(error.value)
        assert error.value.__suppress_context__
    else:
        answer = generator.answer("Question", results)
        assert answer.answer == "Supported answer"
        assert answer.citations[0].source == "public.md"
    assert closed == [True]


def test_optional_sdk_missing_is_actionable(monkeypatch):
    monkeypatch.setitem(sys.modules, "openai", None)
    results = [
        RetrievalResult(chunk=Chunk(chunk_id="a", source="public.md", text="Context"), score=0.8)
    ]
    with pytest.raises(RuntimeError, match=r"rag-knowledge-assistant\[openai\]"):
        Generator(mode="openai").answer("Question", results)
