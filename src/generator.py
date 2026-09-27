"""Answer generation with grounded extractive and optional OpenAI modes."""

from __future__ import annotations

import os

from .models import Answer, Citation, RetrievalResult


SYSTEM_INSTRUCTIONS = """You are a grounded knowledge assistant.
Answer using only the supplied context. Do not invent facts that are absent from the context.
When the context is insufficient, say that the available documents do not contain the answer.
Keep the answer concise and cite sources using the provided source names."""


class Generator:
    def __init__(self, mode: str | None = None, model: str | None = None) -> None:
        self.mode = mode or os.getenv("RAG_LLM_MODE", "extractive")
        self.model = model or os.getenv("OPENAI_MODEL", "gpt-5.5")

    def answer(self, question: str, results: list[RetrievalResult]) -> Answer:
        if not results:
            return Answer(answer="No relevant context was retrieved.", citations=[], mode=self.mode)
        citations = [
            Citation(source=r.chunk.source, chunk_id=r.chunk.chunk_id, score=round(r.score, 4))
            for r in results
            if r.score > 0
        ]
        if self.mode == "openai":
            return self._openai_answer(question, results, citations)
        return self._extractive_answer(results, citations)

    @staticmethod
    def _extractive_answer(results: list[RetrievalResult], citations: list[Citation]) -> Answer:
        snippets = []
        for result in results[:3]:
            if result.score <= 0:
                continue
            snippets.append(f"[{result.chunk.source}] {result.chunk.text}")
        if not snippets:
            answer = "The available documents do not contain enough relevant information to answer this question."
        else:
            answer = "\n\n".join(snippets)
        return Answer(answer=answer, citations=citations, mode="extractive")

    def _openai_answer(
        self,
        question: str,
        results: list[RetrievalResult],
        citations: list[Citation],
    ) -> Answer:
        try:
            from openai import OpenAI
        except ImportError as exc:  # pragma: no cover
            raise RuntimeError("Install the openai package to use RAG_LLM_MODE=openai") from exc

        context = "\n\n".join(
            f"SOURCE: {result.chunk.source}\n{result.chunk.text}" for result in results if result.score > 0
        )
        client = OpenAI()
        response = client.responses.create(
            model=self.model,
            instructions=SYSTEM_INSTRUCTIONS,
            input=f"Question:\n{question}\n\nContext:\n{context}",
        )
        return Answer(
            answer=response.output_text,
            citations=citations,
            mode="openai",
        )
