"""Generic pipeline: retrieve with any `Retriever`, then generate with an `AnswerGenerator`.

Every one of the five pipelines (P1-P5) is an instance of this same class;
only the injected `Retriever` differs between them (see `pipelines/factory.py`).
"""

from __future__ import annotations

from agentic_rag.domain.evaluation import GenerationResult
from agentic_rag.domain.retrieval import RetrievalResult, RetrievalStrategy
from agentic_rag.generation.answer_generator import AnswerGenerator
from agentic_rag.pipelines.base import RagPipeline
from agentic_rag.retrieval.base import Retriever

_DEFAULT_CONTEXT_CHUNKS_FOR_GENERATION = 5


class RetrieveThenGeneratePipeline(RagPipeline):
    """Composes any `Retriever` with the shared `AnswerGenerator`."""

    def __init__(
        self,
        strategy: RetrievalStrategy,
        retriever: Retriever,
        answer_generator: AnswerGenerator,
        context_chunk_count: int = _DEFAULT_CONTEXT_CHUNKS_FOR_GENERATION,
    ) -> None:
        self.strategy = strategy
        self._retriever = retriever
        self._answer_generator = answer_generator
        self._context_chunk_count = context_chunk_count

    @property
    def retriever(self) -> Retriever:
        return self._retriever

    def answer(
        self, query_id: str, question: str, top_k: int, dataset: str | None = None
    ) -> tuple[RetrievalResult, GenerationResult]:
        retrieval_result = self._retriever.retrieve(question, query_id, top_k)
        generation_result = self._answer_generator.generate(
            query_id=query_id,
            strategy=self.strategy,
            question=question,
            context_chunks=retrieval_result.top_k_texts(self._context_chunk_count),
            dataset=dataset,
        )
        return retrieval_result, generation_result

    def retrieve_only(self, query_id: str, question: str, top_k: int) -> RetrievalResult:
        return self._retriever.retrieve(question, query_id, top_k)
