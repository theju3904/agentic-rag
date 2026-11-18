"""Turns retrieved context + a question into a final answer (or an abstention)."""

from __future__ import annotations

import time

from agentic_rag.domain.evaluation import GenerationResult
from agentic_rag.domain.retrieval import RetrievalStrategy
from agentic_rag.generation.llm_client import LLMClient
from agentic_rag.generation.prompts import (
    ANSWER_SYSTEM_PROMPT,
    NUMERIC_ANSWER_SYSTEM_PROMPT,
    build_answer_prompt,
)

_NUMERIC_DATASETS = {"t2_ragbench"}

_ABSTENTION_MARKER = "INSUFFICIENT_CONTEXT"
ABSTENTION_MESSAGE = "I don't have enough information in the retrieved context to answer this."


class AnswerGenerator:
    """Wraps an `LLMClient` with the RAG answer-generation prompt and abstention logic."""

    def __init__(self, llm_client: LLMClient, max_new_tokens: int = 200) -> None:
        self._llm_client = llm_client
        self._max_new_tokens = max_new_tokens

    def generate(
        self,
        query_id: str,
        strategy: RetrievalStrategy,
        question: str,
        context_chunks: list[str],
        dataset: str | None = None,
    ) -> GenerationResult:
        """Generate an answer for `question` given already-retrieved `context_chunks`.

        Abstains immediately without calling the LLM if `context_chunks` is
        empty, since there would be nothing to ground an answer in.
        """
        start_time = time.perf_counter()
        if not context_chunks:
            return GenerationResult(
                query_id=query_id,
                strategy=strategy,
                answer_text=ABSTENTION_MESSAGE,
                abstained=True,
                prompt_tokens=0,
                completion_tokens=0,
                latency_seconds=time.perf_counter() - start_time,
            )

        system_prompt = (
            NUMERIC_ANSWER_SYSTEM_PROMPT if dataset in _NUMERIC_DATASETS else ANSWER_SYSTEM_PROMPT
        )
        prompt = build_answer_prompt(question, context_chunks)
        completion = self._llm_client.complete(
            prompt, system_prompt=system_prompt, max_new_tokens=self._max_new_tokens
        )
        abstained = _ABSTENTION_MARKER in completion.text.upper()
        answer_text = ABSTENTION_MESSAGE if abstained else completion.text

        return GenerationResult(
            query_id=query_id,
            strategy=strategy,
            answer_text=answer_text,
            abstained=abstained,
            prompt_tokens=completion.prompt_tokens,
            completion_tokens=completion.completion_tokens,
            latency_seconds=time.perf_counter() - start_time,
        )
