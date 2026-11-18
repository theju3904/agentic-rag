"""Corrective query rewriting: produces an alternative phrasing when the
initial retrieval pass comes back ambiguous.
"""

from __future__ import annotations

from agentic_rag.generation.llm_client import LLMClient
from agentic_rag.generation.prompts import QUERY_REWRITE_SYSTEM_PROMPT, build_query_rewrite_prompt


class QueryRewriter:
    """Delegates to an `LLMClient` to rephrase a query that retrieved poorly."""

    def __init__(self, llm_client: LLMClient) -> None:
        self._llm_client = llm_client

    def rewrite(self, question: str) -> str:
        """Return an alternative phrasing of `question`.

        Falls back to the original question unchanged if the LLM backend
        returns empty text, so a flaky rewrite never breaks the retry step.
        """
        completion = self._llm_client.complete(
            build_query_rewrite_prompt(question),
            system_prompt=QUERY_REWRITE_SYSTEM_PROMPT,
            max_new_tokens=64,
        )
        rewritten = completion.text.strip()
        return rewritten if rewritten else question
