"""Generation-quality metrics: numeric match, exact-match/F1, abstention accuracy, LLM judge."""

from __future__ import annotations

import re
from collections import Counter
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from agentic_rag.generation.llm_client import LLMClient

_NUMBER_PATTERN = re.compile(r"-?\d[\d,]*\.?\d*")
_WORD_PATTERN = re.compile(r"[a-z0-9]+")


def extract_first_number(text: str) -> float | None:
    """Pull the first numeric token out of `text` (commas treated as thousands separators)."""
    match = _NUMBER_PATTERN.search(text.replace(",", ""))
    return float(match.group().replace(",", "")) if match else None


def numeric_match(
    predicted_answer: str, gold_answer: str, relative_tolerance: float = 0.01
) -> bool:
    """True if the first number in `predicted_answer` is within `relative_tolerance` of gold's.

    Mirrors T2-RAGBench's own evaluation protocol (1% relative tolerance).
    """
    predicted_number = extract_first_number(predicted_answer)
    gold_number = extract_first_number(gold_answer)
    if predicted_number is None or gold_number is None:
        return False
    if gold_number == 0:
        return abs(predicted_number) < 1e-9
    return abs(predicted_number - gold_number) / abs(gold_number) <= relative_tolerance


def _normalize_tokens(text: str) -> list[str]:
    return _WORD_PATTERN.findall(text.lower())


def exact_match(predicted_answer: str, gold_answer: str) -> bool:
    """Whitespace/case/punctuation-insensitive exact match."""
    return _normalize_tokens(predicted_answer) == _normalize_tokens(gold_answer)


def token_f1(predicted_answer: str, gold_answer: str) -> float:
    """Token-overlap F1 between predicted and gold answers (SQuAD-style)."""
    predicted_tokens = _normalize_tokens(predicted_answer)
    gold_tokens = _normalize_tokens(gold_answer)
    if not predicted_tokens or not gold_tokens:
        return float(predicted_tokens == gold_tokens)

    num_common = sum((Counter(predicted_tokens) & Counter(gold_tokens)).values())
    if num_common == 0:
        return 0.0
    precision = num_common / len(predicted_tokens)
    recall = num_common / len(gold_tokens)
    return 2 * precision * recall / (precision + recall)


def llm_judge_score(
    question: str,
    context_chunks: list[str],
    answer: str,
    llm_client: LLMClient,
) -> float:
    """LLM-as-judge: score answer quality 0/0.5/1 using the configured LLM.

    The judge receives the question, retrieved context, and generated answer and
    scores on: groundedness in context + direct relevance to the question.
    Returns 0.0 on any LLM failure so one bad call never crashes the eval run.
    """
    from agentic_rag.generation.llm_client import LLMGenerationError
    from agentic_rag.generation.prompts import LLM_JUDGE_SYSTEM_PROMPT, build_judge_prompt

    prompt = build_judge_prompt(question, context_chunks, answer)
    try:
        completion = llm_client.complete(
            prompt, system_prompt=LLM_JUDGE_SYSTEM_PROMPT, max_new_tokens=10
        )
        val = extract_first_number(completion.text.strip())
        return max(0.0, min(1.0, float(val))) if val is not None else 0.0
    except LLMGenerationError:
        return 0.0


def abstention_accuracy(predicted_abstained: list[bool], should_abstain: list[bool]) -> float:
    """Fraction of queries where the pipeline's abstention decision matched the gold label."""
    if not predicted_abstained:
        return 0.0
    correct = sum(p == g for p, g in zip(predicted_abstained, should_abstain, strict=True))
    return correct / len(predicted_abstained)
