"""Prompt templates used by the answer generator and the agentic pipeline.

Kept as plain string-formatting functions (no Jinja/templating dependency)
since every prompt here is short and has a fixed shape.
"""

from __future__ import annotations

ANSWER_SYSTEM_PROMPT = (
    "You are a precise question-answering assistant. Answer using ONLY the "
    "provided context. If the context does not contain the answer, respond "
    'exactly with "INSUFFICIENT_CONTEXT" and nothing else. Be concise.'
)

NUMERIC_ANSWER_SYSTEM_PROMPT = (
    "You are a precise numeric data extraction assistant. Extract the specific "
    "numeric value requested from the provided context. Respond with ONLY the "
    "number and its unit (e.g. '42 MW', '3.5%', '120 rpm'). Do not include any "
    "explanation or surrounding text. If the context does not contain the answer, "
    'respond exactly with "INSUFFICIENT_CONTEXT" and nothing else.'
)

LLM_JUDGE_SYSTEM_PROMPT = (
    "You are an expert evaluator of RAG system answers. Given a question, retrieved "
    "context, and a generated answer, score the answer quality on a 0–1 scale:\n"
    "  1.0 — fully grounded in the context AND directly answers the question\n"
    "  0.5 — partially answers or mostly grounded but missing key details\n"
    "  0.0 — not grounded, hallucinated, refuses to answer, or off-topic\n"
    "Respond with ONLY a single number: 0, 0.5, or 1. No explanation."
)


def build_answer_prompt(question: str, context_chunks: list[str]) -> str:
    """Build the final generation prompt from the retrieved context chunks."""
    numbered_context = "\n\n".join(f"[{i + 1}] {chunk}" for i, chunk in enumerate(context_chunks))
    return f"Context:\n{numbered_context}\n\nQuestion: {question}\nAnswer:"


def build_judge_prompt(question: str, context_chunks: list[str], answer: str) -> str:
    """Build the LLM-as-judge evaluation prompt."""
    context_text = "\n\n".join(f"[{i + 1}] {chunk}" for i, chunk in enumerate(context_chunks))
    return (
        f"Context:\n{context_text}\n\n"
        f"Question: {question}\n"
        f"Answer: {answer}\n"
        "Score:"
    )


RELEVANCE_GRADING_SYSTEM_PROMPT = (
    "You grade whether a passage can answer a question. Respond with exactly "
    'one word: "RELEVANT" or "IRRELEVANT".'
)


def build_relevance_grading_prompt(question: str, passage: str) -> str:
    """Build the prompt the (optional) LLM-backed relevance grader sends per passage."""
    return f'Question: "{question}"\nPassage: "{passage}"\nGrading:'


QUERY_REWRITE_SYSTEM_PROMPT = (
    "You rewrite search queries to improve retrieval recall. Given a question "
    "that initial retrieval failed on, produce ONE alternative phrasing that "
    "preserves the original meaning but uses different wording and synonyms. "
    "Respond with only the rewritten query."
)


def build_query_rewrite_prompt(question: str) -> str:
    """Build the prompt the agentic pipeline's corrective query rewriter sends."""
    return f"Original question: {question}\nRewritten question:"
