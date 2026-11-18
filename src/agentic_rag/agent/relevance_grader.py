"""Lightweight relevance grading for the agentic pipeline's corrective step.

Deliberately uses the cross-encoder's own score rather than a separate LLM
call. CRAG (Corrective RAG) calls this component a "retrieval evaluator"; a
small trained cross-encoder is both faster and cheaper than an LLM call per
query while still being a calibrated relevance signal -- it's the same score
the reranker itself already trusts to reorder candidates. Thresholds should
be calibrated per corpus; see `agentic-rag calibrate` (`agent/calibration.py`).
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum

from agentic_rag.domain.retrieval import ScoredChunk


class RelevanceGrade(StrEnum):
    """The three-way outcome CRAG's retrieval evaluator distinguishes."""

    CONFIDENT_RELEVANT = "confident_relevant"
    AMBIGUOUS = "ambiguous"
    CONFIDENT_IRRELEVANT = "confident_irrelevant"


@dataclass(frozen=True, slots=True)
class GradingResult:
    """The grade assigned to a candidate pool plus the score it was based on."""

    grade: RelevanceGrade
    top_score: float


class RelevanceGrader:
    """Grades a reranked candidate pool by thresholding its top score."""

    def __init__(
        self, confident_relevant_threshold: float, confident_irrelevant_threshold: float
    ) -> None:
        """
        Args:
            confident_relevant_threshold: scores at or above this are trusted as-is.
            confident_irrelevant_threshold: scores at or below this trigger abstention.

        Raises:
            ValueError: if the thresholds are not properly ordered.
        """
        if confident_irrelevant_threshold >= confident_relevant_threshold:
            raise ValueError(
                "confident_irrelevant_threshold must be lower than confident_relevant_threshold"
            )
        self._confident_relevant_threshold = confident_relevant_threshold
        self._confident_irrelevant_threshold = confident_irrelevant_threshold

    def grade(self, reranked_candidates: list[ScoredChunk]) -> GradingResult:
        """Grade a pool that has already been scored by the cross-encoder (best first)."""
        if not reranked_candidates:
            return GradingResult(grade=RelevanceGrade.CONFIDENT_IRRELEVANT, top_score=float("-inf"))

        return GradingResult(
            grade=self._grade_score(reranked_candidates[0].score),
            top_score=reranked_candidates[0].score,
        )

    def grade_each(self, chunks: list[ScoredChunk]) -> list[RelevanceGrade]:
        """Grade each candidate independently against the thresholds.

        Per-chunk grading is what CRAG's "retrieval evaluator" actually specifies:
        each candidate gets its own relevance label, and the agentic pipeline can
        then drop confidently-irrelevant chunks from the pool instead of returning
        them to the generator. Returned list is parallel to `chunks`.
        """
        return [self._grade_score(c.score) for c in chunks]

    def _grade_score(self, score: float) -> RelevanceGrade:
        if score >= self._confident_relevant_threshold:
            return RelevanceGrade.CONFIDENT_RELEVANT
        if score <= self._confident_irrelevant_threshold:
            return RelevanceGrade.CONFIDENT_IRRELEVANT
        return RelevanceGrade.AMBIGUOUS
