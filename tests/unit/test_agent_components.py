"""Tests for the agentic pipeline's routing/grading/decision components."""

from __future__ import annotations

import pytest

from agentic_rag.agent.corrective_controller import CorrectiveAction, CorrectiveController
from agentic_rag.agent.relevance_grader import RelevanceGrade, RelevanceGrader
from agentic_rag.domain.document import Chunk, DatasetSource
from agentic_rag.domain.retrieval import ScoredChunk


def _scored_chunk(score: float) -> ScoredChunk:
    chunk = Chunk(
        chunk_id="c1", document_id="d1", source=DatasetSource.SCIFACT, text="x", position=0
    )
    return ScoredChunk(chunk=chunk, score=score, rank=0)


class TestRelevanceGrader:
    def _grader(self) -> RelevanceGrader:
        return RelevanceGrader(
            confident_relevant_threshold=0.0, confident_irrelevant_threshold=-5.0
        )

    def test_high_score_is_confident_relevant(self):
        assert self._grader().grade([_scored_chunk(5.0)]).grade == RelevanceGrade.CONFIDENT_RELEVANT

    def test_low_score_is_confident_irrelevant(self):
        grade = self._grader().grade([_scored_chunk(-10.0)]).grade
        assert grade == RelevanceGrade.CONFIDENT_IRRELEVANT

    def test_mid_score_is_ambiguous(self):
        assert self._grader().grade([_scored_chunk(-2.0)]).grade == RelevanceGrade.AMBIGUOUS

    def test_empty_candidates_are_confident_irrelevant(self):
        assert self._grader().grade([]).grade == RelevanceGrade.CONFIDENT_IRRELEVANT

    def test_rejects_misordered_thresholds(self):
        with pytest.raises(ValueError):
            RelevanceGrader(confident_relevant_threshold=-5.0, confident_irrelevant_threshold=0.0)


class TestCorrectiveController:
    def test_confident_relevant_proceeds(self):
        controller = CorrectiveController()
        action = controller.decide(RelevanceGrade.CONFIDENT_RELEVANT, already_retried=False)
        assert action == CorrectiveAction.PROCEED

    def test_confident_irrelevant_abstains(self):
        controller = CorrectiveController()
        action = controller.decide(RelevanceGrade.CONFIDENT_IRRELEVANT, already_retried=False)
        assert action == CorrectiveAction.ABSTAIN

    def test_ambiguous_triggers_retry_once(self):
        controller = CorrectiveController()
        action = controller.decide(RelevanceGrade.AMBIGUOUS, already_retried=False)
        assert action == CorrectiveAction.REWRITE_AND_RETRY

    def test_ambiguous_after_retry_abstains(self):
        controller = CorrectiveController()
        action = controller.decide(RelevanceGrade.AMBIGUOUS, already_retried=True)
        assert action == CorrectiveAction.ABSTAIN
