"""Decision logic mapping a relevance grade to a corrective action.

This is the central "agentic" decision point: given the same initial
retrieval, the controller chooses to proceed, retry with a rewritten query,
or abstain, rather than always running a fixed, linear pipeline.
"""

from __future__ import annotations

from enum import StrEnum

from agentic_rag.agent.relevance_grader import RelevanceGrade


class CorrectiveAction(StrEnum):
    """The three actions the agentic pipeline can take after grading a pool."""

    PROCEED = "proceed"
    REWRITE_AND_RETRY = "rewrite_and_retry"
    ABSTAIN = "abstain"


class CorrectiveController:
    """Maps a `RelevanceGrade` to the next `CorrectiveAction`."""

    def decide(self, grade: RelevanceGrade, already_retried: bool) -> CorrectiveAction:
        """
        Args:
            grade: the `RelevanceGrade` of the current candidate pool.
            already_retried: whether a rewrite-and-retry was already attempted
                for this query, to bound the number of retries at exactly one.
        """
        if grade == RelevanceGrade.CONFIDENT_RELEVANT:
            return CorrectiveAction.PROCEED
        if grade == RelevanceGrade.CONFIDENT_IRRELEVANT:
            return CorrectiveAction.ABSTAIN
        return CorrectiveAction.ABSTAIN if already_retried else CorrectiveAction.REWRITE_AND_RETRY
