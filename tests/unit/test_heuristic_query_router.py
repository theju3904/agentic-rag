"""Unit tests for the heuristic query router."""

from __future__ import annotations

import pytest

from agentic_rag.agent.heuristic_query_router import HeuristicQueryRouter, QueryRoute


@pytest.fixture
def router() -> HeuristicQueryRouter:
    return HeuristicQueryRouter()


class TestIdentifierHeavy:
    """Queries with concrete identifiers (dates/numbers/money + entities) go to BM25.

    Identifier-heavy classification now REQUIRES at least one numeric signal
    (number, year, or money token) alongside the density rule. Pure capital-token
    density is no longer sufficient - that caused NFCorpus/FiQA false positives
    on queries like "ACE inhibitors" or "best ETF for IRA".
    """

    def test_year_and_entity(self, router: HeuristicQueryRouter) -> None:
        assert router.route("Apple revenue 2024") == QueryRoute.IDENTIFIER_HEAVY

    def test_money_and_number(self, router: HeuristicQueryRouter) -> None:
        assert router.route("Tesla reported $1.2B in Q3 2023") == QueryRoute.IDENTIFIER_HEAVY

    def test_proper_nouns_with_year(self, router: HeuristicQueryRouter) -> None:
        # Proper nouns + a year => still identifier-heavy.
        assert router.route("Nvidia AMD Intel release 2023") == QueryRoute.IDENTIFIER_HEAVY


class TestIdentifierRequiresNumericCoOccurrence:
    """Capital-token density alone must NOT trigger identifier-heavy routing.

    These queries used to misroute to BM25-only and lost recall on NFCorpus and
    FiQA, where the hybrid+rerank path is strictly better.
    """

    def test_brand_list_without_numeric_falls_through(
        self, router: HeuristicQueryRouter
    ) -> None:
        # Four capitalised brand tokens, zero numbers/years/money. The old rule
        # routed this to BM25; the new rule falls through to hybrid.
        assert router.route("Nvidia AMD Intel GPU") != QueryRoute.IDENTIFIER_HEAVY

    def test_proper_nouns_only_falls_through(self, router: HeuristicQueryRouter) -> None:
        assert router.route("Apple Microsoft Google") != QueryRoute.IDENTIFIER_HEAVY


class TestCapitalStopwordsIgnored:
    """Common all-caps initialisms (ETF, ACE, NIH, ...) must not count as identifiers."""

    def test_ace_inhibitors_is_not_identifier(self, router: HeuristicQueryRouter) -> None:
        # NFCorpus-style medical query that previously misrouted.
        assert router.route("ACE inhibitors effect on blood pressure") != (
            QueryRoute.IDENTIFIER_HEAVY
        )

    def test_etf_question_is_not_identifier(self, router: HeuristicQueryRouter) -> None:
        # FiQA-style retirement question. "best" is a semantic marker, so this
        # should land in SEMANTIC, not IDENTIFIER_HEAVY.
        assert router.route("best ETF for a Roth IRA") == QueryRoute.SEMANTIC

    def test_nih_guidelines_is_not_identifier(self, router: HeuristicQueryRouter) -> None:
        assert router.route("NIH guidelines vitamin D supplementation") != (
            QueryRoute.IDENTIFIER_HEAVY
        )


class TestSemantic:
    """Queries that are natural-language, no identifiers, with wh-markers go to dense."""

    def test_how_question(self, router: HeuristicQueryRouter) -> None:
        assert router.route("how do i explain this to a beginner") == QueryRoute.SEMANTIC

    def test_why_question(self, router: HeuristicQueryRouter) -> None:
        assert router.route("why does the stock price drop") == QueryRoute.SEMANTIC

    def test_mean_question(self, router: HeuristicQueryRouter) -> None:
        assert router.route("what does inflation mean for me") == QueryRoute.SEMANTIC


class TestHybrid:
    """Queries without strong identifier or semantic signals fall back to hybrid."""

    def test_short_declarative(self, router: HeuristicQueryRouter) -> None:
        assert router.route("inflation and interest rates") == QueryRoute.HYBRID

    def test_empty_string(self, router: HeuristicQueryRouter) -> None:
        assert router.route("") == QueryRoute.HYBRID


class TestFirstWordNotCountedAsCapital:
    """The sentence-first capital token should not drive the identifier-heavy signal
    on its own - that would misclassify every well-written question."""

    def test_single_leading_capital_is_not_enough(self, router: HeuristicQueryRouter) -> None:
        # "What causes diabetes" has one leading capital, no numbers, no money, and
        # 'what' is a semantic marker -> should go semantic.
        assert router.route("What causes diabetes") == QueryRoute.SEMANTIC
