"""Heuristic query router for the `heuristic_router` agentic mode.

Classifies a query into one of three retrieval regimes based ONLY on text
features of the query itself (not benchmark metadata), so the router is
reproducible in production where the dataset label would not exist.

Motivated by the full-set finding (see `report/RESULTS_TECHNICAL.md`): on
MultiHop-RAG and T2-RAGBench, cross-encoder reranking actually HURTS vs raw
BM25 (0.484 / 0.409 reranked vs 0.535 / 0.535 BM25). So the single largest
achievable lift for the agentic pipeline is to route identifier/number-heavy
queries to BM25 and SKIP reranking entirely; keep hybrid+rerank for everything
else where the cross-encoder genuinely helps (SciFact, NFCorpus, FiQA).
"""

from __future__ import annotations

import re
from enum import StrEnum


class QueryRoute(StrEnum):
    """Which retrieval path the agent should take for a given query."""

    IDENTIFIER_HEAVY = "identifier_heavy"  # BM25 top-k, no rerank (wins on exact-token corpora)
    SEMANTIC = "semantic"                   # Dense top-k (wins on paraphrase-heavy prose)
    HYBRID = "hybrid"                       # Hybrid retrieve + rerank (the fallback / general case)


# A number: optional sign, digits, optional thousands separators, optional decimal.
_NUMBER = re.compile(r"-?\d[\d,]*\.?\d*")
# A 4-digit year, surfaced as a strong temporal/identifier signal.
_YEAR = re.compile(r"\b(?:19|20)\d{2}\b")
# A money token: a $, EUR, USD, or similar, usually paired with digits.
_MONEY = re.compile(r"[\$£€]|\b(?:USD|EUR|GBP|CHF|JPY|CNY)\b", re.IGNORECASE)
# A capital-first alphanumeric token: a proper-noun / ticker / brand signal.
_CAPITAL_TOKEN = re.compile(r"\b[A-Z][A-Za-z0-9]*\b")
# A rough tokenizer; splits on whitespace plus common punctuation.
_WORD = re.compile(r"[A-Za-z0-9]+")

# Marker words typical of natural-language, paraphrase-heavy questions:
# wh- questions, polite framings, broad topic references. Case-insensitive.
_SEMANTIC_MARKERS = {
    "what", "why", "how", "explain", "describe", "mean", "means", "meaning",
    "about", "think", "feel", "opinion", "difference", "vs", "versus",
    "should", "would", "could", "best", "better", "good", "bad",
}

# Capitalised tokens that LOOK like identifiers but are everyday function words
# or domain terms. On NFCorpus and FiQA these drove false-positive BM25 routing
# (e.g. "ACE inhibitors", "best ETF for IRA", "NIH guidelines") which costs
# recall vs. the hybrid+rerank path. The signal we want is a real identifier
# (ticker + date, model + release year), not any uppercase token.
_CAPITAL_STOPWORDS = frozenset({
    # Medicine / science
    "ACE", "DNA", "RNA", "HIV", "AIDS", "COVID", "SARS", "MRSA", "PCR",
    "MRI", "CT", "ICU", "ER", "WHO", "CDC", "FDA", "NIH", "NHS",
    # Finance / regulation
    "ETF", "IRA", "ROTH", "GDP", "CPI", "IPO", "VAT", "GST", "SEC",
    "FICO", "APR", "APY", "P2P", "B2B", "B2C", "CEO", "CFO", "COO", "CTO",
    # Tech / general initialisms
    "AI", "ML", "LLM", "RAG", "GPU", "CPU", "RAM", "SSD", "USB", "LED",
    "LCD", "HTML", "CSS", "JSON", "XML", "API", "URL", "URI", "SDK",
    "TV", "DVD", "CD", "FM", "AM", "PM", "AD", "BC", "GPS", "VPN", "VR", "AR",
    # Geo
    "EU", "US", "USA", "UK", "UN", "USSR", "UAE",
})


class HeuristicQueryRouter:
    """Text-only query classifier.

    No dataset metadata is used; only features extracted from the query string
    itself, so the classification would work the same in production as it does
    on the benchmark.
    """

    def __init__(
        self,
        identifier_min_ratio: float = 0.15,
        semantic_min_marker_count: int = 1,
    ) -> None:
        """
        Args:
            identifier_min_ratio: minimum (identifier_signals / word_count) at or
                above which the query is routed to BM25. 0.15 means "at least
                about 1 identifier-looking token per 7 words."
            semantic_min_marker_count: minimum number of semantic-marker words
                (what/why/how/...) a query must contain to be considered semantic
                when no identifier signal is present.
        """
        self._identifier_min_ratio = identifier_min_ratio
        self._semantic_min_marker_count = semantic_min_marker_count

    def route(self, query_text: str) -> QueryRoute:
        """Classify `query_text` into one of three retrieval regimes."""
        words = _WORD.findall(query_text)
        if not words:
            return QueryRoute.HYBRID

        n_numbers = len(_NUMBER.findall(query_text))
        n_years = len(_YEAR.findall(query_text))
        n_money = len(_MONEY.findall(query_text))
        # Only count capitalized tokens that are NOT the sentence-first word
        # (which would always capitalize), to avoid false-positives on normal
        # English prose. Also exclude single-letter tokens and common
        # all-caps function words/initialisms (see `_CAPITAL_STOPWORDS`).
        capital_tokens = [
            m.group()
            for m in _CAPITAL_TOKEN.finditer(query_text)
            if len(m.group()) >= 2 and m.group().upper() not in _CAPITAL_STOPWORDS
        ]
        if capital_tokens and query_text.strip().startswith(capital_tokens[0]):
            capital_tokens = capital_tokens[1:]
        n_capital = len(capital_tokens)

        identifier_signal = n_numbers + n_years + n_money + n_capital
        identifier_ratio = identifier_signal / len(words)
        # Require at least one *numeric* identifier signal (number, year, or
        # money) before classifying as identifier-heavy. Capital tokens alone
        # misroute NFCorpus/FiQA natural-language questions like "ACE inhibitors"
        # or "best ETF for retirement" to BM25-only; real identifier queries
        # (tickers + dates, SKUs + quantities) almost always include a digit.
        has_numeric_signal = (n_numbers + n_years + n_money) >= 1
        if has_numeric_signal and identifier_ratio >= self._identifier_min_ratio:
            return QueryRoute.IDENTIFIER_HEAVY

        lowered = query_text.lower()
        semantic_hits = sum(1 for marker in _SEMANTIC_MARKERS if marker in lowered.split())
        if semantic_hits >= self._semantic_min_marker_count:
            return QueryRoute.SEMANTIC

        return QueryRoute.HYBRID
