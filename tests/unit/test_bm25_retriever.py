"""Tests for the pure in-memory BM25 retriever (no DB/network required)."""

from __future__ import annotations

import pytest

from agentic_rag.domain.document import Chunk, DatasetSource
from agentic_rag.domain.retrieval import RetrievalStrategy
from agentic_rag.retrieval.bm25_retriever import Bm25Retriever, tokenize


def _chunk(chunk_id: str, text: str) -> Chunk:
    return Chunk(
        chunk_id=chunk_id, document_id=chunk_id, source=DatasetSource.SCIFACT, text=text, position=0
    )


def test_tokenize_lowercases_and_strips_punctuation():
    assert tokenize("Hello, World! 123") == ["hello", "world", "123"]


def test_bm25_retriever_ranks_exact_keyword_match_first():
    chunks = [
        _chunk("c1", "The mitochondria is the powerhouse of the cell."),
        _chunk("c2", "Paris is the capital of France."),
        _chunk("c3", "Photosynthesis converts light into chemical energy."),
    ]
    retriever = Bm25Retriever(chunks)

    result = retriever.retrieve("What is the capital of France?", query_id="q1", top_k=2)

    assert result.strategy == RetrievalStrategy.SPARSE_BM25
    assert result.ranked_chunks[0].chunk.chunk_id == "c2"
    assert result.ranked_chunks[0].rank == 0


def test_bm25_retriever_rejects_empty_corpus():
    with pytest.raises(ValueError):
        Bm25Retriever([])


def test_bm25_retriever_respects_top_k():
    chunks = [_chunk(f"c{i}", f"document number {i} about topic {i}") for i in range(10)]
    retriever = Bm25Retriever(chunks)

    result = retriever.retrieve("topic 3", query_id="q1", top_k=3)

    assert len(result.ranked_chunks) == 3
