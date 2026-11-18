"""Tests for the document chunking strategies."""

from __future__ import annotations

import pytest

from agentic_rag.data.chunking import (
    SentenceWindowChunker,
    WholeDocumentChunker,
    build_chunker_for_source,
)
from agentic_rag.domain.document import DatasetSource, Document


def _make_document(text: str, title: str = "") -> Document:
    return Document(
        document_id="doc1", source=DatasetSource.SCIFACT, title=title, text=text, metadata={}
    )


def test_whole_document_chunker_returns_single_chunk():
    document = _make_document("Sentence one. Sentence two. Sentence three.")
    chunks = WholeDocumentChunker().chunk(document)
    assert len(chunks) == 1
    assert chunks[0].chunk_id == "doc1::chunk0"
    assert chunks[0].position == 0


def test_sentence_window_chunker_splits_long_document_into_multiple_chunks():
    long_text = " ".join(f"This is sentence number {i} with some words." for i in range(40))
    document = _make_document(long_text)
    chunks = SentenceWindowChunker(target_words=50, overlap_sentences=1).chunk(document)

    assert len(chunks) > 1
    assert [chunk.position for chunk in chunks] == list(range(len(chunks)))


def test_sentence_window_chunker_keeps_short_document_as_one_chunk():
    document = _make_document("Only one short sentence here.")
    chunks = SentenceWindowChunker(target_words=180).chunk(document)
    assert len(chunks) == 1


def test_sentence_window_chunker_rejects_invalid_arguments():
    with pytest.raises(ValueError):
        SentenceWindowChunker(target_words=0)
    with pytest.raises(ValueError):
        SentenceWindowChunker(overlap_sentences=-1)


def test_build_chunker_for_source_uses_whole_document_for_t2_ragbench():
    assert isinstance(build_chunker_for_source(DatasetSource.T2_RAGBENCH), WholeDocumentChunker)


def test_build_chunker_for_source_uses_sentence_window_for_beir():
    assert isinstance(build_chunker_for_source(DatasetSource.SCIFACT), SentenceWindowChunker)


def test_empty_document_produces_no_chunks():
    assert SentenceWindowChunker().chunk(_make_document("")) == []
