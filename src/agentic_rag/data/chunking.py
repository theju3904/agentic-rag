"""Chunking strategies that turn a `Document` into a list of `Chunk`s.

Chunking policy is frozen per dataset source (README, "Practical dataset
notes"): BEIR and MultiHop-RAG documents are split into sentence-aware,
overlapping windows, while T2-RAGBench documents are indexed whole because
the benchmark's authors already pre-segmented them to ~900 tokens each.
"""

from __future__ import annotations

import re
from abc import ABC, abstractmethod

from agentic_rag.domain.document import Chunk, DatasetSource, Document

_SENTENCE_BOUNDARY = re.compile(r"(?<=[.!?])\s+")


class Chunker(ABC):
    """Splits one `Document` into an ordered list of `Chunk`s."""

    @abstractmethod
    def chunk(self, document: Document) -> list[Chunk]:
        """Return the chunks for `document`, in reading order."""


class WholeDocumentChunker(Chunker):
    """Indexes a document as a single chunk (used for pre-segmented corpora)."""

    def chunk(self, document: Document) -> list[Chunk]:
        text = f"{document.title}\n{document.text}".strip() if document.title else document.text
        return [
            Chunk(
                chunk_id=f"{document.document_id}::chunk0",
                document_id=document.document_id,
                source=document.source,
                text=text,
                position=0,
                metadata=dict(document.metadata),
            )
        ]


class SentenceWindowChunker(Chunker):
    """Groups consecutive sentences into fixed-size, overlapping windows.

    A sentence-aware splitter avoids cutting a sentence in half at a chunk
    boundary, which would otherwise hand the reranker/generator a dangling
    clause with little standalone meaning.
    """

    def __init__(self, target_words: int = 180, overlap_sentences: int = 1) -> None:
        """
        Args:
            target_words: approximate word budget per chunk before a new one starts.
            overlap_sentences: trailing sentences repeated at the start of the
                next chunk, so context isn't lost at a chunk boundary.

        Raises:
            ValueError: if either argument is out of range.
        """
        if target_words <= 0:
            raise ValueError("target_words must be positive")
        if overlap_sentences < 0:
            raise ValueError("overlap_sentences cannot be negative")
        self._target_words = target_words
        self._overlap_sentences = overlap_sentences

    def chunk(self, document: Document) -> list[Chunk]:
        sentences = self._split_sentences(document.text)
        if not sentences:
            return []

        chunks: list[Chunk] = []
        current_sentences: list[str] = []
        current_word_count = 0

        for sentence in sentences:
            current_sentences.append(sentence)
            current_word_count += len(sentence.split())
            if current_word_count >= self._target_words:
                chunks.append(self._build_chunk(document, current_sentences, len(chunks)))
                current_sentences = (
                    current_sentences[-self._overlap_sentences :]
                    if self._overlap_sentences
                    else []
                )
                current_word_count = sum(len(s.split()) for s in current_sentences)

        if current_sentences:
            chunks.append(self._build_chunk(document, current_sentences, len(chunks)))
        return chunks

    def _build_chunk(self, document: Document, sentences: list[str], position: int) -> Chunk:
        body = " ".join(sentences)
        text = f"{document.title}\n{body}".strip() if document.title else body
        return Chunk(
            chunk_id=f"{document.document_id}::chunk{position}",
            document_id=document.document_id,
            source=document.source,
            text=text,
            position=position,
            metadata=dict(document.metadata),
        )

    @staticmethod
    def _split_sentences(text: str) -> list[str]:
        return [s.strip() for s in _SENTENCE_BOUNDARY.split(text.strip()) if s.strip()]


def build_chunker_for_source(source: DatasetSource) -> Chunker:
    """Return the frozen chunking policy for `source` (README, "Practical dataset notes")."""
    if source == DatasetSource.T2_RAGBENCH:
        return WholeDocumentChunker()
    return SentenceWindowChunker()
