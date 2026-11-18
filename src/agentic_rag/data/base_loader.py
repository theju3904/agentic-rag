"""Abstract base class every dataset loader implements."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass

from agentic_rag.domain.document import Document
from agentic_rag.domain.evaluation import RelevanceJudgment
from agentic_rag.domain.query import Query


@dataclass(slots=True)
class LoadedDataset:
    """The three parallel collections every retrieval benchmark is built from."""

    documents: list[Document]
    queries: list[Query]
    judgments: list[RelevanceJudgment]


class DatasetLoadError(RuntimeError):
    """Raised when a dataset cannot be downloaded, parsed, or validated."""


class DatasetLoader(ABC):
    """Downloads/parses one benchmark dataset into the shared domain model."""

    @abstractmethod
    def load(self) -> LoadedDataset:
        """Fetch (from the Hugging Face cache or network) and parse the dataset.

        Raises:
            DatasetLoadError: if the dataset cannot be downloaded, parsed, or
                turns out to be empty after parsing.
        """
