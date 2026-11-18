"""The `LLMClient` strategy interface plus its zero-dependency fallback."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class LlmCompletion:
    """The result of one `LLMClient.complete` call, with token accounting."""

    text: str
    prompt_tokens: int
    completion_tokens: int


class LLMGenerationError(RuntimeError):
    """Raised when an `LLMClient` cannot produce a completion."""


class LLMClient(ABC):
    """Strategy interface for turning a prompt into text.

    See the `generation` package docstring for the concrete implementations
    and `generation/factory.py` for how one is selected at runtime.
    """

    @abstractmethod
    def complete(
        self, prompt: str, system_prompt: str | None = None, max_new_tokens: int = 256
    ) -> LlmCompletion:
        """Generate a completion for `prompt`.

        Raises:
            LLMGenerationError: if the backend fails to produce a completion.
        """


class NullLLMClient(LLMClient):
    """Zero-dependency fallback: deterministic, makes no model calls.

    Used when `LLM_BACKEND=null` (e.g. in unit tests), or as an automatic
    safety net if a real backend fails to initialize (see
    `generation/factory.py`). It never hallucinates an answer; it clearly
    states that no backend is configured instead.
    """

    _STUB_ANSWER = (
        "[NullLLMClient] no LLM backend is configured; cannot generate a free-text answer."
    )

    def complete(
        self, prompt: str, system_prompt: str | None = None, max_new_tokens: int = 256
    ) -> LlmCompletion:
        return LlmCompletion(
            text=self._STUB_ANSWER,
            prompt_tokens=_approximate_token_count(prompt),
            completion_tokens=0,
        )


def _approximate_token_count(text: str) -> int:
    """A word-count approximation; avoids pulling in a tokenizer just for logging."""
    return len(text.split())
