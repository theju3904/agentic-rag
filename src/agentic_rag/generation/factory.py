"""Constructs the configured `LLMClient` implementation from `Settings`."""

from __future__ import annotations

import logging

from agentic_rag.config import Settings
from agentic_rag.generation.llm_client import LLMClient, NullLLMClient

logger = logging.getLogger(__name__)


def build_llm_client(settings: Settings) -> LLMClient:
    """Return the `LLMClient` selected by `settings.llm_backend`.

    Backend options:
      ``gemma``  — self-hosted Gemma via Ollama /api/chat (default).
      ``openai`` — OpenAI chat-completions (requires ``OPENAI_API_KEY``).
      ``local``  — local Hugging Face seq2seq model (e.g. flan-t5-base).
      ``null``   — no-op stub; used in tests or retrieval-only workflows.

    Falls back to ``NullLLMClient`` (with a logged warning) if the requested
    backend fails to initialise, so a misconfigured backend never crashes a
    retrieval-only run that never calls generation.
    """
    backend = settings.llm_backend.lower()
    try:
        if backend == "gemma":
            from agentic_rag.generation.gemma_client import GemmaClient

            return GemmaClient(base_url=settings.gemma_base_url, model=settings.gemma_model)

        if backend == "openai":
            from agentic_rag.generation.openai_client import OpenAIChatClient

            if not settings.openai_api_key:
                raise ValueError("LLM_BACKEND=openai requires OPENAI_API_KEY to be set")
            return OpenAIChatClient(api_key=settings.openai_api_key, model=settings.openai_model)

        if backend == "local":
            from agentic_rag.generation.local_transformers_client import LocalTransformersClient

            return LocalTransformersClient(settings.local_generation_model)

        if backend != "null":
            logger.warning("Unrecognised LLM_BACKEND '%s'; falling back to null", backend)

    except Exception:
        logger.exception("Failed to initialise LLM backend '%s'; falling back to null", backend)
        return NullLLMClient()

    return NullLLMClient()
