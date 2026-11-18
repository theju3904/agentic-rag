"""Local, key-free LLM backend using a small instruction-tuned Hugging Face model.

This is the default backend for this project since it requires no API key:
`google/flan-t5-base` runs acceptably on CPU for short RAG-style generation.

Uses `AutoModelForSeq2SeqLM` directly rather than `transformers.pipeline(...)`:
the high-level pipeline's unified "text-generation" task in this transformers
version only recognizes causal-LM architectures, and silently echoes the
input back unchanged for an encoder-decoder model like T5 instead of raising
-- calling `.generate()` on the model ourselves avoids that failure mode.
"""

from __future__ import annotations

import logging

from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

from agentic_rag.generation.llm_client import LLMClient, LlmCompletion, LLMGenerationError

logger = logging.getLogger(__name__)

_MAX_INPUT_TOKENS = 512


class LocalTransformersClient(LLMClient):
    """Wraps a local Hugging Face sequence-to-sequence model (e.g. flan-t5)."""

    def __init__(self, model_name: str) -> None:
        """
        Raises:
            LLMGenerationError: if the model cannot be downloaded or loaded.
        """
        try:
            self._tokenizer = AutoTokenizer.from_pretrained(model_name)
            self._model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
        except Exception as exc:
            raise LLMGenerationError(
                f"Failed to load local generation model '{model_name}'"
            ) from exc
        self._model_name = model_name

    def complete(
        self, prompt: str, system_prompt: str | None = None, max_new_tokens: int = 256
    ) -> LlmCompletion:
        full_prompt = f"{system_prompt}\n\n{prompt}" if system_prompt else prompt
        try:
            inputs = self._tokenizer(
                full_prompt, return_tensors="pt", truncation=True, max_length=_MAX_INPUT_TOKENS
            )
            output_ids = self._model.generate(**inputs, max_new_tokens=max_new_tokens)
        except Exception as exc:
            raise LLMGenerationError(
                f"Local model '{self._model_name}' failed to generate"
            ) from exc

        decoded = self._tokenizer.decode(output_ids[0], skip_special_tokens=True)
        generated_text = decoded if isinstance(decoded, str) else " ".join(decoded)
        return LlmCompletion(
            text=generated_text.strip(),
            prompt_tokens=inputs["input_ids"].shape[1],
            completion_tokens=output_ids.shape[1],
        )
