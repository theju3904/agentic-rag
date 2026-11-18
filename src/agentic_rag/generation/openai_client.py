"""OpenAI-compatible LLM backend.

Works with OpenAI itself, or any OpenAI-compatible endpoint (Azure OpenAI, a
local vLLM/Ollama shim, ...) by overriding `base_url`.
"""

from __future__ import annotations

from openai import OpenAI

from agentic_rag.generation.llm_client import LLMClient, LlmCompletion, LLMGenerationError


class OpenAIChatClient(LLMClient):
    """Wraps the OpenAI chat-completions API behind the shared `LLMClient` interface."""

    def __init__(self, api_key: str, model: str, base_url: str | None = None) -> None:
        if not api_key:
            raise ValueError("OpenAIChatClient requires a non-empty api_key")
        self._client = OpenAI(api_key=api_key, base_url=base_url)
        self._model = model

    def complete(
        self, prompt: str, system_prompt: str | None = None, max_new_tokens: int = 256
    ) -> LlmCompletion:
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})

        try:
            response = self._client.chat.completions.create(
                model=self._model,
                messages=messages,  # type: ignore[arg-type] # dicts match the runtime schema
                max_tokens=max_new_tokens,
                temperature=0.0,
            )
        except Exception as exc:
            raise LLMGenerationError(f"OpenAI completion failed for model '{self._model}'") from exc

        usage = response.usage
        return LlmCompletion(
            text=(response.choices[0].message.content or "").strip(),
            prompt_tokens=usage.prompt_tokens if usage else 0,
            completion_tokens=usage.completion_tokens if usage else 0,
        )
