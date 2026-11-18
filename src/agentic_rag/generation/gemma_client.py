"""Gemma backend via an Ollama-compatible /api/chat endpoint.

Targets any server that speaks the Ollama /api/chat protocol, including a
locally hosted Gemma model.  Uses only stdlib HTTP so no new production
dependency is required.
"""

from __future__ import annotations

import json
import logging
import urllib.error
import urllib.request

from agentic_rag.generation.llm_client import LLMClient, LlmCompletion, LLMGenerationError

logger = logging.getLogger(__name__)

_DEFAULT_SYSTEM = (
    "You are a reliable AI assistant. Analyze the context if provided and answer the user's "
    "latest request accurately. Use the conversation context when needed. "
    "Do not add any preamble, only return the relevant answer."
)


class GemmaClient(LLMClient):
    """Wraps an Ollama-compatible /api/chat endpoint (e.g. a self-hosted Gemma model)."""

    def __init__(self, base_url: str, model: str) -> None:
        self._url = f"{base_url.rstrip('/')}/api/chat"
        self._model = model
        logger.info("GemmaClient initialised — endpoint: %s  model: %s", self._url, model)

    def complete(
        self, prompt: str, system_prompt: str | None = None, max_new_tokens: int = 256
    ) -> LlmCompletion:
        payload = {
            "model": self._model,
            "messages": [
                {"role": "system", "content": system_prompt or _DEFAULT_SYSTEM},
                {"role": "user", "content": prompt},
            ],
            "stream": False,
            "max_tokens": max_new_tokens,
            "temperature": 0.0,
            "top_p": 1.0,
            "truncate": True,
        }
        body = json.dumps(payload).encode()
        req = urllib.request.Request(
            self._url,
            data=body,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        try:
            with urllib.request.urlopen(req, timeout=120) as resp:
                data = json.loads(resp.read())
        except urllib.error.URLError as exc:
            raise LLMGenerationError(
                f"Gemma endpoint unreachable at '{self._url}': {exc}"
            ) from exc
        except json.JSONDecodeError as exc:
            raise LLMGenerationError(
                f"Gemma endpoint returned non-JSON response: {exc}"
            ) from exc

        try:
            # Support both Ollama format {"message": {"content": ...}}
            # and OpenAI-compatible format {"choices": [{"message": {"content": ...}}]}
            if "message" in data:
                text = data["message"]["content"].strip()
                prompt_tokens = data.get("prompt_eval_count", 0)
                completion_tokens = data.get("eval_count", 0)
            elif "choices" in data and data["choices"]:
                text = data["choices"][0]["message"]["content"].strip()
                usage = data.get("usage", {})
                prompt_tokens = usage.get("prompt_tokens", 0)
                completion_tokens = usage.get("completion_tokens", 0)
            else:
                raise KeyError("neither 'message' nor 'choices' found in response")
        except (KeyError, TypeError) as exc:
            raise LLMGenerationError(
                f"Unexpected response shape from Gemma endpoint: {data!r}"
            ) from exc

        return LlmCompletion(
            text=text,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
        )
