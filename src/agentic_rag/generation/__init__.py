"""Pluggable text-generation layer.

`LLMClient` is the one interface every generation call in this project goes
through. Three implementations are provided, selected by `Settings.llm_backend`:

- `NullLLMClient` ("null"): zero-dependency deterministic stub, used in tests.
- `LocalTransformersClient` ("local", default): a small local Hugging Face
  seq2seq model (e.g. flan-t5-base); needs no API key, runs on CPU.
- `OpenAIChatClient` ("openai"): OpenAI (or an OpenAI-compatible endpoint);
  needs `OPENAI_API_KEY`.
"""
