"""Complete RAG pipelines: a `Retriever` plus an `AnswerGenerator`.

All five strategies (P1-P5) share the exact same `RetrieveThenGeneratePipeline`
wrapper; what differs between them is which `Retriever` implementation gets
injected. See `pipelines/factory.py` for how each of the five is assembled.
"""
