"""Smoke test for FastAPI app construction (routes registered, no lifespan run)."""

from __future__ import annotations

from agentic_rag.api.main import create_app


def test_create_app_registers_expected_routes():
    # Use the OpenAPI schema (a stable public API) rather than walking
    # `app.routes` directly, whose internal wrapper types vary across
    # FastAPI/Starlette versions.
    app = create_app()
    paths = set(app.openapi()["paths"].keys())

    assert "/health" in paths
    assert "/query" in paths
    assert "/compare" in paths
    assert "/pipelines" in paths
    assert "/documents/{document_id}" in paths
    assert "/chunks/{chunk_id}" in paths
