"""Endpoints for inspecting raw documents and their chunks.

This is the "see the chunk/document" requirement: every chunk the retrievers
operate on, and the source document it came from, is browsable here.
"""

from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session

from agentic_rag.api.dependencies import get_session
from agentic_rag.api.schemas import ChunkResponse, DocumentResponse
from agentic_rag.storage.postgres.repository import ChunkRepository, DocumentRepository

documents_router = APIRouter(prefix="/documents", tags=["documents"])
chunks_router = APIRouter(prefix="/chunks", tags=["documents"])


@documents_router.get("/{document_id}", response_model=DocumentResponse)
def get_document(document_id: str, session: Session = Depends(get_session)) -> DocumentResponse:
    document = DocumentRepository(session).get(document_id)
    if document is None:
        raise HTTPException(status_code=404, detail=f"Document '{document_id}' not found")
    return DocumentResponse(
        document_id=document.document_id,
        source=document.source.value,
        title=document.title,
        text=document.text,
        metadata=document.metadata,
    )


@documents_router.get("/{document_id}/chunks", response_model=list[ChunkResponse])
def list_document_chunks(
    document_id: str, session: Session = Depends(get_session)
) -> list[ChunkResponse]:
    chunks = ChunkRepository(session).list_by_document(document_id)
    if not chunks:
        raise HTTPException(status_code=404, detail=f"No chunks found for document '{document_id}'")
    return [_to_chunk_response(chunk) for chunk in chunks]


@chunks_router.get("/{chunk_id}", response_model=ChunkResponse)
def get_chunk(chunk_id: str, session: Session = Depends(get_session)) -> ChunkResponse:
    chunk = ChunkRepository(session).get(chunk_id)
    if chunk is None:
        raise HTTPException(status_code=404, detail=f"Chunk '{chunk_id}' not found")
    return _to_chunk_response(chunk)


def _to_chunk_response(chunk) -> ChunkResponse:
    return ChunkResponse(
        chunk_id=chunk.chunk_id,
        document_id=chunk.document_id,
        source=chunk.source.value,
        text=chunk.text,
        position=chunk.position,
        metadata=chunk.metadata,
    )
