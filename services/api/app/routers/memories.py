from fastapi import APIRouter, HTTPException

from app.models import (
    CreateMemoryRequest,
    CreateMemoryResponse,
    ReportRequest,
    UpdateMemoryRequest,
)

router = APIRouter()


@router.post("", response_model=CreateMemoryResponse)
def create_memory(body: CreateMemoryRequest) -> CreateMemoryResponse:
    """Create a draft memory and return presigned upload URLs for each file."""
    raise HTTPException(501, "Not implemented")


@router.post("/{memory_id}/process")
def start_processing(memory_id: str) -> dict:
    """Enqueue the processing workflow (see services/workers)."""
    raise HTTPException(501, "Not implemented")


@router.get("/{memory_id}")
def get_memory(memory_id: str) -> dict:
    """Return the memory's manifest (packages/manifest) and processing status."""
    raise HTTPException(501, "Not implemented")


@router.get("/{memory_id}/events")
def stream_progress(memory_id: str):
    """Server-Sent Events stream of per-stage pipeline progress."""
    raise HTTPException(501, "Not implemented")


@router.patch("/{memory_id}")
def update_memory(memory_id: str, body: UpdateMemoryRequest) -> dict:
    """Update title, description, visibility and location."""
    raise HTTPException(501, "Not implemented")


@router.post("/{memory_id}/report")
def report_memory(memory_id: str, body: ReportRequest) -> dict:
    """Flag a memory for moderation."""
    raise HTTPException(501, "Not implemented")
