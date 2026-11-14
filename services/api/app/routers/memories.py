import json
import re

from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import FileResponse, RedirectResponse, Response

from app.models import (
    CreateMemoryRequest,
    CreateMemoryResponse,
    ReportRequest,
    UpdateMemoryRequest,
)

from app.storage import LocalStore, get_store

router = APIRouter()

# Pipeline stages in DAG order (services/workers/memoryscapes_workers/pipeline.py).
STAGES = ["ingest", "poses", "splat", "separate", "localize", "soundscape", "publish"]
# Outputs each stage writes last, so their presence means the stage finished. The visual
# stages and publish write a _stage.json marker; the audio stages don't use one.
DONE_FILE = {"separate": "stems.json", "localize": "sources.json", "soundscape": "scene.json"}
_ID = re.compile(r"^[A-Za-z0-9_-]{1,128}$")
# Never served: the raw upload (unprocessed video) and stage bookkeeping.
_PRIVATE = ("upload/",)


@router.post("", response_model=CreateMemoryResponse)
def create_memory(body: CreateMemoryRequest) -> CreateMemoryResponse:
    """Create a draft memory and return presigned upload URLs for each file."""
    raise HTTPException(501, "Not implemented")


@router.post("/{memory_id}/process")
def start_processing(memory_id: str) -> dict:
    """Enqueue the processing workflow (see services/workers)."""
    raise HTTPException(501, "Not implemented")


@router.get("/{memory_id}")
def get_memory(memory_id: str, request: Request) -> dict:
    """Return the memory's manifest (packages/manifest), per-stage status, and the base
    URL that the manifest's relative asset URLs resolve against.

    No auth yet: every memory is readable, so run this locally only.
    """
    _check_id(memory_id)
    store = get_store()
    stages = {s: _stage_done(store, memory_id, s) for s in STAGES}
    if not any(stages.values()) and not store.list(f"{memory_id}/upload/"):
        raise HTTPException(404, f"No memory {memory_id!r}")
    manifest_key = f"{memory_id}/publish/manifest.json"
    manifest = json.loads(store.read_bytes(manifest_key)) if store.exists(manifest_key) else None
    return {
        "id": memory_id,
        "status": "ready" if manifest else "processing",
        "stages": stages,
        "manifest": manifest,
        "asset_base": str(request.url_for("get_memory_file", memory_id=memory_id, key="")),
    }


@router.get("/{memory_id}/files/{key:path}", name="get_memory_file")
def get_memory_file(memory_id: str, key: str) -> Response:
    """One stage output (splat LODs, audio, path, ...): a file locally, or a redirect to
    a short-lived presigned URL on S3."""
    _check_id(memory_id)
    parts = key.split("/")
    if not key or ".." in parts or key.startswith(_PRIVATE) or parts[-1] == "_stage.json":
        raise HTTPException(404, "Not found")
    store = get_store()
    full = f"{memory_id}/{key}"
    if isinstance(store, LocalStore):
        try:
            path = store.path(full)
        except ValueError:
            raise HTTPException(404, "Not found") from None
        if not path.is_file():
            raise HTTPException(404, "Not found")
        media = "application/octet-stream" if path.suffix == ".spz" else None
        return FileResponse(path, media_type=media)
    if not store.exists(full):
        raise HTTPException(404, "Not found")
    return RedirectResponse(store.presigned_url(full), status_code=307)


def _check_id(memory_id: str) -> None:
    if not _ID.match(memory_id):
        raise HTTPException(404, "Not found")


def _stage_done(store, memory_id: str, stage: str) -> bool:
    return store.exists(f"{memory_id}/{stage}/{DONE_FILE.get(stage, '_stage.json')}")


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
