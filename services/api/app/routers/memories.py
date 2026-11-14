import json
import re
from pathlib import PurePosixPath

from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import FileResponse, RedirectResponse, Response

from app import db, pipeline
from app.models import (
    CreateMemoryRequest,
    CreateMemoryResponse,
    MemoryStatus,
    MemorySummary,
    ReportRequest,
    UpdateMemoryRequest,
    UploadResult,
    UploadTarget,
)
from app.settings import get_settings
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

# What ingest accepts (services/workers/memoryscapes_workers/visual/video.py). It finds the
# files in upload/ by extension, so we save them as video.<ext> and audio.<ext>.
VIDEO_EXTS = {".mp4", ".mov", ".m4v", ".mkv", ".webm", ".avi", ".3gp"}
AUDIO_EXTS = {".wav", ".flac", ".m4a", ".mp3", ".aac", ".ogg", ".opus", ".caf", ".aiff", ".aif"}
# Same rule as the debugger's uploader (tools/debugger/server/jobs.ts).
_FILE_NAME = re.compile(r"^[\w.\- ()]{1,128}$")
UPLOAD_META = "upload/meta.json"
_CHUNK = 1 << 20


# --------------------------------------------------------------------------- create + upload


@router.post("", response_model=CreateMemoryResponse, status_code=201)
def create_memory(body: CreateMemoryRequest, request: Request) -> CreateMemoryResponse:
    """Create a draft memory and say where to upload each file (docs/webapp-plan/backend.md).

    Locally each ``url`` is this API's own upload endpoint; with S3 it will be a presigned
    URL. Either way the client PUTs the file's bytes to it.
    """
    if body.input_kind != "video":
        raise HTTPException(422, "Only video input is supported for now (panoramas come later).")
    stored = _plan_uploads(body.file_names)

    memory = db.create_memory(get_settings().dev_user_id, title=body.title)
    _write_meta(memory["id"], {"expected": stored})
    return CreateMemoryResponse(id=memory["id"], uploads=[
        UploadTarget(file_name=name, url=str(request.url_for(
            "upload_file", memory_id=memory["id"], file_name=name)))
        for name in body.file_names])


def _plan_uploads(file_names: list[str]) -> dict[str, str]:
    """original name -> stored name (video.<ext> / audio.<ext>). Exactly one video, at most
    one audio file."""
    if not file_names:
        raise HTTPException(422, "List the files you'll upload (one video, optionally one audio file).")
    stored: dict[str, str] = {}
    for name in file_names:
        if not _FILE_NAME.match(name) or name in (".", ".."):
            raise HTTPException(422, f"File name {name!r} has characters we don't accept. Rename it and try again.")
        if name in stored:
            raise HTTPException(422, f"{name!r} is listed twice.")
        stored[name] = _stored_name(name)
    kinds = [v.split(".")[0] for v in stored.values()]
    if kinds.count("video") != 1:
        raise HTTPException(422, "Upload exactly one video.")
    if kinds.count("audio") > 1:
        raise HTTPException(422, "Upload at most one separate audio file.")
    return stored


def _stored_name(name: str) -> str:
    ext = PurePosixPath(name).suffix.lower()
    if ext in VIDEO_EXTS:
        return f"video{ext}"
    if ext in AUDIO_EXTS:
        return f"audio{ext}"
    raise HTTPException(422, f"{name!r}: unsupported file type. Videos: {', '.join(sorted(VIDEO_EXTS))}. "
                             f"Audio: {', '.join(sorted(AUDIO_EXTS))}.")


@router.put("/{memory_id}/upload/{file_name}", name="upload_file", response_model=UploadResult)
async def upload_file(memory_id: str, file_name: str, request: Request) -> UploadResult:
    """Receive one file's raw bytes (not a form) and save it to ``{id}/upload/``."""
    _check_id(memory_id)
    memory = _require_memory(memory_id)
    if memory["status"] == MemoryStatus.processing.value or pipeline.is_running(memory_id):
        raise HTTPException(409, "This memory is being processed. Wait for it to finish.")
    meta = _read_meta(memory_id)
    stored = meta.get("expected", {}).get(file_name)
    if stored is None:
        raise HTTPException(422, f"{file_name!r} wasn't listed when the memory was created.")

    max_bytes = get_settings().max_upload_mb * 1024 * 1024
    too_large = HTTPException(413, f"Files can be at most {get_settings().max_upload_mb} MB.")
    declared = request.headers.get("content-length")
    if declared and declared.isdigit() and int(declared) > max_bytes:
        raise too_large

    store = get_store()
    if not isinstance(store, LocalStore):
        raise HTTPException(501, "Uploads through the API need local storage (STORAGE_BACKEND=local).")
    db.set_status(memory_id, MemoryStatus.uploading)
    size = 0
    with store.writer(f"{memory_id}/upload/{stored}") as out:
        async for chunk in request.stream():
            size += len(chunk)
            if size > max_bytes:
                raise too_large  # the half-written file is discarded
            out.write(chunk)
    if size == 0:
        store.delete(f"{memory_id}/upload/{stored}")
        raise HTTPException(422, "The file is empty.")

    # Only one video (and one audio file) per memory: ingest picks by extension.
    kind = stored.split(".")[0]
    for key in store.list(f"{memory_id}/upload/"):
        name = key.rsplit("/", 1)[-1]
        if name.startswith(kind + ".") and name != stored:
            store.delete(key)
    meta[kind] = stored
    meta[f"{kind}_original_name"] = file_name
    _write_meta(memory_id, meta)
    return UploadResult(file_name=file_name, stored_as=stored, bytes=size)


@router.post("/{memory_id}/process", status_code=202)
def start_processing(memory_id: str) -> dict:
    """Start the workers' pipeline on the uploaded files (app/pipeline.py)."""
    _check_id(memory_id)
    _require_memory(memory_id)
    meta = _read_meta(memory_id)
    if not meta.get("video") or not get_store().exists(f"{memory_id}/upload/{meta['video']}"):
        raise HTTPException(422, "Upload the video first.")
    try:
        pipeline.start(memory_id)
    except pipeline.AlreadyRunning:
        raise HTTPException(409, "This memory is already being processed.") from None
    except pipeline.PipelineError as err:
        raise HTTPException(503, str(err)) from None
    db.upsert_jobs(memory_id, {})  # forget the previous run's stages
    db.set_status(memory_id, MemoryStatus.processing)
    return {"id": memory_id, "status": MemoryStatus.processing.value}


# --------------------------------------------------------------------------- read


@router.get("", response_model=list[MemorySummary])
def list_memories() -> list[dict]:
    """The dev user's memories, newest first."""
    return db.list_memories(get_settings().dev_user_id)


@router.get("/{memory_id}")
def get_memory(memory_id: str, request: Request) -> dict:
    """Return the memory's manifest (packages/manifest), per-stage status, the base URL that
    the manifest's relative asset URLs resolve against, and (for memories created through
    the API) its details from the database.

    The first five fields are what the viewer uses; keep their meaning. Memories made with
    the worker CLI have no database row and get ``null`` for the rest.
    No auth yet: every memory is readable, so run this locally only.
    """
    _check_id(memory_id)
    store = get_store()
    row = db.get_memory(memory_id)
    stages = {s: _stage_done(store, memory_id, s) for s in STAGES}
    if row is None and not any(stages.values()) and not store.list(f"{memory_id}/upload/"):
        raise HTTPException(404, f"No memory {memory_id!r}")
    manifest_key = f"{memory_id}/publish/manifest.json"
    manifest = json.loads(store.read_bytes(manifest_key)) if store.exists(manifest_key) else None
    if row is not None:
        status = row["status"]  # the database decides; a re-run may sit on an old manifest
    else:
        status = "ready" if manifest else "processing"
    out = {
        "id": memory_id,
        "status": status,
        "stages": stages,
        "manifest": manifest,
        "asset_base": str(request.url_for("get_memory_file", memory_id=memory_id, key="")),
        "stage_detail": _stage_detail(memory_id, row is not None),
    }
    details = ("title", "description", "visibility", "lat", "lon", "location_source",
               "captured_at", "created_at", "updated_at", "error")
    out.update({k: (row[k] if row else None) for k in details})
    out["suggested_location"] = _suggested_location(memory_id) if row and row["lat"] is None else None
    return out


def _stage_detail(memory_id: str, has_row: bool) -> dict | None:
    """Per-stage progress: the runner's live job/status.json, else the last run's rows."""
    status = pipeline.read_status(memory_id)
    if status:
        return {
            "state": status.get("state"),
            "current": status.get("current"),
            "detail": status.get("detail") or None,
            "stages": {name: _stage_fields(info) for name, info in (status.get("stages") or {}).items()},
        }
    rows = db.list_jobs(memory_id) if has_row else []
    if rows:
        return {"state": None, "current": None, "detail": None,
                "stages": {r["stage"]: _stage_fields(r) for r in rows}}
    return None


def _stage_fields(info) -> dict:
    return {k: info[k] for k in ("status", "seconds", "error") if info.get(k) is not None}


def _suggested_location(memory_id: str) -> dict | None:
    """The video's own GPS (written by ingest), rounded like any stored location."""
    try:
        geo = json.loads(get_store().read_bytes(f"{memory_id}/ingest/meta.json")).get("geo")
        lat, lon = db.blur_location(geo["lat"], geo["lon"])
    except (OSError, ValueError, KeyError, TypeError, AttributeError):
        return None
    return {"lat": lat, "lon": lon}


def _require_memory(memory_id: str) -> dict:
    memory = db.get_memory(memory_id)
    if memory is None:
        raise HTTPException(404, f"No memory {memory_id!r}")
    return memory


def _read_meta(memory_id: str) -> dict:
    store = get_store()
    try:
        return json.loads(store.read_bytes(f"{memory_id}/{UPLOAD_META}"))
    except (OSError, ValueError):
        return {}


def _write_meta(memory_id: str, meta: dict) -> None:
    get_store().write_bytes(f"{memory_id}/{UPLOAD_META}", json.dumps(meta, indent=2).encode())


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
