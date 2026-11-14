from datetime import datetime
from enum import Enum

from pydantic import BaseModel


class Visibility(str, Enum):
    private = "private"
    unlisted = "unlisted"
    public = "public"


class MemoryStatus(str, Enum):
    draft = "draft"
    uploading = "uploading"
    processing = "processing"
    ready = "ready"
    failed = "failed"


class CreateMemoryRequest(BaseModel):
    title: str | None = None
    has_audio_track: bool = True
    input_kind: str = "video"  # "video" | "panoramas"
    file_names: list[str]


class UploadTarget(BaseModel):
    file_name: str
    url: str  # presigned PUT URL


class CreateMemoryResponse(BaseModel):
    id: str
    uploads: list[UploadTarget]


class UpdateMemoryRequest(BaseModel):
    title: str | None = None
    description: str | None = None
    visibility: Visibility | None = None
    lat: float | None = None
    lon: float | None = None


class MemorySummary(BaseModel):
    id: str
    title: str | None
    status: MemoryStatus
    visibility: Visibility
    thumbnail_url: str | None
    captured_at: datetime | None


class ReportRequest(BaseModel):
    reason: str
