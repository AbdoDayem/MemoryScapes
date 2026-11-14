from datetime import datetime
from enum import Enum
from typing import Literal

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
    # "video_gps" when the client used suggested_location; otherwise a pin ("user_pin").
    location_source: Literal["video_gps", "user_pin"] | None = None


class MemorySummary(BaseModel):
    id: str
    title: str | None
    status: MemoryStatus
    visibility: Visibility
    thumbnail_url: str | None
    captured_at: datetime | None


class GlobeMemory(BaseModel):
    id: str
    title: str | None
    lat: float
    lon: float
    status: MemoryStatus
    captured_at: datetime | None
    created_at: datetime
    thumbnail_url: str | None


class GlobeResponse(BaseModel):
    memories: list[GlobeMemory]


class UploadResult(BaseModel):
    file_name: str
    stored_as: str  # video.<ext> or audio.<ext> in {id}/upload/
    bytes: int


class ReportRequest(BaseModel):
    reason: str
