"""Database tables (docs/webapp-plan/database.md § Tables).

Portable column types only, so the same tables work on SQLite now and Postgres later.
Times are stored as UTC and always come back timezone-aware.
"""

from __future__ import annotations

from datetime import datetime, timezone

from sqlalchemy import (
    Column,
    DateTime,
    Float,
    ForeignKey,
    Index,
    Integer,
    MetaData,
    String,
    Table,
    Text,
    TypeDecorator,
    UniqueConstraint,
)


class UTCDateTime(TypeDecorator):
    """Stores naive UTC (SQLite has no time zones); returns aware UTC datetimes."""

    impl = DateTime
    cache_ok = True

    def process_bind_param(self, value: datetime | None, dialect) -> datetime | None:
        if value is None:
            return None
        if value.tzinfo is None:
            raise ValueError("Use timezone-aware datetimes (UTC)")
        return value.astimezone(timezone.utc).replace(tzinfo=None)

    def process_result_value(self, value: datetime | None, dialect) -> datetime | None:
        return value.replace(tzinfo=timezone.utc) if value is not None else None


metadata = MetaData()

users = Table(
    "users",
    metadata,
    Column("id", String(64), primary_key=True),
    Column("handle", String(64), nullable=False, unique=True),
    Column("created_at", UTCDateTime, nullable=False),
)

memories = Table(
    "memories",
    metadata,
    Column("id", String(64), primary_key=True),
    Column("owner_id", String(64), ForeignKey("users.id"), nullable=False),
    Column("title", Text),
    Column("description", Text),
    Column("visibility", String(16), nullable=False, default="private"),
    Column("status", String(16), nullable=False, default="draft"),
    Column("lat", Float),  # rounded to ~100 m before saving
    Column("lon", Float),
    Column("location_source", String(16)),  # video_gps | user_pin
    Column("captured_at", UTCDateTime),
    Column("error", Text),
    Column("thumbnail_url", Text),
    Column("created_at", UTCDateTime, nullable=False),
    Column("updated_at", UTCDateTime, nullable=False),
    Index("memories_lat_lon", "lat", "lon"),
    Index("memories_owner_created", "owner_id", "created_at"),
)

jobs = Table(
    "jobs",
    metadata,
    Column("id", Integer, primary_key=True, autoincrement=True),
    Column("memory_id", String(64), ForeignKey("memories.id", ondelete="CASCADE"), nullable=False),
    Column("stage", String(32), nullable=False),
    Column("status", String(16), nullable=False),
    Column("error", Text),
    Column("seconds", Float),
    Column("started_at", UTCDateTime),
    Column("finished_at", UTCDateTime),
    UniqueConstraint("memory_id", "stage", name="jobs_memory_stage"),
)
