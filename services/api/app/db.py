"""All database access for the API (docs/webapp-plan/database.md).

Nothing else in the API imports SQLAlchemy. Functions take and return plain dicts, so the
routers don't depend on the database library, and switching SQLite to Postgres only
touches this file (and ``DATABASE_URL``).

    init_db()                      create tables, seed the dev user (call once at startup)
    create_memory / get_memory / list_memories / update_memory / set_status
    upsert_jobs / list_jobs        per-stage results copied from job/status.json
    list_in_bbox                   memories inside a map rectangle, for the globe
    blur_location                  round a location to about 100 m
"""

from __future__ import annotations

import math
import re
import secrets
import string
from collections.abc import Mapping
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from sqlalchemy import Engine, and_, create_engine, delete, event, insert, or_, select, update
from sqlalchemy.engine import make_url
from sqlalchemy.exc import IntegrityError

from app.models import MemoryStatus, Visibility
from app.schema import jobs, memories, metadata, users
from app.settings import get_settings

_engine: Engine | None = None

MEMORY_ID_PREFIX = "mem_"
# Memory ids double as storage folder names (same rule as tools/debugger/server/jobs.ts).
MEMORY_ID_PATTERN = re.compile(r"^[a-z0-9][a-z0-9_-]{0,63}$")
# Only private memories until the privacy pass (face/plate blur) exists. Add "unlisted" and
# "public" here when it does.
ALLOWED_VISIBILITY = {Visibility.private.value}
_ID_ALPHABET = string.ascii_lowercase + string.digits
LOCATION_SOURCES = {"video_gps", "user_pin"}
# Stage statuses written by services/workers/memoryscapes_workers/run.py.
JOB_STATUSES = {"pending", "running", "done", "failed", "skipped", "blocked", "not requested"}
_UPDATABLE = {"title", "description", "visibility", "lat", "lon", "location_source",
              "captured_at", "thumbnail_url"}
# Grid step for rounding a location: 0.001° of latitude is about 111 m.
_BLUR_DEG = 0.001


# ---------------------------------------------------------------------------- setup


def init_db(url: str | None = None) -> Engine:
    """Connect, create any missing tables, and make sure the dev user exists."""
    global _engine
    settings = get_settings()
    url = url or settings.database_url
    parsed = make_url(url)
    is_sqlite = parsed.get_backend_name() == "sqlite"

    if is_sqlite and parsed.database and parsed.database != ":memory:":
        Path(parsed.database).parent.mkdir(parents=True, exist_ok=True)

    engine = create_engine(url, connect_args={"check_same_thread": False} if is_sqlite else {})
    if is_sqlite:
        event.listen(engine, "connect", _sqlite_pragmas)

    metadata.create_all(engine)
    if _engine is not None and _engine is not engine:
        _engine.dispose()
    _engine = engine
    ensure_user(settings.dev_user_id, settings.dev_user_handle)
    return engine


def _sqlite_pragmas(dbapi_conn, _record) -> None:
    cur = dbapi_conn.cursor()
    cur.execute("PRAGMA foreign_keys=ON")  # SQLite ignores foreign keys unless asked
    cur.execute("PRAGMA journal_mode=WAL")  # readers don't block the writer
    cur.close()


def get_engine() -> Engine:
    if _engine is None:
        init_db()
    assert _engine is not None
    return _engine


def _now() -> datetime:
    return datetime.now(timezone.utc)


def _row(row) -> dict[str, Any] | None:
    return dict(row._mapping) if row is not None else None


# ---------------------------------------------------------------------------- users


def ensure_user(user_id: str, handle: str) -> dict[str, Any]:
    """Create the user if it doesn't exist. Returns the row."""
    query = select(users).where(users.c.id == user_id)
    with get_engine().begin() as conn:
        row = conn.execute(query).first()
    if row is None:
        try:
            with get_engine().begin() as conn:
                conn.execute(insert(users).values(id=user_id, handle=handle, created_at=_now()))
        except IntegrityError:
            pass  # another process created it at the same moment
        with get_engine().connect() as conn:
            row = conn.execute(query).first()
    return _row(row)


def get_user(user_id: str) -> dict[str, Any] | None:
    with get_engine().connect() as conn:
        return _row(conn.execute(select(users).where(users.c.id == user_id)).first())


# ---------------------------------------------------------------------------- memories


def new_memory_id() -> str:
    """``mem_`` + 12 random lowercase letters/digits: safe as a storage folder name."""
    return MEMORY_ID_PREFIX + "".join(secrets.choice(_ID_ALPHABET) for _ in range(12))


def create_memory(owner_id: str, title: str | None = None, description: str | None = None,
                  memory_id: str | None = None) -> dict[str, Any]:
    """Insert a new ``draft`` memory and return it."""
    memory_id = memory_id or new_memory_id()
    if not MEMORY_ID_PATTERN.match(memory_id):
        raise ValueError(f"Invalid memory id {memory_id!r}")
    now = _now()
    values = {
        "id": memory_id,
        "owner_id": owner_id,
        "title": title,
        "description": description,
        "visibility": Visibility.private.value,
        "status": MemoryStatus.draft.value,
        "created_at": now,
        "updated_at": now,
    }
    with get_engine().begin() as conn:
        conn.execute(insert(memories).values(**values))
        row = conn.execute(select(memories).where(memories.c.id == values["id"])).first()
    return _row(row)


def get_memory(memory_id: str) -> dict[str, Any] | None:
    with get_engine().connect() as conn:
        return _row(conn.execute(select(memories).where(memories.c.id == memory_id)).first())


def list_memories(owner_id: str) -> list[dict[str, Any]]:
    """The owner's memories, newest first."""
    query = (select(memories).where(memories.c.owner_id == owner_id)
             .order_by(memories.c.created_at.desc(), memories.c.id))
    with get_engine().connect() as conn:
        return [_row(r) for r in conn.execute(query)]


def update_memory(memory_id: str, **fields: Any) -> dict[str, Any] | None:
    """Change any of: title, description, visibility, lat + lon (together), location_source,
    captured_at, thumbnail_url. Locations are rounded to about 100 m here, so an exact
    location can never be stored. Pass ``lat=None, lon=None`` to clear the location.

    Returns the updated row, or None if there's no such memory. Raises ValueError for
    unknown fields or invalid values.
    """
    unknown = set(fields) - _UPDATABLE
    if unknown:
        raise ValueError(f"Can't update {sorted(unknown)}")
    values = dict(fields)

    if "visibility" in values:
        values["visibility"] = Visibility(values["visibility"]).value
        if values["visibility"] not in ALLOWED_VISIBILITY:
            raise ValueError("Only private memories for now: sharing needs the privacy pass")

    if values.get("captured_at") is not None:
        captured = values["captured_at"]
        if not isinstance(captured, datetime) or captured.tzinfo is None:
            raise ValueError("captured_at must be a timezone-aware datetime")

    if "lat" in values or "lon" in values:
        if "lat" not in values or "lon" not in values:
            raise ValueError("lat and lon must be set together")
        if values["lat"] is None and values["lon"] is None:
            values["location_source"] = None
        elif values["lat"] is None or values["lon"] is None:
            raise ValueError("lat and lon must both be numbers, or both None")
        else:
            values["lat"], values["lon"] = blur_location(values["lat"], values["lon"])

    if values.get("location_source") is not None and values["location_source"] not in LOCATION_SOURCES:
        raise ValueError(f"location_source must be one of {sorted(LOCATION_SOURCES)}")

    values["updated_at"] = _now()
    with get_engine().begin() as conn:
        result = conn.execute(update(memories).where(memories.c.id == memory_id).values(**values))
        if result.rowcount == 0:
            return None
        row = conn.execute(select(memories).where(memories.c.id == memory_id)).first()
    return _row(row)


def set_status(memory_id: str, status: str | MemoryStatus, error: str | None = None) -> bool:
    """Change a memory's status. ``error`` is kept only for ``failed`` and cleared otherwise.
    Returns False if there's no such memory."""
    status = MemoryStatus(status).value
    values = {"status": status, "error": error if status == MemoryStatus.failed.value else None,
              "updated_at": _now()}
    with get_engine().begin() as conn:
        result = conn.execute(update(memories).where(memories.c.id == memory_id).values(**values))
    return result.rowcount > 0


# ---------------------------------------------------------------------------- jobs


def upsert_jobs(memory_id: str, stages: Mapping[str, Mapping[str, Any]]) -> None:
    """Replace a memory's job rows with the ``stages`` block of ``job/status.json``:

        {"ingest": {"status": "done", "started_at": 1731600001.2, "seconds": 12.3}, ...}

    ``started_at`` is a Unix timestamp (as the runner writes it). ``finished_at`` is
    worked out from ``started_at + seconds`` when both are there.
    """
    rows = []
    for stage, info in stages.items():
        status = info.get("status", "pending")
        if status not in JOB_STATUSES:
            raise ValueError(f"Unknown job status {status!r} for stage {stage!r}")
        started = _from_unix(info.get("started_at"))
        seconds = info.get("seconds")
        finished = (datetime.fromtimestamp(info["started_at"] + seconds, timezone.utc)
                    if started is not None and seconds is not None else None)
        rows.append({"memory_id": memory_id, "stage": stage, "status": status,
                     "error": info.get("error"), "seconds": seconds,
                     "started_at": started, "finished_at": finished})
    with get_engine().begin() as conn:
        conn.execute(delete(jobs).where(jobs.c.memory_id == memory_id))
        if rows:
            conn.execute(insert(jobs), rows)


def list_jobs(memory_id: str) -> list[dict[str, Any]]:
    with get_engine().connect() as conn:
        return [_row(r) for r in conn.execute(
            select(jobs).where(jobs.c.memory_id == memory_id).order_by(jobs.c.id))]


def _from_unix(value: float | None) -> datetime | None:
    return datetime.fromtimestamp(value, timezone.utc) if value is not None else None


# ---------------------------------------------------------------------------- location


def blur_location(lat: float, lon: float) -> tuple[float, float]:
    """Snap a location to a grid of about 110 m × 110 m (design doc § 8: never store
    exact home locations). The longitude step widens away from the equator, where a
    degree of longitude gets shorter, so the cells stay roughly square."""
    lat, lon = float(lat), float(lon)
    if not (math.isfinite(lat) and math.isfinite(lon)):
        raise ValueError("Location must be finite numbers")
    if not -90 <= lat <= 90:
        raise ValueError("lat must be between -90 and 90")
    if not -180 <= lon <= 180:
        raise ValueError("lon must be between -180 and 180")

    lat_r = round(round(lat / _BLUR_DEG) * _BLUR_DEG, 6)
    # Near the poles cos(lat) → 0; cap the step at 1° so it stays sensible. The step is
    # adjusted so an even number of cells fits exactly into 360°: then ±180° are both grid
    # points, and rounding a rounded value never jumps across the dateline.
    wanted = min(_BLUR_DEG / max(math.cos(math.radians(lat_r)), 1e-9), 1.0)
    cells = max(2, 2 * round(180 / wanted))
    lon_step = 360 / cells
    k = round(lon / lon_step)
    if k == cells // 2:  # +180° and -180° are the same meridian; store it as -180°
        k = -k
    lon_r = round(k * lon_step, 6)
    return lat_r, lon_r


def list_in_bbox(min_lon: float, min_lat: float, max_lon: float, max_lat: float,
                 owner_id: str | None = None) -> list[dict[str, Any]]:
    """Memories with a location inside the rectangle. A box with ``min_lon > max_lon``
    crosses the ±180° line (e.g. 170 → -170 covers the Pacific) and is split in two."""
    for name, value, limit in (("min_lat", min_lat, 90), ("max_lat", max_lat, 90),
                               ("min_lon", min_lon, 180), ("max_lon", max_lon, 180)):
        if not -limit <= value <= limit:
            raise ValueError(f"{name} must be between -{limit} and {limit}")
    if min_lat > max_lat:
        raise ValueError("min_lat must not be greater than max_lat")

    lat_ok = memories.c.lat.between(min_lat, max_lat)
    if min_lon <= max_lon:
        lon_ok = memories.c.lon.between(min_lon, max_lon)
    else:
        lon_ok = or_(memories.c.lon >= min_lon, memories.c.lon <= max_lon)
    conditions = [memories.c.lat.is_not(None), memories.c.lon.is_not(None), lat_ok, lon_ok]
    if owner_id is not None:
        conditions.append(memories.c.owner_id == owner_id)

    query = select(memories).where(and_(*conditions)).order_by(memories.c.created_at.desc())
    with get_engine().connect() as conn:
        return [_row(r) for r in conn.execute(query)]
