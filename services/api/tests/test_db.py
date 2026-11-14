import math
import random
from datetime import datetime, timedelta, timezone

import pytest
from fastapi.testclient import TestClient

from app import db, seed, settings
from app.models import MemoryStatus


@pytest.fixture
def database(tmp_path, monkeypatch):
    """A fresh SQLite file per test."""
    url = "sqlite:///" + (tmp_path / "test.db").as_posix()
    monkeypatch.setenv("DATABASE_URL", url)
    settings.get_settings.cache_clear()
    db.init_db()
    yield url
    _reset_engine()
    settings.get_settings.cache_clear()


def _reset_engine():
    if db._engine is not None:
        db._engine.dispose()
    db._engine = None


DEV = "usr_dev"


# ---------------------------------------------------------------- setup


def test_init_creates_tables_and_dev_user(database):
    assert db.ensure_user(DEV, "dev")["handle"] == "dev"
    db.init_db()  # running again is harmless
    with db.get_engine().connect() as conn:
        assert conn.exec_driver_sql("PRAGMA foreign_keys").scalar() == 1
        assert conn.exec_driver_sql("PRAGMA journal_mode").scalar() == "wal"


def test_init_creates_the_database_folder(tmp_path):
    path = tmp_path / "nested" / "dir" / "memories.db"
    try:
        db.init_db("sqlite:///" + path.as_posix())
        assert path.exists()
    finally:
        _reset_engine()


def test_app_startup_initialises_the_database(tmp_path, monkeypatch):
    """No fixture: the database must not exist until the app starts."""
    from app.main import app

    path = tmp_path / "startup" / "app.db"
    monkeypatch.setenv("DATABASE_URL", "sqlite:///" + path.as_posix())
    settings.get_settings.cache_clear()
    _reset_engine()
    try:
        with TestClient(app) as client:
            assert client.get("/health").json() == {"status": "ok"}
        assert path.exists()
        assert db.get_user(DEV)["handle"] == "dev"
    finally:
        _reset_engine()
        settings.get_settings.cache_clear()


# ---------------------------------------------------------------- memories


def test_create_and_get(database):
    m = db.create_memory(DEV, title="Golden Gate walk")
    assert m["id"].startswith("mem_") and len(m["id"]) == 16
    assert m["id"][4:].isalnum() and m["id"].islower()
    assert m["status"] == "draft" and m["visibility"] == "private"
    assert m["lat"] is None and m["error"] is None
    assert m["created_at"].tzinfo is not None
    assert db.get_memory(m["id"]) == m
    assert db.get_memory("mem_missing") is None


@pytest.mark.parametrize("bad", ["../escape", "Mem_Upper", "_leading", "a" * 65, "has space"])
def test_create_rejects_bad_ids(database, bad):
    with pytest.raises(ValueError):
        db.create_memory(DEV, memory_id=bad)


def test_ids_are_unique(database):
    ids = {db.create_memory(DEV)["id"] for _ in range(50)}
    assert len(ids) == 50


def test_owner_must_exist(database):
    with pytest.raises(Exception):  # foreign key
        db.create_memory("usr_nobody")


def test_list_newest_first(database, monkeypatch):
    # Fixed, distinct times: Windows' clock can give two quick calls the same timestamp.
    t0 = datetime(2026, 11, 1, tzinfo=timezone.utc)
    ticks = iter(range(1000))
    monkeypatch.setattr(db, "_now", lambda: t0 + timedelta(seconds=next(ticks)))
    a = db.create_memory(DEV, title="a")
    b = db.create_memory(DEV, title="b")
    db.ensure_user("usr_other", "other")
    db.create_memory("usr_other", title="theirs")
    listed = [m["id"] for m in db.list_memories(DEV)]
    assert set(listed) == {a["id"], b["id"]}
    assert listed == [b["id"], a["id"]]


def test_update_fields(database):
    m = db.create_memory(DEV)
    captured = datetime(2026, 11, 10, 18, 30, tzinfo=timezone.utc)
    out = db.update_memory(m["id"], title="New", description="Foggy", captured_at=captured)
    assert out["title"] == "New" and out["description"] == "Foggy"
    assert out["captured_at"] == captured
    assert out["updated_at"] >= m["updated_at"]
    assert db.update_memory("mem_missing", title="x") is None


def test_update_blurs_location(database):
    m = db.create_memory(DEV)
    out = db.update_memory(m["id"], lat=37.819912, lon=-122.478255, location_source="video_gps")
    assert (out["lat"], out["lon"]) == db.blur_location(37.819912, -122.478255)
    assert out["lat"] != 37.819912
    assert out["location_source"] == "video_gps"
    cleared = db.update_memory(m["id"], lat=None, lon=None)
    assert cleared["lat"] is None and cleared["lon"] is None and cleared["location_source"] is None


@pytest.mark.parametrize("fields", [
    {"lat": 10.0},                        # lon missing
    {"lat": 10.0, "lon": None},           # half a location
    {"lat": 91.0, "lon": 0.0},            # out of range
    {"lat": 0.0, "lon": 181.0},
    {"lat": float("nan"), "lon": 0.0},
    {"visibility": "secret"},
    {"visibility": "public"},             # not until the privacy pass
    {"visibility": "unlisted"},
    {"captured_at": datetime(2026, 11, 10)},  # no time zone
    {"captured_at": "2026-11-10"},
    {"location_source": "guess"},
    {"status": "ready"},                  # status goes through set_status
    {"owner_id": "usr_other"},
])
def test_update_rejects_bad_values(database, fields):
    m = db.create_memory(DEV)
    with pytest.raises(ValueError):
        db.update_memory(m["id"], **fields)


def test_set_status(database):
    m = db.create_memory(DEV)
    assert db.set_status(m["id"], MemoryStatus.processing)
    assert db.get_memory(m["id"])["status"] == "processing"
    assert db.set_status(m["id"], "failed", error="poses: too few matches")
    assert db.get_memory(m["id"])["error"] == "poses: too few matches"
    db.set_status(m["id"], "processing", error="ignored")  # error only kept for failed
    assert db.get_memory(m["id"])["error"] is None
    assert not db.set_status("mem_missing", "ready")
    with pytest.raises(ValueError):
        db.set_status(m["id"], "finished")


# ---------------------------------------------------------------- jobs


STATUS_STAGES = {  # the "stages" block of job/status.json, as run.py writes it
    "ingest": {"status": "done", "started_at": 1_731_600_001.0, "seconds": 12.5},
    "poses": {"status": "failed", "started_at": 1_731_600_014.0, "seconds": 3.0,
              "error": "COLMAP found too few matches"},
    "splat": {"status": "blocked", "error": "needs poses"},
    "separate": {"status": "running", "started_at": 1_731_600_014.0},
    "publish": {"status": "pending"},
}


def test_upsert_jobs(database):
    m = db.create_memory(DEV)
    db.upsert_jobs(m["id"], STATUS_STAGES)
    rows = {r["stage"]: r for r in db.list_jobs(m["id"])}
    assert set(rows) == set(STATUS_STAGES)
    ingest = rows["ingest"]
    assert ingest["status"] == "done" and ingest["seconds"] == 12.5
    assert ingest["started_at"] == datetime.fromtimestamp(1_731_600_001.0, timezone.utc)
    assert ingest["finished_at"] == datetime.fromtimestamp(1_731_600_013.5, timezone.utc)
    assert rows["poses"]["error"] == "COLMAP found too few matches"
    assert rows["separate"]["finished_at"] is None
    assert rows["publish"]["started_at"] is None


def test_upsert_jobs_replaces_previous_run(database):
    m = db.create_memory(DEV)
    db.upsert_jobs(m["id"], STATUS_STAGES)
    db.upsert_jobs(m["id"], {"ingest": {"status": "running", "started_at": 1_731_700_000.0}})
    rows = db.list_jobs(m["id"])
    assert [(r["stage"], r["status"]) for r in rows] == [("ingest", "running")]


def test_upsert_jobs_rejects_unknown_status(database):
    m = db.create_memory(DEV)
    with pytest.raises(ValueError):
        db.upsert_jobs(m["id"], {"ingest": {"status": "exploded"}})


# ---------------------------------------------------------------- location


@pytest.mark.parametrize("lat,lon", [
    (37.819912, -122.478255), (0.0, 0.0), (-33.8568, 151.2153), (64.1466, -21.9426),
    (89.9999, 45.0), (-89.95, -179.99), (12.3456, 179.9999),
])
def test_blur_location_moves_at_most_about_100m(lat, lon):
    blat, blon = db.blur_location(lat, lon)
    assert -90 <= blat <= 90 and -180 <= blon <= 180
    assert _metres(lat, lon, blat, blon) < 110
    assert db.blur_location(blat, blon) == (blat, blon)  # stable: blurring twice changes nothing


def test_blur_location_is_stable_everywhere():
    """Blurring a blurred location changes nothing, including next to ±180°."""
    rng = random.Random(1)
    points = [(rng.uniform(-90, 90), rng.uniform(-180, 180)) for _ in range(20_000)]
    points += [(rng.uniform(-90, 90), rng.choice([-1, 1]) * rng.uniform(179.9, 180))
               for _ in range(5_000)]
    points += [(-88.158, -179.99978), (0.0, 180.0), (0.0, -180.0)]
    for lat, lon in points:
        once = db.blur_location(lat, lon)
        assert db.blur_location(*once) == once, (lat, lon)
        assert -180 <= once[1] < 180
        assert _metres(lat, lon, *once) < 110, (lat, lon)


def test_blur_location_hides_the_exact_spot():
    a = db.blur_location(37.81991, -122.47825)
    b = db.blur_location(37.81994, -122.47829)  # a few metres away
    assert a == b


def _metres(lat1, lon1, lat2, lon2):
    r = 6_371_000
    p1, p2 = math.radians(lat1), math.radians(lat2)
    dp, dl = p2 - p1, math.radians(lon2 - lon1)
    h = math.sin(dp / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(dl / 2) ** 2
    return 2 * r * math.asin(math.sqrt(h))


# ---------------------------------------------------------------- bbox


def _at(lat, lon, owner=DEV):
    m = db.create_memory(owner)
    db.update_memory(m["id"], lat=lat, lon=lon)
    return m["id"]


def test_list_in_bbox(database):
    sf = _at(37.8, -122.4)
    paris = _at(48.86, 2.35)
    db.create_memory(DEV)  # no location: never on the globe
    assert {m["id"] for m in db.list_in_bbox(-130, 30, -110, 45)} == {sf}
    assert {m["id"] for m in db.list_in_bbox(-180, -90, 180, 90)} == {sf, paris}
    assert db.list_in_bbox(100, -10, 120, 10) == []


def test_list_in_bbox_across_the_dateline(database):
    fiji = _at(-17.7, 178.1)
    samoa = _at(-13.8, -171.8)
    sf = _at(37.8, -122.4)
    found = {m["id"] for m in db.list_in_bbox(170, -30, -160, 0)}  # 170°E → 160°W
    assert found == {fiji, samoa}
    assert sf not in found


def test_list_in_bbox_by_owner(database):
    db.ensure_user("usr_other", "other")
    mine = _at(10, 10)
    _at(10, 10.01, owner="usr_other")
    assert [m["id"] for m in db.list_in_bbox(0, 0, 20, 20, owner_id=DEV)] == [mine]
    assert len(db.list_in_bbox(0, 0, 20, 20)) == 2


@pytest.mark.parametrize("box", [(0, 0, 10, 91), (-181, 0, 10, 10), (0, 20, 10, 10)])
def test_list_in_bbox_rejects_bad_boxes(database, box):
    with pytest.raises(ValueError):
        db.list_in_bbox(*box)


# ---------------------------------------------------------------- seed


def test_seed_completes_a_half_made_memory(database):
    db.create_memory(DEV, memory_id="seed_golden-gate")  # as if a run stopped right here
    seed.seed()
    row = db.get_memory("seed_golden-gate")
    assert row["status"] == "ready" and row["lat"] is not None


def test_seed_is_idempotent(database):
    added = seed.seed()
    assert added == len(seed.SEED_MEMORIES)
    assert seed.seed() == 0
    rows = db.list_memories(DEV)
    assert len(rows) == len(seed.SEED_MEMORIES)
    assert all(r["status"] == "ready" and r["lat"] is not None for r in rows)
    assert len(db.list_in_bbox(170, -30, -160, 0)) == 1  # the Fiji seed sits near ±180°
