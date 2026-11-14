"""The soundscapes table: what a run's SATIE soundscape looks like in the database."""

import json

import pytest
from fastapi.testclient import TestClient
from sqlalchemy.exc import IntegrityError

from app import db, pipeline, soundscapes, storage
from app.main import app
from app.settings import get_settings

SCRIPT = "loop memory/s6_water\n    at s6_water\n    spl 57\n"
SAMPLES = {"Audio/memory/s6_water": {"url": "http://localhost:8000/memories/mem_1/files/separate/stems/s6_water.wav",
                                     "source": "provided", "licence": "provided"}}


@pytest.fixture
def store(tmp_path, monkeypatch):
    monkeypatch.setenv("STORAGE_BACKEND", "local")
    monkeypatch.setenv("STORAGE_ROOT", str(tmp_path / "storage"))
    storage.get_store.cache_clear()
    db.init_db()
    yield storage.get_store()
    storage.get_store.cache_clear()


def _memory(memory_id="mem_1"):
    return db.create_memory(get_settings().dev_user_id, title="Canal walk", memory_id=memory_id)


def _put(store, key, value):
    store.write_bytes(key, value if isinstance(value, bytes) else json.dumps(value).encode())


def _satie_outputs(store, memory_id="mem_1"):
    """What the workers' soundscape stage writes for a SATIE soundscape."""
    _put(store, f"{memory_id}/soundscape/scene.json", {
        "provider": {"name": "satie", "notes": [], "realize_id": "r_1"},
        "sources": [], "ambient_bed": {"key": f"{memory_id}/separate/ambient.wav", "format": "stereo"},
        "satie": {"script_key": f"{memory_id}/soundscape/satie/scene.satie",
                  "samples_key": f"{memory_id}/soundscape/satie/samples.json",
                  "host_map": {"s6_water": "s6_water"}, "contract_id": "sc_1", "realize_id": "r_1"}})
    _put(store, f"{memory_id}/soundscape/satie/scene.satie", SCRIPT.encode())
    _put(store, f"{memory_id}/soundscape/satie/samples.json", SAMPLES)
    _put(store, f"{memory_id}/soundscape/satie/raw.json",
         {"realize": {"generated": 0, "fromLibrary": 0, "maxSpentCents": 0}})


def test_a_soundscape_round_trips_and_is_replaced_on_rerun(store):
    _memory()
    first = db.upsert_soundscape("mem_1", {"provider": "satie", "script": SCRIPT, "samples": SAMPLES,
                                           "host_map": {"s6_water": "s6_water"}, "notes": []})
    assert first["samples"] == SAMPLES and first["host_map"] == {"s6_water": "s6_water"}
    assert first["created_at"].tzinfo is not None
    second = db.upsert_soundscape("mem_1", {"provider": "stub", "fallback_reason": "satie failed: not signed in"})
    assert second["provider"] == "stub" and second["script"] is None and second["samples"] is None
    assert second["created_at"] == first["created_at"] and second["updated_at"] >= first["updated_at"]
    assert [r["memory_id"] for r in db.list_soundscapes(provider="stub")] == ["mem_1"]
    assert db.delete_soundscape("mem_1") and db.get_soundscape("mem_1") is None


def test_a_soundscape_needs_its_memory(store):
    with pytest.raises(IntegrityError):
        db.upsert_soundscape("mem_missing", {"provider": "satie"})
    with pytest.raises(ValueError):
        db.upsert_soundscape("mem_missing", {"provider": "satie", "colour": "blue"})


def test_sync_copies_the_satie_soundscape_from_storage(store):
    _memory()
    _satie_outputs(store)
    row = soundscapes.sync("mem_1")
    assert row["provider"] == "satie" and row["script"] == SCRIPT and row["samples"] == SAMPLES
    assert row["host_map"] == {"s6_water": "s6_water"} and row["contract_id"] == "sc_1"
    assert (row["realize_id"], row["spent_cents"], row["generated"], row["from_library"]) == ("r_1", 0, 0, 0)


def test_sync_records_why_there_is_no_satie_soundscape(store):
    _memory()
    _put(store, "mem_1/soundscape/scene.json", {
        "provider": {"name": "stub", "notes": [], "fallback_reason": "satie failed: satie_compose returned no script"},
        "sources": [], "ambient_bed": {}, "satie": None})
    row = soundscapes.sync("mem_1")
    assert row["provider"] == "stub" and row["script"] is None
    assert "returned no script" in row["fallback_reason"]
    store.delete("mem_1/soundscape/scene.json")  # the stage's output is gone: so is the row
    assert soundscapes.sync("mem_1") is None and db.get_soundscape("mem_1") is None


def test_sync_only_reads_this_memorys_files(store):
    _memory()
    _satie_outputs(store)
    scene = json.loads(store.read_bytes("mem_1/soundscape/scene.json"))
    scene["satie"]["script_key"] = "mem_2/upload/clip.mp4"
    _put(store, "mem_1/soundscape/scene.json", scene)
    with pytest.raises(ValueError):
        soundscapes.sync("mem_1")


def test_a_finished_run_stores_its_soundscape(store):
    _memory()
    _satie_outputs(store)
    _put(store, "mem_1/publish/manifest.json", {"id": "mem_1", "version": 4})
    _put(store, "mem_1/job/status.json", {"state": "done", "stages": {
        "soundscape": {"status": "done", "started_at": 1_731_600_000.0, "seconds": 90.0}}})
    assert pipeline.finish("mem_1", 0) == "ready"
    assert db.get_soundscape("mem_1")["script"] == SCRIPT


def test_a_broken_soundscape_never_fails_the_run(store):
    _memory()
    _put(store, "mem_1/soundscape/scene.json", b"{not json")
    _put(store, "mem_1/publish/manifest.json", {"id": "mem_1", "version": 4})
    _put(store, "mem_1/job/status.json", {"state": "done", "stages": {}})
    assert pipeline.finish("mem_1", 0) == "ready"
    assert db.get_soundscape("mem_1") is None


def test_register_backfills_a_cli_memory(store):
    _put(store, "canada/publish/manifest.json", {"id": "canada", "version": 4, "meta": {"title": "canada"}})
    _satie_outputs(store, "canada")
    assert soundscapes.main(["canada"]) == 1  # no row yet, and no --register: nothing stored
    assert db.get_soundscape("canada") is None
    assert soundscapes.main(["canada", "--register"]) == 0
    assert db.get_memory("canada")["status"] == "ready"
    assert db.get_soundscape("canada")["samples"] == SAMPLES


def test_the_memory_response_is_unchanged_by_a_stored_soundscape(store):
    """The frontend reads GET /memories/{id}; storing the soundscape mustn't change it."""
    _memory()
    _satie_outputs(store)
    _put(store, "mem_1/publish/manifest.json", {"id": "mem_1", "version": 4})
    db.set_status("mem_1", "ready")
    client = TestClient(app)
    before = client.get("/memories/mem_1").json()
    soundscapes.sync("mem_1")
    after = client.get("/memories/mem_1").json()
    assert after == before
