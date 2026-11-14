"""memory_files: a memory's files (manifest, splat, path, sound) stored in the database."""

import hashlib
import json

import pytest
from fastapi.testclient import TestClient

from app import db, memory_files, pipeline, storage
from app.main import app
from app.settings import get_settings

MANIFEST = {
    "id": "mem_1", "version": 4,
    "coord_frame": {"units": "meters", "up": "+y"},
    "splat": {"lods": [{"url": "splat/lods/lod0.spz", "count": 3}, {"url": "splat/lods/lod1.spz", "count": 1}]},
    "navigation": {"original_path": "poses/path.json", "spawn": {"position": [0, 1.6, 0]}},
    "audio": {
        "sources": [{"id": "s1", "url": "separate/stems/s1.wav", "position": [1, 0, 2]}],
        "ambient_bed": {"url": "separate/ambient.wav", "format": "stereo"},
        "satie": {"script_url": "soundscape/satie/scene.satie", "samples_url": "soundscape/satie/samples.json",
                  "host_map": {"s1": "s1"}},
    },
    "meta": {"title": "Canal walk", "thumbnail_url": "https://cdn.example/t.jpg"},
}
FILES = {
    "splat/lods/lod0.spz": b"SPZ0" * 100,
    "splat/lods/lod1.spz": b"SPZ1" * 10,
    "poses/path.json": b'{"samples": []}',
    "separate/stems/s1.wav": b"RIFF....WAVE" + bytes(range(256)) * 4,
    "separate/ambient.wav": b"RIFF....WAVE" + bytes(512),
    "soundscape/satie/scene.satie": b"loop memory/s1\n    at s1\n",
    "soundscape/satie/samples.json": b'{"Audio/memory/s1": {"source": "provided"}}',
}
WORKING_FILES = {"upload/clip.mp4": b"raw video", "poses/frames/0001.jpg": b"JPEG"}


@pytest.fixture
def store(tmp_path, monkeypatch):
    monkeypatch.setenv("STORAGE_BACKEND", "local")
    monkeypatch.setenv("STORAGE_ROOT", str(tmp_path / "storage"))
    storage.get_store.cache_clear()
    db.init_db()
    blobs = storage.get_store()
    db.create_memory(get_settings().dev_user_id, title="Canal walk", memory_id="mem_1")
    blobs.write_bytes("mem_1/publish/manifest.json", json.dumps(MANIFEST).encode())
    for key, data in {**FILES, **WORKING_FILES}.items():
        blobs.write_bytes(f"mem_1/{key}", data)
    yield blobs
    storage.get_store.cache_clear()


def test_the_manifest_names_every_file_of_the_memory():
    files = memory_files.referenced_files(MANIFEST)
    assert files == {
        "splat/lods/lod0.spz": "splat", "splat/lods/lod1.spz": "splat", "poses/path.json": "path",
        "separate/stems/s1.wav": "stem", "separate/ambient.wav": "ambient",
        "soundscape/satie/scene.satie": "satie_script", "soundscape/satie/samples.json": "satie_samples",
    }  # the thumbnail is a web address, not a file of the memory
    assert memory_files.referenced_files({"splat": {"lods": [{"url": "../mem_2/x.spz"}, {"url": "/etc/passwd"}]}}) == {}


def test_store_copies_every_file_with_its_bytes_and_checksum(store):
    summary = memory_files.store("mem_1")
    assert summary["files"] == 8 and summary["stored"] == 8 and summary["missing"] == []
    listed = {f["key"]: f for f in db.list_files("mem_1")}
    assert set(listed) == {"publish/manifest.json", *FILES}  # no upload, no frames
    assert "data" not in listed["splat/lods/lod0.spz"]  # listing never loads the bytes
    for key, data in FILES.items():
        row = db.get_file("mem_1", key)
        assert row["data"] == data and row["bytes"] == len(data)
        assert row["sha256"] == hashlib.sha256(data).hexdigest()
    assert listed["separate/stems/s1.wav"]["content_type"] == "audio/wav"
    assert listed["publish/manifest.json"]["kind"] == "manifest"


def test_storing_again_only_writes_what_changed(store):
    memory_files.store("mem_1")
    before = {f["key"]: f["updated_at"] for f in db.list_files("mem_1")}
    store.write_bytes("mem_1/splat/lods/lod0.spz", b"retrained")
    again = memory_files.store("mem_1")
    assert (again["stored"], again["unchanged"]) == (1, 7)
    after = {f["key"]: f["updated_at"] for f in db.list_files("mem_1")}
    assert after["splat/lods/lod0.spz"] > before["splat/lods/lod0.spz"]
    assert after["poses/path.json"] == before["poses/path.json"]
    assert db.get_file("mem_1", "splat/lods/lod0.spz")["data"] == b"retrained"


def test_files_the_manifest_drops_are_removed(store):
    memory_files.store("mem_1")
    manifest = json.loads(json.dumps(MANIFEST))
    manifest["audio"]["satie"] = None  # a re-run without a SATIE soundscape
    store.write_bytes("mem_1/publish/manifest.json", json.dumps(manifest).encode())
    summary = memory_files.store("mem_1")
    assert summary["removed"] == 2
    assert db.get_file("mem_1", "soundscape/satie/scene.satie") is None


def test_missing_files_are_reported_not_stored(store):
    store.delete("mem_1/separate/ambient.wav")
    summary = memory_files.store("mem_1")
    assert summary["missing"] == ["separate/ambient.wav"] and summary["files"] == 7


def test_restore_writes_identical_files(store, tmp_path):
    memory_files.store("mem_1")
    written = memory_files.restore("mem_1", tmp_path / "restored")
    assert len(written) == 8
    for key, data in FILES.items():
        assert (tmp_path / "restored" / "mem_1" / key).read_bytes() == data
    assert memory_files.verify("mem_1") == []
    store.write_bytes("mem_1/poses/path.json", b"edited")
    assert memory_files.verify("mem_1") == ["poses/path.json: differs from storage"]


def test_a_finished_run_stores_the_memory(store):
    store.write_bytes("mem_1/job/status.json", json.dumps({"state": "done", "stages": {}}).encode())
    assert pipeline.finish("mem_1", 0) == "ready"
    assert len(db.list_files("mem_1")) == 8


def test_a_failed_run_stores_nothing(store):
    store.write_bytes("mem_1/job/status.json", json.dumps({"state": "failed", "stages": {
        "poses": {"status": "failed", "error": "too few frames"}}}).encode())
    assert pipeline.finish("mem_1", 1) == "failed"
    assert db.list_files("mem_1") == []


def test_deleting_a_memory_row_deletes_its_files(store):
    memory_files.store("mem_1")
    from sqlalchemy import delete

    from app.schema import memories
    with db.get_engine().begin() as conn:
        conn.execute(delete(memories).where(memories.c.id == "mem_1"))
    assert db.list_files("mem_1") == []


def test_the_memory_response_is_unchanged_by_stored_files(store):
    db.set_status("mem_1", "ready")
    client = TestClient(app)
    before = client.get("/memories/mem_1").json()
    lod = client.get("/memories/mem_1/files/splat/lods/lod0.spz").content
    memory_files.store("mem_1")
    assert client.get("/memories/mem_1").json() == before
    assert client.get("/memories/mem_1/files/splat/lods/lod0.spz").content == lod


def test_the_api_serves_a_stored_memory_from_the_database(store):
    memory_files.store("mem_1")
    db.set_status("mem_1", "ready")
    for key in ("publish/manifest.json", "splat/lods/lod0.spz", "separate/stems/s1.wav"):
        store.delete(f"mem_1/{key}")  # gone from storage: the database still has the memory
    client = TestClient(app)
    body = client.get("/memories/mem_1").json()
    assert body["status"] == "ready" and body["manifest"] == MANIFEST
    r = client.get("/memories/mem_1/files/splat/lods/lod0.spz")
    assert r.status_code == 200 and r.content == FILES["splat/lods/lod0.spz"]
    assert r.headers["content-type"] == "application/octet-stream"
    wav = client.get("/memories/mem_1/files/separate/stems/s1.wav")
    assert wav.content == FILES["separate/stems/s1.wav"] and wav.headers["content-type"] == "audio/wav"


def test_the_database_copy_wins_over_storage(store):
    """Mid re-run, storage holds new outputs; the memory stays what was stored last."""
    memory_files.store("mem_1")
    store.write_bytes("mem_1/splat/lods/lod0.spz", b"half-trained")
    manifest = json.loads(json.dumps(MANIFEST))
    manifest["version"] = 5
    store.write_bytes("mem_1/publish/manifest.json", json.dumps(manifest).encode())
    client = TestClient(app)
    assert client.get("/memories/mem_1").json()["manifest"]["version"] == 4
    assert client.get("/memories/mem_1/files/splat/lods/lod0.spz").content == FILES["splat/lods/lod0.spz"]


def test_files_that_are_not_stored_come_from_storage(store):
    memory_files.store("mem_1")
    store.write_bytes("mem_1/ingest/meta.json", b'{"fps": 30}')  # a stage output, not part of the memory
    client = TestClient(app)
    assert client.get("/memories/mem_1/files/ingest/meta.json").content == b'{"fps": 30}'
    assert client.get("/memories/mem_1/files/upload/clip.mp4").status_code == 404  # still private


def test_stored_files_support_etags_and_ranges(store):
    memory_files.store("mem_1")
    client = TestClient(app)
    url = "/memories/mem_1/files/separate/stems/s1.wav"
    data = FILES["separate/stems/s1.wav"]
    etag = client.get(url).headers["etag"]
    assert etag == f'"{hashlib.sha256(data).hexdigest()}"'
    assert client.get(url, headers={"If-None-Match": etag}).status_code == 304
    part = client.get(url, headers={"Range": "bytes=4-11"})
    assert part.status_code == 206 and part.content == data[4:12]
    assert part.headers["content-range"] == f"bytes 4-11/{len(data)}"
    assert client.get(url, headers={"Range": "bytes=-4"}).content == data[-4:]
    assert client.get(url, headers={"Range": "bytes=99999-"}).status_code == 416
