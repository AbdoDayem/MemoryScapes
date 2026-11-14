import json

import pytest
from fastapi.testclient import TestClient

from app import storage
from app.main import app

MANIFEST = {"id": "mem_1", "version": 1, "coord_frame": {"units": "meters", "up": "+y"},
            "splat": {"lods": [{"url": "splat/lods/lod0.spz", "count": 10}]},
            "audio": {"sources": []}, "meta": {"visibility": "private"}}


@pytest.fixture
def client(tmp_path, monkeypatch):
    root = tmp_path / "storage"

    def put(key, data=b"x"):
        path = root / key
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(data)

    for stage in ("ingest", "poses", "splat"):
        put(f"mem_1/{stage}/_stage.json", b"{}")
    put("mem_1/upload/clip.mp4", b"raw video")
    put("mem_1/splat/lods/lod0.spz", b"SPZDATA")
    put("mem_2/upload/clip.mp4", b"raw video")  # uploaded, nothing processed yet
    monkeypatch.setenv("STORAGE_BACKEND", "local")
    monkeypatch.setenv("STORAGE_ROOT", str(root))
    storage.get_store.cache_clear()
    yield TestClient(app), put
    storage.get_store.cache_clear()


def test_processing_then_ready(client):
    c, put = client
    r = c.get("/memories/mem_1")
    assert r.status_code == 200
    body = r.json()
    assert body["status"] == "processing" and body["manifest"] is None
    assert body["stages"] == {"ingest": True, "poses": True, "splat": True, "separate": False,
                              "localize": False, "soundscape": False, "publish": False}
    assert body["asset_base"] == "http://testserver/memories/mem_1/files/"

    put("mem_1/publish/_stage.json", b"{}")
    put("mem_1/publish/manifest.json", json.dumps(MANIFEST).encode())
    body = c.get("/memories/mem_1").json()
    assert body["status"] == "ready" and body["manifest"]["splat"]["lods"][0]["url"] == "splat/lods/lod0.spz"
    # The manifest's relative URL resolves against asset_base.
    lod = c.get(body["asset_base"] + body["manifest"]["splat"]["lods"][0]["url"])
    assert lod.status_code == 200 and lod.content == b"SPZDATA"


def test_uploaded_but_unprocessed_and_unknown(client):
    c, _ = client
    assert c.get("/memories/mem_2").json()["status"] == "processing"
    assert c.get("/memories/nope").status_code == 404
    assert c.get("/memories/..%2Fetc").status_code == 404


@pytest.mark.parametrize("key", ["upload/clip.mp4", "splat/_stage.json", "splat/missing.spz",
                                 "../mem_2/upload/clip.mp4", "splat/../../mem_2/upload/clip.mp4"])
def test_files_refuses_private_or_missing(client, key):
    c, _ = client
    assert c.get(f"/memories/mem_1/files/{key}").status_code == 404


def test_cors_allows_the_web_app(client):
    c, _ = client
    r = c.get("/memories/mem_1", headers={"Origin": "http://localhost:3000"})
    assert r.headers["access-control-allow-origin"] == "http://localhost:3000"


def test_range_requests_for_audio_seeking(client):
    c, put = client
    put("mem_1/ingest/audio.wav", bytes(range(256)) * 4)
    r = c.get("/memories/mem_1/files/ingest/audio.wav", headers={"Range": "bytes=10-19"})
    assert r.status_code == 206 and r.content == bytes(range(10, 20))
