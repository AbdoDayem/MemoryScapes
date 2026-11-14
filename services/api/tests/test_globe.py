"""PATCH /memories/{id} and GET /globe."""

import pytest
from fastapi.testclient import TestClient

from app import db, seed, storage
from app.main import app


@pytest.fixture
def client(tmp_path, monkeypatch):
    monkeypatch.setenv("STORAGE_BACKEND", "local")
    monkeypatch.setenv("STORAGE_ROOT", str(tmp_path / "storage"))
    storage.get_store.cache_clear()
    with TestClient(app) as c:
        yield c
    storage.get_store.cache_clear()


def new(client, title="Golden Gate walk"):
    return client.post("/memories", json={"title": title, "file_names": ["a.mp4"]}).json()["id"]


# ---------------------------------------------------------------- PATCH


def test_patch_title_and_location(client):
    mid = new(client)
    r = client.patch(f"/memories/{mid}", json={"title": "Fog", "lat": 37.819912, "lon": -122.478255})
    assert r.status_code == 200
    m = r.json()
    assert m["title"] == "Fog"
    assert (m["lat"], m["lon"]) == db.blur_location(37.819912, -122.478255)  # rounded, not exact
    assert m["location_source"] == "user_pin"
    for field in ("id", "status", "stages", "manifest", "asset_base"):  # same shape as GET
        assert field in m


def test_patch_only_changes_what_is_sent(client):
    mid = new(client, title="Keep me")
    client.patch(f"/memories/{mid}", json={"lat": 10, "lon": 10})
    m = client.patch(f"/memories/{mid}", json={"description": "Windy"}).json()
    assert m["title"] == "Keep me" and m["lat"] is not None and m["description"] == "Windy"


def test_patch_video_location_and_clearing(client):
    mid = new(client)
    m = client.patch(f"/memories/{mid}", json={"lat": 1, "lon": 2, "location_source": "video_gps"}).json()
    assert m["location_source"] == "video_gps"
    m = client.patch(f"/memories/{mid}", json={"lat": None, "lon": None}).json()
    assert m["lat"] is None and m["lon"] is None and m["location_source"] is None


def test_patch_empty_body_is_a_no_op(client):
    mid = new(client)
    r = client.patch(f"/memories/{mid}", json={})
    assert r.status_code == 200 and r.json()["title"] == "Golden Gate walk"


@pytest.mark.parametrize("body,needle", [
    ({"lat": 10}, "together"),
    ({"lon": 10}, "together"),
    ({"lat": 10, "lon": None}, "lat and lon"),
    ({"lat": 95, "lon": 0}, "lat must be"),
    ({"lat": 0, "lon": 200}, "lon must be"),
    ({"visibility": "public"}, "privacy pass"),
    ({"visibility": "unlisted"}, "privacy pass"),
    ({"visibility": None}, "null"),
    ({"location_source": "video_gps"}, "goes with"),
])
def test_patch_rejects(client, body, needle):
    mid = new(client)
    r = client.patch(f"/memories/{mid}", json=body)
    assert r.status_code == 422 and needle in r.json()["detail"]


def test_patch_rejects_bad_types_and_unknown_memory(client):
    mid = new(client)
    assert client.patch(f"/memories/{mid}", json={"lat": "north", "lon": 0}).status_code == 422
    assert client.patch(f"/memories/{mid}", json={"location_source": "guess", "lat": 1, "lon": 1}).status_code == 422
    assert client.patch("/memories/mem_missing0000", json={"title": "x"}).status_code == 404


def test_cors_allows_patch(client):
    r = client.options("/memories/x", headers={
        "Origin": "http://localhost:3000", "Access-Control-Request-Method": "PATCH"})
    assert r.status_code == 200


# ---------------------------------------------------------------- GET /globe


def test_globe_lists_memories_with_a_location(client):
    paris = new(client, "Paris")
    client.patch(f"/memories/{paris}", json={"lat": 48.8584, "lon": 2.2945})
    new(client, "No location yet")
    r = client.get("/globe")
    assert r.status_code == 200
    memories = r.json()["memories"]
    assert [m["id"] for m in memories] == [paris]
    m = memories[0]
    assert set(m) == {"id", "title", "lat", "lon", "status", "captured_at", "created_at", "thumbnail_url"}
    assert m["title"] == "Paris" and m["status"] == "draft"


def test_globe_with_seed_data_and_bbox(client):
    seed.seed()
    everything = client.get("/globe").json()["memories"]
    assert len(everything) == len(seed.SEED_MEMORIES)
    assert all(m["status"] == "ready" for m in everything)
    europe = {m["id"] for m in client.get("/globe", params={"bbox": "-10,35,30,60"}).json()["memories"]}
    assert europe == {"seed_paris-seine", "seed_london-market"}
    pacific = {m["id"] for m in client.get("/globe", params={"bbox": "170,-30,-160,0"}).json()["memories"]}
    assert pacific == {"seed_fiji-reef"}  # crosses ±180°


def test_globe_only_shows_the_dev_users_memories(client):
    db.ensure_user("usr_other", "other")
    theirs = db.create_memory("usr_other", title="theirs")
    db.update_memory(theirs["id"], lat=1, lon=1)
    assert client.get("/globe").json()["memories"] == []


@pytest.mark.parametrize("bbox", ["1,2,3", "a,b,c,d", "0,0,10,95", "0,20,10,10", "nan,0,1,1", "inf,0,1,1"])
def test_globe_rejects_bad_bbox(client, bbox):
    assert client.get("/globe", params={"bbox": bbox}).status_code == 422
