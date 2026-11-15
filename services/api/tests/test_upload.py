"""Create → upload → process, against a fake runner (tests/fake_runner.py)."""

import json
import sys
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from app import db, pipeline, settings, storage
from app.main import app

FAKE_RUNNER = Path(__file__).with_name("fake_runner.py")
VIDEO = b"\x00\x00\x00\x18ftypmp42" + bytes(range(256)) * 40


@pytest.fixture
def api(tmp_path, monkeypatch):
    root = tmp_path / "storage"
    root.mkdir()
    monkeypatch.setenv("STORAGE_BACKEND", "local")
    monkeypatch.setenv("STORAGE_ROOT", str(root))
    storage.get_store.cache_clear()
    mode = {"value": "done"}
    monkeypatch.setattr(pipeline, "build_command", lambda memory_id, r: [
        sys.executable, str(FAKE_RUNNER), str(r), memory_id, mode["value"]])
    with TestClient(app) as client:
        yield client, root, mode
    storage.get_store.cache_clear()


def create(client, names=("IMG_1234.MOV",), title="Golden Gate walk"):
    r = client.post("/memories", json={"title": title, "file_names": list(names)})
    assert r.status_code == 201, r.text
    return r.json()


def upload(client, body, name, data=VIDEO):
    url = next(u["url"] for u in body["uploads"] if u["file_name"] == name)
    return client.put(url, content=data, headers={"Content-Type": "application/octet-stream"})


def process_and_wait(client, memory_id):
    r = client.post(f"/memories/{memory_id}/process")
    assert r.status_code == 202, r.text
    assert pipeline.wait(memory_id, timeout=20)
    return client.get(f"/memories/{memory_id}").json()


# ---------------------------------------------------------------- happy path


def test_create_upload_process_ready(api):
    client, root, _ = api
    body = create(client, names=("IMG_1234.MOV", "recorder.wav"))
    mid = body["id"]
    assert mid.startswith("mem_")
    assert [u["file_name"] for u in body["uploads"]] == ["IMG_1234.MOV", "recorder.wav"]
    assert body["uploads"][0]["url"] == f"http://testserver/memories/{mid}/upload/IMG_1234.MOV"
    assert db.get_memory(mid)["status"] == "draft"

    r = upload(client, body, "IMG_1234.MOV")
    assert r.status_code == 200
    assert r.json() == {"file_name": "IMG_1234.MOV", "stored_as": "video.mov", "bytes": len(VIDEO)}
    assert (root / mid / "upload" / "video.mov").read_bytes() == VIDEO
    assert db.get_memory(mid)["status"] == "uploading"
    assert upload(client, body, "recorder.wav", b"RIFF....WAVE").json()["stored_as"] == "audio.wav"
    meta = json.loads((root / mid / "upload" / "meta.json").read_text())
    assert meta["video"] == "video.mov" and meta["audio_original_name"] == "recorder.wav"

    m = process_and_wait(client, mid)
    assert m["status"] == "ready" and m["error"] is None
    assert m["manifest"] == {"id": mid, "version": 1}
    assert m["title"] == "Golden Gate walk"
    assert m["stage_detail"]["state"] == "done"
    assert m["stage_detail"]["stages"]["poses"] == {"status": "done", "seconds": 1.0}
    assert {j["stage"]: j["status"] for j in db.list_jobs(mid)} == {s: "done" for s in m["stages"]}


@pytest.mark.parametrize("base", [None, "/api"])
def test_upload_url_is_encoded_for_names_with_spaces(api, monkeypatch, base):
    """Phone apps name videos like this. The URL must be ready to use as given (curl and
    other non-browser clients don't encode it), and the name must come back unchanged."""
    client, root, _ = api
    if base:
        monkeypatch.setenv("PUBLIC_API_URL", base)
    name = "WhatsApp Video 2026-11-14 at 5.20.26 PM (1).mp4"
    body = create(client, names=(name,))
    url = body["uploads"][0]["url"]
    assert " " not in url
    assert url.endswith("/upload/WhatsApp%20Video%202026-11-14%20at%205.20.26%20PM%20%281%29.mp4")
    r = client.put(url.removeprefix(base or ""), content=VIDEO)
    assert r.status_code == 200 and r.json()["file_name"] == name
    assert (root / body["id"] / "upload" / "video.mp4").read_bytes() == VIDEO


def test_get_memory_keeps_the_viewer_fields(api):
    client, _, _ = api
    body = create(client)
    m = client.get(f"/memories/{body['id']}").json()
    for field in ("id", "status", "stages", "manifest", "asset_base"):
        assert field in m
    assert m["status"] == "draft" and m["manifest"] is None and m["stage_detail"] is None
    assert m["asset_base"] == f"http://testserver/memories/{body['id']}/files/"
    assert m["visibility"] == "private" and m["lat"] is None


def test_list_memories(api):
    client, _, _ = api
    a = create(client, title="a")["id"]
    b = create(client, title="b")["id"]
    ids = [m["id"] for m in client.get("/memories").json()]
    assert set(ids) == {a, b}


def test_reupload_replaces_the_video(api):
    client, root, _ = api
    body = create(client, names=("first.mp4",))
    upload(client, body, "first.mp4", b"one")
    upload(client, body, "first.mp4", b"two")
    files = sorted(p.name for p in (root / body["id"] / "upload").iterdir())
    assert files == ["meta.json", "video.mp4"]
    assert (root / body["id"] / "upload" / "video.mp4").read_bytes() == b"two"


def test_suggested_location_from_video_gps(api):
    client, root, _ = api
    body = create(client)
    ingest = root / body["id"] / "ingest"
    ingest.mkdir(parents=True)
    (ingest / "meta.json").write_text(json.dumps({"geo": {"lat": 37.819912, "lon": -122.478255, "alt": 10}}))
    m = client.get(f"/memories/{body['id']}").json()
    lat, lon = db.blur_location(37.819912, -122.478255)
    assert m["suggested_location"] == {"lat": lat, "lon": lon}


# ---------------------------------------------------------------- failures


def test_failed_run_records_the_reason(api):
    client, _, mode = api
    mode["value"] = "fail"
    body = create(client)
    upload(client, body, "IMG_1234.MOV")
    m = process_and_wait(client, body["id"])
    assert m["status"] == "failed"
    assert m["error"].startswith("poses: COLMAP registered too few frames")
    assert m["stage_detail"]["stages"]["poses"]["status"] == "failed"
    assert m["stage_detail"]["stages"]["splat"] == {"status": "blocked", "error": "needs poses"}


def test_crashed_runner_is_failed_with_its_output(api):
    client, _, mode = api
    mode["value"] = "crash"
    body = create(client)
    upload(client, body, "IMG_1234.MOV")
    m = process_and_wait(client, body["id"])
    assert m["status"] == "failed"
    assert "exited with code 3" in m["error"] and "ModuleNotFoundError" in m["error"]
    assert m["stage_detail"] is None


def test_run_stopped_mid_stage(api):
    client, _, mode = api
    mode["value"] = "killed"
    body = create(client)
    upload(client, body, "IMG_1234.MOV")
    m = process_and_wait(client, body["id"])
    assert m["status"] == "failed" and "stopped unexpectedly" in m["error"]
    jobs = {j["stage"]: j for j in db.list_jobs(body["id"])}
    assert jobs["splat"]["status"] == "failed" and "stopped during" in jobs["splat"]["error"]
    assert jobs["publish"]["status"] == "pending"


def test_reprocess_after_failure(api):
    client, _, mode = api
    body = create(client)
    upload(client, body, "IMG_1234.MOV")
    mode["value"] = "fail"
    assert process_and_wait(client, body["id"])["status"] == "failed"
    mode["value"] = "done"
    m = process_and_wait(client, body["id"])
    assert m["status"] == "ready" and m["error"] is None
    assert m["stage_detail"]["stages"]["poses"]["status"] == "done"


def test_while_processing(api):
    client, root, mode = api
    mode["value"] = "wait"
    body = create(client)
    mid = body["id"]
    upload(client, body, "IMG_1234.MOV")
    assert client.post(f"/memories/{mid}/process").status_code == 202
    try:
        m = client.get(f"/memories/{mid}").json()
        assert m["status"] == "processing"
        assert client.post(f"/memories/{mid}/process").status_code == 409
        assert upload(client, body, "IMG_1234.MOV").status_code == 409
    finally:
        (root / mid / "release").touch()
        assert pipeline.wait(mid, timeout=20)
    assert client.get(f"/memories/{mid}").json()["status"] == "ready"


def test_interrupted_runs_are_failed_on_startup(api):
    client, _, _ = api
    mid = create(client)["id"]
    db.set_status(mid, "processing")  # as if the API died mid-run
    assert pipeline.recover_interrupted() == 1
    m = db.get_memory(mid)
    assert m["status"] == "failed" and "interrupted" in m["error"]


def test_pipeline_not_installed(api, monkeypatch):
    client, _, _ = api
    monkeypatch.setattr(pipeline, "build_command", lambda mid, r: ["/nonexistent/python", "-m", "x"])
    body = create(client)
    upload(client, body, "IMG_1234.MOV")
    r = client.post(f"/memories/{body['id']}/process")
    assert r.status_code == 503 and "not installed" in r.json()["detail"]
    assert db.get_memory(body["id"])["status"] == "uploading"


# ---------------------------------------------------------------- bad requests


@pytest.mark.parametrize("names,needle", [
    ([], "one video"),
    (["notes.txt"], "unsupported"),
    (["a.wav"], "exactly one video"),
    (["a.mp4", "b.mov"], "exactly one video"),
    (["a.mp4", "a.wav", "b.flac"], "at most one"),
    (["a.mp4", "a.mp4"], "twice"),
    (["../evil.mp4"], "characters"),
    (["clip?.mp4"], "characters"),
])
def test_create_rejects_bad_file_lists(api, names, needle):
    client, _, _ = api
    r = client.post("/memories", json={"file_names": names})
    assert r.status_code == 422 and needle in r.json()["detail"]


def test_create_rejects_panoramas(api):
    client, _, _ = api
    r = client.post("/memories", json={"file_names": ["a.mp4"], "input_kind": "panoramas"})
    assert r.status_code == 422


def test_upload_errors(api):
    client, root, _ = api
    body = create(client)
    mid = body["id"]
    assert client.put(f"/memories/{mid}/upload/other.mp4", content=VIDEO).status_code == 422
    assert client.put("/memories/mem_missing0000/upload/IMG_1234.MOV", content=VIDEO).status_code == 404
    assert upload(client, body, "IMG_1234.MOV", b"").status_code == 422
    assert not (root / mid / "upload" / "video.mov").exists()


def test_upload_too_large(api, monkeypatch):
    client, root, _ = api
    monkeypatch.setenv("MAX_UPLOAD_MB", "1")
    settings.get_settings.cache_clear()
    body = create(client)
    big = b"x" * (1024 * 1024 + 1)
    assert upload(client, body, "IMG_1234.MOV", big).status_code == 413
    # Streamed without a size header: stopped partway, nothing left behind.
    url = body["uploads"][0]["url"]
    r = client.put(url, content=iter([b"x" * 600_000, b"x" * 600_000]))
    assert r.status_code == 413
    assert sorted(p.name for p in (root / body["id"] / "upload").iterdir()) == ["meta.json"]


def test_process_needs_a_video(api):
    client, _, _ = api
    mid = create(client)["id"]
    assert client.post(f"/memories/{mid}/process").status_code == 422
    assert client.post("/memories/mem_missing0000/process").status_code == 404


def test_cors_allows_uploads(api):
    client, _, _ = api
    r = client.options("/memories/x/upload/a.mp4", headers={
        "Origin": "http://localhost:3000", "Access-Control-Request-Method": "PUT"})
    assert r.status_code == 200 and "PUT" in r.headers["access-control-allow-methods"]


def test_cli_memories_without_a_row_still_work(api):
    """Memories made with memoryscapes-worker (no database row) read as before."""
    client, root, _ = api
    (root / "demo" / "ingest").mkdir(parents=True)
    (root / "demo" / "ingest" / "_stage.json").write_text("{}")
    m = client.get("/memories/demo").json()
    assert m["status"] == "processing" and m["stages"]["ingest"] is True
    assert m["title"] is None and m["suggested_location"] is None
