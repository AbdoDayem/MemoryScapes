import json

import pytest
from pydantic import BaseModel

from memoryscapes_workers.stagekit import STAGE_MARKER, StageError, run_stage
from memoryscapes_workers.storage import LocalStorage, download_prefix, upload_dir


def test_local_storage_prefix_semantics(storage: LocalStorage):
    storage.write_bytes("m/a/x.txt", b"1")
    storage.write_bytes("m/a/sub/y.txt", b"2")
    storage.write_bytes("m/ab/z.txt", b"3")
    assert storage.list("m/a/") == ["m/a/sub/y.txt", "m/a/x.txt"]
    assert storage.list("m/a") == ["m/a/sub/y.txt", "m/a/x.txt", "m/ab/z.txt"]  # S3-style
    assert storage.list("") == ["m/a/sub/y.txt", "m/a/x.txt", "m/ab/z.txt"]
    storage.delete_prefix("m/a")
    assert storage.list("m/") == ["m/ab/z.txt"]


def test_local_storage_rejects_escape(storage: LocalStorage):
    with pytest.raises(ValueError):
        storage.write_bytes("../outside.txt", b"x")


def test_upload_download_roundtrip(storage, tmp_path):
    src = tmp_path / "src"
    (src / "d").mkdir(parents=True)
    (src / "a.bin").write_bytes(b"a")
    (src / "d" / "b.bin").write_bytes(b"b")
    keys = upload_dir(storage, src, "m/stage")
    assert keys == ["m/stage/a.bin", "m/stage/d/b.bin"]
    dest = tmp_path / "dest"
    download_prefix(storage, "m/stage/", dest)
    assert (dest / "d" / "b.bin").read_bytes() == b"b"


class Cfg(BaseModel):
    value: int = 1


def _body(calls: list, name="out.txt"):
    def body(io):
        calls.append(io.stage)
        (io.out_dir / name).write_text("hello")
        return {"ok": True}
    return body


def test_run_stage_skips_when_unchanged_and_reruns_on_config_change(ctx):
    calls: list = []
    ctx.storage.write_bytes(f"{ctx.memory_id}/upload/v.mp4", b"video")
    run_stage(ctx, "s", version="1", config=Cfg(), upstream=["upload"], body=_body(calls))
    run_stage(ctx, "s", version="1", config=Cfg(), upstream=["upload"], body=_body(calls))
    assert calls == ["s"]
    marker = json.loads(ctx.storage.read_bytes(ctx.prefix("s") + STAGE_MARKER))
    assert marker["summary"] == {"ok": True}

    run_stage(ctx, "s", version="1", config=Cfg(value=2), upstream=["upload"],
              body=_body(calls, name="new.txt"))
    assert calls == ["s", "s"]
    # Stale outputs from the previous run are gone.
    assert ctx.storage.list(ctx.prefix("s")) == [ctx.prefix("s") + STAGE_MARKER, ctx.prefix("s") + "new.txt"]


def test_downstream_reruns_when_upstream_changes(ctx):
    calls: list = []
    ctx.storage.write_bytes(f"{ctx.memory_id}/upload/v.mp4", b"video")
    run_stage(ctx, "a", version="1", config=Cfg(), upstream=["upload"], body=_body(calls))
    run_stage(ctx, "b", version="1", config=Cfg(), upstream=["a"], body=_body(calls))
    run_stage(ctx, "a", version="2", config=Cfg(), upstream=["upload"], body=_body(calls))
    run_stage(ctx, "b", version="1", config=Cfg(), upstream=["a"], body=_body(calls))
    assert calls == ["a", "b", "a", "b"]


def test_failed_stage_leaves_no_marker(ctx):
    ctx.storage.write_bytes(f"{ctx.memory_id}/upload/v.mp4", b"video")

    def boom(io):
        (io.out_dir / "partial.txt").write_text("x")
        raise RuntimeError("crash")

    with pytest.raises(RuntimeError):
        run_stage(ctx, "s", version="1", config=Cfg(), upstream=["upload"], body=boom)
    assert ctx.storage.list(ctx.prefix("s")) == []


def test_missing_upstream_raises(ctx):
    with pytest.raises(StageError):
        run_stage(ctx, "b", version="1", config=Cfg(), upstream=["a"], body=_body([]))
    with pytest.raises(StageError, match="No uploaded files"):
        run_stage(ctx, "ingest", version="1", config=Cfg(), upstream=["upload"], body=_body([]))
