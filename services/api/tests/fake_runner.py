"""Stands in for ``python -m memoryscapes_workers.run`` in tests, so they never run the
real pipeline. Writes job/status.json the way services/workers/memoryscapes_workers/run.py
does.

    python fake_runner.py <root> <memory id> <mode>

Modes: ``done`` (every stage done, writes a manifest), ``fail`` (poses fails),
``crash`` (exits without a status file), ``killed`` (dies mid-stage, status left "running"),
``wait`` (runs until <root>/<id>/release exists).
"""

import json
import sys
import time
from pathlib import Path

STAGES = ["ingest", "poses", "splat", "separate", "localize", "soundscape", "publish"]


def write_status(job: Path, state: str, stages: dict, current=None) -> None:
    job.mkdir(parents=True, exist_ok=True)
    status = {"memory_id": job.parent.name, "state": state, "started_at": 1_731_600_000.0,
              "finished_at": None if state == "running" else 1_731_600_100.0, "current": current,
              "detail": "fake runner", "options": {}, "stages": stages}
    tmp = job / "status.tmp"
    tmp.write_text(json.dumps(status))
    tmp.replace(job / "status.json")


def main() -> int:
    root, memory_id, mode = Path(sys.argv[1]), sys.argv[2], sys.argv[3]
    mem = root / memory_id
    job = mem / "job"
    assert (mem / "upload" / "meta.json").exists(), "runner started before the upload"

    if mode == "crash":
        print("Traceback: ModuleNotFoundError: No module named 'memoryscapes_workers'", file=sys.stderr)
        return 3

    if mode == "killed":
        write_status(job, "running", {
            "ingest": {"status": "done", "started_at": 1_731_600_000.0, "seconds": 1.0},
            "splat": {"status": "running", "started_at": 1_731_600_001.0},
            "publish": {"status": "pending"}}, current="splat")
        return -9

    if mode == "wait":
        write_status(job, "running", {"ingest": {"status": "running", "started_at": 1_731_600_000.0}},
                     current="ingest")
        deadline = time.time() + 20
        while not (mem / "release").exists() and time.time() < deadline:
            time.sleep(0.02)
        mode = "done"

    if mode == "fail":
        write_status(job, "failed", {
            "ingest": {"status": "done", "started_at": 1_731_600_000.0, "seconds": 5.0},
            "poses": {"status": "failed", "started_at": 1_731_600_005.0, "seconds": 2.0,
                      "error": "COLMAP registered too few frames. Hint: move more slowly."},
            **{s: {"status": "blocked", "error": "needs poses"} for s in STAGES[2:]},
        })
        return 1

    publish = mem / "publish"
    publish.mkdir(parents=True, exist_ok=True)
    (publish / "manifest.json").write_text(json.dumps({"id": memory_id, "version": 1}))
    (publish / "_stage.json").write_text("{}")
    write_status(job, "done", {s: {"status": "done", "started_at": 1_731_600_000.0 + i, "seconds": 1.0}
                               for i, s in enumerate(STAGES)})
    return 0


if __name__ == "__main__":
    sys.exit(main())
