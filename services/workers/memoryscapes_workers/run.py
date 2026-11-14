"""Run the pipeline for one memory on this machine, writing progress for the debugger.

    python -m memoryscapes_workers.run --root <data root> --memory <id> [--max-seconds 30] [--no-detector]

Stages run in DAG order. A stage that isn't implemented yet (splat, publish) is marked
skipped; a stage whose dependency failed is marked blocked; the rest keep going, so an
SfM failure still leaves you the separated audio to inspect.

Progress goes to <root>/<id>/job/status.json (rewritten as stages start and finish, and
with the latest log line while one runs) and the full log to <root>/<id>/job/log.txt.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import time
import traceback
from pathlib import Path

from memoryscapes_workers.context import StageContext
from memoryscapes_workers.pipeline import DAG, topological_order
from memoryscapes_workers.storage import LocalStorage


class StatusWriter:
    def __init__(self, path: Path, memory_id: str, options: dict) -> None:
        self.path = path
        self.state = {
            "memory_id": memory_id,
            "state": "running",
            "started_at": time.time(),
            "finished_at": None,
            "options": options,
            "current": None,
            "detail": "",
            "stages": {name: {"status": "pending"} for name in topological_order()},
        }
        self._last_write = 0.0
        self.write(force=True)

    def write(self, force: bool = False) -> None:
        now = time.time()
        if not force and now - self._last_write < 0.5:
            return
        self._last_write = now
        tmp = self.path.with_suffix(".tmp")
        tmp.write_text(json.dumps(self.state, indent=2), encoding="utf-8")
        os.replace(tmp, self.path)

    def stage(self, name: str, **fields) -> None:
        self.state["stages"][name].update(fields)
        self.write(force=True)


class DetailHandler(logging.Handler):
    """Puts the latest log line in status.json so the UI shows what a long stage is doing."""

    def __init__(self, status: StatusWriter) -> None:
        super().__init__(logging.INFO)
        self.status = status

    def emit(self, record: logging.LogRecord) -> None:
        self.status.state["detail"] = record.getMessage()[:300]
        self.status.write()


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", required=True, help="data root (one folder per memory)")
    ap.add_argument("--memory", required=True)
    ap.add_argument("--max-seconds", type=float, default=60.0, help="process at most this much of the video")
    ap.add_argument("--target-frames", type=int, default=60)
    ap.add_argument("--no-detector", action="store_true", help="skip Grounding DINO (much faster on CPU)")
    ap.add_argument("--only", help="comma-separated stages to run (others are left as they are)")
    args = ap.parse_args(argv)

    root = Path(args.root)
    job_dir = root / args.memory / "job"
    job_dir.mkdir(parents=True, exist_ok=True)
    options = {"max_seconds": args.max_seconds, "target_frames": args.target_frames, "detector": not args.no_detector}
    status = StatusWriter(job_dir / "status.json", args.memory, options)

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        handlers=[logging.FileHandler(job_dir / "log.txt", mode="w", encoding="utf-8"), logging.StreamHandler(), DetailHandler(status)],
        force=True,
    )
    log = logging.getLogger("memoryscapes.run")
    if args.no_detector:
        os.environ["DETECTOR"] = "none"

    from memoryscapes_workers.stages import ingest, localize, poses, separate

    ctx = StageContext(memory_id=args.memory, storage=LocalStorage(root))
    overrides = {
        "ingest": lambda c: ingest.run(c, ingest.IngestConfig(max_duration_s=args.max_seconds, target_frames=args.target_frames)),
        "poses": lambda c: poses.run(c),
        "separate": lambda c: separate.run(c),
        "localize": lambda c: localize.run(c),
    }
    only = set(args.only.split(",")) if args.only else None
    failed: set[str] = set()
    for name in topological_order():
        fn, deps = DAG[name]
        if only and name not in only:
            status.stage(name, status="not requested")
            continue
        blocked = [d for d in deps if d in failed]
        if blocked:
            failed.add(name)
            status.stage(name, status="blocked", error=f"needs {', '.join(blocked)}")
            log.warning("%s blocked by %s", name, blocked)
            continue
        status.state["current"] = name
        status.state["detail"] = ""
        status.stage(name, status="running", started_at=time.time())
        t0 = time.perf_counter()
        try:
            (overrides.get(name) or fn)(ctx)
        except NotImplementedError:
            failed.add(name)
            status.stage(name, status="skipped", error="not implemented yet", seconds=round(time.perf_counter() - t0, 2))
            log.info("%s skipped (not implemented yet)", name)
            continue
        except Exception as exc:  # noqa: BLE001
            failed.add(name)
            status.stage(name, status="failed", error=f"{type(exc).__name__}: {exc}", seconds=round(time.perf_counter() - t0, 2))
            log.error("%s failed:\n%s", name, traceback.format_exc())
            continue
        status.stage(name, status="done", seconds=round(time.perf_counter() - t0, 2))
        log.info("%s done in %.1f s", name, time.perf_counter() - t0)

    real_failures = [n for n, s in status.state["stages"].items() if s["status"] in ("failed", "blocked") and n not in ("publish",)]
    status.state.update(state="failed" if real_failures else "done", current=None, finished_at=time.time(), detail="")
    status.write(force=True)
    return 1 if real_failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
