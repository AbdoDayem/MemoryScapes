"""Run the whole pipeline for one uploaded memory, writing progress for the debugger.

    python -m memoryscapes_workers.run --root <data root> --memory <id> [--max-seconds 30] [--no-detector]

Each stage runs in its own process through the worker CLI's child mode
(``python -m memoryscapes_workers run <id> --stages <s> --in-process --result-file ...``),
because pycolmap and torch bundle different OpenMP runtimes that can abort when loaded
into one process (see cli.py). Their output is streamed into this run's log.

Stages run in DAG order. A stage that isn't implemented yet is marked skipped; a stage
whose dependency failed is marked blocked; the rest keep going, so an SfM failure still
leaves you the separated audio to inspect. Every stage is forced: an upload replaces the
memory's files under the same names, which the stage fingerprints wouldn't notice.

Progress: <root>/<id>/job/status.json (rewritten as stages start and finish, and with the
latest log line while one runs), full log: <root>/<id>/job/log.txt.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import subprocess
import sys
import tempfile
import time
from collections.abc import Callable
from pathlib import Path

from memoryscapes_workers.pipeline import DAG, topological_order

log = logging.getLogger("memoryscapes.run")


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
        # On Windows the replace fails while a reader (the debugger polling this file,
        # an indexer, antivirus) has it open; it's free again within milliseconds.
        for attempt in range(20):
            try:
                os.replace(tmp, self.path)
                return
            except PermissionError:
                if attempt == 19:
                    raise
                time.sleep(0.05)

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


def stage_config(max_seconds: float, max_keyframes: int) -> dict:
    """Per-stage options for a quick CPU run (the stages' own defaults target a GPU worker)."""
    return {
        "ingest": {"max_duration_s": max_seconds, "max_keyframes": max_keyframes, "min_keyframes": min(40, max_keyframes)},
        "poses": {"use_gpu": False},
    }


def run_stage_process(stage: str, memory_id: str, root: Path, config: dict, env: dict, on_line: Callable[[str], None]) -> tuple[str, str | None]:
    """Run one stage in a child process. Returns (status, error)."""
    with tempfile.TemporaryDirectory(prefix="ms-job-") as tmp:
        cfg_path, out_path = Path(tmp) / "config.json", Path(tmp) / "result.json"
        cfg_path.write_text(json.dumps(config))
        cmd = [sys.executable, "-m", "memoryscapes_workers", "run", memory_id, "--stages", stage, "--in-process",
               "--force", "--config", str(cfg_path), "--result-file", str(out_path)]
        child_env = {**os.environ, **env, "STORAGE_BACKEND": "local", "STORAGE_ROOT": str(root), "PYTHONIOENCODING": "utf-8"}
        proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, env=child_env, text=True, encoding="utf-8", errors="replace")
        assert proc.stdout is not None
        tail: list[str] = []
        for line in proc.stdout:
            line = line.rstrip()
            if line:
                on_line(line)
                tail = (tail + [line])[-15:]
        code = proc.wait()
        out = json.loads(out_path.read_text()) if out_path.exists() else {}

    if "result" in out:
        return "done", None
    if "stage_error" in out:
        e = out["stage_error"]
        return "failed", e["message"] + (f" Hint: {e['hint']}" if e.get("hint") else "")
    error = out.get("error") or f"exited with code {code}: " + " | ".join(tail[-3:])
    if error.startswith("NotImplementedError"):
        return "skipped", "not implemented yet"
    return "failed", error


def main(argv: list[str] | None = None, run_stage: Callable[..., tuple[str, str | None]] = run_stage_process) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", required=True, help="data root (one folder per memory)")
    ap.add_argument("--memory", required=True)
    ap.add_argument("--max-seconds", type=float, default=60.0, help="process at most this much of the video")
    ap.add_argument("--max-keyframes", type=int, default=120, help="cap keyframes (COLMAP time grows with them)")
    ap.add_argument("--no-detector", action="store_true", help="skip Grounding DINO (much faster on CPU)")
    ap.add_argument("--only", help="comma-separated stages to run (others are left as they are)")
    args = ap.parse_args(argv)

    root = Path(args.root).resolve()
    job_dir = root / args.memory / "job"
    job_dir.mkdir(parents=True, exist_ok=True)
    options = {"max_seconds": args.max_seconds, "max_keyframes": args.max_keyframes, "detector": not args.no_detector}
    status = StatusWriter(job_dir / "status.json", args.memory, options)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(message)s",
        datefmt="%H:%M:%S",
        handlers=[logging.FileHandler(job_dir / "log.txt", mode="w", encoding="utf-8"), logging.StreamHandler(), DetailHandler(status)],
        force=True,
    )
    env = {"DETECTOR": "none"} if args.no_detector else {}
    config = stage_config(args.max_seconds, args.max_keyframes)

    only = set(args.only.split(",")) if args.only else None
    unavailable: set[str] = set()  # failed, skipped or blocked: dependents can't run
    for name in topological_order():
        deps = DAG[name][1]
        if only and name not in only:
            status.stage(name, status="not requested")
            continue
        blocked = [d for d in deps if d in unavailable]
        if blocked:
            unavailable.add(name)
            status.stage(name, status="blocked", error=f"needs {', '.join(blocked)}")
            log.warning("%s blocked by %s", name, ", ".join(blocked))
            continue
        status.state.update(current=name, detail="")
        status.stage(name, status="running", started_at=time.time())
        t0 = time.perf_counter()
        result, error = run_stage(name, args.memory, root, config, env, lambda line: log.info("%s", line))
        seconds = round(time.perf_counter() - t0, 2)
        status.stage(name, status=result, seconds=seconds, **({"error": error} if error else {}))
        log.info("%s %s in %.1f s%s", name, result, seconds, f": {error}" if error else "")
        if result != "done":
            unavailable.add(name)

    stages = status.state["stages"]
    problems = [n for n, s in stages.items() if s["status"] in ("failed", "blocked") and not _only_blocked_by_skips(n, stages)]
    status.state.update(state="failed" if problems else "done", current=None, finished_at=time.time(), detail="")
    status.write(force=True)
    return 1 if problems else 0


def _only_blocked_by_skips(name: str, stages: dict) -> bool:
    """publish waits on stages that don't exist yet; that isn't a failed run."""
    s = stages[name]
    if s["status"] != "blocked":
        return False
    return all(stages[d]["status"] in ("skipped",) or _only_blocked_by_skips(d, stages) for d in DAG[name][1] if stages[d]["status"] != "done")


if __name__ == "__main__":
    raise SystemExit(main())
