"""Start the processing pipeline for a memory and record how it went.

The pipeline itself belongs to the workers (services/workers). This module only:

1. starts their runner in the background:
       python -m memoryscapes_workers.run --root <storage folder> --memory <id>
   (the same command the debugger uses, tools/debugger/server/jobs.ts),
2. reads the progress file it keeps up to date, ``{id}/job/status.json``,
3. when the run ends, copies the per-stage results into the ``jobs`` table and sets the
   memory to ``ready`` or ``failed``.

How much of the video is processed is the runner's own default (``--max-seconds``); we
don't override it. See docs/webapp-plan/backend.md § Starting the pipeline.
"""

from __future__ import annotations

import json
import logging
import os
import shutil
import subprocess
import threading
from pathlib import Path
from typing import Any

from app import db
from app.models import MemoryStatus
from app.settings import REPO_DIR, get_settings
from app.storage import LocalStore, get_store

log = logging.getLogger("memoryscapes.api.pipeline")

WORKERS_DIR = REPO_DIR / "services" / "workers"
MANIFEST = "publish/manifest.json"
STATUS = "job/status.json"
# The runner's own stdout/stderr. Its log is job/log.txt; this catches anything before that
# exists (e.g. the workers' Python can't import the package).
RUNNER_OUTPUT = "job/runner-output.txt"

_runs: dict[str, subprocess.Popen] = {}
_watchers: dict[str, threading.Thread] = {}
_lock = threading.Lock()


class PipelineError(Exception):
    """The pipeline can't be started (not installed, wrong storage backend, ...)."""


class AlreadyRunning(Exception):
    pass


def build_command(memory_id: str, root: Path) -> list[str]:
    """The runner command. Tests replace this with a fake runner."""
    return [get_settings().workers_python, "-m", "memoryscapes_workers.run",
            "--root", str(root), "--memory", memory_id]


def is_running(memory_id: str) -> bool:
    with _lock:
        proc = _runs.get(memory_id)
        return proc is not None and proc.poll() is None


def start(memory_id: str) -> None:
    """Start a run in the background. Raises AlreadyRunning or PipelineError."""
    store = get_store()
    if not isinstance(store, LocalStore):
        raise PipelineError("The pipeline runs on local storage only for now (STORAGE_BACKEND=local)")
    cmd = build_command(memory_id, store.root)
    if not Path(cmd[0]).exists() and shutil.which(cmd[0]) is None:
        raise PipelineError(f"Pipeline not installed: {cmd[0]} not found. "
                            "Set up services/workers/.venv or set WORKERS_PYTHON.")

    with _lock:
        running = _runs.get(memory_id)
        if running is not None and running.poll() is None:
            raise AlreadyRunning(memory_id)
        # Clear the previous run's progress, so nobody reads a stale status.json.
        store.delete_prefix(f"{memory_id}/job/")
        out_path = store.path(f"{memory_id}/{RUNNER_OUTPUT}")
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, "wb") as out:
            proc = subprocess.Popen(cmd, cwd=WORKERS_DIR if WORKERS_DIR.is_dir() else None,
                                    stdout=out, stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL,
                                    env={**os.environ, "PYTHONIOENCODING": "utf-8"})
        _runs[memory_id] = proc
        watcher = threading.Thread(target=_watch, args=(memory_id, proc), daemon=True,
                                   name=f"pipeline-{memory_id}")
        _watchers[memory_id] = watcher
        watcher.start()
    log.info("started pipeline for %s (pid %s)", memory_id, proc.pid)


def wait(memory_id: str, timeout: float | None = None) -> bool:
    """Block until the run and its bookkeeping are finished. For tests and scripts."""
    with _lock:
        watcher = _watchers.get(memory_id)
    if watcher is None:
        return True
    watcher.join(timeout)
    return not watcher.is_alive()


def _watch(memory_id: str, proc: subprocess.Popen) -> None:
    code = proc.wait()
    try:
        finish(memory_id, code)
    except Exception:  # never let bookkeeping kill the thread silently
        log.exception("could not record the result of the run for %s", memory_id)
    finally:
        with _lock:
            if _runs.get(memory_id) is proc:
                del _runs[memory_id]


def finish(memory_id: str, returncode: int) -> str:
    """Record a finished run: copy stage results into ``jobs``, then set ``ready`` (the run
    ended ``done`` and wrote a manifest) or ``failed`` with the reason. Returns the status."""
    status = read_status(memory_id)
    if status and isinstance(status.get("stages"), dict):
        stages = status["stages"]
        if status.get("state") == "running":
            # The runner died mid-stage (killed, crashed): that stage will never finish.
            stages = {name: ({**info, "status": "failed", "error": "The pipeline stopped during this stage."}
                             if info.get("status") == "running" else info)
                      for name, info in stages.items()}
        db.upsert_jobs(memory_id, stages)

    if status and status.get("state") == "done" and get_store().exists(f"{memory_id}/{MANIFEST}"):
        db.set_status(memory_id, MemoryStatus.ready)
        return MemoryStatus.ready.value

    db.set_status(memory_id, MemoryStatus.failed, error=failure_reason(status, returncode, memory_id))
    return MemoryStatus.failed.value


def failure_reason(status: dict[str, Any] | None, returncode: int, memory_id: str) -> str:
    """A short, human-readable reason for a failed run."""
    if status:
        for name, stage in (status.get("stages") or {}).items():
            if stage.get("status") == "failed":
                return f"{name}: {stage.get('error') or 'failed'}"
        if status.get("state") == "done":
            return "The pipeline finished but didn't produce a manifest (publish didn't run)."
        if status.get("state") == "running":
            return f"The pipeline stopped unexpectedly (exit code {returncode})."
    tail = _runner_output_tail(memory_id)
    return f"The pipeline exited with code {returncode}" + (f": {tail}" if tail else ".")


def read_status(memory_id: str) -> dict[str, Any] | None:
    """The runner's ``job/status.json``, or None if there isn't a readable one."""
    store = get_store()
    key = f"{memory_id}/{STATUS}"
    try:
        if not store.exists(key):
            return None
        data = json.loads(store.read_bytes(key))
    except (OSError, ValueError):
        return None  # missing, or caught mid-write
    return data if isinstance(data, dict) else None


def recover_interrupted() -> int:
    """Mark memories left ``processing`` by an earlier API process as failed: their runs
    are no longer watched. Returns how many."""
    count = 0
    for memory in db.list_by_status(MemoryStatus.processing):
        if not is_running(memory["id"]):
            db.set_status(memory["id"], MemoryStatus.failed,
                          error="Processing was interrupted (the API restarted). Process it again.")
            count += 1
    return count


def _runner_output_tail(memory_id: str, lines: int = 3) -> str:
    try:
        text = get_store().read_bytes(f"{memory_id}/{RUNNER_OUTPUT}").decode("utf-8", "replace")
    except (OSError, ValueError):
        return ""
    return " | ".join([line for line in text.splitlines() if line.strip()][-lines:])[:500]
