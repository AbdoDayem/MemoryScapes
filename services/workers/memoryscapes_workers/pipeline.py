"""Pipeline DAG definition.

    ingest ─┬─> poses ──> splat ─┬─> localize ──> soundscape ──> publish
            └─> separate ────────┘

The visual and audio branches run in parallel and meet at localization, which needs
camera poses and scene geometry. Wire this into Temporal (or Celery for the MVP).
"""

from collections.abc import Callable

from memoryscapes_workers.context import StageContext
from memoryscapes_workers.stages import (
    ingest,
    localize,
    poses,
    publish,
    separate,
    soundscape,
    splat,
)

Stage = Callable[[StageContext], None]

# stage name -> (function, upstream dependencies)
DAG: dict[str, tuple[Stage, list[str]]] = {
    "ingest": (ingest.run, []),
    "poses": (poses.run, ["ingest"]),
    "splat": (splat.run, ["poses"]),
    "separate": (separate.run, ["ingest"]),
    "localize": (localize.run, ["splat", "separate"]),
    "soundscape": (soundscape.run, ["localize"]),
    "publish": (publish.run, ["splat", "soundscape"]),
}


def topological_order() -> list[str]:
    done: list[str] = []
    pending = dict(DAG)
    while pending:
        ready = [name for name, (_, deps) in pending.items() if all(d in done for d in deps)]
        if not ready:
            raise ValueError("Cycle in pipeline DAG")
        for name in ready:
            done.append(name)
            del pending[name]
    return done


def run_local(ctx: StageContext) -> None:
    """Run every stage sequentially. For local development only."""
    for name in topological_order():
        fn, _ = DAG[name]
        fn(ctx)
