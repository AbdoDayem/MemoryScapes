"""Pipeline DAG definition.

    ingest ─┬─> poses ──> splat ─┬─> localize ──> soundscape ──> publish
            └─> separate ────────┘

The visual and audio branches run in parallel and meet at localization, which needs
camera poses and scene geometry. Wire this into Temporal (or Celery for the MVP).
"""

from collections.abc import Callable, Iterable

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

Stage = Callable[[StageContext], object]

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

# The implemented MVP path: video -> keyframes -> poses -> splat.
VISUAL_STAGES = ["ingest", "poses", "splat"]


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


def ancestors(stage: str) -> list[str]:
    """All stages ``stage`` depends on, in topological order."""
    seen: set[str] = set()
    stack = list(DAG[stage][1])
    while stack:
        s = stack.pop()
        if s not in seen:
            seen.add(s)
            stack.extend(DAG[s][1])
    return [s for s in topological_order() if s in seen]


def run_local(ctx: StageContext, stages: Iterable[str] | None = None) -> dict[str, object]:
    """Run stages sequentially (all of them by default). For local development only.

    Completed stages whose inputs and config are unchanged are skipped automatically.
    """
    wanted = list(stages) if stages is not None else topological_order()
    unknown = [s for s in wanted if s not in DAG]
    if unknown:
        raise ValueError(f"Unknown stages: {unknown}")
    results = {}
    for name in topological_order():
        if name in wanted:
            fn, _ = DAG[name]
            results[name] = fn(ctx)
    return results
