"""Example memories through the real visual pipeline: video -> ingest -> COLMAP -> splat.

    .venv/Scripts/python examples/run_visual.py                    # all scenarios
    .venv/Scripts/python examples/run_visual.py park_walk --steps 300

For each scenario this
1. renders the scene to an .mp4 (examples/make_video.py),
2. runs the real ingest, poses and splat stages on it (the worker CLI, one process each),
3. fits the similarity transform from the scene's true cameras to the recovered ones
   (rotation from the camera orientations, then scale and offset from the positions),
   so the ground truth can be expressed in the pipeline's world frame,
4. runs separate, localize and soundscape on the real ingest and poses outputs, with the
   oracle stand-in models (the detector projects the transformed truth through the
   recovered cameras),
5. writes debug/truth.json, report.md and map.svg like run_example.py.

Memories are named <scenario>_real, next to the stand-in examples in examples/out/, so the
debugger lists both. hard_mode's trained splat is scaled by 0.5 afterwards, so the
debugger's alignment check has a misaligned splat to catch. A scenario whose SfM fails
(courtyard_static turns in place, with almost no parallax) is reported, not forced.
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import os
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from evaluate import BAD_VERDICTS, evaluate, summary  # noqa: E402
from make_video import render_video  # noqa: E402
from run_example import write_map, write_report  # noqa: E402
from scenes import SCENARIOS, OracleSeparator, OracleTagger, camera_path, project  # noqa: E402

from memoryscapes_workers.audio.vocabulary import BY_LABEL  # noqa: E402
from memoryscapes_workers.cli import import_upload, run_isolated  # noqa: E402
from memoryscapes_workers.context import StageContext  # noqa: E402
from memoryscapes_workers.models import Camera  # noqa: E402
from memoryscapes_workers.providers.soundscape import StubProvider  # noqa: E402
from memoryscapes_workers.stagekit import StageError  # noqa: E402
from memoryscapes_workers.stages import localize, separate, soundscape  # noqa: E402
from memoryscapes_workers.storage import LocalStorage, read_json, write_json  # noqa: E402
from memoryscapes_workers.vision.grounding import Detection  # noqa: E402

FPS = 8.0


@dataclasses.dataclass
class Similarity:
    """x_pipeline = scale * R @ x_true + t"""

    scale: float
    R: np.ndarray
    t: np.ndarray
    rms_m: float  # camera position residual after alignment, in pipeline metres

    def apply(self, p) -> np.ndarray:
        return self.scale * self.R @ np.asarray(p, float) + self.t


def fit_similarity(true_c2w: np.ndarray, pipe_c2w: np.ndarray) -> Similarity:
    """Similarity from true to recovered cameras. Rotation comes from the camera
    orientations (robust even when the path is nearly a straight line), then scale and
    offset from the positions by least squares."""
    M = sum(p[:3, :3] @ q[:3, :3].T for p, q in zip(pipe_c2w, true_c2w))
    U, _, Vt = np.linalg.svd(M)
    R = U @ np.diag([1, 1, np.sign(np.linalg.det(U @ Vt))]) @ Vt
    src, dst = true_c2w[:, :3, 3], pipe_c2w[:, :3, 3]
    sc, dc = src - src.mean(0), dst - dst.mean(0)
    rs = sc @ R.T
    denom = float((sc**2).sum())
    scale = float((dc * rs).sum() / denom) if denom > 1e-9 else 1.0
    t = dst.mean(0) - scale * R @ src.mean(0)
    resid = dst - (scale * src @ R.T + t)
    return Similarity(scale, R, t, float(np.sqrt((resid**2).sum(1).mean())))


class ProjectingOracleDetector:
    """Like scenes.OracleDetector, but projects the truth through the *recovered*
    cameras, with positions already moved into the pipeline frame."""

    def __init__(self, sources, sim: Similarity, cameras: list[Camera], storage, conditions, seed: int):
        self.sources = [(s, sim.apply(s.position), s.size_m * sim.scale) for s in sources]
        self.by_bytes = {storage.read_bytes(c.image_key): c for c in cameras if c.image_key}
        self.c = conditions
        self.rng = np.random.default_rng(seed + 200)

    def detect(self, image_bytes, prompt):
        cam = self.by_bytes[image_bytes]
        out = []
        for s, pos, size in self.sources:
            if BY_LABEL[s.label].detect_prompt != prompt or self.rng.random() < self.c.detector_miss_rate:
                continue
            uv, z = project(cam, pos)
            if z <= 0.5:
                continue
            half = cam.K[0, 0] * size / 2 / z
            uv = uv + self.rng.standard_normal(2) * half * self.c.box_jitter
            box = (uv[0] - half, uv[1] - half, uv[0] + half, uv[1] + half)
            if box[2] < 0 or box[3] < 0 or box[0] > cam.width or box[1] > cam.height:
                continue
            box = (max(box[0], 0), max(box[1], 0), min(box[2], cam.width), min(box[3], cam.height))
            out.append(Detection(box, float(self.rng.uniform(0.45, 0.9)), s.label))
        if self.rng.random() < self.c.detector_false_rate:
            u, v = self.rng.uniform(0, cam.width), self.rng.uniform(0, cam.height)
            out.append(Detection((u - 30, v - 30, u + 30, v + 30), float(self.rng.uniform(0.3, 0.6)), "false"))
        return out


def scale_splat(storage, memory_id: str, factor: float) -> None:
    """Scale the trained splat about the origin (positions and gaussian sizes)."""
    from memoryscapes_workers.visual.splat_io import read_ply, write_ply

    path = Path(storage.root) / memory_id / "splat" / "point_cloud.ply"
    cloud = read_ply(path)
    cloud.means = (cloud.means * factor).astype(np.float32)
    cloud.scales = (cloud.scales + np.log(factor)).astype(np.float32)
    write_ply(path, cloud)


def run_scenario(name: str, root: Path, steps: int, max_keyframes: int) -> dict:
    scenario = SCENARIOS[name]
    memory = f"{name}_real"
    storage = LocalStorage(root)
    os.environ.update(STORAGE_BACKEND="local", STORAGE_ROOT=str(storage.root))
    work = root / "_videos"
    work.mkdir(parents=True, exist_ok=True)
    timings: dict[str, float] = {}

    t0 = time.perf_counter()
    video = work / f"{name}.mp4"
    r = render_video(scenario, video, FPS)
    timings["video"] = time.perf_counter() - t0
    import_upload(storage, memory, video)

    config = {
        "ingest": {"max_keyframes": max_keyframes, "min_keyframes": min(40, max_keyframes)},
        "poses": {"use_gpu": False},
        "splat": {"trainer": "torch", "train": {"steps": steps, "preview_steps": [], "render_max_side": 320, "max_image_side": 480}},
    }
    result: dict = {"scenario": name, "memory": memory}
    for stage in ("ingest", "poses", "splat"):
        t0 = time.perf_counter()
        try:
            run_isolated(memory, [stage], config, force=True)
        except StageError as err:
            if stage == "poses" and config["poses"].get("mapper") != "incremental":
                # the global mapper needs focal-length priors; incremental copes without
                print(f"  poses (global) failed, retrying with the incremental mapper: {err.message}")
                config["poses"]["mapper"] = "incremental"
                try:
                    run_isolated(memory, [stage], config, force=True)
                except StageError as err2:
                    err = err2
                else:
                    timings[stage] = time.perf_counter() - t0
                    continue
            timings[stage] = time.perf_counter() - t0
            result.update(failed_stage=stage, error=err.message, hint=err.hint, timings=timings)
            write_json(storage, f"{memory}/debug/truth.json", _truth_json(r, None, scenario))
            return result
        timings[stage] = time.perf_counter() - t0

    ctx = StageContext(memory_id=memory, storage=storage)
    cams = [Camera.from_json(c) for c in read_json(storage, ctx.key("poses", "cameras.json"))]
    dense = camera_path(scenario.conditions.camera, scenario.duration_s, n=int(scenario.duration_s * 100) + 1)
    true_c2w = np.stack([min(dense, key=lambda c, t=cam.t: abs(c.t - t)).c2w for cam in cams])
    sim = fit_similarity(true_c2w, np.stack([c.c2w for c in cams]))

    if scenario.conditions.splat_scale != 1.0:
        scale_splat(storage, memory, scenario.conditions.splat_scale)

    # audio stages on the real ingest + poses, with oracle models
    t0 = time.perf_counter()
    separate.run(ctx, tagger=OracleTagger(r), separator=OracleSeparator(r))
    detector = ProjectingOracleDetector(scenario.sources, sim, cams, storage, scenario.conditions, scenario.seed)
    localized = localize.run(ctx, detector=detector)
    soundscape.run(ctx, provider=StubProvider())
    timings["audio"] = time.perf_counter() - t0

    # score in the pipeline frame
    moved = dataclasses.replace(scenario, sources=[dataclasses.replace(s, position=tuple(sim.apply(s.position).tolist())) for s in scenario.sources])
    r_pipe = dataclasses.replace(r, scenario=moved, cameras=cams)
    rows = evaluate(r_pipe, localized)
    write_json(storage, ctx.key("debug", "truth.json"), _truth_json(r, sim, moved))
    out_dir = root / memory
    write_report(out_dir / "report.md", r_pipe, rows, {"separate": timings["audio"], "localize": 0.0})
    write_map(out_dir / "map.svg", r_pipe, rows)
    report = read_json(storage, ctx.key("splat", "metrics.json")) if storage.exists(ctx.key("splat", "metrics.json")) else {}
    result.update(
        registered=len(cams),
        alignment={"scale": round(sim.scale, 4), "rms_m": round(sim.rms_m, 4)},
        final_psnr=report.get("psnr", [None])[-1] if report.get("psnr") else None,
        verdicts=summary(rows),
        bad=[(row.label, row.verdict) for row in rows if row.verdict in BAD_VERDICTS],
        rows=[(row.label, row.method, round(row.confidence, 2), row.error_m, row.verdict) for row in rows],
        timings={k: round(v, 1) for k, v in timings.items()},
    )
    return result


def _truth_json(r, sim: Similarity | None, scenario) -> dict:
    out = {
        "scenario": f"{r.scenario.name} (real visual pipeline)",
        "description": r.scenario.description + " Rendered to video and run through the real ingest, COLMAP poses and torch splat training; ground truth moved into the recovered frame.",
        "sources": [{"label": s.label, "position": [round(float(v), 3) for v in s.position]} for s in scenario.sources],
        "diffuse": [label for label, _ in r.scenario.diffuse],
        "splat_transform": {"scale": r.scenario.conditions.splat_scale, "offset": list(r.scenario.conditions.splat_offset)},
    }
    if sim is not None:
        out["frame_alignment"] = {"scale": sim.scale, "R": sim.R.tolist(), "t": sim.t.tolist(), "camera_rms_m": sim.rms_m}
    else:
        out["frame_alignment"] = None
        out["note"] = "Poses failed, so the truth is in the scene's own frame."
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("scenarios", nargs="*", default=list(SCENARIOS))
    ap.add_argument("--out", default=str(HERE / "out"))
    ap.add_argument("--steps", type=int, default=300, help="torch splat training steps (CPU: keep it short)")
    ap.add_argument("--max-keyframes", type=int, default=60)
    args = ap.parse_args()
    root = Path(args.out)
    results = []
    for name in args.scenarios:
        print(f"\n== {name}")
        res = run_scenario(name, root, args.steps, args.max_keyframes)
        results.append(res)
        if "failed_stage" in res:
            print(f"  {res['failed_stage']} failed: {res['error']}")
        else:
            print(f"  {res['registered']} cameras, frame scale {res['alignment']['scale']}, camera rms {res['alignment']['rms_m']} m, PSNR {res['final_psnr']}")
            for row in res["rows"]:
                print("   ", row)
        print(f"  timings: {res['timings']}")
    (root / "real_pipeline_summary.json").write_text(json.dumps(results, indent=2, default=str), encoding="utf-8")
    return 1 if any(r.get("bad") for r in results) else 0


if __name__ == "__main__":
    raise SystemExit(main())
