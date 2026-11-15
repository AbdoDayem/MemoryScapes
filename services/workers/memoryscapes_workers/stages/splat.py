"""Gaussian splat training and packaging.

- Train with gsplat (MCMC densification) on an NVIDIA GPU (production), or the same
  loop on a pure-PyTorch rasterizer on Apple Silicon for testing (``trainer="torch"``).
  ``trainer="auto"`` (default) picks whichever the machine supports.
- Clean up for walking through: path floaters, oversized blobs, needles, isolated
  specks; crop generously around the camera trajectory.
- Compress to .spz and produce 3 LOD levels.
- Publish an early low-iteration preview splat so the user sees something sooner.

Reads  {memory_id}/poses/ (undistorted world-frame model + images)
Writes {memory_id}/splat/ (the contract is docs/SPLAT_OUTPUTS.md; the debugger reads these)
    meta.json            {trainer, trainer_version, iterations, sh_degree, units, up, eval_frames}
    metrics.json         training curves: {iteration, loss, psnr, num_gaussians, elapsed_s}
    point_cloud.ply      raw trained splat, standard 3DGS PLY (kept for re-processing)
    renders/<frame_id>.png  the trained splat from each keyframe's camera (trainers that
                         render; not "points")
    lods/lod0.spz ...    cleaned splat, most detailed first
    lods.json            [{file, count, sh_degree, bytes}], plus bounds and cleanup report
    preview.spz          early checkpoint (published before training finishes)
    train_report.json    final trainer summary (device, PSNR, timings)

Choose the trainer with ``config["splat"]["trainer"]`` or ``SPLAT_TRAINER``
(auto | gsplat | torch | points). "auto" is resolved before fingerprinting, so a splat
trained on a Mac is not mistaken for an up-to-date production splat.
"""

from __future__ import annotations

import math
import os

import numpy as np
from pydantic import BaseModel, Field

from memoryscapes_workers.context import StageContext
from memoryscapes_workers.stagekit import StageError, StageIO, run_stage, write_json
from memoryscapes_workers.visual.splat_post import DEFAULT_LODS, CleanupConfig, LodSpec
from memoryscapes_workers.visual.train import TRAINER_PRESETS, TrainConfig, TrainerName

STAGE = "splat"
VERSION = "4"  # 2: gsplat-matching rasterizer; 3: SPLAT_OUTPUTS.md layout; 4: open scenes (sky dome)


class SplatConfig(BaseModel):
    trainer: TrainerName = Field(default_factory=lambda: os.environ.get("SPLAT_TRAINER", "auto"))
    train: TrainConfig = TrainConfig()
    cleanup: CleanupConfig = CleanupConfig()
    lods: list[LodSpec] = Field(default_factory=lambda: list(DEFAULT_LODS))
    publish_preview: bool = True


def resolve_config(raw: dict) -> SplatConfig:
    """Resolve trainer="auto" and apply that trainer's presets under explicit settings."""
    from memoryscapes_workers.visual.train import detect_trainer

    cfg = SplatConfig(**raw)
    trainer = detect_trainer() if cfg.trainer == "auto" else cfg.trainer
    train = {**TRAINER_PRESETS.get(trainer, {}), **dict(raw.get("train", {}))}
    return SplatConfig(**{**dict(raw), "trainer": trainer, "train": train})


def _expected_dome(scene, train_cfg) -> float | None:
    """The dome radius the trainer will use (previews are cleaned before it reports)."""
    far = float(np.percentile(np.linalg.norm(scene.points - scene.cam_centers.mean(axis=0), axis=1), 95))
    unbounded = train_cfg.unbounded == "on" or far > train_cfg.unbounded_ratio * scene.scene_scale
    return max(3.0 * far, 50.0) if unbounded and train_cfg.sky_dome_points > 0 else None


def _trainer_version(trainer: str) -> str:
    from importlib.metadata import PackageNotFoundError, version

    lib = {"gsplat": "gsplat", "torch": "torch"}.get(trainer)
    try:
        dep = f"{lib} {version(lib)}" if lib else "no training"
    except PackageNotFoundError:
        dep = f"{lib} (version unknown)"
    return f"memoryscapes splat v{VERSION}, {trainer} ({dep})"


def run(ctx: StageContext) -> dict:
    cfg = resolve_config(dict(ctx.stage_config(STAGE)))
    return run_stage(ctx, STAGE, version=VERSION, config=cfg, upstream=["poses"],
                     body=lambda io: _splat(io, cfg))


def _splat(io: StageIO, cfg: SplatConfig) -> dict:
    from memoryscapes_workers.visual import splat_io, splat_post
    from memoryscapes_workers.visual.train import get_trainer, load_scene

    scene = load_scene(io.input_dir("poses"), stereo_views=cfg.train.stereo_views)
    if len(scene.points) < 100:
        raise StageError(STAGE, f"Only {len(scene.points)} SfM points; too few to initialize.")
    out = io.out_dir
    trainer = get_trainer(cfg.trainer)

    def on_checkpoint(step: int, cloud: splat_io.GaussianCloud) -> None:
        if not cfg.publish_preview:
            return
        preview, _ = splat_post.cleanup(cloud, scene.cam_centers, cleanup_cfg(), scene.points)
        splat_io.write_spz(out / "preview.spz", splat_post.make_lod(preview, cfg.lods[-1]))
        io.publish_early("preview.spz")
        io.progress(0.1 + 0.8 * step / max(cfg.train.steps, 1), f"preview at step {step}")

    def on_render(frame_id: str, rgb: np.ndarray) -> None:
        from memoryscapes_workers.visual.video import imwrite_rgb

        (out / "renders").mkdir(exist_ok=True)
        imwrite_rgb(out / "renders" / f"{frame_id}.png", rgb)

    def cleanup_cfg(metrics: dict | None = None):
        """Keep a sky dome: crop beyond it (an open scene's dome is far past the SfM points)."""
        dome = (metrics or {}).get("sky_dome_radius_m") or (
            None if cfg.train.unbounded == "off" else _expected_dome(scene, cfg.train))
        if not dome or cfg.cleanup.crop_radius_m is not None:
            return cfg.cleanup
        return cfg.cleanup.model_copy(update={"crop_radius_m": 1.3 * dome})

    result = trainer.train(
        scene, cfg.train, on_checkpoint=on_checkpoint,
        on_progress=lambda f, msg: io.progress(0.05 + 0.85 * f, msg), on_render=on_render,
    )
    raw = result.cloud
    splat_io.write_ply(out / "point_cloud.ply", raw)

    cleaned, report = splat_post.cleanup(raw, scene.cam_centers, cleanup_cfg(result.metrics), scene.points)
    if len(cleaned) == 0:
        raise StageError(STAGE, "Cleanup removed every gaussian; check the poses stage output.")
    io.progress(0.95, f"cleaned {len(raw)} -> {len(cleaned)} gaussians")

    lods = []
    (out / "lods").mkdir()
    for i, spec in enumerate(cfg.lods):
        lod = splat_post.make_lod(cleaned, spec)
        name = f"lods/lod{i}.spz"
        size = splat_io.write_spz(out / name, lod)
        lods.append({"file": name, "count": len(lod), "sh_degree": lod.sh_degree, "bytes": size})
    if not (out / "preview.spz").exists() and cfg.publish_preview:
        # Trainers without checkpoints (e.g. "points"): the coarsest LOD doubles as preview.
        (out / "preview.spz").write_bytes((out / lods[-1]["file"]).read_bytes())

    bounds = splat_post.bounds(cleaned)
    write_json(out / "lods.json", {
        "order": "most detailed first",
        "format": "spz v3",
        "coord_frame": {"units": "meters", "up": "+y"},
        "lods": lods,
        "bounds": bounds,
        "cleanup": report.to_dict(),
    })
    write_json(out / "train_report.json", result.metrics)
    write_json(out / "meta.json", {
        "trainer": cfg.trainer,
        "trainer_version": _trainer_version(cfg.trainer),
        "iterations": cfg.train.steps if cfg.trainer != "points" else 0,
        "sh_degree": raw.sh_degree,
        "units": "meters",
        "up": "+y",
        "eval_frames": result.eval_frames,
    })
    # Non-finite values (a diverged run) become null: still valid JSON, and the debugger
    # flags them.
    write_json(out / "metrics.json", {
        k: [v if isinstance(v, int) or math.isfinite(v) else None for v in vals]
        for k, vals in result.curves.items()
    })
    return {
        "trainer": cfg.trainer,
        "gaussians_raw": len(raw),
        "gaussians_clean": len(cleaned),
        "lods": [{k: v for k, v in lod.items() if k != "file"} for lod in lods],
        "bounds": bounds,
        "metrics": {k: v for k, v in result.metrics.items() if isinstance(v, (int, float, str))},
        "camera_count": int(len(scene.w2c)),
        "_raw_mb": round((out / "point_cloud.ply").stat().st_size / 1e6, 2),
        "_scene_scale": round(float(np.asarray(scene.scene_scale)), 3),
    }
