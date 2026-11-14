"""Gaussian splat training and packaging.

- Train with gsplat (MCMC densification) on an NVIDIA GPU (production), or the same
  loop on a pure-PyTorch rasterizer on Apple Silicon for testing (``trainer="torch"``).
  ``trainer="auto"`` (default) picks whichever the machine supports.
- Clean up for walking through: path floaters, oversized blobs, needles, isolated
  specks; crop generously around the camera trajectory.
- Compress to .spz and produce 3 LOD levels.
- Publish an early low-iteration preview splat so the user sees something sooner.

Reads  {memory_id}/poses/ (undistorted world-frame model + images)
Writes {memory_id}/splat/
    point_cloud.ply      raw trained splat (full precision, kept for re-processing)
    preview.spz          early checkpoint (published before training finishes)
    lod0.spz ...         cleaned splat, most detailed first
    lods.json            [{file, count, sh_degree, bytes}], plus bounds and cleanup report
    train_report.json    trainer metrics

Choose the trainer with ``config["splat"]["trainer"]`` or ``SPLAT_TRAINER``
(auto | gsplat | torch | points). "auto" is resolved before fingerprinting, so a splat
trained on a Mac is not mistaken for an up-to-date production splat.
"""

from __future__ import annotations

import os
import numpy as np
from pydantic import BaseModel, Field

from memoryscapes_workers.context import StageContext
from memoryscapes_workers.stagekit import StageError, StageIO, run_stage, write_json
from memoryscapes_workers.visual.splat_post import DEFAULT_LODS, CleanupConfig, LodSpec
from memoryscapes_workers.visual.train import TRAINER_PRESETS, TrainConfig, TrainerName

STAGE = "splat"
VERSION = "1"


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


def run(ctx: StageContext) -> dict:
    cfg = resolve_config(dict(ctx.stage_config(STAGE)))
    return run_stage(ctx, STAGE, version=VERSION, config=cfg, upstream=["poses"],
                     body=lambda io: _splat(io, cfg))


def _splat(io: StageIO, cfg: SplatConfig) -> dict:
    from memoryscapes_workers.visual import splat_io, splat_post
    from memoryscapes_workers.visual.train import get_trainer, load_scene

    scene = load_scene(io.input_dir("poses"))
    if len(scene.points) < 100:
        raise StageError(STAGE, f"Only {len(scene.points)} SfM points; too few to initialize.")
    out = io.out_dir
    trainer = get_trainer(cfg.trainer)

    def on_checkpoint(step: int, cloud: splat_io.GaussianCloud) -> None:
        if not cfg.publish_preview:
            return
        preview, _ = splat_post.cleanup(cloud, scene.cam_centers, cfg.cleanup, scene.points)
        splat_io.write_spz(out / "preview.spz", splat_post.make_lod(preview, cfg.lods[-1]))
        io.publish_early("preview.spz")
        io.progress(0.1 + 0.8 * step / max(cfg.train.steps, 1), f"preview at step {step}")

    result = trainer.train(
        scene, cfg.train, on_checkpoint=on_checkpoint,
        on_progress=lambda f, msg: io.progress(0.05 + 0.85 * f, msg),
    )
    raw = result.cloud
    splat_io.write_ply(out / "point_cloud.ply", raw)

    cleaned, report = splat_post.cleanup(raw, scene.cam_centers, cfg.cleanup, scene.points)
    if len(cleaned) == 0:
        raise StageError(STAGE, "Cleanup removed every gaussian; check the poses stage output.")
    io.progress(0.95, f"cleaned {len(raw)} -> {len(cleaned)} gaussians")

    lods = []
    for i, spec in enumerate(cfg.lods):
        lod = splat_post.make_lod(cleaned, spec)
        name = f"lod{i}.spz"
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
