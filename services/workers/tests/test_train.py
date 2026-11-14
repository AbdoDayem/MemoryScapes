"""Splat training loop on a tiny synthetic scene (CPU, a few seconds).

The torch backend runs the whole loop: loss, densification, opacity reset, checkpoints,
evaluation. The gsplat backend needs CUDA, so here its rasterizer is swapped for the torch
one while every call is checked against gsplat's real ``rasterization`` signature: that
catches API drift (renamed or removed arguments) without a GPU.
"""

import inspect

import numpy as np
import pytest

torch = pytest.importorskip("torch")
cv2 = pytest.importorskip("cv2")
gsplat = pytest.importorskip("gsplat")

from memoryscapes_workers.visual import rasterize_torch as rt  # noqa: E402
from memoryscapes_workers.visual.train import (  # noqa: E402
    TRAINER_PRESETS,
    GaussianTrainer,
    SceneData,
    TrainConfig,
    get_trainer,
)

W, H = 64, 48


def look_at(eye, target):
    """OpenCV world-to-camera (x right, y down, z forward) with world +y down."""
    eye, target = np.asarray(eye, float), np.asarray(target, float)
    fwd = target - eye
    fwd /= np.linalg.norm(fwd)
    right = np.cross(fwd, [0.0, -1.0, 0.0])
    right /= np.linalg.norm(right)
    down = np.cross(fwd, right)
    R = np.stack([right, down, fwd])
    m = np.eye(4)
    m[:3, :3], m[:3, 3] = R, -R @ eye
    return m


@pytest.fixture(scope="module")
def scene(tmp_path_factory) -> SceneData:
    """Ground-truth gaussians rendered from 10 cameras on an arc; SfM-like noisy points.

    An object of small gaussians in front of a backdrop wall that fills every view, like
    a real capture (no empty background to learn).
    """
    rng = np.random.default_rng(0)
    n_obj, n_wall = 400, 600
    n = n_obj + n_wall
    wall = np.stack([rng.uniform(-3, 3, n_wall), rng.uniform(-2, 2, n_wall),
                     rng.uniform(1.0, 1.1, n_wall)], -1)
    gt = {
        "means": np.concatenate([rng.uniform([-0.6, -0.4, -0.3], [0.6, 0.4, 0.3], (n_obj, 3)), wall]),
        "quats": rng.normal(size=(n, 4)),
        "scales": np.concatenate([rng.uniform(0.02, 0.07, (n_obj, 3)),
                                  rng.uniform(0.12, 0.2, (n_wall, 3))]),
        "opacities": np.concatenate([rng.uniform(0.6, 0.95, n_obj), np.full(n_wall, 0.99)]),
        "colors": rng.uniform(0, 1, (n, 3)),
    }
    gt_t = {k: torch.tensor(v, dtype=torch.float32) for k, v in gt.items()}
    K = np.array([[55.0, 0, W / 2], [0, 55.0, H / 2], [0, 0, 1]])
    out = tmp_path_factory.mktemp("train_scene")
    paths, w2cs = [], []
    for i, ang in enumerate(np.linspace(-0.6, 0.6, 10)):
        w2c = look_at([2.2 * np.sin(ang), -0.2, -2.2 * np.cos(ang)], [0, 0, 0])
        rgb, _, _ = rt.rasterization(**{k: gt_t[k] for k in gt_t},
                                     viewmats=torch.tensor(w2c, dtype=torch.float32)[None],
                                     Ks=torch.tensor(K, dtype=torch.float32)[None], width=W, height=H,
                                     backgrounds=torch.ones(1, 3))
        img = (rgb[0].clamp(0, 1).numpy() * 255).round().astype(np.uint8)
        path = out / f"frame_{i:03d}.png"
        cv2.imwrite(str(path), cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
        paths.append(path)
        w2cs.append(w2c)
    keep = rng.random(n) < 0.6
    points = gt["means"][keep] + rng.normal(0, 0.02, (int(keep.sum()), 3))
    colors = np.clip(gt["colors"][keep] + rng.normal(0, 0.1, (int(keep.sum()), 3)), 0, 1)
    return SceneData(paths, np.repeat(K[None], 10, axis=0), np.array(w2cs), [(W, H)] * 10,
                     points.astype(np.float32), colors.astype(np.float32))


def small_config(**over) -> TrainConfig:
    cfg = dict(steps=120, max_image_side=W, sh_degree=1, sh_degree_interval=40, strategy="default",
               refine_start_iter=15, refine_every=15, reset_every=10_000, preview_steps=[60])
    return TrainConfig(**{**cfg, **over})


def test_torch_trainer_learns_and_densifies(scene):
    checkpoints, progress = [], []
    baseline = GaussianTrainer("torch", device="cpu").train(scene, small_config(steps=1))
    result = GaussianTrainer("torch", device="cpu").train(
        scene, small_config(), on_checkpoint=lambda step, cloud: checkpoints.append((step, len(cloud))),
        on_progress=lambda f, msg: progress.append(f))
    m = result.metrics
    assert m["trainer"] == "torch" and m["strategy"] == "default" and m["device"] == "cpu"
    assert m["psnr"] > baseline.metrics["psnr"] + 2.5, (baseline.metrics["psnr"], m["psnr"])
    assert m["gaussians"] != len(scene.points)  # densification grew or pruned
    assert [s for s, _ in checkpoints] == [60]
    assert progress and progress == sorted(progress)
    cloud = result.cloud
    assert cloud.shN.shape == (len(cloud), 3, 3)  # sh_degree 1 -> 3 extra coefficients
    assert np.isfinite(cloud.means).all() and np.isfinite(cloud.opacities).all()


def test_curves_eval_frames_and_renders(scene):
    """What splat/metrics.json, meta.json and renders/ are built from (docs/SPLAT_OUTPUTS.md)."""
    renders = {}
    result = GaussianTrainer("torch", device="cpu").train(
        scene, small_config(steps=30, render_max_side=32),
        on_render=lambda frame_id, rgb: renders.setdefault(frame_id, rgb))
    curves = result.curves
    assert set(curves) == {"iteration", "loss", "psnr", "num_gaussians", "elapsed_s"}
    assert len({len(v) for v in curves.values()}) == 1 and len(curves["iteration"]) >= 2
    assert curves["iteration"] == sorted(curves["iteration"]) and curves["iteration"][-1] == 30
    assert all(np.isfinite(v).all() for v in curves.values())
    assert curves["psnr"][-1] == pytest.approx(result.metrics["psnr"], abs=0.01)  # last = eval
    ids = [p.stem for p in scene.image_paths]
    assert result.eval_frames and set(result.eval_frames) <= set(ids)
    assert list(renders) == ids
    assert all(r.dtype == np.uint8 and r.shape == (24, 32, 3) for r in renders.values())


def test_opacity_reset_runs(scene, monkeypatch):
    """gsplat 1.5.3 never resets opacity itself; our workaround must fire every reset_every."""
    from memoryscapes_workers.visual import train

    calls = []
    real = train._maybe_reset_opacity

    def spy(strategy, params, optimizers, state, step):
        before = torch.sigmoid(params["opacities"]).max().item()
        real(strategy, params, optimizers, state, step)
        after = torch.sigmoid(params["opacities"]).max().item()
        if after < before:
            calls.append((step, after))

    monkeypatch.setattr(train, "_maybe_reset_opacity", spy)
    GaussianTrainer("torch", device="cpu").train(scene, small_config(steps=100, reset_every=30))
    # Refinement stops at steps // 2 = 50, so only the reset at step 30 happens.
    assert [s for s, _ in calls] == [30]
    assert calls[0][1] <= 0.01 + 1e-6  # reset to 2 * prune_opa


def test_gsplat_backend_calls_match_gsplat_api(scene, monkeypatch):
    """Run the gsplat code path with the torch rasterizer, validating each call's
    arguments against gsplat.rasterization's signature."""
    sig = inspect.signature(gsplat.rasterization)
    seen = []

    def checked(**kwargs):
        sig.bind(**kwargs)  # TypeError on unknown or missing arguments
        seen.append(set(kwargs))
        return rt.rasterization(**kwargs)

    monkeypatch.setattr(GaussianTrainer, "_setup", lambda self: (torch.device("cpu"), checked))
    result = GaussianTrainer("cuda").train(scene, small_config(steps=20, antialiased=True))
    assert seen and result.metrics["trainer"] == "gsplat"
    # The MCMC strategy is the gsplat default; its post-backward hook takes the means lr.
    from gsplat.strategy import DefaultStrategy, MCMCStrategy

    assert "lr" in inspect.signature(MCMCStrategy.step_post_backward).parameters
    assert "packed" in inspect.signature(DefaultStrategy.step_post_backward).parameters


def test_presets_and_trainer_names():
    assert get_trainer("torch").name == "torch"
    assert get_trainer("gsplat").name == "gsplat"
    assert get_trainer("points").name == "points"
    with pytest.raises(ValueError):
        get_trainer("nope")
    assert set(TRAINER_PRESETS["torch"]) <= set(TrainConfig.model_fields)
