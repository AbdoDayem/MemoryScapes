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
    data = SceneData(paths, np.repeat(K[None], 10, axis=0), np.array(w2cs), [(W, H)] * 10,
                     points.astype(np.float32), colors.astype(np.float32))
    data._gt, data._dir = gt_t, out  # for tests that need the true scene (e.g. depth maps)
    return data


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


def test_open_scene_adds_a_sky_dome_and_keeps_far_gaussians(scene):
    """Outdoor footage reaches far past the camera path; the sky has no SfM points."""
    from memoryscapes_workers.visual.train import _load_images, sky_dome

    images, Ks = _load_images(scene, W)
    pts, cols, radius = sky_dome(scene, images, Ks, n=500, far=20.0)
    center = scene.cam_centers.mean(axis=0)
    dist = np.linalg.norm(pts - center, axis=1)
    assert radius == 60.0 and np.allclose(dist, 60.0, atol=1e-3)
    elev = np.degrees(np.arcsin((pts[:, 1] - center[1]) / dist))  # world +y is up
    assert elev.max() > 85 and -16 < elev.min() < -14  # overhead down to 15 below the horizon
    assert cols.min() >= 0 and cols.max() <= 1

    bounded = GaussianTrainer("torch", device="cpu").train(scene, small_config(steps=5))
    assert bounded.metrics["unbounded"] is False and bounded.metrics["sky_dome_radius_m"] is None
    opened = GaussianTrainer("torch", device="cpu").train(
        scene, small_config(steps=5, unbounded="on", sky_dome_points=300))
    assert opened.metrics["unbounded"] is True and opened.metrics["sky_dome_radius_m"] >= 50
    assert opened.metrics["gaussians"] >= bounded.metrics["gaussians"] + 250


def test_stereo_depth_supervision_runs_and_scores_geometry(scene):
    """Depth maps like poses/stereo_depth (uint16 inverse depth): the left-eye images get
    the depth loss, and the result reports the rendered depth's error against the points."""
    import dataclasses

    scale = 10_000.0
    depth_paths = []
    for i, w2c in enumerate(scene.w2c):
        out, alpha, _ = rt.rasterization(**scene._gt, viewmats=torch.tensor(w2c, dtype=torch.float32)[None],
                                         Ks=torch.tensor(scene.Ks[i], dtype=torch.float32)[None],
                                         width=W, height=H, render_mode="RGB+ED")
        inv = np.where(alpha[0, ..., 0].numpy() > 0.5, 1.0 / np.maximum(out[0, ..., 3].numpy(), 1e-3), 0.0)
        path = scene._dir / f"depth_{i:03d}.png"
        cv2.imwrite(str(path), np.clip(inv * scale, 0, 65535).astype(np.uint16))
        depth_paths.append(path)
    with_depth = dataclasses.replace(scene, depth_paths=depth_paths, depth_scale=scale)
    supervised = GaussianTrainer("torch", device="cpu").train(with_depth, small_config(depth_lambda=1.0, holdout_every=5))
    plain = GaussianTrainer("torch", device="cpu").train(with_depth, small_config(depth_lambda=0.0, holdout_every=5))
    assert supervised.metrics["depth_supervised_images"] == 8 and plain.metrics["depth_supervised_images"] == 0
    assert supervised.metrics["depth_error_vs_sfm"] is not None
    assert supervised.metrics["depth_error_vs_sfm"] <= plain.metrics["depth_error_vs_sfm"] * 1.05


def test_exposure_compensation_absorbs_per_frame_exposure(scene):
    """Frames exposed differently, like a phone's auto exposure and white balance at night.
    With exposure_comp each frame's learned gain follows the camera's, and the splat learns
    one consistent scene, so held-out views score better."""
    import dataclasses

    rng = np.random.default_rng(1)
    gains = [1.0, 0.55, 0.8, 0.6, 0.9, 1.0, 0.65, 0.85, 0.7, 0.95]  # frames 0 and 5 are held out
    paths = []
    for i, path in enumerate(scene.image_paths):
        tint = np.ones(3) if i % 5 == 0 else rng.uniform(0.85, 1.0, 3)
        img = cv2.imread(str(path)).astype(np.float32) * gains[i] * tint
        out = scene._dir / f"exposed_{i:03d}.png"
        cv2.imwrite(str(out), np.clip(img, 0, 255).round().astype(np.uint8))
        paths.append(out)
    exposed = dataclasses.replace(scene, image_paths=paths)
    # No densification: this is about colour, and it keeps the runs to seconds.
    cfg = dict(holdout_every=5, steps=150, refine_start_iter=100_000)
    on = GaussianTrainer("torch", device="cpu").train(
        exposed, small_config(exposure_comp=True, exposure_lr=2e-2, **cfg)).metrics
    off = GaussianTrainer("torch", device="cpu").train(exposed, small_config(**cfg)).metrics
    learned = on["exposure_gains"]
    assert sorted(learned) == [p.stem for i, p in enumerate(paths) if i % 5]  # training frames only
    applied = [gains[int(k.rsplit("_", 1)[1])] for k in sorted(learned)]
    assert np.corrcoef([learned[k] for k in sorted(learned)], applied)[0, 1] > 0.9, learned
    assert on["psnr"] > off["psnr"] + 0.5, (on["psnr"], off["psnr"])  # measured +0.9 dB
    assert on["psnr_cc"] > off["psnr_cc"] + 0.1, (on["psnr_cc"], off["psnr_cc"])  # measured +0.3 dB
    assert off["exposure_gains"] is None and off["exposure_comp"] is False


def test_a_collapsed_splat_says_what_went_wrong():
    """With every gaussian transparent, gsplat's MCMC relocation fails with a bare "CUDA
    error: invalid configuration argument" (an empty sample). The trainer names the cause."""
    from memoryscapes_workers.visual.train import SplatCollapsed, _check_not_collapsed

    alive = {"opacities": torch.tensor([-8.0, 2.0])}  # sigmoid: ~0.0003 and ~0.88
    _check_not_collapsed(alive, 0.005, 600)
    with pytest.raises(SplatCollapsed, match="poses"):
        _check_not_collapsed({"opacities": torch.full((5,), -8.0)}, 0.005, 600)
