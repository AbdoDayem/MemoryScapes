"""Splat trainers behind one interface.

One 3DGS training loop (``GaussianTrainer``) with two rasterizer backends:

- ``trainer="gsplat"``: gsplat's CUDA kernels on an NVIDIA GPU. Production. MCMC
  densification by default: it caps the gaussian count (predictable web file sizes)
  and tends to leave fewer floaters than the classic heuristics.
- ``trainer="torch"``: the pure-PyTorch rasterizer in ``rasterize_torch`` on Apple
  Silicon (MPS) or CPU. Same loss, init, learning rates and gsplat's DefaultStrategy
  densification; runs at reduced resolution and steps for testing on a Mac.

Plus ``trainer="points"`` (no optimization: SfM points as gaussians, seconds, anywhere)
and ``trainer="auto"``, which picks gsplat when CUDA is available, else torch on MPS,
else points. Per-trainer presets (steps, resolution, strategy) apply unless the stage
config sets them explicitly.

Training happens directly in the metric world frame from the poses stage (no
normalization), so the trained splat lines up with the trajectory and audio sources.
"""

from __future__ import annotations

import logging
import math
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal, Protocol

import numpy as np
from pydantic import BaseModel

from memoryscapes_workers.visual.splat_io import SH_C0, SH_DIM, GaussianCloud, logit
from memoryscapes_workers.visual.splat_post import knn_mean_dist_np

log = logging.getLogger("memoryscapes.train")

TrainerName = Literal["auto", "gsplat", "torch", "points"]


class TrainConfig(BaseModel):
    steps: int = 30_000
    strategy: Literal["mcmc", "default"] = "mcmc"  # torch backend always uses "default"
    cap_max: int = 1_500_000  # MCMC gaussian budget
    sh_degree: int = 3
    sh_degree_interval: int = 1000
    ssim_lambda: float = 0.2
    max_image_side: int = 1600
    # Random background colors penalize see-through regions, which show up as holes
    # when you walk around.
    random_background: bool = True
    antialiased: bool = False  # Mip-Splatting style; the viewer must support the flag
    preview_steps: list[int] = [7_000]
    # 0: train on every frame and report PSNR on training frames.
    # N > 0: hold out every Nth frame and report PSNR on those (honest novel-view score).
    holdout_every: int = 0
    seed: int = 0
    # Learning rates (gsplat simple_trainer defaults); means lr is scaled by scene size.
    means_lr: float = 1.6e-4
    scales_lr: float = 5e-3
    quats_lr: float = 1e-3
    opacities_lr: float = 5e-2
    sh0_lr: float = 2.5e-3
    shN_lr: float = 2.5e-3 / 20
    # MCMC regularizers
    opacity_reg: float = 0.01
    scale_reg: float = 0.01
    # Densification schedule (gsplat defaults). reset_every=0: every steps/3, 500..3000.
    refine_start_iter: int = 500
    refine_every: int = 100
    reset_every: int = 0
    # Renders of the trained splat from every keyframe (splat/renders/), long side <= this.
    # 0 disables. Used by the debugger for side-by-side checks against the frames.
    render_max_side: int = 800
    # Open (outdoor) scenes reach far beyond the camera path: buildings across a square,
    # the sky. "auto": when the SfM points reach unbounded_ratio x the camera spread. Then
    # a sky dome is added (the sky has no SfM points) and gaussians aren't pruned for
    # their 3D size, which is relative to the camera spread and would delete everything
    # distant (cleanup still drops oversized blobs, with distance-aware limits).
    unbounded: Literal["auto", "on", "off"] = "auto"
    unbounded_ratio: float = 5.0
    sky_dome_points: int = 4000


# Overrides applied per trainer unless the stage config sets the key explicitly.
TRAINER_PRESETS: dict[str, dict] = {
    "gsplat": {},  # TrainConfig defaults are the production settings
    "torch": {"steps": 5_000, "max_image_side": 800, "strategy": "default",
              "preview_steps": [1_500]},
    "points": {},
}


def detect_trainer() -> str:
    """gsplat on CUDA, else the torch backend on Apple Silicon, else points."""
    try:
        import torch
    except ImportError:
        return "points"
    if torch.cuda.is_available():
        try:
            import gsplat  # noqa: F401

            return "gsplat"
        except ImportError:
            log.warning("CUDA is available but gsplat is not installed; using the torch backend")
            return "torch"
    if torch.backends.mps.is_available():
        return "torch"
    return "points"


@dataclass
class SceneData:
    image_paths: list[Path]
    Ks: np.ndarray  # [C, 3, 3]
    w2c: np.ndarray  # [C, 4, 4] OpenCV
    sizes: list[tuple[int, int]]  # (width, height)
    points: np.ndarray  # [N, 3]
    colors: np.ndarray  # [N, 3] 0..1

    @property
    def cam_centers(self) -> np.ndarray:
        return np.array([np.linalg.inv(m)[:3, 3] for m in self.w2c])

    @property
    def frame_ids(self) -> list[str]:
        """Same ids as poses/cameras.json (the image file stem)."""
        return [p.stem for p in self.image_paths]

    @property
    def scene_scale(self) -> float:
        c = self.cam_centers
        return float(np.linalg.norm(c - c.mean(axis=0), axis=1).max()) * 1.1


def load_scene(poses_dir: Path) -> SceneData:
    """Load the undistorted, world-frame COLMAP model written by the poses stage.

    Uses the numpy reader, not pycolmap: on macOS pycolmap and torch bundle conflicting
    OpenMP runtimes and cannot share a process.
    """
    from memoryscapes_workers.visual.colmap_io import read_model

    model = read_model(poses_dir / "sparse" / "0")
    images = sorted(model.images.values(), key=lambda im: im.name)
    paths, Ks, w2cs, sizes = [], [], [], []
    for im in images:
        cam = model.cameras[im.camera_id]
        paths.append(poses_dir / "images" / im.name)
        Ks.append(cam.K())
        w2cs.append(im.w2c())
        sizes.append((cam.width, cam.height))
    return SceneData(paths, np.array(Ks), np.array(w2cs), sizes,
                     model.points.astype(np.float32), model.colors.astype(np.float32) / 255.0)


CURVE_KEYS = ("iteration", "loss", "psnr", "num_gaussians", "elapsed_s")


@dataclass
class TrainResult:
    cloud: GaussianCloud
    metrics: dict = field(default_factory=dict)  # final summary
    # Training curves as parallel arrays (CURVE_KEYS). psnr is on the training view of
    # each logged step; the last entry is the evaluation PSNR over eval_frames.
    curves: dict[str, list[float]] = field(default_factory=lambda: {k: [] for k in CURVE_KEYS})
    eval_frames: list[str] = field(default_factory=list)


CheckpointFn = Callable[[int, GaussianCloud], None]
ProgressFn = Callable[[float, str], None]
RenderFn = Callable[[str, np.ndarray], None]  # (frame_id, uint8 HxWx3 RGB)


class Trainer(Protocol):
    name: str

    def train(self, scene: SceneData, cfg: TrainConfig, on_checkpoint: CheckpointFn | None = None,
              on_progress: ProgressFn | None = None,
              on_render: RenderFn | None = None) -> TrainResult: ...


def init_from_points(points: np.ndarray, colors: np.ndarray, sh_degree: int, opacity: float,
                     scale_mult: float, knn: np.ndarray) -> GaussianCloud:
    n = len(points)
    return GaussianCloud(
        means=points.astype(np.float32),
        scales=np.repeat(np.log(np.maximum(knn * scale_mult, 1e-4))[:, None], 3, axis=1).astype(np.float32),
        quats=np.tile(np.array([1, 0, 0, 0], np.float32), (n, 1)),
        opacities=np.full(n, logit(np.array(opacity)), np.float32),
        sh0=((colors - 0.5) / SH_C0).astype(np.float32),
        shN=np.zeros((n, SH_DIM[sh_degree], 3), np.float32),
    )


class PointsTrainer:
    name = "points"

    def train(self, scene, cfg, on_checkpoint=None, on_progress=None, on_render=None) -> TrainResult:
        knn = knn_mean_dist_np(scene.points)
        cloud = init_from_points(scene.points, scene.colors, cfg.sh_degree, 0.8, 1.0, knn)
        return TrainResult(cloud, {"trainer": self.name, "note": "no optimization"})


class GaussianTrainer:
    """3DGS training loop; mirrors gsplat's examples/simple_trainer.py.

    backend="cuda": gsplat CUDA rasterizer. backend="torch": pure-PyTorch rasterizer on
    ``device`` (default: mps, else cuda, else cpu).
    """

    def __init__(self, backend: Literal["cuda", "torch"], device: str | None = None) -> None:
        self.backend = backend
        self.name = "gsplat" if backend == "cuda" else "torch"
        self._device = device

    def _setup(self):
        import torch

        if self.backend == "cuda":
            if not torch.cuda.is_available():
                raise RuntimeError("trainer='gsplat' needs an NVIDIA GPU with CUDA; "
                                   "use trainer='torch' (Mac) or 'auto'")
            from gsplat import rasterization

            return torch.device(self._device or "cuda"), rasterization
        from memoryscapes_workers.visual.rasterize_torch import rasterization

        if self._device:
            dev = self._device
        elif torch.backends.mps.is_available():
            dev = "mps"
        else:
            dev = "cuda" if torch.cuda.is_available() else "cpu"
        return torch.device(dev), rasterization

    def train(self, scene, cfg, on_checkpoint=None, on_progress=None, on_render=None) -> TrainResult:
        import torch
        import torch.nn.functional as F
        from gsplat.strategy import DefaultStrategy, MCMCStrategy

        dev, rasterize = self._setup()
        torch.manual_seed(cfg.seed)
        rng = np.random.default_rng(cfg.seed)
        t0 = time.time()
        mcmc = cfg.strategy == "mcmc"
        if mcmc and self.backend == "torch":
            log.info("MCMC relocation needs gsplat's CUDA kernels; torch backend uses 'default'")
            mcmc = False

        images, Ks = _load_images(scene, cfg.max_image_side)
        all_idx = np.arange(len(images))
        if cfg.holdout_every > 0:
            test_idx = all_idx[all_idx % cfg.holdout_every == 0]
            train_idx = all_idx[all_idx % cfg.holdout_every != 0]
        else:
            train_idx, test_idx = all_idx, all_idx[:: 8]
        gts = [torch.from_numpy(im).to(dev) for im in images]  # uint8 HxWx3
        Ks_t = torch.tensor(Ks, dtype=torch.float32, device=dev)
        viewmats = torch.tensor(scene.w2c, dtype=torch.float32, device=dev)
        scene_scale = scene.scene_scale

        far = float(np.percentile(np.linalg.norm(scene.points - scene.cam_centers.mean(axis=0), axis=1), 95))
        unbounded = cfg.unbounded == "on" or (cfg.unbounded == "auto" and far > cfg.unbounded_ratio * scene_scale)
        init_pts, init_cols, dome_radius = scene.points, scene.colors, None
        if unbounded and cfg.sky_dome_points > 0:
            dome, dome_cols, dome_radius = sky_dome(scene, images, Ks, cfg.sky_dome_points, far)
            init_pts = np.concatenate([scene.points, dome])
            init_cols = np.concatenate([scene.colors, dome_cols])
            log.info("open scene (points reach %.0f m, cameras span %.1f m): sky dome of %d at %.0f m",
                     far, scene_scale, len(dome), dome_radius)
        pts = torch.tensor(init_pts, dtype=torch.float32, device=dev)
        knn = _knn_mean_dist_torch(pts, 3).cpu().numpy()
        init = init_from_points(init_pts, init_cols, cfg.sh_degree,
                                opacity=0.5 if mcmc else 0.1, scale_mult=0.1 if mcmc else 1.0, knn=knn)
        n = len(init)

        def param(a):
            return torch.nn.Parameter(torch.tensor(a, dtype=torch.float32, device=dev))

        params = torch.nn.ParameterDict({
            "means": param(init.means),
            "scales": param(init.scales),
            "quats": torch.nn.Parameter(torch.rand((n, 4), device=dev)),
            "opacities": param(init.opacities),
            "sh0": param(init.sh0[:, None, :]),
            "shN": torch.nn.Parameter(torch.zeros((n, SH_DIM[cfg.sh_degree], 3), device=dev)),
        })
        lrs = {"means": cfg.means_lr * scene_scale, "scales": cfg.scales_lr, "quats": cfg.quats_lr,
               "opacities": cfg.opacities_lr, "sh0": cfg.sh0_lr, "shN": cfg.shN_lr}
        optimizers = {k: torch.optim.Adam([{"params": params[k], "lr": lr, "name": k}], eps=1e-15)
                      for k, lr in lrs.items()}
        sched = torch.optim.lr_scheduler.ExponentialLR(optimizers["means"], gamma=0.01 ** (1.0 / cfg.steps))

        if mcmc:
            strategy = MCMCStrategy(cap_max=cfg.cap_max, refine_start_iter=cfg.refine_start_iter,
                                    refine_stop_iter=int(cfg.steps * 0.83),
                                    refine_every=cfg.refine_every)
            state = strategy.initialize_state()
        else:
            strategy = DefaultStrategy(
                refine_start_iter=cfg.refine_start_iter, refine_stop_iter=cfg.steps // 2,
                refine_every=cfg.refine_every,
                reset_every=cfg.reset_every or min(3000, max(cfg.steps // 3, 500)),
                prune_scale3d=1e6 if unbounded else 0.1)
            state = strategy.initialize_state(scene_scale=scene_scale)
        strategy.check_sanity(params, optimizers)

        def render(i: int, sh_degree: int, bg, scale: float = 1.0):
            h, w = gts[i].shape[:2]
            K = Ks_t[i : i + 1]
            if scale != 1.0:
                rw, rh = max(1, round(w * scale)), max(1, round(h * scale))
                K = K.clone()
                K[:, 0] *= rw / w
                K[:, 1] *= rh / h
                w, h = rw, rh
            return rasterize(
                means=params["means"], quats=params["quats"], scales=torch.exp(params["scales"]),
                opacities=torch.sigmoid(params["opacities"]),
                colors=torch.cat([params["sh0"], params["shN"]], dim=1),
                viewmats=viewmats[i : i + 1], Ks=K, width=w, height=h,
                sh_degree=sh_degree, packed=False,
                rasterize_mode="antialiased" if cfg.antialiased else "classic",
                backgrounds=bg[None] if bg is not None else None,
            )

        curves: dict[str, list[float]] = {k: [] for k in CURVE_KEYS}

        def log_curve(step: int, loss_v: float, psnr_v: float) -> None:
            for key, v in zip(CURVE_KEYS, (step, loss_v, psnr_v, len(params["means"]),
                                           round(time.time() - t0, 2)), strict=True):
                curves[key].append(round(float(v), 5) if isinstance(v, float) else v)

        order: list[int] = []
        log_every = max(1, cfg.steps // 50)
        loss_v = float("nan")
        for step in range(cfg.steps):
            if not order:
                order = list(rng.permutation(train_idx))
            i = int(order.pop())
            gt = gts[i].float() / 255.0
            bg = torch.rand(3, device=dev) if cfg.random_background else torch.zeros(3, device=dev)
            renders, _, info = render(i, min(step // cfg.sh_degree_interval, cfg.sh_degree), bg)
            pred = renders[0]
            if not mcmc:
                strategy.step_pre_backward(params, optimizers, state, step, info)
            l1 = F.l1_loss(pred, gt)
            ssim = _ssim(pred.permute(2, 0, 1)[None], gt.permute(2, 0, 1)[None])
            loss = (1 - cfg.ssim_lambda) * l1 + cfg.ssim_lambda * (1 - ssim)
            if mcmc:
                loss = loss + cfg.opacity_reg * torch.sigmoid(params["opacities"]).mean()
                loss = loss + cfg.scale_reg * torch.exp(params["scales"]).mean()
            loss.backward()
            for opt in optimizers.values():
                opt.step()
                opt.zero_grad(set_to_none=True)
            sched.step()
            if mcmc:
                strategy.step_post_backward(params, optimizers, state, step, info,
                                            lr=sched.get_last_lr()[0])
            else:
                strategy.step_post_backward(params, optimizers, state, step, info, packed=False)
                _maybe_reset_opacity(strategy, params, optimizers, state, step)

            if step % log_every == 0 or step == cfg.steps - 1:
                loss_v = loss.item()
                mse = F.mse_loss(pred.detach().clamp(0, 1), gt).item()
                log_curve(step, loss_v, -10 * math.log10(max(mse, 1e-10)))
                if on_progress:
                    on_progress(step / cfg.steps,
                                f"step {step}/{cfg.steps} loss {loss_v:.4f} gaussians {len(params['means'])}")
            if on_checkpoint and (step + 1) in cfg.preview_steps:
                on_checkpoint(step + 1, _to_cloud(params, cfg.antialiased))

        psnr = _evaluate(render, gts, test_idx, cfg.sh_degree)
        log_curve(cfg.steps, loss_v, psnr)
        if on_render and cfg.render_max_side > 0:
            with torch.no_grad():
                for i, frame_id in enumerate(scene.frame_ids):
                    h, w = gts[i].shape[:2]
                    out, _, _ = render(i, cfg.sh_degree, None, min(1.0, cfg.render_max_side / max(h, w)))
                    on_render(frame_id, (out[0].clamp(0, 1) * 255).round().byte().cpu().numpy())
        cloud = _to_cloud(params, cfg.antialiased)
        return TrainResult(cloud, curves=curves,
                           eval_frames=[scene.frame_ids[int(i)] for i in test_idx], metrics={
            "trainer": self.name, "device": str(dev), "strategy": "mcmc" if mcmc else "default",
            "steps": cfg.steps, "gaussians": len(cloud),
            "psnr": round(psnr, 2), "psnr_split": "holdout" if cfg.holdout_every > 0 else "train",
            "image_size": list(gts[0].shape[1::-1]), "train_images": int(len(train_idx)),
            "seconds": round(time.time() - t0, 1), "scene_scale": round(scene_scale, 3),
            "unbounded": bool(unbounded),
            "sky_dome_radius_m": None if dome_radius is None else round(dome_radius, 1),
            "gpu": torch.cuda.get_device_name(0) if dev.type == "cuda" else str(dev),
        })


def sky_dome(scene: SceneData, images: list[np.ndarray], Ks: np.ndarray, n: int,
             far: float) -> tuple[np.ndarray, np.ndarray, float]:
    """Points on a far hemisphere (down to 15 degrees below the horizon) around the
    cameras, coloured from the frames they project into. World frame: +y up."""
    center = scene.cam_centers.mean(axis=0)
    radius = max(3.0 * far, 50.0)
    i = np.arange(n) + 0.5
    y = 1.0 - (i / n) * (1.0 + np.sin(np.radians(15.0)))
    phi = i * np.pi * (3.0 - np.sqrt(5.0))  # Fibonacci spiral: even coverage
    ring = np.sqrt(np.maximum(0.0, 1.0 - y * y))
    pts = center + radius * np.stack([ring * np.cos(phi), y, ring * np.sin(phi)], axis=-1)
    acc, cnt = np.zeros((n, 3)), np.zeros(n)
    for k in range(0, len(images), max(1, len(images) // 12)):
        img, K, w2c = images[k], Ks[k], scene.w2c[k]
        h, w = img.shape[:2]
        pc = pts @ w2c[:3, :3].T + w2c[:3, 3]
        uv = (pc @ K.T)[:, :2] / np.maximum(pc[:, 2:3], 1e-9)
        ok = (pc[:, 2] > 0) & (uv[:, 0] >= 0) & (uv[:, 0] < w - 1) & (uv[:, 1] >= 0) & (uv[:, 1] < h - 1)
        acc[ok] += img[uv[ok, 1].astype(int), uv[ok, 0].astype(int)] / 255.0
        cnt[ok] += 1
    sky_blue = np.array([0.72, 0.8, 0.9])
    cols = np.where(cnt[:, None] > 0, acc / np.maximum(cnt[:, None], 1), sky_blue)
    return pts.astype(np.float32), cols.astype(np.float32), float(radius)


def _load_images(scene: SceneData, max_side: int) -> tuple[list[np.ndarray], np.ndarray]:
    import cv2

    from memoryscapes_workers.visual.video import imread_rgb

    images, Ks = [], []
    for path, K in zip(scene.image_paths, scene.Ks, strict=True):
        img = imread_rgb(path)
        h, w = img.shape[:2]
        f = min(1.0, max_side / max(h, w))
        K = K.copy()
        if f < 1.0:
            nw, nh = int(round(w * f)), int(round(h * f))
            img = cv2.resize(img, (nw, nh), interpolation=cv2.INTER_AREA)
            K[0] *= nw / w
            K[1] *= nh / h
        images.append(img)
        Ks.append(K)
    return images, np.array(Ks)


def _knn_mean_dist_torch(pts, k: int):
    import torch

    out = torch.empty(len(pts), device=pts.device)
    for i in range(0, len(pts), 1024):
        d = torch.cdist(pts[i : i + 1024], pts)
        out[i : i + 1024] = d.topk(k + 1, largest=False).values[:, 1:].mean(dim=1)
    return out


def _ssim(x, y, window: int = 11, sigma: float = 1.5):
    """SSIM on [1, C, H, W] in [0, 1] ('valid' padding, Gaussian window)."""
    import torch
    import torch.nn.functional as F

    c = x.shape[1]
    g = torch.exp(-((torch.arange(window, device=x.device) - window // 2) ** 2) / (2 * sigma**2))
    g = (g / g.sum()).float()
    kernel = (g[:, None] @ g[None, :]).expand(c, 1, window, window).contiguous()

    def blur(t):
        return F.conv2d(t, kernel, groups=c)

    mu_x, mu_y = blur(x), blur(y)
    sxx = blur(x * x) - mu_x**2
    syy = blur(y * y) - mu_y**2
    sxy = blur(x * y) - mu_x * mu_y
    c1, c2 = 0.01**2, 0.03**2
    s = ((2 * mu_x * mu_y + c1) * (2 * sxy + c2)) / ((mu_x**2 + mu_y**2 + c1) * (sxx + syy + c2))
    return s.mean()


def _empty_cache() -> None:
    import torch

    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    elif torch.backends.mps.is_available():
        torch.mps.empty_cache()


def _maybe_reset_opacity(strategy, params, optimizers, state, step: int) -> None:
    """gsplat 1.5.3's DefaultStrategy never resets opacity (operator-precedence bug in
    `step % reset_every == 0 & step > 0`). Resetting is a key floater defense, so do it."""
    from gsplat.strategy.ops import reset_opa

    if step > 0 and step % strategy.reset_every == 0 and step < strategy.refine_stop_iter:
        reset_opa(params=params, optimizers=optimizers, state=state, value=strategy.prune_opa * 2.0)
        _empty_cache()


def _evaluate(render, gts, idx, sh_degree: int) -> float:
    import torch

    psnrs = []
    with torch.no_grad():
        for i in idx:
            renders, _, _ = render(int(i), sh_degree, None)
            gt = gts[int(i)].float() / 255.0
            mse = torch.mean((renders[0].clamp(0, 1) - gt) ** 2).item()
            psnrs.append(-10 * math.log10(max(mse, 1e-10)))
    return float(np.mean(psnrs)) if psnrs else float("nan")


def _to_cloud(params, antialiased: bool) -> GaussianCloud:
    def np_(t):
        return t.detach().float().cpu().numpy()

    return GaussianCloud(
        means=np_(params["means"]), scales=np_(params["scales"]), quats=np_(params["quats"]),
        opacities=np_(params["opacities"]), sh0=np_(params["sh0"])[:, 0, :],
        shN=np_(params["shN"]), antialiased=antialiased,
    )


def get_trainer(name: str) -> Trainer:
    if name == "auto":
        name = detect_trainer()
    if name == "gsplat":
        return GaussianTrainer("cuda")
    if name == "torch":
        return GaussianTrainer("torch")
    if name == "points":
        return PointsTrainer()
    raise ValueError(f"Unknown splat trainer: {name}")
