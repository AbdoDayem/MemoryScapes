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

import json
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
    # Spatial video: weight of an L1 loss between the rendered and the stereo inverse depth
    # (poses/stereo_depth) on the left-eye images. Off until it's tuned on a GPU
    # (docs/SPATIAL_VIDEO.md); try 0.05-0.5 with --set splat.train.depth_lambda=...
    depth_lambda: float = 0.0
    # Spatial video: also train on the right eye (poses/stereo_right.json), one more view per
    # instant, 17.7 mm to the side.
    stereo_views: bool = True
    # Per-image colour compensation: a 3x3 colour matrix + offset per training image,
    # learned with the splat and applied to its render before the loss. It absorbs what
    # changes between frames but isn't the scene: a phone's auto exposure and white
    # balance (strong at night), and the right eye's brightness offset in spatial video.
    # Without it the splat bakes those changes into view-dependent colour and floaters.
    # The average transform is held at identity, so the splat's colours are the average
    # frame's. Off until it's measured on a GPU (docs/SPATIAL_VIDEO.md); try
    # --set splat.train.exposure_comp=true.
    exposure_comp: bool = False
    exposure_lr: float = 2e-3
    exposure_reg: float = 1.0  # weight of the average-transform-is-identity penalty
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
    # distant; they're pruned by their size on screen instead.
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
    # Spatial video: "left"/"right" per image, and the instant (left frame index) each image
    # shows, so both eyes of a held-out instant stay out of training. Empty: one eye.
    eyes: list[str] = field(default_factory=list)
    instants: list[int] = field(default_factory=list)
    # Stereo inverse-depth maps (poses/stereo_depth, uint16 PNG of 1/m x depth_scale) for
    # the left-eye images, None for the rest. Empty: no depth supervision.
    depth_paths: list[Path | None] = field(default_factory=list)
    depth_scale: float = 10_000.0

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


def load_scene(poses_dir: Path, stereo_views: bool = True) -> SceneData:
    """Load the undistorted, world-frame COLMAP model written by the poses stage, plus, for
    a spatial video, the right eye's cameras (poses/stereo_right.json) as extra views.

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
    eyes, instants, depth_paths, depth_scale = [], [], [], 10_000.0
    right_file = poses_dir / "stereo_right.json"
    stereo = json.loads(right_file.read_text(encoding="utf-8")) if right_file.exists() else None
    if stereo and stereo.get("depth"):
        depth_dir = poses_dir / stereo["depth"]["dir"]
        depth_scale = float(stereo["depth"]["scale"])
        depth_paths = [(depth_dir / f"{p.stem}.png") if (depth_dir / f"{p.stem}.png").exists() else None
                       for p in paths]
    if stereo_views and stereo:
        index = {p.stem: i for i, p in enumerate(paths)}
        eyes, instants = ["left"] * len(paths), list(range(len(paths)))
        for fr in stereo["frames"]:
            if fr["left_frame_id"] not in index or not (poses_dir / fr["image"]).exists():
                continue
            paths.append(poses_dir / fr["image"])
            Ks.append(np.array(fr["K"], float))
            w2cs.append(np.linalg.inv(np.array(fr["c2w"], float)))
            sizes.append((fr["width"], fr["height"]))
            eyes.append("right")
            instants.append(index[fr["left_frame_id"]])
            if depth_paths:
                depth_paths.append(None)
    return SceneData(paths, np.array(Ks), np.array(w2cs), sizes,
                     model.points.astype(np.float32), model.colors.astype(np.float32) / 255.0,
                     eyes=eyes, instants=instants, depth_paths=depth_paths, depth_scale=depth_scale)


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
        # Held out by instant: both eyes of a held-out instant stay out of training, and
        # scores are on the left eye, so runs with and without the right eye compare.
        instants = np.array(scene.instants or list(all_idx))
        is_right = np.array([e == "right" for e in scene.eyes] or [False] * len(all_idx))
        if cfg.holdout_every > 0:
            held = instants % cfg.holdout_every == 0
            test_idx = all_idx[held & ~is_right]
            train_idx = all_idx[~held]
        else:
            train_idx, test_idx = all_idx, all_idx[~is_right][:: 8]
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
                # Open scenes: prune by size on screen (> 15% of the image) instead of
                # in metres, which keeps far buildings and still kills huge floaters.
                prune_scale3d=1e6 if unbounded else 0.1,
                refine_scale2d_stop_iter=cfg.steps if unbounded else 0)
            state = strategy.initialize_state(scene_scale=scene_scale)
        strategy.check_sanity(params, optimizers)

        depth_tgts = _load_depths(scene, gts, dev) if cfg.depth_lambda > 0 else [None] * len(gts)

        # Per-image colour transforms (cfg.exposure_comp): [C, 3, 4] = (A - I | b), zero =
        # identity. Held-out images are never trained, so they're scored as the splat is.
        color_tf = torch.nn.Parameter(torch.zeros((len(gts), 3, 4), device=dev)) if cfg.exposure_comp else None
        color_opt = torch.optim.Adam([color_tf], lr=cfg.exposure_lr) if color_tf is not None else None
        train_rows = torch.as_tensor(train_idx, dtype=torch.long, device=dev)

        def correct(img, i: int):
            """The render as frame i's camera would have exposed it."""
            if color_tf is None:
                return img
            t = color_tf[i]
            return img @ (torch.eye(3, device=dev) + t[:, :3]).T + t[:, 3]

        def render(i: int, sh_degree: int, bg, scale: float = 1.0, depth: bool = False):
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
                render_mode="RGB+ED" if depth else "RGB",
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
            tgt = depth_tgts[i]
            renders, _, info = render(i, min(step // cfg.sh_degree_interval, cfg.sh_degree), bg,
                                      depth=tgt is not None)
            pred = correct(renders[0][..., :3], i)
            if not mcmc:
                strategy.step_pre_backward(params, optimizers, state, step, info)
            l1 = F.l1_loss(pred, gt)
            ssim = _ssim(pred.permute(2, 0, 1)[None], gt.permute(2, 0, 1)[None])
            loss = (1 - cfg.ssim_lambda) * l1 + cfg.ssim_lambda * (1 - ssim)
            if tgt is not None:  # inverse depth: stereo's noise is constant in it
                valid = tgt > 0
                if bool(valid.any()):
                    inv = 1.0 / renders[0][..., 3].clamp_min(0.05)
                    loss = loss + cfg.depth_lambda * (inv[valid] - tgt[valid]).abs().mean()
            if mcmc:
                loss = loss + cfg.opacity_reg * torch.sigmoid(params["opacities"]).mean()
                loss = loss + cfg.scale_reg * torch.exp(params["scales"]).mean()
            if color_tf is not None:  # keep the average exposure at identity (no global drift)
                loss = loss + cfg.exposure_reg * color_tf[train_rows].mean(dim=0).square().sum()
            loss.backward()
            for opt in [*optimizers.values(), *([color_opt] if color_opt else [])]:
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

        psnr, psnr_cc = _evaluate(render, gts, test_idx, cfg.sh_degree, correct)
        depth_err = _depth_error(render, scene, Ks, gts, test_idx, cfg.sh_degree)
        log_curve(cfg.steps, loss_v, psnr)
        if on_render and cfg.render_max_side > 0:
            with torch.no_grad():
                for i, frame_id in enumerate(scene.frame_ids):
                    if is_right[i]:  # renders are per poses/cameras.json frame (the left eye)
                        continue
                    h, w = gts[i].shape[:2]
                    out, _, _ = render(i, cfg.sh_degree, None, min(1.0, cfg.render_max_side / max(h, w)))
                    on_render(frame_id, (out[0].clamp(0, 1) * 255).round().byte().cpu().numpy())
        cloud = _to_cloud(params, cfg.antialiased)
        return TrainResult(cloud, curves=curves,
                           eval_frames=[scene.frame_ids[int(i)] for i in test_idx], metrics={
            "trainer": self.name, "device": str(dev), "strategy": "mcmc" if mcmc else "default",
            "steps": cfg.steps, "gaussians": len(cloud),
            "psnr": round(psnr, 2), "psnr_split": "holdout" if cfg.holdout_every > 0 else "train",
            # PSNR after the best affine colour fit per scored image: compares structure,
            # not exposure (fair between runs with and without exposure_comp).
            "psnr_cc": round(psnr_cc, 2),
            "exposure_comp": bool(cfg.exposure_comp),
            # Learned brightness per training image (mean colour-matrix diagonal; 1 = the
            # splat's own, the average): how the camera's exposure varied, as far as
            # training could tell. None without exposure_comp.
            "exposure_gains": _gains(color_tf, train_idx, scene.frame_ids),
            "image_size": list(gts[0].shape[1::-1]), "train_images": int(len(train_idx)),
            "right_eye_images": int(is_right[train_idx].sum()),
            "depth_supervised_images": int(sum(depth_tgts[int(i)] is not None for i in train_idx)),
            # Rendered depth on the scored frames vs the reconstruction's 3D points: median
            # relative error (geometry, which colour PSNR near the capture path can miss).
            "depth_error_vs_sfm": None if depth_err is None else round(depth_err, 4),
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


def _load_depths(scene: SceneData, gts, dev) -> list:
    """Stereo inverse-depth targets at each training image's size (None where there's none)."""
    import cv2
    import torch

    out = []
    for i, gt in enumerate(gts):
        path = scene.depth_paths[i] if i < len(scene.depth_paths) else None
        raw = cv2.imread(str(path), cv2.IMREAD_UNCHANGED) if path is not None else None
        if raw is None:
            out.append(None)
            continue
        h, w = gt.shape[:2]
        inv = cv2.resize(raw.astype(np.float32) / scene.depth_scale, (w, h), interpolation=cv2.INTER_NEAREST)
        out.append(torch.from_numpy(inv).to(dev))
    return out


def _depth_error(render, scene: SceneData, Ks, gts, idx, sh_degree: int) -> float | None:
    import torch

    errs = []
    with torch.no_grad():
        for i in idx:
            i = int(i)
            h, w = gts[i].shape[:2]
            xc = (scene.w2c[i][:3, :3] @ scene.points.T + scene.w2c[i][:3, 3:4]).T
            z = xc[:, 2]
            uv = (Ks[i] @ xc.T).T[:, :2] / np.maximum(z[:, None], 1e-9)
            ok = (z > 0.1) & (uv[:, 0] >= 0) & (uv[:, 0] < w - 1) & (uv[:, 1] >= 0) & (uv[:, 1] < h - 1)
            if ok.sum() < 20:
                continue
            renders, alphas, _ = render(i, sh_degree, None, depth=True)
            d = renders[0][..., 3].cpu().numpy()
            a = alphas[0][..., 0].cpu().numpy()
            u, v = np.round(uv[ok]).astype(int).T
            seen = a[v, u] > 0.5
            if seen.sum() >= 20:
                errs.append(np.abs(d[v, u][seen] - z[ok][seen]) / z[ok][seen])
    return float(np.median(np.concatenate(errs))) if errs else None


def _evaluate(render, gts, idx, sh_degree: int, correct=None) -> tuple[float, float]:
    """Mean PSNR on the scored images, and the same after the best affine colour fit per
    image (least squares from the render's RGB to the frame's)."""
    import torch

    def psnr(a, b) -> float:
        return -10 * math.log10(max(torch.mean((a - b) ** 2).item(), 1e-10))

    plain, cc = [], []
    with torch.no_grad():
        for i in idx:
            renders, _, _ = render(int(i), sh_degree, None)
            img = renders[0][..., :3]
            if correct is not None:
                img = correct(img, int(i))  # identity for held-out images (never trained)
            img = img.clamp(0, 1)
            gt = gts[int(i)].float() / 255.0
            plain.append(psnr(img, gt))
            # On the CPU in float64 (MPS has no float64).
            rgb = img.reshape(-1, 3).cpu().double()
            x = torch.cat([rgb, torch.ones_like(rgb[:, :1])], dim=1)
            y = gt.reshape(-1, 3).cpu().double()
            fit = (x @ torch.linalg.lstsq(x, y).solution).clamp(0, 1)
            cc.append(psnr(fit, y))
    if not plain:
        return float("nan"), float("nan")
    return float(np.mean(plain)), float(np.mean(cc))


def _gains(color_tf, train_idx, frame_ids) -> dict[str, float] | None:
    if color_tf is None:
        return None
    diag = color_tf.detach().cpu().diagonal(dim1=1, dim2=2)  # [C, 3]
    return {frame_ids[int(i)]: round(1.0 + float(diag[int(i)].mean()), 3) for i in train_idx}


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
