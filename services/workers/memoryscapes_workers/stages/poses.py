"""Camera poses and a sparse point cloud with COLMAP (pycolmap, CPU).

Inputs:  ingest/frames.json, ingest/frames/*.jpg
Outputs (docs/SPLAT_OUTPUTS.md):
    poses/cameras.json       registered keyframes, OpenCV convention, metric-ish world frame
    poses/points3d.npy       (N, 3) float32
    poses/points3d_rgb.npy   (N, 3) uint8
    poses/meta.json          registration stats, how the frame was normalized, warnings

COLMAP's world frame is arbitrary in rotation and scale, so it's normalized to the
pipeline's convention:
- rotation: the cameras' mean "down" axis becomes -Y (phones are held roughly upright),
  and the first camera's forward direction becomes +Z;
- origin: first camera at x = z = 0, the ground at y = 0;
- scale: the median camera height above the ground becomes `camera_height_m` (people
  film at about chest height). Monocular SfM has no true scale; this is an estimate,
  reported in meta.json, and the debugger's walking-speed check is the sanity test.
"""

from __future__ import annotations

import logging
import tempfile
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from memoryscapes_workers.context import StageContext
from memoryscapes_workers.storage import read_json, write_json, write_npy

log = logging.getLogger(__name__)


class PosesError(RuntimeError):
    pass


@dataclass(frozen=True)
class PosesConfig:
    camera_height_m: float = 1.5
    min_registered: int = 5
    exhaustive_below: int = 50  # match all pairs for short clips, sequential windows above


def rotation_between(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Smallest rotation taking unit vector a to unit vector b."""
    a, b = a / np.linalg.norm(a), b / np.linalg.norm(b)
    v, c = np.cross(a, b), float(a @ b)
    if c < -0.999999:  # opposite: rotate 180 degrees about any perpendicular axis
        axis = np.cross(a, [1.0, 0, 0]) if abs(a[0]) < 0.9 else np.cross(a, [0, 1.0, 0])
        axis /= np.linalg.norm(axis)
        return 2 * np.outer(axis, axis) - np.eye(3)
    vx = np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])
    return np.eye(3) + vx + vx @ vx / (1 + c)


def normalize_frame(c2ws: np.ndarray, points: np.ndarray, camera_height_m: float) -> tuple[np.ndarray, np.ndarray, dict]:
    """c2ws (K, 4, 4) and points (N, 3) in COLMAP's frame -> the pipeline's frame."""
    down = c2ws[:, :3, 1].mean(axis=0)
    R = rotation_between(down, np.array([0.0, -1.0, 0.0]))
    fwd = R @ c2ws[0, :3, 2]
    yaw = np.arctan2(fwd[0], fwd[2])  # rotate about +Y so the first camera looks along +Z
    cy, sy = np.cos(-yaw), np.sin(-yaw)
    R = np.array([[cy, 0, sy], [0, 1, 0], [-sy, 0, cy]]) @ R

    pos = (R @ c2ws[:, :3, 3].T).T
    pts = (R @ points.T).T if len(points) else points.reshape(0, 3)
    # how much the cameras' down axes disagree: large values mean "upright" is a guess
    downs = c2ws[:, :3, 1] / np.linalg.norm(c2ws[:, :3, 1], axis=1, keepdims=True)
    unit_down = down / np.linalg.norm(down)
    info: dict = {"camera_tilt_spread_deg": float(np.degrees(np.median(np.arccos(np.clip(downs @ unit_down, -1, 1)))))}

    scale, method = 1.0, "none (no points below the cameras)"
    if len(pts) >= 20:
        below = pts[pts[:, 1] < np.median(pos[:, 1])]
        if len(below) >= 10:
            ground = float(np.percentile(below[:, 1], 5))
            height = float(np.median(pos[:, 1]) - ground)
            if height > 1e-6:
                scale, method = camera_height_m / height, f"median camera height above the ground = {camera_height_m} m"
                info["ground_y_raw"] = ground
    if method.startswith("none"):
        log.warning("couldn't find the ground; leaving COLMAP's scale")
    origin = np.array([pos[0, 0], info.get("ground_y_raw", float(np.min(pos[:, 1]) - 1.5 / scale)), pos[0, 2]])

    out = np.repeat(np.eye(4)[None], len(c2ws), axis=0)
    out[:, :3, :3] = np.einsum("ij,kjl->kil", R, c2ws[:, :3, :3])
    out[:, :3, 3] = (pos - origin) * scale
    pts = (pts - origin) * scale
    info.update({"scale": scale, "scale_method": method})
    return out, pts.astype(np.float32), info


def _cam_from_world(image) -> np.ndarray:
    cfw = image.cam_from_world() if callable(image.cam_from_world) else image.cam_from_world
    return np.asarray(cfw.matrix())


def reconstruct(image_dir: Path, work: Path, n_images: int, cfg: PosesConfig):
    import pycolmap

    db = work / "database.db"
    reader = pycolmap.ImageReaderOptions()
    reader.camera_model = "SIMPLE_RADIAL"
    try:  # pycolmap >= 3.11: camera model lives in the reader options
        pycolmap.extract_features(db, image_dir, camera_mode=pycolmap.CameraMode.SINGLE, reader_options=reader)
    except TypeError:
        pycolmap.extract_features(db, image_dir, camera_mode=pycolmap.CameraMode.SINGLE, camera_model="SIMPLE_RADIAL")
    if n_images <= cfg.exhaustive_below:
        pycolmap.match_exhaustive(db)
    else:
        pycolmap.match_sequential(db)
    maps = pycolmap.incremental_mapping(db, image_dir, work / "sparse")
    if not maps:
        raise PosesError("COLMAP couldn't reconstruct the scene: too little texture or overlap between frames")
    return max(maps.values(), key=lambda r: r.num_reg_images())


def run(ctx: StageContext, cfg: PosesConfig = PosesConfig()) -> dict:
    storage = ctx.storage
    frames = read_json(storage, ctx.key("ingest", "frames.json"))
    by_name = {f"{f['frame_id']}.jpg": f for f in frames}

    with tempfile.TemporaryDirectory() as tmpdir:
        work = Path(tmpdir)
        images = work / "images"
        images.mkdir()
        for name, f in by_name.items():
            (images / name).write_bytes(storage.read_bytes(f["key"]))
        log.info("SfM on %d frames", len(frames))
        rec = reconstruct(images, work, len(frames), cfg)

        regs = sorted(rec.images.values(), key=lambda im: by_name[im.name]["t"])
        if len(regs) < cfg.min_registered:
            raise PosesError(f"only {len(regs)} of {len(frames)} frames registered; try a slower, steadier video")
        c2ws, intrinsics = [], []
        for im in regs:
            M = _cam_from_world(im)
            R, t = M[:, :3], M[:, 3]
            c2w = np.eye(4)
            c2w[:3, :3], c2w[:3, 3] = R.T, -R.T @ t
            c2ws.append(c2w)
            cam = rec.cameras[im.camera_id]
            intrinsics.append((np.asarray(cam.calibration_matrix()), int(cam.width), int(cam.height)))
        pts = np.array([p.xyz for p in rec.points3D.values()], dtype=float).reshape(-1, 3)
        rgb = np.array([p.color for p in rec.points3D.values()], dtype=np.uint8).reshape(-1, 3)
        reproj = float(np.mean([p.error for p in rec.points3D.values()])) if len(pts) else None

    c2ws_n, pts_n, frame_info = normalize_frame(np.stack(c2ws), pts, cfg.camera_height_m)
    cameras = [
        {"frame_id": by_name[im.name]["frame_id"], "t": by_name[im.name]["t"], "width": w, "height": h, "K": K.tolist(), "c2w": c2w.tolist(), "image_key": by_name[im.name]["key"]}
        for im, c2w, (K, w, h) in zip(regs, c2ws_n, intrinsics)
    ]
    write_json(storage, ctx.key("poses", "cameras.json"), cameras)
    write_npy(storage, ctx.key("poses", "points3d.npy"), pts_n)
    write_npy(storage, ctx.key("poses", "points3d_rgb.npy"), rgb)

    warnings = []
    if len(regs) < 0.7 * len(frames):
        warnings.append(f"only {len(regs)} of {len(frames)} frames registered; localization uses those")
    meta = {
        "method": "COLMAP incremental SfM (pycolmap, CPU)",
        "registered": len(regs),
        "frames": len(frames),
        "points": int(len(pts_n)),
        "mean_reprojection_error_px": reproj,
        **frame_info,
        "units": "meters (estimated)",
        "up": "+y",
        "warnings": warnings,
    }
    write_json(storage, ctx.key("poses", "meta.json"), meta)
    log.info("poses: %d/%d registered, %d points, scale x%.3f", len(regs), len(frames), len(pts_n), frame_info["scale"])
    return meta
