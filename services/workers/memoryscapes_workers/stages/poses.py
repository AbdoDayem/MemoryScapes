"""Camera poses.

- Quality path: GLOMAP / COLMAP (both through pycolmap).
- Fast path: feed-forward VGGT or MASt3R-SfM (poses + dense points in minutes). Not yet.
- Estimate metric scale (monocular metric depth, or IMU/ARKit later) so audio
  distance attenuation works in meters. For now: eye height above a RANSAC ground plane.

Reads  {memory_id}/ingest/frames/, metadata.json
Writes {memory_id}/poses/
    sparse/0/{cameras,images,points3D}.bin   undistorted PINHOLE model, in the world frame
    images/frame_*.jpg                        undistorted images matching sparse/0
    sparse_points.ply                         SfM points (xyz + rgb), world frame
    cameras.json                              per-frame c2w (OpenCV) + intrinsics + time
    path.json                                 capture path for the viewer (three.js camera)
    navigation.json                           spawn, floor, walkable bounds + corridor
    alignment.json                            SfM -> world similarity and how it was found

World frame: meters, +y up, ground at y=0, origin under the first camera, which looks
along -z. Splat, audio sources and geolocation all live in this frame.
"""

from __future__ import annotations

import shutil
from pathlib import Path
from typing import Literal

import numpy as np
from pydantic import BaseModel, Field

from memoryscapes_workers.context import StageContext
from memoryscapes_workers.stagekit import StageError, StageIO, run_stage, write_json

STAGE = "poses"
VERSION = "1"

CAPTURE_HINT = (
    "Move slowly and steadily, walk through the space or orbit the subject instead of "
    "turning in place, and avoid blank walls, mirrors and very dark scenes."
)


class PosesConfig(BaseModel):
    mapper: Literal["global", "incremental"] = "global"
    camera_model: Literal["SIMPLE_RADIAL", "OPENCV", "PINHOLE", "SIMPLE_PINHOLE"] = "SIMPLE_RADIAL"
    max_num_features: int = 8192
    matcher: Literal["sequential", "exhaustive"] = "sequential"
    sequential_overlap: int = 15
    min_registered_fraction: float = Field(0.5, ge=0, le=1)
    # Metric scale: handheld eye height above the ground plane. Override when known.
    eye_height_m: float = 1.5
    fixed_scale: float | None = None
    corridor_radius_m: float = 1.5
    use_gpu: bool = True  # SIFT on CUDA when the pycolmap build supports it


def run(ctx: StageContext) -> dict:
    cfg = PosesConfig(**ctx.stage_config(STAGE))
    return run_stage(ctx, STAGE, version=VERSION, config=cfg, upstream=["ingest"],
                     body=lambda io: _poses(io, cfg))


def _largest(recs: dict) -> object | None:
    if not recs:
        return None
    return max(recs.values(), key=lambda r: r.num_reg_images())


def _sfm(io: StageIO, cfg: PosesConfig, image_dir: Path, n_images: int):
    import pycolmap

    work = io.work_dir / "colmap"
    work.mkdir()
    db = work / "database.db"
    device = pycolmap.Device.auto if cfg.use_gpu else pycolmap.Device.cpu

    extraction = pycolmap.FeatureExtractionOptions()
    extraction.sift.max_num_features = cfg.max_num_features
    pycolmap.extract_features(
        db, image_dir,
        camera_mode=pycolmap.CameraMode.SINGLE,
        reader_options=pycolmap.ImageReaderOptions(camera_model=cfg.camera_model),
        extraction_options=extraction,
        device=device,
    )
    io.progress(0.2, "features extracted")

    if cfg.matcher == "exhaustive":
        pycolmap.match_exhaustive(db, device=device)
    else:
        pairing = pycolmap.SequentialPairingOptions()
        pairing.overlap = cfg.sequential_overlap
        pairing.quadratic_overlap = True
        pycolmap.match_sequential(db, pairing_options=pairing, device=device)
    io.progress(0.4, "features matched")

    sparse = work / "sparse"
    sparse.mkdir()
    if cfg.mapper == "global":
        recs = pycolmap.global_mapping(db, image_dir, sparse)
    else:
        recs = pycolmap.incremental_mapping(db, image_dir, sparse)
    rec = _largest(recs)
    registered = rec.num_reg_images() if rec is not None else 0
    if registered < max(10, cfg.min_registered_fraction * n_images):
        raise StageError(
            STAGE,
            f"Could not recover the camera path: only {registered} of {n_images} frames "
            f"were placed ({len(recs)} disconnected pieces).",
            CAPTURE_HINT,
        )
    io.progress(0.7, f"registered {registered}/{n_images} frames")
    return rec, work


def _poses(io: StageIO, cfg: PosesConfig) -> dict:
    import pycolmap

    from memoryscapes_workers.visual import alignment as al

    ingest_dir = io.input_dir("ingest")
    meta = io.read_upstream_json("ingest", "metadata.json")
    frame_times = {Path(f["file"]).name: f["t"] for f in meta["keyframes"]["frames"]}
    image_dir = ingest_dir / "frames"
    n_images = len(frame_times)

    rec, work = _sfm(io, cfg, image_dir, n_images)

    # ---- gravity, metric scale, origin
    images = sorted((im for im in rec.images.values() if im.has_pose), key=lambda im: im.name)
    centers = np.array([im.projection_center() for im in images])
    rots = np.array([im.cam_from_world().rotation.matrix().T for im in images])
    pts = np.array([p.xyz for p in rec.points3D.values()])
    align = al.estimate_alignment(centers, rots, pts, eye_height_m=cfg.eye_height_m,
                                  fixed_scale=cfg.fixed_scale)

    # ---- undistort, then move the undistorted model into the world frame
    raw_model = work / "model_raw"
    raw_model.mkdir()
    rec.write(raw_model)
    undist = work / "undistorted"
    pycolmap.undistort_images(undist, raw_model, image_dir)
    model = pycolmap.Reconstruction(undist / "sparse")
    model.transform(pycolmap.Sim3d(align.matrix()))
    io.progress(0.85, "undistorted and aligned")

    out = io.out_dir
    (out / "sparse" / "0").mkdir(parents=True)
    model.write(out / "sparse" / "0")
    shutil.copytree(undist / "images", out / "images")
    model.export_PLY(str(out / "sparse_points.ply"))

    # ---- per-frame exports (capture order)
    frames, c2ws = [], []
    for im in sorted((im for im in model.images.values() if im.has_pose), key=lambda im: im.name):
        cam = model.cameras[im.camera_id]
        c2w = np.eye(4)
        c2w[:3, :] = im.cam_from_world().inverse().matrix()
        c2ws.append(c2w)
        k = cam.calibration_matrix()  # undistorted cameras are PINHOLE
        fx, fy, cx, cy = k[0, 0], k[1, 1], k[0, 2], k[1, 2]
        frames.append({
            "file": f"images/{im.name}",
            "t": frame_times.get(im.name),
            "c2w_opencv": np.round(c2w, 6).tolist(),
            "fx": float(fx), "fy": float(fy), "cx": float(cx), "cy": float(cy),
            "width": int(cam.width), "height": int(cam.height),
        })
    write_json(out / "cameras.json", {
        "convention": "c2w, OpenCV camera axes (x right, y down, z forward); world meters, +y up",
        "frames": frames,
    })
    path = []
    for f, c2w in zip(frames, c2ws, strict=True):
        pos, quat = al.camera_to_threejs(c2w)
        path.append({"t": f["t"], "position": np.round(pos, 4).tolist(),
                     "quaternion_xyzw": np.round(quat, 6).tolist()})
    write_json(out / "path.json", {
        "convention": "three.js camera (looks along -z, +y up); world meters",
        "fov_y_deg": float(np.degrees(2 * np.arctan(frames[0]["height"] / (2 * frames[0]["fy"])))),
        "samples": path,
    })
    nav = al.navigation_from_trajectory(c2ws, corridor_radius_m=cfg.corridor_radius_m)
    nav["original_path"] = "path.json"
    write_json(out / "navigation.json", nav)

    world_pts = np.array([p.xyz for p in model.points3D.values()])
    stats = {
        "registered": len(frames),
        "frames": n_images,
        "points3d": int(model.num_points3D()),
        "mean_reprojection_error_px": round(float(model.compute_mean_reprojection_error()), 3),
        "mean_track_length": round(float(model.compute_mean_track_length()), 2),
        "path_length_m": round(float(np.linalg.norm(np.diff([c[:3, 3] for c in c2ws], axis=0), axis=1).sum()), 2),
        "scene_extent_m": np.round(np.percentile(world_pts, 98, axis=0) - np.percentile(world_pts, 2, axis=0), 2).tolist(),
    }
    write_json(out / "alignment.json", {**align.to_dict(), "stats": stats, "mapper": cfg.mapper})

    warnings = []
    if align.scale_confidence < 0.4:
        warnings.append(
            f"Metric scale is a rough guess ({align.scale_method}); distances may be off."
        )
    io.progress(1.0, "done")
    return {
        **stats,
        "up_method": align.up_method,
        "scale_method": align.scale_method,
        "scale_confidence": align.scale_confidence,
        "warnings": warnings,
    }
