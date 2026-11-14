"""Camera poses.

- Quality path: GLOMAP / COLMAP (both through pycolmap).
- Fast path: feed-forward VGGT or MASt3R-SfM (poses + dense points in minutes). Not yet.
- Estimate metric scale (monocular metric depth, or IMU/ARKit later) so audio
  distance attenuation works in meters. For now: eye height above a RANSAC ground plane.

Reads  {memory_id}/ingest/frames/, meta.json
Writes {memory_id}/poses/
    sparse/0/{cameras,images,points3D}.bin   undistorted PINHOLE model, in the world frame
    images/frame_*.jpg                        undistorted images matching sparse/0
    sparse_points.ply                         SfM points (xyz + rgb), world frame
    points3d.npy                              the same points, (N, 3) float32
    points3d_rgb.npy                          their colours, (N, 3) uint8
    cameras.json                              [{frame_id, t, width, height, K, c2w,
                                               image_key}]: OpenCV c2w, t on the audio clock
    path.json                                 capture path for the viewer (three.js camera)
    navigation.json                           spawn, floor, walkable bounds + corridor
    alignment.json                            SfM -> world similarity and how it was found

World frame: meters, +y up, ground at y=0, origin under the first camera, which looks
along -z. Splat, audio sources and geolocation all live in this frame.

cameras.json and points3d.npy are the audio stages' inputs (contract in
services/workers/README.md).
"""

from __future__ import annotations

import logging
import shutil
from pathlib import Path
from typing import Literal

import numpy as np
from pydantic import BaseModel, Field

from memoryscapes_workers.context import StageContext
from memoryscapes_workers.stagekit import StageError, StageIO, run_stage, write_json

STAGE = "poses"
log = logging.getLogger("memoryscapes.poses")
VERSION = "5"  # 2: audio contract; 3: points3d_rgb.npy; 4: focal prior, mapper fallback,
#                sky masks, walking-speed scale check; 5: edited videos (cross-shot)

CAPTURE_HINT = (
    "Move slowly and steadily, walk through the space or orbit the subject instead of "
    "turning in place, and avoid blank walls, mirrors and very dark scenes."
)


class PosesConfig(BaseModel):
    mapper: Literal["global", "incremental"] = "global"
    camera_model: Literal["SIMPLE_RADIAL", "OPENCV", "PINHOLE", "SIMPLE_PINHOLE"] = "SIMPLE_RADIAL"
    # Starting focal length x the long image side when the video doesn't record one.
    # Extracted frames carry no EXIF, and COLMAP's own default (1.2, about 45 degrees) is
    # far narrower than phone cameras (about 0.7-0.8): on wide footage it fails to
    # initialize or breaks the path into pieces. Bundle adjustment refines it either way.
    focal_length_factor: float = Field(0.8, gt=0.2, lt=3)
    # Ignore features on the sky (clouds have no parallax and drift); see visual/sky_mask.py.
    sky_mask: bool = True
    # Edited videos: frames sampled per shot and matched across shots, so shots that see
    # the same scenery can join one reconstruction.
    cross_shot_samples: int = 10
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


def cross_shot_pairs(shot_of: dict[str, int], per_shot: int) -> list[tuple[str, str]]:
    """Image pairs linking different shots: up to ``per_shot`` evenly spaced frames from
    each shot, every combination across each pair of shots."""
    groups: dict[int, list[str]] = {}
    for name in sorted(shot_of):
        groups.setdefault(shot_of[name], []).append(name)
    picks = {}
    for k, names in groups.items():
        idx = np.unique(np.linspace(0, len(names) - 1, min(per_shot, len(names))).round().astype(int))
        picks[k] = [names[i] for i in idx]
    keys = sorted(picks)
    return [(a, b) for i, ka in enumerate(keys) for kb in keys[i + 1:]
            for a in picks[ka] for b in picks[kb]]


def _sfm(io: StageIO, cfg: PosesConfig, image_dir: Path, n_images: int, focal_factor: float,
         shot_of: dict[str, int] | None = None):
    import pycolmap

    work = io.work_dir / "colmap"
    work.mkdir()
    db = work / "database.db"
    device = pycolmap.Device.auto if cfg.use_gpu else pycolmap.Device.cpu

    extraction = pycolmap.FeatureExtractionOptions()
    extraction.sift.max_num_features = cfg.max_num_features
    reader = pycolmap.ImageReaderOptions(camera_model=cfg.camera_model,
                                         default_focal_length_factor=focal_factor)
    if cfg.sky_mask:
        from memoryscapes_workers.visual.sky_mask import write_masks

        sky = write_masks(image_dir, work / "masks")
        reader.mask_path = str(work / "masks")
        log.info("[%s] sky masks: %.0f%% of pixels on average", STAGE, 100 * sky)
    pycolmap.extract_features(
        db, image_dir,
        camera_mode=pycolmap.CameraMode.SINGLE,
        reader_options=reader,
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
    n_shots = len(set((shot_of or {}).values()))
    if n_shots > 1 and cfg.matcher != "exhaustive":
        pairs = cross_shot_pairs(shot_of, cfg.cross_shot_samples)
        pair_file = work / "cross_shot_pairs.txt"
        pair_file.write_text("\n".join(f"{a} {b}" for a, b in pairs) + "\n", encoding="utf-8")
        pycolmap.match_image_pairs(db, pairing_options=pycolmap.ImportedPairingOptions(
            match_list_path=pair_file), device=device)
        log.info("[%s] %d shots: matched %d cross-shot pairs", STAGE, n_shots, len(pairs))
    io.progress(0.4, "features matched")

    needed = max(10, cfg.min_registered_fraction * n_images)
    if n_shots > 1:
        # An edited video can't always be joined: accept the largest linked part if it
        # covers at least half of the longest shot (the stage then warns what's left out).
        longest = max(list(shot_of.values()).count(k) for k in set(shot_of.values()))
        needed = max(10, min(needed, 0.5 * longest))

    def mapping(kind: str):
        sparse = work / f"sparse_{kind}"
        sparse.mkdir()
        try:
            if kind == "global":
                recs = pycolmap.global_mapping(db, image_dir, sparse)
            else:
                recs = pycolmap.incremental_mapping(db, image_dir, sparse)
        except Exception as err:  # noqa: BLE001 - a failed mapper falls back below
            log.warning("[%s] %s mapping raised: %s", STAGE, kind, err)
            recs = {}
        rec = _largest(recs)
        return rec, (rec.num_reg_images() if rec is not None else 0), len(recs)

    used = cfg.mapper
    rec, registered, pieces = mapping(used)
    if registered < needed and used == "global":
        # Global SfM is fast but fragile on some footage (e.g. a gimbal walking straight
        # ahead); incremental is slower and more robust. Same database, no re-extraction.
        log.warning("[%s] global mapping placed %d/%d frames; retrying incrementally",
                    STAGE, registered, n_images)
        io.progress(0.45, "global mapping failed; trying incremental")
        used = "incremental"
        rec, registered, pieces = mapping(used)
    if registered < needed:
        raise StageError(
            STAGE,
            f"Could not recover the camera path: only {registered} of {n_images} frames "
            f"were placed ({pieces} disconnected pieces).",
            CAPTURE_HINT,
        )
    io.progress(0.7, f"registered {registered}/{n_images} frames ({used} mapping)")
    return rec, work, used


def shot_coverage(meta: dict, placed: set[str]) -> list[dict]:
    """Per shot: its time span and the fraction of its keyframes the reconstruction placed."""
    out = []
    for k, shot in enumerate(meta.get("shots") or []):
        names = [Path(f["file"]).name for f in meta["keyframes"]["frames"] if f.get("shot", 0) == k]
        if names:
            out.append({**shot, "frames": len(names), "placed": sum(n in placed for n in names) / len(names)})
    return out


def focal_prior(keyframes: dict, cfg: PosesConfig) -> dict:
    """Starting focal length for SfM, as a factor of the long image side."""
    if keyframes.get("focal_px"):
        long_side = max(keyframes["width"], keyframes["height"])
        return {"source": "video metadata", "factor": round(keyframes["focal_px"] / long_side, 4)}
    return {"source": "default", "factor": cfg.focal_length_factor}


def _poses(io: StageIO, cfg: PosesConfig) -> dict:
    import pycolmap

    from memoryscapes_workers.visual import alignment as al

    ingest_dir = io.input_dir("ingest")
    meta = io.read_upstream_json("ingest", "meta.json")
    frame_times = {Path(f["file"]).name: f["t"] for f in meta["keyframes"]["frames"]}
    image_dir = ingest_dir / "frames"
    n_images = len(frame_times)

    focal = focal_prior(meta["keyframes"], cfg)
    shot_of = {Path(f["file"]).name: f.get("shot", 0) for f in meta["keyframes"]["frames"]}
    rec, work, mapper_used = _sfm(io, cfg, image_dir, n_images, focal["factor"], shot_of)

    # ---- gravity, metric scale, origin
    images = sorted((im for im in rec.images.values() if im.has_pose), key=lambda im: im.name)
    centers = np.array([im.projection_center() for im in images])
    rots = np.array([im.cam_from_world().rotation.matrix().T for im in images])
    pts = np.array([p.xyz for p in rec.points3D.values()])
    times = np.array([frame_times[im.name] for im in images], float)
    shot_ids = np.array([shot_of.get(im.name, 0) for im in images])
    align = al.estimate_alignment(centers, rots, pts, eye_height_m=cfg.eye_height_m, times=times,
                                  segments=shot_ids,
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
        frames.append({
            "frame_id": Path(im.name).stem,
            "t": float(frame_times[im.name]),
            "width": int(cam.width), "height": int(cam.height),
            "K": np.round(k, 6).tolist(),
            "c2w": np.round(c2w, 6).tolist(),
            # The undistorted image, which is what K describes.
            "image_key": io.ctx.key(STAGE, f"images/{im.name}"),
        })
    write_json(out / "cameras.json", frames)
    path = []
    for f, c2w in zip(frames, c2ws, strict=True):
        pos, quat = al.camera_to_threejs(c2w)
        path.append({"t": f["t"], "position": np.round(pos, 4).tolist(),
                     "quaternion_xyzw": np.round(quat, 6).tolist()})
    write_json(out / "path.json", {
        "convention": "three.js camera (looks along -z, +y up); world meters",
        "fov_y_deg": float(np.degrees(2 * np.arctan(frames[0]["height"] / (2 * frames[0]["K"][1][1])))),
        "samples": path,
    })
    nav = al.navigation_from_trajectory(c2ws, corridor_radius_m=cfg.corridor_radius_m)
    nav["original_path"] = "path.json"
    write_json(out / "navigation.json", nav)

    world_pts = np.array([p.xyz for p in model.points3D.values()])
    np.save(out / "points3d.npy", world_pts.astype(np.float32))
    np.save(out / "points3d_rgb.npy", np.array([p.color for p in model.points3D.values()], np.uint8))
    stats = {
        "registered": len(frames),
        "frames": n_images,
        "points3d": int(model.num_points3D()),
        "mean_reprojection_error_px": round(float(model.compute_mean_reprojection_error()), 3),
        "mean_track_length": round(float(model.compute_mean_track_length()), 2),
        "path_length_m": round(float(np.linalg.norm(np.diff([c[:3, 3] for c in c2ws], axis=0), axis=1).sum()), 2),
        "scene_extent_m": np.round(np.percentile(world_pts, 98, axis=0) - np.percentile(world_pts, 2, axis=0), 2).tolist(),
    }
    coverage = shot_coverage(meta, {im.name for im in images})
    write_json(out / "alignment.json", {**align.to_dict(), "stats": stats, "mapper": mapper_used,
                                        "shots": coverage,
                                        "focal_prior": focal})

    warnings = []
    if len(coverage) > 1 and any(c["placed"] < 0.5 for c in coverage):
        kept = [c for c in coverage if c["placed"] >= 0.5]
        spans = ", ".join(f"{c['start_s']:.0f}-{c['end_s']:.0f} s" for c in kept)
        warnings.append(f"Only part of this edited video could be joined into one space ({spans}); "
                        "the other shots don't share enough scenery with it and are left out.")
    if align.scale_confidence < 0.4:
        warnings.append(
            f"Metric scale is a rough guess ({align.scale_method}); distances may be off."
            + (f" {align.scale_note[0].upper()}{align.scale_note[1:]}." if align.scale_note else "")
        )
    io.progress(1.0, "done")
    return {
        **stats,
        "up_method": align.up_method,
        "scale_method": align.scale_method,
        "scale_confidence": align.scale_confidence,
        "warnings": warnings,
    }
