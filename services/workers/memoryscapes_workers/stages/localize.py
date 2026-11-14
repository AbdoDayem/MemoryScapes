"""Audio source localization in the splat's world frame.

Inputs:
    separate/stems.json, separate/stems/*.wav
    poses/cameras.json    list of Camera (metric, OpenCV convention, see models.py)
    poses/points3d.npy    (N, 3) scene points in world metres
    ingest/meta.json      mic layout and spacing

Output:
    localize/sources.json  {"bounds": SceneBounds, "sources": [LocalizedSource]}

Per stem, in order of preference:
1. Audio-visual grounding: detect the class's visual prompt in the frames where the stem
   is loudest, lift each detection to 3D, cluster, keep the heaviest cluster.
2. Stereo DOA: bearings per window, triangulated along the camera path. Used to choose
   between AV clusters, or alone when nothing was detected.
3. Diffuse: nothing trustworthy, or a class with no location (wind, rain).
Moving sources are placed at their dominant location for now (trajectory = None).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from memoryscapes_workers.audio import doa as doa_mod
from memoryscapes_workers.audio.fusion import FusionConfig, cluster_candidates, fuse
from memoryscapes_workers.audio.vocabulary import BY_LABEL
from memoryscapes_workers.context import StageContext
from memoryscapes_workers.models import Camera, LocalizedSource, MicInfo, SceneBounds, StemInfo, StemsIndex
from memoryscapes_workers.storage import read_json, read_npy, read_wav, write_json
from memoryscapes_workers.vision.grounding import Detection, Detector, lift_box


@dataclass(frozen=True)
class LocalizationConfig:
    max_frames: int = 12
    min_frame_activity: float = 0.3
    det_threshold: float = 0.3
    cluster_radius_m: float = 1.5
    default_stereo_spacing_m: float = 0.12  # phone top/bottom mics; override via meta.json
    fusion: FusionConfig = FusionConfig()


_DEFAULT = object()


def run(ctx: StageContext, detector: Detector | None = _DEFAULT, cfg: LocalizationConfig = LocalizationConfig()) -> dict:
    """detector=None disables audio-visual grounding; omit it to use the configured one."""
    from memoryscapes_workers import backends

    if detector is _DEFAULT:
        detector = backends.get_detector()
    storage = ctx.storage

    index = StemsIndex.from_json(read_json(storage, ctx.key("separate", "stems.json")))
    cameras = [Camera.from_json(c) for c in read_json(storage, ctx.key("poses", "cameras.json"))]
    points = read_npy(storage, ctx.key("poses", "points3d.npy")).astype(float)
    mic = MicInfo.from_json(read_json(storage, ctx.key("ingest", "meta.json"))["mic"])

    cam_positions = np.stack([c.position for c in cameras])
    bounds = SceneBounds.from_points(np.concatenate([points, cam_positions]) if len(points) else cam_positions)
    path = doa_mod.CameraPath(cameras)
    candidates = doa_mod.candidate_points(points, np.array(bounds.min), np.array(bounds.max))
    use_doa = mic.layout == "stereo" and index.channels >= 2
    spacing = mic.spacing_m or cfg.default_stereo_spacing_m

    detection_cache: dict[tuple[str, str], list[Detection]] = {}
    sources: list[LocalizedSource] = []
    for stem in index.stems:
        cls = BY_LABEL.get(stem.label)
        activity = lambda t, s=stem: index.activity_at(s, t)  # noqa: E731

        if cls is None or cls.diffuse:
            sources.append(_diffuse(stem))
            continue

        # 1. audio-visual candidates
        clusters = []
        if detector is not None and cls.detect_prompt:
            pts, weights = _av_candidates(storage, stem, cameras, points, cls.detect_prompt, activity, detector, detection_cache, cfg)
            clusters = cluster_candidates(pts, weights, cfg.cluster_radius_m)

        # 2. stereo DOA
        doa_scorer, doa_est = None, None
        if use_doa:
            audio, sr = read_wav(storage, stem.key)
            bearings = doa_mod.stereo_bearings(audio[:2], sr, spacing, activity)
            if bearings:
                doa_scorer = lambda p, b=bearings: doa_mod.bearing_likelihood(p, b, path)  # noqa: E731
                doa_est = doa_mod.triangulate(candidates, doa_scorer(candidates))

        placement = fuse(clusters, doa_scorer, doa_est, cfg.fusion)
        if placement.position is None:
            sources.append(_diffuse(stem))
            continue

        pos = placement.position
        sources.append(
            LocalizedSource(
                id=stem.id,
                label=stem.label,
                kind=stem.kind,
                stem_key=stem.key,
                position=np.round(pos, 3).tolist(),
                spread_m=round(float(placement.spread_m or 0.0), 3),
                confidence=placement.confidence,
                method=placement.method,
                level_dbfs=stem.level_dbfs,
                level_db_at_1m=round(stem.level_dbfs + 20 * np.log10(_mean_distance(pos, cameras, activity)), 2),
            )
        )

    result = {"bounds": {"min": bounds.min, "max": bounds.max}, "sources": [s.to_json() for s in sources]}
    write_json(storage, ctx.key("localize", "sources.json"), result)
    return result


def _av_candidates(storage, stem: StemInfo, cameras, points, prompt, activity, detector, cache, cfg):
    ranked = sorted(cameras, key=lambda c: activity(c.t), reverse=True)
    frames = [c for c in ranked if activity(c.t) >= cfg.min_frame_activity and c.image_key][: cfg.max_frames]
    pts, weights = [], []
    for cam in frames:
        cache_key = (cam.frame_id, prompt)
        if cache_key not in cache:
            cache[cache_key] = detector.detect(storage.read_bytes(cam.image_key), prompt)
        for det in cache[cache_key]:
            if det.score < cfg.det_threshold:
                continue
            p = lift_box(det.box, cam, points)
            if p is not None:
                pts.append(p)
                weights.append(det.score * activity(cam.t))
    if not pts:
        return np.zeros((0, 3)), np.zeros(0)
    return np.stack(pts), np.array(weights)


def _mean_distance(pos: np.ndarray, cameras: list[Camera], activity) -> float:
    w = np.array([max(activity(c.t), 1e-3) for c in cameras])
    d = np.array([np.linalg.norm(c.position - pos) for c in cameras])
    return float(max((w * d).sum() / w.sum(), 0.5))


def _diffuse(stem: StemInfo) -> LocalizedSource:
    return LocalizedSource(
        id=stem.id,
        label=stem.label,
        kind=stem.kind,
        stem_key=stem.key,
        position=None,
        spread_m=None,
        confidence=0.0,
        method="diffuse",
        level_dbfs=stem.level_dbfs,
        level_db_at_1m=None,
    )
