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
from memoryscapes_workers.audio.fusion import Candidate, FusionConfig, cluster_candidates, fuse, resolve_stacks
from memoryscapes_workers.audio.vocabulary import BY_LABEL
from memoryscapes_workers.context import StageContext
from memoryscapes_workers.models import Camera, LocalizedSource, MicInfo, SceneBounds, StemInfo, StemsIndex
from memoryscapes_workers.storage import read_json, read_npy, read_wav, write_json
from memoryscapes_workers.vision.grounding import Detection, Detector, lift_box, touches_border


@dataclass(frozen=True)
class LocalizationConfig:
    max_frames: int = 12
    min_frame_activity: float = 0.3
    det_threshold: float = 0.3
    cluster_radius_m: float = 1.5
    # Lifting a box gets less precise with distance, so far sources cluster more loosely.
    cluster_radius_rel: float = 0.2
    default_stereo_spacing_m: float = 0.12  # phone top/bottom mics; override via meta.json
    fusion: FusionConfig = FusionConfig()
    # Stems placed closer than this are on one spot (see fusion.resolve_stacks).
    stack_radius_m: float = 0.75
    # Speech is narration (no position: heard everywhere at its recorded level) unless the
    # evidence that it comes from one visible spot is strong. Whoever is filming talks from
    # right at the moving mic, which can't be triangulated, and a person in frame may not
    # be the one talking. Pinned to the wrong spot, a voice faded as the visitor walked
    # away (an iPhone clip: the filmer's voice placed on a bystander 4 m away, conf 0.68).
    unseen_speech_as_narration: bool = True
    speech_place_min_confidence: float = 0.75
    # Footsteps in a handheld walk are the filmer's own, under the moving mic. Placed on one
    # spot (a Tampere underpass: 36 m down the tunnel, conf 0.57) they fade as the visitor
    # walks on. Like speech they need strong evidence of one visible spot; otherwise they're
    # diffuse (heard everywhere). Empty to place footsteps like any other sound.
    filmer_labels: tuple[str, ...] = ("footsteps",)
    # Scene points this far under the walkable floor (poses/navigation.json floor_y) are left
    # out of placement: on a shiny floor they're mirror images of the scene above. A Tampere
    # underpass (wet floor) had points 10 m underground, and a train placed on them.
    below_floor_m: float = 0.5


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
    points = _above_floor(storage, ctx, points, cfg.below_floor_m)
    mic = MicInfo.from_json(read_json(storage, ctx.key("ingest", "meta.json"))["mic"])

    cam_positions = np.stack([c.position for c in cameras])
    bounds = SceneBounds.from_points(np.concatenate([points, cam_positions]) if len(points) else cam_positions)
    path = doa_mod.CameraPath(cameras)
    candidates = doa_mod.candidate_points(points, np.array(bounds.min), np.array(bounds.max))
    use_doa = mic.layout == "stereo" and index.channels >= 2
    spacing = mic.spacing_m or cfg.default_stereo_spacing_m

    detection_cache: dict[tuple[str, str], list[Detection]] = {}
    # Every stem's placement first, then stacks resolved across stems, then the output.
    entries: list[tuple[StemInfo, Candidate | None]] = []
    for stem in index.stems:
        cls = BY_LABEL.get(stem.label)
        activity = lambda t, s=stem: index.activity_at(s, t)  # noqa: E731

        if cls is None or cls.diffuse:
            entries.append((stem, None))
            continue

        # 1. audio-visual candidates
        clusters = []
        radius = cfg.cluster_radius_m
        if detector is not None and cls.detect_prompt:
            pts, weights, dists = _av_candidates(storage, stem, cameras, points, cls.detect_prompt, activity, detector, detection_cache, cfg)
            if len(dists):
                radius = max(cfg.cluster_radius_m, cfg.cluster_radius_rel * float(np.median(dists)))
            clusters = cluster_candidates(pts, weights, radius)

        # 2. stereo DOA
        doa_scorer, doa_est = None, None
        if use_doa:
            audio, sr = read_wav(storage, stem.key)
            bearings = doa_mod.stereo_bearings(audio[:2], sr, spacing, activity)
            if bearings:
                doa_scorer = lambda p, b=bearings: doa_mod.bearing_likelihood(p, b, path)  # noqa: E731
                doa_est = doa_mod.triangulate(candidates, doa_scorer(candidates))

        carried = _carried(stem, cfg)
        if carried and not clusters:
            entries.append((stem, None))  # narration or diffuse (see _diffuse)
            continue

        agree = max(2.0, 1.5 * radius)
        placement = fuse(clusters, doa_scorer, doa_est, cfg.fusion, agree_radius_m=agree)
        if carried and placement.confidence < cfg.speech_place_min_confidence:
            entries.append((stem, None))  # not sure enough where it is: heard everywhere
            continue
        entries.append((stem, Candidate(stem.id, placement, clusters, doa_scorer, doa_est, agree)))

    # 3. stems piled onto one point: move apart, or share the point as one object
    resolve_stacks([c for _, c in entries if c is not None], cfg.fusion, cfg.stack_radius_m)

    sources: list[LocalizedSource] = []
    for stem, cand in entries:
        if cand is None or cand.placement.position is None:
            narration = stem.kind == "speech" and cfg.unseen_speech_as_narration
            sources.append(_diffuse(stem, "narration" if narration else "diffuse"))
            continue
        placement, pos = cand.placement, cand.placement.position
        activity = lambda t, s=stem: index.activity_at(s, t)  # noqa: E731
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
                shares_with=cand.shares_with,
            )
        )

    result = {"bounds": {"min": bounds.min, "max": bounds.max}, "sources": [s.to_json() for s in sources]}
    write_json(storage, ctx.key("localize", "sources.json"), result)
    return result


def _av_candidates(storage, stem: StemInfo, cameras, points, prompt, activity, detector, cache, cfg):
    ranked = sorted(cameras, key=lambda c: activity(c.t), reverse=True)
    frames = [c for c in ranked if activity(c.t) >= cfg.min_frame_activity and c.image_key][: cfg.max_frames]
    pts, weights, dists = [], [], []
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
                # a box cut off by the frame edge only shows part of the object
                truncated = 0.5 if touches_border(det.box, cam) else 1.0
                weights.append(det.score * activity(cam.t) * truncated)
                dists.append(float(np.linalg.norm(p - cam.position)))
    if not pts:
        return np.zeros((0, 3)), np.zeros(0), np.zeros(0)
    return np.stack(pts), np.array(weights), np.array(dists)


def _mean_distance(pos: np.ndarray, cameras: list[Camera], activity) -> float:
    w = np.array([max(activity(c.t), 1e-3) for c in cameras])
    d = np.array([np.linalg.norm(c.position - pos) for c in cameras])
    return float(max((w * d).sum() / w.sum(), 0.5))


def _above_floor(storage, ctx: StageContext, points: np.ndarray, below_m: float) -> np.ndarray:
    """Points no lower than below_m under the walkable floor, when poses wrote one."""
    key = ctx.key("poses", "navigation.json")
    if not len(points) or not storage.exists(key):
        return points
    floor_y = read_json(storage, key).get("floor_y")
    if floor_y is None:
        return points
    return points[points[:, 1] >= float(floor_y) - below_m]


def _carried(stem: StemInfo, cfg: LocalizationConfig) -> bool:
    """Sounds that usually come from whoever is filming: unplaced unless clearly seen."""
    return ((stem.kind == "speech" and cfg.unseen_speech_as_narration)
            or stem.label in cfg.filmer_labels)


def _diffuse(stem: StemInfo, method: str = "diffuse") -> LocalizedSource:
    """No position: heard everywhere. "narration" is speech no frame shows (the person
    filming, usually); "diffuse" is a sound with no single place (wind, rain, a crowd)."""
    return LocalizedSource(
        id=stem.id,
        label=stem.label,
        kind=stem.kind,
        stem_key=stem.key,
        position=None,
        spread_m=None,
        confidence=0.0,
        method=method,
        level_dbfs=stem.level_dbfs,
        level_db_at_1m=None,
    )
