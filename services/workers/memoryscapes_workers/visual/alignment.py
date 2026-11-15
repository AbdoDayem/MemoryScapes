"""Put an SfM reconstruction into the memory's metric, gravity-aligned world frame.

SfM output has arbitrary rotation, translation and scale. A walkable memory needs:

- **Up = +y**, so the viewer can walk on a floor and the horizon is level.
- **Meters**, so walking speed, eye height and audio distance falloff are believable.
- **Origin on the ground** under the first camera, with that camera looking along -z
  (three.js "forward"), so the spawn point is natural.

Up comes from a RANSAC ground plane (the dominant plane roughly below the cameras),
falling back to the average camera "up" vector (people hold phones upright). Scale comes
from the cameras' height above that plane, assuming a handheld eye height (default
1.5 m). Without a ground plane the scale is a low-confidence guess from scene depth;
``scale_confidence`` records which. A metric-depth estimator can replace this later.

All camera rotations here are camera-to-world in the OpenCV convention
(x right, y down, z forward).
"""

from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np

UP = np.array([0.0, 1.0, 0.0])
# Handheld capture is a walk: average speed and the range we accept from other cues.
WALK_SPEED_MPS = 1.2
PLAUSIBLE_SPEED_MPS = (0.2, 2.5)
CV_TO_GL = np.diag([1.0, -1.0, -1.0])  # OpenCV camera axes -> OpenGL/three.js camera axes


def _normalize(v: np.ndarray) -> np.ndarray:
    n = np.linalg.norm(v)
    return v / n if n > 1e-12 else v


def rotation_between(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Smallest rotation taking unit vector a onto unit vector b."""
    a, b = _normalize(a), _normalize(b)
    v = np.cross(a, b)
    c = float(np.dot(a, b))
    if c < -1 + 1e-9:  # opposite: rotate 180 deg about any axis orthogonal to a
        axis = _normalize(np.cross(a, [1.0, 0, 0] if abs(a[0]) < 0.9 else [0, 1.0, 0]))
        return 2 * np.outer(axis, axis) - np.eye(3)
    vx = np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])
    return np.eye(3) + vx + vx @ vx / (1 + c)


def quat_xyzw(r: np.ndarray) -> list[float]:
    """Rotation matrix -> unit quaternion [x, y, z, w] with w >= 0."""
    m = r
    tr = np.trace(m)
    if tr > 0:
        s = np.sqrt(tr + 1.0) * 2
        w, x, y, z = 0.25 * s, (m[2, 1] - m[1, 2]) / s, (m[0, 2] - m[2, 0]) / s, (m[1, 0] - m[0, 1]) / s
    elif m[0, 0] > m[1, 1] and m[0, 0] > m[2, 2]:
        s = np.sqrt(1.0 + m[0, 0] - m[1, 1] - m[2, 2]) * 2
        w, x, y, z = (m[2, 1] - m[1, 2]) / s, 0.25 * s, (m[0, 1] + m[1, 0]) / s, (m[0, 2] + m[2, 0]) / s
    elif m[1, 1] > m[2, 2]:
        s = np.sqrt(1.0 + m[1, 1] - m[0, 0] - m[2, 2]) * 2
        w, x, y, z = (m[0, 2] - m[2, 0]) / s, (m[0, 1] + m[1, 0]) / s, 0.25 * s, (m[1, 2] + m[2, 1]) / s
    else:
        s = np.sqrt(1.0 + m[2, 2] - m[0, 0] - m[1, 1]) * 2
        w, x, y, z = (m[1, 0] - m[0, 1]) / s, (m[0, 2] + m[2, 0]) / s, (m[1, 2] + m[2, 1]) / s, 0.25 * s
    q = np.array([x, y, z, w])
    q = q / np.linalg.norm(q)
    return (q if q[3] >= 0 else -q).tolist()


@dataclass
class Plane:
    normal: np.ndarray  # unit, oriented toward the cameras
    point: np.ndarray
    inliers: int
    candidates: int


def fit_ground_plane(
    points: np.ndarray,
    cam_centers: np.ndarray,
    approx_up: np.ndarray,
    *,
    max_tilt_deg: float = 30.0,
    iters: int = 400,
    seed: int = 0,
) -> Plane | None:
    """RANSAC the dominant plane below the cameras whose normal is near ``approx_up``."""
    if len(points) < 50:
        return None
    rng = np.random.default_rng(seed)
    cam_h = cam_centers @ approx_up
    cand = points[points @ approx_up < np.median(cam_h)]
    if len(cand) < 50:
        return None
    # Inlier threshold relative to scene size (SfM units are arbitrary).
    extent = np.median(np.linalg.norm(cand - np.median(cam_centers, axis=0), axis=1))
    tau = 0.02 * extent
    cos_tilt = np.cos(np.radians(max_tilt_deg))
    best_n, best_p, best_count = None, None, 0
    for _ in range(iters):
        a, b, c = cand[rng.choice(len(cand), 3, replace=False)]
        n = np.cross(b - a, c - a)
        if np.linalg.norm(n) < 1e-12:
            continue
        n = _normalize(n)
        if n @ approx_up < 0:
            n = -n
        if n @ approx_up < cos_tilt:
            continue
        count = int((np.abs((cand - a) @ n) < tau).sum())
        if count > best_count:
            best_n, best_p, best_count = n, a, count
    if best_n is None or best_count < max(30, 0.05 * len(cand)):
        return None
    # Refine with least squares on the inliers.
    inl = cand[np.abs((cand - best_p) @ best_n) < tau]
    centroid = inl.mean(axis=0)
    _, _, vt = np.linalg.svd(inl - centroid, full_matrices=False)
    n = vt[-1]
    if n @ approx_up < 0:
        n = -n
    heights = (cam_centers - centroid) @ n
    if (heights > 0).mean() < 0.9:  # cameras should be above the floor
        return None
    return Plane(normal=n, point=centroid, inliers=len(inl), candidates=len(cand))


@dataclass
class Alignment:
    """world = scale * rotation @ sfm + translation"""

    rotation: np.ndarray
    scale: float
    translation: np.ndarray
    up_method: str  # "ground_plane" | "camera_up"
    scale_method: str  # "eye_height" | "scene_depth" | "walking_speed" | "fixed"
    scale_confidence: float  # 0..1
    eye_height_m: float
    ground_inlier_ratio: float | None = None
    scale_note: str | None = None  # why a fallback scale was used

    def apply(self, pts: np.ndarray) -> np.ndarray:
        return self.scale * pts @ self.rotation.T + self.translation

    def apply_c2w(self, c2w: np.ndarray) -> np.ndarray:
        """Transform a 4x4 camera-to-world pose (rotation stays orthonormal)."""
        out = np.eye(4)
        out[:3, :3] = self.rotation @ c2w[:3, :3]
        out[:3, 3] = self.apply(c2w[:3, 3][None])[0]
        return out

    def matrix(self) -> np.ndarray:
        """3x4 [sR | t], the form pycolmap.Sim3d accepts."""
        return np.hstack([self.scale * self.rotation, self.translation[:, None]])

    def to_dict(self) -> dict:
        d = asdict(self)
        d["rotation"] = self.rotation.tolist()
        d["translation"] = self.translation.tolist()
        d["matrix_3x4"] = self.matrix().tolist()
        return d


def estimate_alignment(
    cam_centers: np.ndarray,
    cam_c2w_rot: np.ndarray,
    points: np.ndarray,
    *,
    eye_height_m: float = 1.5,
    fixed_scale: float | None = None,
    measured_scale: tuple[float, float, str] | None = None,
    times: np.ndarray | None = None,
    segments: np.ndarray | None = None,
) -> Alignment:
    """Compute the SfM -> world similarity. Cameras must be in capture order.

    With ``times`` (seconds, one per camera) the scale is sanity-checked against walking
    pace: when the eye-height scale implies an implausible speed (the "ground" plane was
    really distant structure, e.g. a reconstruction of far buildings with no floor), the
    scale assumes a WALK_SPEED_MPS walk instead and the level comes from the cameras.
    ``segments`` (one id per camera, e.g. the shot of an edited video) keeps the jumps
    between segments out of the walking-speed estimate.
    ``measured_scale`` = (scale, confidence, method), e.g. from a stereo pair
    (visual/stereo.py), replaces the eye-height and walking-speed guesses; an implausible
    walking pace is only noted. ``fixed_scale`` (set by hand) still wins over it.
    """
    cam_up = _normalize((cam_c2w_rot @ np.array([0.0, -1.0, 0.0])).mean(axis=0))
    plane = fit_ground_plane(points, cam_centers, cam_up)
    note = None

    if plane is not None:
        heights = (cam_centers - plane.point) @ plane.normal
        med_h = float(np.median(heights))
        spread = float(np.std(heights) / max(med_h, 1e-9))
        scale = eye_height_m / med_h
        ratio = plane.inliers / plane.candidates
        # Confident when many points lie on the plane and the camera height is steady.
        confidence = float(np.clip(0.4 + ratio, 0, 0.9) * np.clip(1.5 - 2 * spread, 0.2, 1.0))
        scale_method = "eye_height"
    else:
        # No floor: assume typical handheld subject distance (~3 m). Low confidence.
        d = np.linalg.norm(points[:, None, :] - cam_centers[None, :: max(1, len(cam_centers) // 50)], axis=2)
        scale = 3.0 / max(float(np.median(d.min(axis=1))), 1e-9)
        confidence, ratio, scale_method = 0.15, None, "scene_depth"

    walk = _walking_speed_scale(cam_centers, times, segments)
    if fixed_scale is None and measured_scale is not None:
        scale, confidence, scale_method = measured_scale
        if walk is not None:
            speed = walk[1] * scale / walk[2]
            if not (PLAUSIBLE_SPEED_MPS[0] <= speed <= PLAUSIBLE_SPEED_MPS[1]):
                note = f"{scale_method} scale implies {speed:.2f} m/s over {walk[2]:.0f} s"
    elif fixed_scale is None and walk is not None:
        walk_scale, path_units, span = walk
        speed = path_units * scale / span
        if not (PLAUSIBLE_SPEED_MPS[0] <= speed <= PLAUSIBLE_SPEED_MPS[1]):
            note = (f"{scale_method} scale implied {speed:.2f} m/s over {span:.0f} s; "
                    f"assumed a {WALK_SPEED_MPS} m/s walk instead")
            scale, confidence, scale_method = walk_scale, 0.3, "walking_speed"
            plane = None  # its normal isn't trustworthy either: level from the cameras

    up = plane.normal if plane is not None else cam_up
    if plane is not None:
        origin = cam_centers[0] - ((cam_centers[0] - plane.point) @ plane.normal) * plane.normal
    else:
        origin = cam_centers[0] - (eye_height_m / scale) * up

    if fixed_scale is not None:
        scale, confidence, scale_method = fixed_scale, 1.0, "fixed"
        if plane is None:
            origin = cam_centers[0] - (eye_height_m / scale) * up

    # Level the scene, then yaw so the first camera looks along -z.
    r_level = rotation_between(up, UP)
    fwd = r_level @ cam_c2w_rot[0] @ np.array([0.0, 0.0, 1.0])
    fwd_h = np.array([fwd[0], 0.0, fwd[2]])
    r = r_level
    if np.linalg.norm(fwd_h) > 1e-3:
        r = rotation_between(_normalize(fwd_h), np.array([0.0, 0.0, -1.0])) @ r_level

    t = -scale * r @ origin
    return Alignment(
        rotation=r,
        scale=float(scale),
        translation=t,
        up_method="ground_plane" if plane is not None else "camera_up",
        scale_method=scale_method,
        scale_confidence=round(confidence, 3),
        eye_height_m=eye_height_m,
        ground_inlier_ratio=None if ratio is None else round(float(ratio), 3),
        scale_note=note,
    )


def _walking_speed_scale(cam_centers: np.ndarray, times: np.ndarray | None,
                         segments: np.ndarray | None = None):
    """(scale for a WALK_SPEED_MPS walk, path length in SfM units, seconds), if usable.
    Path and time are summed within segments only."""
    if times is None or len(times) != len(cam_centers) or len(times) < 3:
        return None
    seg = np.zeros(len(times), int) if segments is None else np.asarray(segments)
    path = span = 0.0
    for k in np.unique(seg):
        m = seg == k
        if m.sum() >= 2:
            path += float(np.linalg.norm(np.diff(cam_centers[m], axis=0), axis=1).sum())
            span += float(times[m][-1] - times[m][0])
    if span < 3.0 or path <= 0:
        return None
    return WALK_SPEED_MPS * span / path, path, span


def camera_to_threejs(c2w_cv: np.ndarray) -> tuple[list[float], list[float]]:
    """World-frame OpenCV c2w -> (position, quaternion_xyzw) for a three.js camera."""
    return c2w_cv[:3, 3].tolist(), quat_xyzw(c2w_cv[:3, :3] @ CV_TO_GL)


def navigation_from_trajectory(
    c2w_world: list[np.ndarray],
    *,
    floor_y: float = 0.0,
    margin_m: float = 1.0,
    corridor_radius_m: float = 1.5,
    max_path_points: int = 200,
) -> dict:
    """Spawn pose and walkable region from the capture trajectory (world frame).

    The capture path is where someone physically walked, so it is known to be free
    space. ``walkable_bounds`` is an axis-aligned box around it plus a corridor
    (polyline + radius) the viewer can clamp the camera to.
    """
    centers = np.array([c[:3, 3] for c in c2w_world])
    pos, quat = camera_to_threejs(c2w_world[0])
    fwd = c2w_world[0][:3, :3] @ np.array([0.0, 0.0, 1.0])
    lo = centers.min(axis=0) - margin_m
    hi = centers.max(axis=0) + margin_m
    lo[1] = floor_y
    hi[1] = max(hi[1], centers[:, 1].max() + 0.5)
    step = max(1, len(centers) // max_path_points)
    path = centers[::step]
    if not np.allclose(path[-1], centers[-1]):
        path = np.vstack([path, centers[-1]])
    return {
        "spawn": {
            "position": np.round(pos, 4).tolist(),
            "quaternion_xyzw": np.round(quat, 6).tolist(),
            "look_at": np.round(centers[0] + fwd, 4).tolist(),
        },
        "floor_y": floor_y,
        "eye_height_m": round(float(np.median(centers[:, 1] - floor_y)), 3),
        "walkable_bounds": {
            "type": "aabb+corridor",
            "min": np.round(lo, 3).tolist(),
            "max": np.round(hi, 3).tolist(),
            "corridor": {"radius_m": corridor_radius_m, "points": np.round(path, 3).tolist()},
        },
    }
