"""Splat cleanup and LODs for a scene people walk through (numpy only).

Trained splats look fine from the training views but show artifacts once you move
off the capture path. The cleanup targets what is most visible while exploring:

1. Non-finite or near-transparent gaussians.
2. **Path floaters:** gaussians in the space the camera physically passed through
   (it was empty air) are almost always optimization noise hanging in front of a view.
3. **Oversized blobs and needles:** the size allowed grows with distance from the
   path, so distant background may stay coarse but nothing big smears past your face.
4. **Sparse outliers** near the walkable area (isolated specks with no neighbors).
5. A generous **crop** to the region around the trajectory (drops far-away junk but
   keeps the background you will see while moving).

LODs keep the most important gaussians (opacity x projected area) and truncate SH
bands, ordered from most detailed (lod0) to coarsest.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from pydantic import BaseModel

from memoryscapes_workers.visual.splat_io import GaussianCloud, sigmoid


class CleanupConfig(BaseModel):
    min_opacity: float = 0.01
    path_clear_radius_m: float = 0.3
    # Max allowed size (largest axis, meters) = base + per_m * distance to path.
    max_scale_base_m: float = 0.15
    max_scale_per_m: float = 0.12
    needle_ratio: float = 25.0  # largest / middle axis
    needle_min_size_m: float = 0.08
    # Voxel for the isolated-speck test: at least this, and at least 2x the cloud's
    # typical neighbor spacing, so sparse clouds aren't mistaken for outliers.
    outlier_voxel_m: float = 0.15
    outlier_min_neighbors: int = 3
    outlier_radius_m: float = 8.0  # only check near the walkable area
    crop_radius_m: float | None = None  # None: derived from the SfM points
    crop_min_radius_m: float = 25.0


class LodSpec(BaseModel):
    fraction: float = 1.0
    sh_degree: int = 3
    max_count: int | None = None


DEFAULT_LODS = [
    LodSpec(fraction=1.0, sh_degree=3, max_count=1_500_000),
    LodSpec(fraction=0.4, sh_degree=1, max_count=500_000),
    LodSpec(fraction=0.12, sh_degree=0, max_count=150_000),
]


def distance_to_points(query: np.ndarray, refs: np.ndarray, chunk: int = 65536) -> np.ndarray:
    """Distance from each query point to the nearest ref point (refs is small: cameras)."""
    out = np.empty(len(query), np.float32)
    refs = refs.astype(np.float32)
    for i in range(0, len(query), chunk):
        q = query[i : i + chunk].astype(np.float32)
        d2 = (q * q).sum(1)[:, None] - 2 * q @ refs.T + (refs * refs).sum(1)[None]
        out[i : i + chunk] = np.sqrt(np.maximum(d2.min(axis=1), 0))
    return out


def knn_mean_dist_np(points: np.ndarray, k: int = 3, max_ref: int = 20_000, seed: int = 0) -> np.ndarray:
    """Mean distance to the k nearest neighbors (approximate above ``max_ref`` points)."""
    n = len(points)
    rng = np.random.default_rng(seed)
    ref = points if n <= max_ref else points[rng.choice(n, max_ref, replace=False)]
    out = np.empty(n, np.float32)
    rr = (ref * ref).sum(1)
    for i in range(0, n, 2048):
        q = points[i : i + 2048]
        d2 = np.maximum((q * q).sum(1)[:, None] - 2 * q @ ref.T + rr[None], 0)
        part = np.sort(np.partition(d2, k, axis=1)[:, : k + 1], axis=1)[:, 1:]  # skip self
        out[i : i + 2048] = np.sqrt(part).mean(axis=1)
    if n > max_ref:  # sparser reference set -> distances too large by ~ (n/max_ref)^(1/3)
        out /= (n / max_ref) ** (1 / 3)
    return out



def _voxel_neighbor_counts(points: np.ndarray, voxel: float) -> np.ndarray:
    """Number of points in each point's 3x3x3 voxel neighborhood (itself included)."""
    keys = np.floor(points / voxel).astype(np.int64)
    keys -= keys.min(axis=0)
    dims = keys.max(axis=0) + 3
    lin = lambda k: (k[:, 0] * dims[1] + k[:, 1]) * dims[2] + k[:, 2]  # noqa: E731
    uniq, counts = np.unique(lin(keys + 1), return_counts=True)
    total = np.zeros(len(points), np.int64)
    for dx in (-1, 0, 1):
        for dy in (-1, 0, 1):
            for dz in (-1, 0, 1):
                probe = lin(keys + 1 + np.array([dx, dy, dz]))
                idx = np.clip(np.searchsorted(uniq, probe), 0, len(uniq) - 1)
                total += np.where(uniq[idx] == probe, counts[idx], 0)
    return total


@dataclass
class CleanupReport:
    input: int
    removed: dict[str, int]
    output: int
    crop_radius_m: float

    def to_dict(self) -> dict:
        return {"input": self.input, "output": self.output, "removed": self.removed,
                "crop_radius_m": round(self.crop_radius_m, 2)}


def cleanup(
    cloud: GaussianCloud,
    cam_centers: np.ndarray,
    cfg: CleanupConfig,
    sfm_points: np.ndarray | None = None,
) -> tuple[GaussianCloud, CleanupReport]:
    n0 = len(cloud)
    removed: dict[str, int] = {}

    def apply(name: str, keep: np.ndarray) -> None:
        nonlocal cloud
        removed[name] = int((~keep).sum())
        cloud = cloud.subset(keep)

    before = len(cloud)
    cloud = cloud.finite()
    removed["non_finite"] = before - len(cloud)
    apply("transparent", sigmoid(cloud.opacities) >= cfg.min_opacity)

    dist = distance_to_points(cloud.means, cam_centers)
    if cfg.crop_radius_m is not None:
        radius = cfg.crop_radius_m
    elif sfm_points is not None and len(sfm_points):
        radius = max(cfg.crop_min_radius_m,
                     1.5 * float(np.percentile(distance_to_points(sfm_points, cam_centers), 98)))
    else:
        radius = cfg.crop_min_radius_m * 2
    keep = dist <= radius
    apply("outside_crop", keep)
    dist = dist[keep]

    keep = dist > cfg.path_clear_radius_m
    apply("path_floaters", keep)
    dist = dist[keep]

    s = np.sort(np.exp(cloud.scales), axis=1)  # ascending: min, mid, max
    allowed = cfg.max_scale_base_m + cfg.max_scale_per_m * dist
    oversized = s[:, 2] > allowed
    needle = (s[:, 2] / np.maximum(s[:, 1], 1e-9) > cfg.needle_ratio) & (s[:, 2] > cfg.needle_min_size_m)
    keep = ~(oversized | needle)
    removed["oversized"] = int(oversized.sum())
    removed["needles"] = int((needle & ~oversized).sum())
    cloud = cloud.subset(keep)
    dist = dist[keep]

    near = dist < cfg.outlier_radius_m
    if near.any():
        near_pts = cloud.means[near]
        sample = near_pts[np.random.default_rng(0).choice(len(near_pts), min(5000, len(near_pts)), replace=False)]
        spacing = float(np.median(knn_mean_dist_np(sample, k=3))) if len(sample) > 3 else 0.0
        counts = _voxel_neighbor_counts(near_pts, max(cfg.outlier_voxel_m, 2.0 * spacing))
        isolated = np.zeros(len(cloud), bool)
        isolated[np.flatnonzero(near)] = counts < cfg.outlier_min_neighbors
        apply("isolated", ~isolated)
    return cloud, CleanupReport(input=n0, removed=removed, output=len(cloud), crop_radius_m=radius)


def importance(cloud: GaussianCloud) -> np.ndarray:
    """Opacity x projected area (product of the two largest axes)."""
    s = np.sort(np.exp(cloud.scales), axis=1)
    return sigmoid(cloud.opacities) * s[:, 1] * s[:, 2]


def make_lod(cloud: GaussianCloud, spec: LodSpec) -> GaussianCloud:
    count = int(round(len(cloud) * spec.fraction))
    if spec.max_count is not None:
        count = min(count, spec.max_count)
    count = max(1, min(count, len(cloud)))
    if count < len(cloud):
        idx = np.argpartition(-importance(cloud), count - 1)[:count]
        cloud = cloud.subset(np.sort(idx))
    return cloud.with_sh_degree(spec.sh_degree)


def bounds(cloud: GaussianCloud) -> dict:
    lo = np.percentile(cloud.means, 1, axis=0)
    hi = np.percentile(cloud.means, 99, axis=0)
    return {"min": np.round(lo, 3).tolist(), "max": np.round(hi, 3).tolist(),
            "center": np.round((lo + hi) / 2, 3).tolist()}
