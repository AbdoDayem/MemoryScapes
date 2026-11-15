"""True (metric) scale from a stereo pair: an iPhone spatial video's two eyes.

COLMAP reconstructs the walk from the left eye up to an unknown scale. For COLMAP's own
feature points, the disparity between the left and right eye at the same instant is
``d = f * B / z`` (f: focal length in pixels, B: distance between the lenses, z: depth in
metres). Their COLMAP depths are ``z / s``, with s the metric scale wanted. So across all
points, d is a straight line in 1/z_colmap with slope ``f * B / s``. The intercept absorbs
any constant horizontal shift between the views.

An iPhone spatial video's two eyes are 17.7 mm apart and already rectified to within about
1.5 px vertically (a small vertical search covers the rest). One pixel of disparity is
then about 23 m of depth at 1600 px wide, so points nearer than ~15 m carry the scale.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class StereoScale:
    scale: float  # metres per COLMAP unit
    confidence: float  # 0..1
    points: int  # points the fit used
    frames: int
    disparity_offset_px: float  # the fit's intercept (a constant shift between the views)
    spread: float  # robust relative spread of the per-point scale estimates

    def to_dict(self) -> dict:
        return {"scale": round(self.scale, 6), "confidence": round(self.confidence, 3),
                "points": self.points, "frames": self.frames,
                "disparity_offset_px": round(self.disparity_offset_px, 3), "spread": round(self.spread, 3)}


def measure_disparity(left: np.ndarray, right: np.ndarray, points: np.ndarray, *, max_disp: int = 48,
                      patch: int = 11, dy: int = 3, min_score: float = 0.85) -> tuple[np.ndarray, np.ndarray]:
    """Disparity (left x - right x, px, sub-pixel) of each (x, y) point of the left image,
    found by normalised cross-correlation along the same row (+-dy rows) of the right image.
    Returns (disparity, score); disparity is NaN where the match is weak or ambiguous."""
    import cv2

    h, w = left.shape
    r = patch // 2
    disp = np.full(len(points), np.nan)
    score = np.zeros(len(points))
    for k, (x, y) in enumerate(np.round(points).astype(int)):
        if x - r - max_disp - 2 < 0 or x + r + 2 >= w or y - r - dy < 0 or y + r + dy >= h:
            continue
        tpl = left[y - r: y + r + 1, x - r: x + r + 1]
        if tpl.std() < 2.0:  # flat patch: no reliable match
            continue
        win = right[y - r - dy: y + r + dy + 1, x - r - max_disp: x + r + 3]  # disparity -2..max_disp
        res = cv2.matchTemplate(win, tpl, cv2.TM_CCOEFF_NORMED)  # [2dy+1, max_disp+3]
        row, col = np.unravel_index(int(np.argmax(res)), res.shape)
        best = float(res[row, col])
        if best < min_score or col == 0 or col == res.shape[1] - 1:
            continue
        a, b, c = res[row, col - 1], res[row, col], res[row, col + 1]
        denom = a - 2 * b + c
        sub = 0.5 * (a - c) / denom if denom < 0 else 0.0
        # col 0 is the right-image window's start (disparity max_disp); x decreases with d
        disp[k] = max_disp - (col + sub)
        score[k] = best
    return disp, score


def fit_scale(disparity: np.ndarray, z_colmap: np.ndarray, fx: np.ndarray, baseline_m: float,
              min_disp: float = 1.0) -> StereoScale | None:
    """Robust line fit of disparity against fx * B / z_colmap (one row per point):
    d = (fx B / z_colmap) / s + d0. Returns None without enough usable points."""
    ok = np.isfinite(disparity) & (z_colmap > 0) & (disparity >= min_disp)
    if ok.sum() < 30:
        return None
    u = fx[ok] * baseline_m / z_colmap[ok]  # disparity the point would have at scale 1
    d = disparity[ok]
    rng = np.random.default_rng(0)
    best = None
    for _ in range(300):  # RANSAC on (1/s, d0), then least squares on the inliers
        i, j = rng.choice(len(u), 2, replace=False)
        if abs(u[i] - u[j]) < 1e-9:
            continue
        inv_s = (d[i] - d[j]) / (u[i] - u[j])
        d0 = d[i] - inv_s * u[i]
        if inv_s <= 0:
            continue
        inl = np.abs(d - (inv_s * u + d0)) < 0.5
        if best is None or inl.sum() > best.sum():
            best = inl
    if best is None or best.sum() < 20:
        return None
    A = np.stack([u[best], np.ones(best.sum())], axis=1)
    (inv_s, d0), *_ = np.linalg.lstsq(A, d[best], rcond=None)
    if inv_s <= 0:
        return None
    per_point = u[best] / np.maximum(d[best] - d0, 1e-6)  # each point's own scale estimate
    med = float(np.median(per_point))
    spread = float(np.median(np.abs(per_point - med)) / med) if med > 0 else 1.0
    n = int(best.sum())
    confidence = float(np.clip(0.95 - 1.5 * spread, 0.3, 0.95) * min(1.0, n / 300))
    return StereoScale(scale=float(1.0 / inv_s), confidence=confidence, points=n, frames=0,
                       disparity_offset_px=float(d0), spread=spread)


def scale_from_reconstruction(rec, left_dir, right_dir, baseline_m: float, *, max_frames: int = 60,
                              max_points_per_frame: int = 400) -> StereoScale | None:
    """Metric scale of a COLMAP reconstruction of the left eye, from the right eye's frames
    (same file names in ``right_dir``)."""
    import cv2

    images = sorted((im for im in rec.images.values() if im.has_pose), key=lambda im: im.name)
    if not images:
        return None
    step = max(1, len(images) // max_frames)
    rng = np.random.default_rng(0)
    all_d, all_z, all_f = [], [], []
    frames = 0
    for im in images[::step]:
        lp, rp = left_dir / im.name, right_dir / im.name
        if not rp.exists():
            continue
        left = cv2.imread(str(lp), cv2.IMREAD_GRAYSCALE)
        right = cv2.imread(str(rp), cv2.IMREAD_GRAYSCALE)
        if left is None or right is None or left.shape != right.shape:
            continue
        obs = list(im.get_observation_points2D())
        if not obs:
            continue
        if len(obs) > max_points_per_frame:
            obs = [obs[i] for i in rng.choice(len(obs), max_points_per_frame, replace=False)]
        pose = im.cam_from_world()
        xy = np.array([p.xy for p in obs])
        z = np.array([(pose * rec.points3D[p.point3D_id].xyz)[2] for p in obs])
        disp, _ = measure_disparity(left, right, xy)
        fx = float(rec.cameras[im.camera_id].calibration_matrix()[0, 0])
        all_d.append(disp)
        all_z.append(z)
        all_f.append(np.full(len(z), fx))
        frames += 1
    if not all_d:
        return None
    fit = fit_scale(np.concatenate(all_d), np.concatenate(all_z), np.concatenate(all_f), baseline_m)
    if fit is not None:
        fit.frames = frames
    return fit
