"""Keyframe selection: sharp frames, spaced evenly in camera motion.

1. For every candidate frame (decoded at a constant sample rate), measure
   - sharpness: variance of the Laplacian, and
   - motion: median sparse optical flow from the previous candidate, as a fraction of
     the image diagonal.
2. Split the cumulative motion into N equal bins and take the sharpest frame in each.
   Standing still adds no motion, so pauses don't flood the set with duplicates, and
   fast pans get proportionally more keyframes.
3. Drop bins whose best frame is still much blurrier than its neighbours.
"""

from __future__ import annotations

import math
from collections.abc import Iterable
from dataclasses import dataclass, field

import numpy as np


@dataclass
class FrameStats:
    sharpness: np.ndarray  # [n]
    motion: np.ndarray  # [n], motion[0] = 0
    tracked: np.ndarray  # [n] number of flow tracks used (0 = flow failed)


@dataclass
class Selection:
    indices: list[int]
    rel_sharpness: np.ndarray
    total_motion: float
    warnings: list[str] = field(default_factory=list)


def analyze(frames: Iterable[np.ndarray]) -> FrameStats:
    """Compute sharpness and inter-frame motion for grayscale uint8 frames."""
    import cv2

    sharp, motion, tracked = [], [], []
    prev = None
    for gray in frames:
        sharp.append(float(cv2.Laplacian(gray, cv2.CV_64F).var()))
        if prev is None:
            motion.append(0.0)
            tracked.append(0)
        else:
            m, n = _flow(prev, gray)
            motion.append(m)
            tracked.append(n)
        prev = gray
    return FrameStats(np.array(sharp), np.array(motion), np.array(tracked))


def _flow(prev: np.ndarray, cur: np.ndarray) -> tuple[float, int]:
    import cv2

    diag = math.hypot(*prev.shape[:2])
    pts = cv2.goodFeaturesToTrack(prev, maxCorners=400, qualityLevel=0.01, minDistance=7)
    if pts is None or len(pts) < 15:
        return float("nan"), 0
    nxt, st, _ = cv2.calcOpticalFlowPyrLK(prev, cur, pts, None, winSize=(21, 21), maxLevel=3)
    back, st_b, _ = cv2.calcOpticalFlowPyrLK(cur, prev, nxt, None, winSize=(21, 21), maxLevel=3)
    fb_err = np.linalg.norm((back - pts).reshape(-1, 2), axis=1)
    ok = (st.ravel() == 1) & (st_b.ravel() == 1) & (fb_err < 1.0)
    if ok.sum() < 15:
        return float("nan"), int(ok.sum())
    disp = np.linalg.norm((nxt - pts).reshape(-1, 2)[ok], axis=1)
    return float(np.median(disp) / diag), int(ok.sum())


def _rolling_median(x: np.ndarray, window: int) -> np.ndarray:
    half = window // 2
    padded = np.pad(x, half, mode="edge")
    view = np.lib.stride_tricks.sliding_window_view(padded, 2 * half + 1)
    return np.median(view, axis=1)


def select(
    stats: FrameStats,
    *,
    target_step: float,
    min_keyframes: int,
    max_keyframes: int,
    blur_reject: float,
    sharpness_window: int,
) -> Selection:
    n = len(stats.sharpness)
    warnings: list[str] = []
    motion = stats.motion.copy()
    bad = ~np.isfinite(motion)
    if bad[1:].any():
        fill = np.nanmedian(motion[1:]) if (~bad[1:]).any() else 0.0
        motion[bad] = fill
        frac = bad[1:].mean()
        if frac > 0.2:
            warnings.append(
                f"Optical flow failed on {frac:.0%} of frames (low texture or heavy blur)."
            )
    motion[0] = 0.0
    cum = np.cumsum(motion)
    total = float(cum[-1])

    local = _rolling_median(stats.sharpness, max(3, sharpness_window | 1))
    rel = stats.sharpness / np.maximum(local, 1e-9)

    k = int(np.clip(math.ceil(total / max(target_step, 1e-6)), min_keyframes, max_keyframes))
    # At least ~2 candidates per bin so there is a sharper alternative to a blurry frame.
    k = max(1, min(k, n // 2))
    edges = np.linspace(0.0, total, k + 1)
    bins = np.clip(np.searchsorted(edges, cum, side="right") - 1, 0, k - 1)

    chosen: list[int] = []
    rejected = 0
    for b in range(k):
        members = np.flatnonzero(bins == b)
        if len(members) == 0:
            continue
        best = members[np.argmax(rel[members])]
        if rel[best] < blur_reject:
            rejected += 1
            continue
        chosen.append(int(best))
    chosen = sorted(set(chosen))

    empty = k - len(np.unique(bins))
    if empty > 0.1 * k:
        warnings.append(
            f"Camera moved too fast in places ({empty} gaps); reconstruction may break there."
            " Move more slowly."
        )
    if rejected > 0.1 * k:
        warnings.append(f"{rejected} stretches were too blurry to use. Hold the camera steadier.")
    return Selection(indices=chosen, rel_sharpness=rel, total_motion=total, warnings=warnings)
