"""Direction of arrival from a stereo recording, triangulated along the camera path.

A stereo pair only measures the angle between the source direction and the mic axis
(a cone of directions). The phone moves and turns while recording, so each window adds
a different cone; scoring candidate 3D points against all of them finds where the cones
intersect. Front/back and cone ambiguity resolve only if the path gives enough variety,
and the result's spread reports how well it was pinned down.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from memoryscapes_workers.models import Camera

SPEED_OF_SOUND = 343.0


def gcc_phat(
    ref: np.ndarray,
    sig: np.ndarray,
    sr: int,
    max_tau: float | None = None,
    interp: int = 16,
    band: tuple[float, float] | None = (100.0, 8000.0),
) -> tuple[float, float]:
    """Delay of `sig` relative to `ref` in seconds, and the PHAT peak height (0..1,
    a coherence-like confidence)."""
    n = len(ref) + len(sig)
    spec = np.fft.rfft(sig, n) * np.conj(np.fft.rfft(ref, n))
    spec /= np.abs(spec) + 1e-12
    if band is not None:
        freqs = np.fft.rfftfreq(n, 1.0 / sr)
        spec[(freqs < band[0]) | (freqs > band[1])] = 0.0
        kept = max(1, int(np.count_nonzero(spec)))
    else:
        kept = len(spec)
    cc = np.fft.irfft(spec, n * interp)
    max_shift = interp * n // 2
    if max_tau is not None:
        max_shift = min(int(interp * sr * max_tau), max_shift)
    cc = np.concatenate((cc[-max_shift:], cc[: max_shift + 1]))
    shift = int(np.argmax(np.abs(cc))) - max_shift
    # Pure delay over the kept band gives a peak of kept / (n_out / 2): normalize to ~1.
    peak = float(np.abs(cc).max() * (n * interp) / (2 * kept))
    return shift / float(interp * sr), min(peak, 1.0)


@dataclass
class Bearing:
    t: float
    cos: float  # cosine of the angle between source direction and mic axis (camera +x)
    weight: float


def stereo_bearings(
    stereo: np.ndarray,
    sr: int,
    spacing_m: float,
    activity,  # callable t -> 0..1 (stem activity)
    window_s: float = 1.0,
    hop_s: float = 0.5,
    min_activity: float = 0.2,
) -> list[Bearing]:
    """One bearing per window. Channel 0 = left mic, channel 1 = right mic.

    With mics at p -/+ (s/2)·a and unit source direction u, the right mic hears the
    source first when u·a > 0: t_R - t_L = -s (u·a) / c, so u·a = -c·tau / s.
    """
    left, right = stereo[0], stereo[1]
    win, hop = int(window_s * sr), int(hop_s * sr)
    max_tau = spacing_m / SPEED_OF_SOUND
    out: list[Bearing] = []
    for start in range(0, max(1, len(left) - win + 1), hop):
        t = (start + win / 2) / sr
        act = float(activity(t))
        if act < min_activity:
            continue
        seg_l, seg_r = left[start : start + win], right[start : start + win]
        if np.sqrt(np.mean(seg_l**2)) < 1e-5:
            continue
        tau, peak = gcc_phat(seg_l, seg_r, sr, max_tau=max_tau)
        cos = float(np.clip(-SPEED_OF_SOUND * tau / spacing_m, -1.0, 1.0))
        out.append(Bearing(t=t, cos=cos, weight=act * peak))
    return out


class CameraPath:
    """Camera position and mic axis at any time, interpolated between keyframes."""

    def __init__(self, cameras: list[Camera]) -> None:
        cams = sorted(cameras, key=lambda c: c.t)
        self.times = np.array([c.t for c in cams])
        self.positions = np.stack([c.position for c in cams])
        self.rights = np.stack([c.right / np.linalg.norm(c.right) for c in cams])

    def at(self, t: float) -> tuple[np.ndarray, np.ndarray]:
        pos = np.array([np.interp(t, self.times, self.positions[:, i]) for i in range(3)])
        right = np.array([np.interp(t, self.times, self.rights[:, i]) for i in range(3)])
        return pos, right / (np.linalg.norm(right) + 1e-12)


def bearing_likelihood(
    candidates: np.ndarray, bearings: list[Bearing], path: CameraPath, sigma: float = 0.12
) -> np.ndarray:
    """Weighted agreement (0..1) of each candidate point (M, 3) with every bearing."""
    total = sum(b.weight for b in bearings)
    if total <= 0:
        return np.zeros(len(candidates))
    scores = np.zeros(len(candidates))
    for b in bearings:
        pos, axis = path.at(b.t)
        d = candidates - pos
        dist = np.linalg.norm(d, axis=1)
        u = d / np.maximum(dist, 1e-6)[:, None]
        err = u @ axis - b.cos
        s = np.exp(-(err**2) / (2 * sigma**2))
        s[dist < 0.3] = 0.0  # too close to the mic for a far-field bearing to mean anything
        scores += b.weight * s
    return scores / total


def candidate_points(
    points3d: np.ndarray, bounds_min: np.ndarray, bounds_max: np.ndarray, max_points: int = 3000, grid: int = 14, seed: int = 0
) -> np.ndarray:
    """Scene points (sources usually sit on surfaces) plus a regular grid for open space."""
    rng = np.random.default_rng(seed)
    pts = points3d
    if len(pts) > max_points:
        pts = pts[rng.choice(len(pts), max_points, replace=False)]
    axes = [np.linspace(bounds_min[i], bounds_max[i], grid) for i in range(3)]
    g = np.stack(np.meshgrid(*axes, indexing="ij"), axis=-1).reshape(-1, 3)
    return np.concatenate([pts, g]) if len(pts) else g


@dataclass
class DoaEstimate:
    position: np.ndarray
    score: float  # peak likelihood, 0..1
    spread_m: float  # how far near-best candidates scatter; large = ambiguous


def triangulate(candidates: np.ndarray, scores: np.ndarray, near_best: float = 0.9) -> DoaEstimate | None:
    if len(scores) == 0 or scores.max() <= 0:
        return None
    best = int(np.argmax(scores))
    near = scores >= near_best * scores[best]
    w = scores[near]
    pts = candidates[near]
    centre = (pts * w[:, None]).sum(0) / w.sum()
    spread = float(np.sqrt((w * np.sum((pts - centre) ** 2, axis=1)).sum() / w.sum()))
    return DoaEstimate(position=candidates[best], score=float(scores[best]), spread_m=spread)
