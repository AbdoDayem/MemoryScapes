"""Combine audio-visual candidates and stereo DOA into one position per source."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import numpy as np

from memoryscapes_workers.audio.doa import DoaEstimate


@dataclass
class Cluster:
    centre: np.ndarray
    weight: float
    spread_m: float
    count: int


def cluster_candidates(points: np.ndarray, weights: np.ndarray, radius_m: float = 1.5) -> list[Cluster]:
    """Greedy weighted clustering: heaviest unassigned point seeds a cluster of all
    unassigned points within radius, refined once around the weighted centre.
    Non-finite points and non-positive weights are ignored (they'd make NaN centres)."""
    points, weights = np.asarray(points, float), np.asarray(weights, float)
    if len(points):
        keep = np.isfinite(points).all(axis=1) & np.isfinite(weights) & (weights > 0)
        points, weights = points[keep], weights[keep]
    if len(points) == 0:
        return []
    order = np.argsort(-weights)
    free = np.ones(len(points), bool)
    clusters: list[Cluster] = []
    for seed in order:
        if not free[seed]:
            continue
        centre = points[seed]
        for _ in range(2):
            members = free & (np.linalg.norm(points - centre, axis=1) <= radius_m)
            w = weights[members]
            centre = (points[members] * w[:, None]).sum(0) / w.sum()
        free &= ~members
        d2 = np.sum((points[members] - centre) ** 2, axis=1)
        clusters.append(
            Cluster(
                centre=centre,
                weight=float(w.sum()),
                spread_m=float(np.sqrt((w * d2).sum() / w.sum())),
                count=int(members.sum()),
            )
        )
    clusters.sort(key=lambda c: c.weight, reverse=True)
    return clusters


@dataclass(frozen=True)
class FusionConfig:
    av_weight: float = 0.6
    doa_weight: float = 0.4
    min_confidence: float = 0.25
    min_av_views: int = 2  # a single-view detection is too easy to get wrong
    doa_spread_scale_m: float = 2.0
    # When the camera saw the source somewhere and DOA alone points elsewhere, DOA is
    # usually the one fooled (leakage, reverb, wrong mic spacing): scale its confidence.
    conflict_penalty: float = 0.4


@dataclass
class Placement:
    position: np.ndarray | None
    spread_m: float | None
    confidence: float
    method: str


def fuse(
    clusters: list[Cluster],
    doa_scorer: Callable[[np.ndarray], np.ndarray] | None,
    doa: DoaEstimate | None,
    cfg: FusionConfig = FusionConfig(),
    agree_radius_m: float = 2.0,
) -> Placement:
    total = sum(c.weight for c in clusters)
    av = [c for c in clusters if c.count >= cfg.min_av_views]

    def av_conf(c: Cluster) -> float:
        return (c.weight / total) * min(1.0, c.count / 4)

    if av and doa_scorer is not None:
        doa_scores = doa_scorer(np.stack([c.centre for c in av]))
        combined = [cfg.av_weight * av_conf(c) + cfg.doa_weight * float(s) for c, s in zip(av, doa_scores)]
        best = int(np.argmax(combined))
        if combined[best] >= cfg.min_confidence:
            c = av[best]
            return Placement(c.centre, c.spread_m, round(combined[best], 4), "av+doa")

    if av:
        c = av[0]
        conf = av_conf(c)
        if conf >= cfg.min_confidence:
            return Placement(c.centre, c.spread_m, round(conf, 4), "av")

    if doa is not None:
        conf = doa.score / (1.0 + doa.spread_m / cfg.doa_spread_scale_m)
        if av and all(np.linalg.norm(doa.position - c.centre) > agree_radius_m for c in av):
            conf *= cfg.conflict_penalty
        if conf >= cfg.min_confidence:
            return Placement(doa.position, doa.spread_m, round(conf, 4), "doa")

    return Placement(None, None, 0.0, "diffuse")
