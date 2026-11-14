"""Shot boundaries in edited videos (cuts and cross-dissolves).

A memory is meant to be one continuous capture, but uploads are often edited: several
clips joined with cuts or dissolves. Structure-from-motion can't follow the camera
through a cut, and a dissolve's frames are double exposures (two views blended) that
poison feature matching. So ingest finds the transitions, drops their frames, and
records the shots; poses then tries to link shots through shared scenery.

Works on tiny grayscale thumbnails of the candidate frames:

- Hard cut: the frame-to-frame difference spikes far above its typical value.
- Dissolve: frame X is close to a blend a*A + (1-a)*B of the frames ``window_s`` before
  and after, those two differ a lot (different views), and the blend explains X much
  better than either endpoint does. Camera motion doesn't pass that test: a panned
  frame is a shifted image, not a mix of its neighbours.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

THUMB_SIZE = (48, 27)  # (width, height)


@dataclass
class Shots:
    transition: np.ndarray  # [n] bool, candidate is part of a cut/dissolve
    segments: list[tuple[int, int]]  # candidate index ranges [start, end) of each shot

    def shot_of(self, i: int) -> int | None:
        for k, (a, b) in enumerate(self.segments):
            if a <= i < b:
                return k
        return None


def thumbnail(gray: np.ndarray) -> np.ndarray:
    import cv2

    return cv2.resize(gray, THUMB_SIZE, interpolation=cv2.INTER_AREA)


def find_shots(thumbs: np.ndarray, fps: float, window_s: float = 0.5,
               min_shot_s: float = 1.0) -> Shots:
    """thumbs [n, h, w] uint8 at ``fps`` frames per second."""
    n = len(thumbs)
    t = thumbs.astype(np.float32)
    flags = np.zeros(n, bool)
    if n < 3:
        return Shots(flags, [(0, n)])

    step = np.abs(np.diff(t, axis=0)).mean(axis=(1, 2))  # [n-1]
    cut_level = max(30.0, 5.0 * float(np.median(step)))
    for i in np.flatnonzero(step > cut_level):
        flags[i : i + 2] = True  # the frames on both sides of a cut

    k = max(2, int(round(window_s * fps)))
    if n > 2 * k:
        a, x, b = t[: n - 2 * k], t[k : n - k], t[2 * k :]
        ab = (a - b).reshape(len(a), -1)
        xb = (x - b).reshape(len(a), -1)
        alpha = np.clip((xb * ab).sum(1) / np.maximum((ab * ab).sum(1), 1e-6), 0.0, 1.0)
        blend = alpha[:, None, None] * a + (1 - alpha[:, None, None]) * b
        resid = np.abs(x - blend).mean(axis=(1, 2))
        d_ab = np.abs(a - b).mean(axis=(1, 2))
        d_end = np.minimum(np.abs(x - a).mean(axis=(1, 2)), np.abs(x - b).mean(axis=(1, 2)))
        change_level = max(20.0, 1.6 * float(np.median(d_ab)))
        dissolve = (d_ab > change_level) & (alpha > 0.15) & (alpha < 0.85) & (resid < 0.7 * d_end)
        idx = np.flatnonzero(dissolve) + k
        # A dissolve's edges are still faint double exposures: widen by half a window.
        for i in idx:
            flags[max(0, i - k // 2) : min(n, i + k // 2 + 1)] = True

    segments, start = [], None
    min_len = max(1, int(round(min_shot_s * fps)))
    for i in range(n + 1):
        inside = i < n and not flags[i]
        if inside and start is None:
            start = i
        elif not inside and start is not None:
            if i - start >= min_len:
                segments.append((start, i))
            else:
                flags[start:i] = True  # too short to be a shot: part of the transition
            start = None
    return Shots(flags, segments or [(0, n)])
