"""Sky masks for structure-from-motion.

Clouds and open sky are effectively at infinity: they have no parallax, and clouds
drift. On outdoor footage they can supply most SIFT features, and then COLMAP sees
mostly "planar or panoramic" pairs (no usable translation) and triangulates almost
nothing. Masking the sky spends the feature budget on nearby structure instead.

Heuristic, no model: sky-coloured pixels (blue, or bright and unsaturated like clouds)
that are connected to the top edge of the frame, grown slightly to cover cloud fringes.
Conservative by design: it only masks what touches the top, so white walls or blue
objects lower in the frame are kept, and a frame with little sky is left alone.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np


def sky_mask(rgb: np.ndarray, min_fraction: float = 0.03) -> np.ndarray:
    """uint8 mask for COLMAP: 255 = use features, 0 = sky (ignore)."""
    import cv2

    h, w = rgb.shape[:2]
    small = cv2.resize(rgb, (max(1, w // 4), max(1, h // 4)), interpolation=cv2.INTER_AREA)
    hsv = cv2.cvtColor(small, cv2.COLOR_RGB2HSV).astype(np.int16)
    hue, sat, val = hsv[..., 0], hsv[..., 1], hsv[..., 2]  # OpenCV: hue 0..179
    r, g, b = (small[..., i].astype(np.int16) for i in range(3))
    blue = (hue >= 90) & (hue <= 130) & (b > r + 15) & (val > 110)
    cloud = (val > 185) & (sat < 45)
    candidate = (blue | cloud).astype(np.uint8)
    # Clean speckle (leaves, railings against the sky) before flood-filling from the top.
    candidate = cv2.morphologyEx(candidate, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    n, labels = cv2.connectedComponents(candidate, connectivity=4)
    top_labels = np.unique(labels[0][candidate[0] > 0])
    sky = np.isin(labels, top_labels[top_labels > 0])
    if sky.mean() < min_fraction:
        return np.full((h, w), 255, np.uint8)
    sky = cv2.dilate(sky.astype(np.uint8), np.ones((5, 5), np.uint8))
    full = cv2.resize(sky, (w, h), interpolation=cv2.INTER_NEAREST)
    return np.where(full > 0, 0, 255).astype(np.uint8)


def write_masks(image_dir: Path, mask_dir: Path) -> float:
    """COLMAP masks for every image (``<name>.png``). Returns the mean sky fraction."""
    from memoryscapes_workers.visual.video import imread_rgb, imwrite_rgb

    mask_dir.mkdir(parents=True, exist_ok=True)
    fractions = []
    for path in sorted(p for p in image_dir.iterdir() if p.suffix.lower() in (".jpg", ".jpeg", ".png")):
        mask = sky_mask(imread_rgb(path))
        fractions.append(float((mask == 0).mean()))
        imwrite_rgb(mask_dir / f"{path.name}.png", np.repeat(mask[..., None], 3, axis=2))
    return float(np.mean(fractions)) if fractions else 0.0
