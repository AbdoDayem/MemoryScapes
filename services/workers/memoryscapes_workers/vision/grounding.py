"""Audio-visual grounding: find a sound's visual source in keyframes and lift it to 3D."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

import numpy as np

from memoryscapes_workers.models import Camera


@dataclass
class Detection:
    box: tuple[float, float, float, float]  # x0, y0, x1, y1 in pixels
    score: float
    phrase: str = ""


class Detector(Protocol):
    def detect(self, image_bytes: bytes, prompt: str) -> list[Detection]:
        """prompt is a Grounding-DINO style phrase list, e.g. "bird. tree."."""
        ...


class GroundingDinoDetector:
    """Open-vocabulary detection with Grounding DINO (transformers). Needs the `visual` extra."""

    def __init__(
        self,
        model_id: str = "IDEA-Research/grounding-dino-tiny",
        device: str | None = None,
        box_threshold: float = 0.3,
        text_threshold: float = 0.25,
    ) -> None:
        self.model_id = model_id
        self.device = device
        self.box_threshold = box_threshold
        self.text_threshold = text_threshold
        self._model = None
        self._processor = None

    def _load(self):
        if self._model is None:
            import torch
            from transformers import AutoModelForZeroShotObjectDetection, AutoProcessor

            self.device = self.device or ("cuda" if torch.cuda.is_available() else "cpu")
            self._processor = AutoProcessor.from_pretrained(self.model_id)
            self._model = AutoModelForZeroShotObjectDetection.from_pretrained(self.model_id).to(self.device).eval()
        return self._model, self._processor

    def detect(self, image_bytes: bytes, prompt: str) -> list[Detection]:
        import io

        import torch
        from PIL import Image

        model, processor = self._load()
        image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        inputs = processor(images=image, text=prompt.lower(), return_tensors="pt").to(self.device)
        with torch.no_grad():
            outputs = model(**inputs)
        kwargs = dict(text_threshold=self.text_threshold, target_sizes=[image.size[::-1]])
        try:  # the threshold argument was renamed across transformers versions
            res = processor.post_process_grounded_object_detection(
                outputs, inputs.input_ids, threshold=self.box_threshold, **kwargs
            )[0]
        except TypeError:
            res = processor.post_process_grounded_object_detection(
                outputs, inputs.input_ids, box_threshold=self.box_threshold, **kwargs
            )[0]
        labels = res.get("text_labels", res.get("labels", []))
        return [
            Detection(box=tuple(float(v) for v in box), score=float(score), phrase=str(label))
            for box, score, label in zip(res["boxes"].tolist(), res["scores"].tolist(), labels)
        ]


def lift_box(
    box: tuple[float, float, float, float],
    camera: Camera,
    points_world: np.ndarray,
    min_points: int = 5,
    depth_percentile: float = 35.0,
    shrink: float = 0.15,
    depth_band: float = 0.2,
    grow: float = 1.0,
) -> np.ndarray | None:
    """3D point for a detection: the centroid of the scene points that project inside
    the (slightly shrunk) box and sit at the object's depth.

    The object's depth is a low percentile of the in-box depths, which prefers the
    object over background showing through the box; points within `depth_band` of it
    (relative) are the object's visible surface. Using their centroid instead of the
    box centre keeps boxes cut off by the frame edge from pointing past the object.
    A sparse SfM cloud (a few thousand points for a phone walk) often leaves fewer than
    `min_points` inside an object's box. Then the box is grown by `grow` times its size
    on each side, around its centre, and tried once more: the points just around an
    object are usually at its depth (the ground it stands on, the wall behind it). The
    low-percentile depth keeps the background those extra points bring from winning.
    Returns None when even the grown box holds too few points to trust the depth.
    """
    w2c = np.linalg.inv(camera.c2w)
    pc = points_world @ w2c[:3, :3].T + w2c[:3, 3]
    z = pc[:, 2]
    front = z > 0.1
    if not front.any():
        return None
    pc, z = pc[front], z[front]
    uv = (pc @ camera.K.T)[:, :2] / z[:, None]
    x0, y0, x1, y1 = box
    dx, dy = (x1 - x0) * shrink / 2, (y1 - y0) * shrink / 2
    inside = (uv[:, 0] >= x0 + dx) & (uv[:, 0] <= x1 - dx) & (uv[:, 1] >= y0 + dy) & (uv[:, 1] <= y1 - dy)
    if inside.sum() < min_points and grow > 0:
        gx, gy = (x1 - x0) * grow, (y1 - y0) * grow
        inside = (uv[:, 0] >= x0 - gx) & (uv[:, 0] <= x1 + gx) & (uv[:, 1] >= y0 - gy) & (uv[:, 1] <= y1 + gy)
    if inside.sum() < min_points:
        return None
    # "nearest": an actual point's depth, so the band below always holds at least one
    # point. Interpolating can land between two far-apart points (sparse far scenes)
    # and leave the band empty.
    depth = float(np.percentile(z[inside], depth_percentile, method="nearest"))
    surface = inside & (np.abs(z - depth) <= depth_band * depth)
    p_cam = pc[surface].mean(axis=0)
    return camera.c2w[:3, :3] @ p_cam + camera.c2w[:3, 3]


def touches_border(box: tuple[float, float, float, float], camera: Camera, margin_px: float = 2.0) -> bool:
    x0, y0, x1, y1 = box
    return x0 <= margin_px or y0 <= margin_px or x1 >= camera.width - margin_px or y1 >= camera.height - margin_px
