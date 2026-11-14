"""Data contracts passed between stages (as JSON in storage).

Coordinate conventions (see docs/ARCHITECTURE.md "coordinate alignment"):
- World frame: metres, +Y up. This is also SATIE's convention, so positions pass through
  unchanged.
- Cameras: OpenCV convention (x right, y down, z forward). `c2w` maps camera coordinates
  to world coordinates. A stereo mic is assumed to lie along the camera's +x axis.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any

import numpy as np

# ---------- produced by ingest / poses (inputs to the audio stages) ----------


@dataclass
class MicInfo:
    """From ingest/meta.json. `layout` drives which localization methods are possible."""

    channels: int
    layout: str = "mono"  # "mono" | "stereo" | "foa"
    spacing_m: float | None = None  # stereo mic spacing; None -> config default

    @classmethod
    def from_json(cls, d: dict[str, Any]) -> MicInfo:
        return cls(channels=d["channels"], layout=d.get("layout", "mono"), spacing_m=d.get("spacing_m"))


@dataclass
class Camera:
    """One posed keyframe, from poses/cameras.json."""

    frame_id: str
    t: float  # seconds since the start of the recording (same clock as the audio)
    width: int
    height: int
    K: np.ndarray  # (3, 3) intrinsics
    c2w: np.ndarray  # (4, 4) camera-to-world
    image_key: str | None = None  # storage key of the frame image

    @property
    def position(self) -> np.ndarray:
        return self.c2w[:3, 3]

    @property
    def right(self) -> np.ndarray:
        return self.c2w[:3, 0]

    @classmethod
    def from_json(cls, d: dict[str, Any]) -> Camera:
        return cls(
            frame_id=d["frame_id"],
            t=float(d["t"]),
            width=int(d["width"]),
            height=int(d["height"]),
            K=np.asarray(d["K"], dtype=float),
            c2w=np.asarray(d["c2w"], dtype=float),
            image_key=d.get("image_key"),
        )

    def to_json(self) -> dict[str, Any]:
        return {
            "frame_id": self.frame_id,
            "t": self.t,
            "width": self.width,
            "height": self.height,
            "K": self.K.tolist(),
            "c2w": self.c2w.tolist(),
            "image_key": self.image_key,
        }


# ---------- produced by separate ----------


@dataclass
class StemInfo:
    id: str
    label: str
    kind: str  # "event" | "speech" | "music"
    key: str  # storage key of the stem WAV (same channel count as the recording)
    level_dbfs: float
    activity: list[float]  # normalized 0..1 envelope sampled at StemsIndex.window_times
    tag_score: float

    def to_json(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class StemsIndex:
    """separate/stems.json"""

    sample_rate: int
    channels: int
    layout: str
    duration_s: float
    window_times: list[float]
    stems: list[StemInfo]
    ambient_key: str
    tags: dict[str, float] = field(default_factory=dict)  # label -> max clip score

    def to_json(self) -> dict[str, Any]:
        d = asdict(self)
        d["stems"] = [s.to_json() for s in self.stems]
        return d

    @classmethod
    def from_json(cls, d: dict[str, Any]) -> StemsIndex:
        return cls(
            sample_rate=d["sample_rate"],
            channels=d["channels"],
            layout=d["layout"],
            duration_s=d["duration_s"],
            window_times=d["window_times"],
            stems=[StemInfo(**s) for s in d["stems"]],
            ambient_key=d["ambient_key"],
            tags=d.get("tags", {}),
        )

    def activity_at(self, stem: StemInfo, t: float) -> float:
        return float(np.interp(t, self.window_times, stem.activity)) if stem.activity else 0.0


# ---------- produced by localize ----------


@dataclass
class LocalizedSource:
    id: str
    label: str
    kind: str
    stem_key: str
    position: list[float] | None  # world metres; None = diffuse
    spread_m: float | None
    confidence: float
    method: str  # "av" | "doa" | "av+doa" | "diffuse"
    level_dbfs: float
    level_db_at_1m: float | None  # level corrected for mean mic distance (for SATIE `spl`)
    trajectory: str | None = None

    def to_json(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_json(cls, d: dict[str, Any]) -> LocalizedSource:
        return cls(**d)


@dataclass
class SceneBounds:
    min: list[float]
    max: list[float]

    @classmethod
    def from_points(cls, points: np.ndarray, pad_m: float = 1.0) -> SceneBounds:
        lo = np.percentile(points, 2, axis=0) - pad_m
        hi = np.percentile(points, 98, axis=0) + pad_m
        return cls(min=lo.tolist(), max=hi.tolist())
