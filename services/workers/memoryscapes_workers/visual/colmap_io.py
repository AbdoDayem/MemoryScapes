"""Minimal reader for COLMAP binary models (cameras.bin, images.bin, points3D.bin).

The splat stage reads the poses stage's model with this instead of pycolmap. On macOS,
pycolmap and torch each bundle their own OpenMP runtime, and loading both into one
process aborts. It also keeps pycolmap out of the training process on GPU servers.

Format: https://colmap.github.io/format.html#binary-file-format
"""

from __future__ import annotations

import struct
from dataclasses import dataclass
from pathlib import Path

import numpy as np

# model_id -> (name, number of params)
CAMERA_MODELS = {
    0: ("SIMPLE_PINHOLE", 3), 1: ("PINHOLE", 4), 2: ("SIMPLE_RADIAL", 4), 3: ("RADIAL", 5),
    4: ("OPENCV", 8), 5: ("OPENCV_FISHEYE", 8), 6: ("FULL_OPENCV", 12), 7: ("FOV", 5),
    8: ("SIMPLE_RADIAL_FISHEYE", 4), 9: ("RADIAL_FISHEYE", 5), 10: ("THIN_PRISM_FISHEYE", 12),
    11: ("RAD_TAN_THIN_PRISM_FISHEYE", 16),
}


@dataclass
class Camera:
    id: int
    model: str
    width: int
    height: int
    params: np.ndarray

    def K(self) -> np.ndarray:
        if self.model == "SIMPLE_PINHOLE":
            f, cx, cy = self.params
            fx = fy = f
        elif self.model == "PINHOLE":
            fx, fy, cx, cy = self.params
        else:
            raise ValueError(f"camera model {self.model} is distorted; undistort first")
        return np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1.0]])


@dataclass
class Image:
    id: int
    camera_id: int
    name: str
    qvec: np.ndarray  # wxyz, world -> camera
    tvec: np.ndarray

    def w2c(self) -> np.ndarray:
        w, x, y, z = self.qvec / np.linalg.norm(self.qvec)
        r = np.array([
            [1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
            [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
            [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)],
        ])
        m = np.eye(4)
        m[:3, :3] = r
        m[:3, 3] = self.tvec
        return m


@dataclass
class Model:
    cameras: dict[int, Camera]
    images: dict[int, Image]
    points: np.ndarray  # [N, 3]
    colors: np.ndarray  # [N, 3] uint8
    errors: np.ndarray  # [N]


def read_cameras(path: Path) -> dict[int, Camera]:
    data = path.read_bytes()
    (n,) = struct.unpack_from("<Q", data, 0)
    off, cams = 8, {}
    for _ in range(n):
        cid, model_id, w, h = struct.unpack_from("<iiQQ", data, off)
        off += 24
        name, n_params = CAMERA_MODELS[model_id]
        params = np.frombuffer(data, "<f8", n_params, off).copy()
        off += 8 * n_params
        cams[cid] = Camera(cid, name, int(w), int(h), params)
    return cams


def read_images(path: Path) -> dict[int, Image]:
    data = path.read_bytes()
    (n,) = struct.unpack_from("<Q", data, 0)
    off, images = 8, {}
    for _ in range(n):
        iid, qw, qx, qy, qz, tx, ty, tz, cid = struct.unpack_from("<i7di", data, off)
        off += 64
        end = data.index(b"\x00", off)
        name = data[off:end].decode()
        off = end + 1
        (n2d,) = struct.unpack_from("<Q", data, off)
        off += 8 + 24 * n2d  # skip (x, y, point3D_id) observations
        images[iid] = Image(iid, cid, name, np.array([qw, qx, qy, qz]), np.array([tx, ty, tz]))
    return images


def read_points3d(path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    data = path.read_bytes()
    (n,) = struct.unpack_from("<Q", data, 0)
    off = 8
    xyz = np.empty((n, 3))
    rgb = np.empty((n, 3), np.uint8)
    err = np.empty(n)
    for i in range(n):
        _, x, y, z, r, g, b, e, track = struct.unpack_from("<Q3d3BdQ", data, off)
        off += 51 + 8 * track
        xyz[i] = (x, y, z)
        rgb[i] = (r, g, b)
        err[i] = e
    return xyz, rgb, err


def read_model(model_dir: Path) -> Model:
    model_dir = Path(model_dir)
    xyz, rgb, err = read_points3d(model_dir / "points3D.bin")
    return Model(read_cameras(model_dir / "cameras.bin"), read_images(model_dir / "images.bin"),
                 xyz, rgb, err)
