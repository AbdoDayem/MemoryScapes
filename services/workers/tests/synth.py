"""Synthetic scenes with known ground truth for the audio stage tests."""

from __future__ import annotations

import numpy as np

from memoryscapes_workers.audio.doa import SPEED_OF_SOUND
from memoryscapes_workers.models import Camera

SR = 16_000
UP = np.array([0.0, 1.0, 0.0])


def look_camera(position, yaw_deg: float, t: float, frame_id: str, width=640, height=480, f=500.0) -> Camera:
    """OpenCV camera at `position`, yawed about world +Y (yaw 0 looks along +Z)."""
    yaw = np.radians(yaw_deg)
    fwd = np.array([np.sin(yaw), 0.0, np.cos(yaw)])
    down = -UP
    right = np.cross(down, fwd)
    c2w = np.eye(4)
    c2w[:3, 0], c2w[:3, 1], c2w[:3, 2], c2w[:3, 3] = right, down, fwd, position
    K = np.array([[f, 0, width / 2], [0, f, height / 2], [0, 0, 1.0]])
    return Camera(frame_id, t, width, height, K, c2w, image_key=None)


def walking_path(duration_s: float = 8.0, n: int = 17) -> list[Camera]:
    """Walk 4 m along +X while panning left and right, like someone filming a place."""
    cams = []
    for i, t in enumerate(np.linspace(0, duration_s, n)):
        x = -2.0 + 4.0 * t / duration_s
        yaw = 40.0 * np.sin(2 * np.pi * t / duration_s * 1.5)
        cams.append(look_camera(np.array([x, 1.6, 0.0]), yaw, float(t), f"f{i:03d}"))
    return cams


def wall_points(z: float = 5.0, step: float = 0.25) -> np.ndarray:
    xs, ys = np.arange(-6, 6 + 1e-9, step), np.arange(0, 4 + 1e-9, step)
    wall = np.array([[x, y, z] for x in xs for y in ys])
    zs = np.arange(0, z + 1e-9, step)
    ground = np.array([[x, 0.0, zz] for x in xs for zz in zs])
    return np.concatenate([wall, ground])


def project(cam: Camera, p: np.ndarray) -> np.ndarray:
    w2c = np.linalg.inv(cam.c2w)
    pc = w2c[:3, :3] @ p + w2c[:3, 3]
    uv = cam.K @ pc
    return uv[:2] / uv[2]


def noise(duration_s: float, seed: int = 0, sr: int = SR) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return (0.1 * rng.standard_normal(int(duration_s * sr))).astype(np.float32)


def frac_delay(x: np.ndarray, delay_s: float, sr: int = SR) -> np.ndarray:
    n = len(x)
    freqs = np.fft.rfftfreq(n, 1.0 / sr)
    return np.fft.irfft(np.fft.rfft(x) * np.exp(-2j * np.pi * freqs * delay_s), n).astype(np.float32)


def stereo_from_source(
    mono: np.ndarray, source: np.ndarray, cameras: list[Camera], spacing_m: float, block_s: float = 0.25, sr: int = SR
) -> np.ndarray:
    """Left/right mics at camera position -/+ spacing/2 along the camera's right axis.
    The delay is held constant within short blocks as the camera moves."""
    from memoryscapes_workers.audio.doa import CameraPath

    path = CameraPath(cameras)
    block = int(block_s * sr)
    left = np.zeros_like(mono)
    right = np.zeros_like(mono)
    for start in range(0, len(mono), block):
        seg = mono[start : start + block]
        pos, axis = path.at((start + len(seg) / 2) / sr)
        pl, pr = pos - axis * spacing_m / 2, pos + axis * spacing_m / 2
        dl, dr = np.linalg.norm(source - pl), np.linalg.norm(source - pr)
        # pad so the circular shift in frac_delay doesn't wrap audible samples
        pad = np.concatenate([seg, np.zeros(64, np.float32)])
        # arrival difference t_L - t_R = (dl - dr) / c, split symmetrically between channels
        left[start : start + len(seg)] = frac_delay(pad, (dl - dr) / 2 / SPEED_OF_SOUND, sr)[: len(seg)]
        right[start : start + len(seg)] = frac_delay(pad, (dr - dl) / 2 / SPEED_OF_SOUND, sr)[: len(seg)]
    return np.stack([left, right])
