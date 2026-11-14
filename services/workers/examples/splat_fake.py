"""Synthetic outputs for the (separately developed) splat pipeline, for example memories.

Writes what docs/SPLAT_OUTPUTS.md specifies: real keyframe images rendered from the
scene points, a 3DGS-format point_cloud.ply, training metrics and splat renders. A
scenario can ask for a misaligned splat (wrong scale or offset), which is exactly the
kind of integration bug the debugger's alignment check is for.
"""

from __future__ import annotations

import struct
import zlib

import numpy as np

from memoryscapes_workers.context import StageContext
from memoryscapes_workers.storage import write_json, write_npy

C0 = 0.28209479177387814  # SH degree-0 constant

LABEL_COLORS = {
    "birdsong": (70, 140, 60),
    "water": (90, 150, 210),
    "traffic": (120, 120, 130),
    "speech": (210, 160, 130),
    "music": (170, 90, 170),
    "dog": (150, 110, 70),
    "bells": (200, 180, 90),
    "waves": (60, 120, 180),
}


def png_bytes(rgb: np.ndarray) -> bytes:
    """Minimal PNG encoder for an (H, W, 3) uint8 image."""
    h, w, _ = rgb.shape
    raw = b"".join(b"\x00" + rgb[y].tobytes() for y in range(h))

    def chunk(tag: bytes, data: bytes) -> bytes:
        return struct.pack(">I", len(data)) + tag + data + struct.pack(">I", zlib.crc32(tag + data) & 0xFFFFFFFF)

    return (
        b"\x89PNG\r\n\x1a\n"
        + chunk(b"IHDR", struct.pack(">IIBBBBB", w, h, 8, 2, 0, 0, 0))
        + chunk(b"IDAT", zlib.compress(raw, 6))
        + chunk(b"IEND", b"")
    )


def point_colors(r) -> np.ndarray:
    """Ground, backdrop, then one blob per source (matches scenes.scene_points order)."""
    pts = r.points
    cols = np.zeros((len(pts), 3), np.uint8)
    n_src = sum(1 for _ in r.scenario.sources) * 400
    base = len(pts) - n_src
    ground = pts[:base, 1] < 0.01
    cols[:base] = np.where(ground[:, None], [96, 118, 72], [150, 160, 175])
    for i, s in enumerate(r.scenario.sources):
        cols[base + i * 400 : base + (i + 1) * 400] = LABEL_COLORS.get(s.label, (200, 80, 80))
    rng = np.random.default_rng(r.scenario.seed + 300)
    jitter = rng.integers(-12, 13, size=cols.shape)
    return np.clip(cols.astype(int) + jitter, 0, 255).astype(np.uint8)


def render_view(cam, points: np.ndarray, colors: np.ndarray, width=320, height=240, radius_m=0.3, blur=0.0, rng=None) -> np.ndarray:
    """Z-buffered disc splats of the points, seen from cam, over a sky gradient."""
    sx, sy = width / cam.width, height / cam.height
    K = cam.K.copy()
    K[0] *= sx
    K[1] *= sy
    w2c = np.linalg.inv(cam.c2w)
    pc = points @ w2c[:3, :3].T + w2c[:3, 3]
    front = pc[:, 2] > 0.2
    pc, cols = pc[front], colors[front]
    uv = (pc @ K.T)[:, :2] / pc[:, 2:3]
    sky = np.linspace([170, 200, 230], [225, 235, 245], height).astype(np.uint8)
    img = np.repeat(sky[:, None, :], width, axis=1)
    depth = np.full((height, width), np.inf)
    rad = np.clip(K[0, 0] * radius_m / pc[:, 2], 1, 12).astype(int)
    order = np.argsort(-pc[:, 2])  # far to near
    for i in order:
        u, v, rr = int(uv[i, 0]), int(uv[i, 1]), rad[i]
        if u + rr < 0 or v + rr < 0 or u - rr >= width or v - rr >= height:
            continue
        y0, y1, x0, x1 = max(v - rr, 0), min(v + rr + 1, height), max(u - rr, 0), min(u + rr + 1, width)
        patch = depth[y0:y1, x0:x1]
        mask = patch > pc[i, 2]
        img[y0:y1, x0:x1][mask] = cols[i]
        patch[mask] = pc[i, 2]
    if blur and rng is not None:  # a splat render is a little soft and noisy
        k = np.array([1, 2, 1], float) / 4
        img = img.astype(float)
        for axis in (0, 1):
            img = np.apply_along_axis(lambda m: np.convolve(m, k, mode="same"), axis, img)
        img += rng.normal(0, 3, img.shape)
    return np.clip(img, 0, 255).astype(np.uint8)


def write_frames(ctx: StageContext, r) -> dict[bytes, str]:
    """Writes ingest/frames/<id>.png and points colours. Returns image bytes -> frame id
    (the oracle detector uses it to know which camera an image came from)."""
    colors = point_colors(r)
    write_npy(ctx.storage, ctx.key("poses", "points3d_rgb.npy"), colors)
    lookup = {}
    for cam in r.cameras:
        data = png_bytes(render_view(cam, r.points, colors))
        cam.image_key = ctx.key("ingest", f"frames/{cam.frame_id}.png")
        ctx.storage.write_bytes(cam.image_key, data, "image/png")
        lookup[data] = cam.frame_id
    return lookup


def gaussians_from_points(points: np.ndarray, colors: np.ndarray, rng) -> dict[str, np.ndarray]:
    n = len(points)
    reps = 2
    xyz = np.repeat(points, reps, axis=0) + rng.normal(0, 0.08, (n * reps, 3))
    rgb = np.repeat(colors, reps, axis=0) / 255.0
    return {
        "xyz": xyz.astype(np.float32),
        "f_dc": ((rgb - 0.5) / C0).astype(np.float32),
        "opacity": rng.normal(2.0, 1.0, n * reps).astype(np.float32),  # logit
        "scale": np.log(rng.uniform(0.03, 0.15, (n * reps, 3))).astype(np.float32),
        "rot": np.tile([1.0, 0.0, 0.0, 0.0], (n * reps, 1)).astype(np.float32),
    }


def ply_bytes(g: dict[str, np.ndarray]) -> bytes:
    n = len(g["xyz"])
    props = ["x", "y", "z", "nx", "ny", "nz", "f_dc_0", "f_dc_1", "f_dc_2", "opacity", "scale_0", "scale_1", "scale_2", "rot_0", "rot_1", "rot_2", "rot_3"]
    header = "ply\nformat binary_little_endian 1.0\n" + f"element vertex {n}\n" + "".join(f"property float {p}\n" for p in props) + "end_header\n"
    data = np.concatenate(
        [g["xyz"], np.zeros((n, 3), np.float32), g["f_dc"], g["opacity"][:, None], g["scale"], g["rot"]], axis=1
    ).astype("<f4")
    return header.encode() + data.tobytes()


def write_splat(ctx: StageContext, r, scale: float = 1.0, offset=(0.0, 0.0, 0.0), iterations: int = 7000) -> None:
    rng = np.random.default_rng(r.scenario.seed + 400)
    colors = point_colors(r)
    g = gaussians_from_points(r.points, colors, rng)
    g["xyz"] = (g["xyz"] * scale + np.asarray(offset, np.float32)).astype(np.float32)
    ctx.storage.write_bytes(ctx.key("splat", "point_cloud.ply"), ply_bytes(g), "application/octet-stream")

    it = np.arange(0, iterations + 1, 100)
    noise = rng.normal(0, 0.004, len(it))
    metrics = {
        "iteration": it.tolist(),
        "loss": np.round(0.22 * np.exp(-it / 1400) + 0.035 + noise, 5).tolist(),
        "psnr": np.round(12 + 15.5 * (1 - np.exp(-it / 1800)) + noise * 40, 3).tolist(),
        "num_gaussians": np.round(len(g["xyz"]) * (0.3 + 0.7 * np.minimum(it / 3000, 1))).astype(int).tolist(),
        "elapsed_s": np.round(it * 0.021, 2).tolist(),
    }
    write_json(ctx.storage, ctx.key("splat", "metrics.json"), metrics)
    eval_frames = [c.frame_id for c in r.cameras[::6]]
    write_json(
        ctx.storage,
        ctx.key("splat", "meta.json"),
        {"trainer": "synthetic (examples/splat_fake.py)", "trainer_version": "0", "iterations": iterations, "sh_degree": 0, "units": "meters", "up": "+y", "eval_frames": eval_frames},
    )
    gcol = np.clip((g["f_dc"] * C0 + 0.5) * 255, 0, 255).astype(np.uint8)
    for cam in r.cameras:
        img = render_view(cam, g["xyz"].astype(float), gcol, radius_m=0.25, blur=1.0, rng=rng)
        ctx.storage.write_bytes(ctx.key("splat", f"renders/{cam.frame_id}.png"), png_bytes(img), "image/png")
    # a viewer deliverable the debugger lists but doesn't parse
    ctx.storage.write_bytes(ctx.key("splat", "lods/lod0.spz"), b"SPZ placeholder", "application/octet-stream")


def write_truth(ctx: StageContext, r, scale: float, offset) -> None:
    write_json(
        ctx.storage,
        ctx.key("debug", "truth.json"),
        {
            "scenario": r.scenario.name,
            "description": r.scenario.description,
            "sources": [{"label": s.label, "position": list(s.position)} for s in r.scenario.sources],
            "diffuse": [label for label, _ in r.scenario.diffuse],
            "splat_transform": {"scale": scale, "offset": list(offset)},
        },
    )
