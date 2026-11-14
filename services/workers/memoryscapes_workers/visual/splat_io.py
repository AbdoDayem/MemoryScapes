"""Gaussian splat containers and file formats (numpy only).

- ``.ply``: the standard 3DGS layout (x y z nx ny nz f_dc_* f_rest_* opacity scale_* rot_*),
  raw values (logit opacity, log scale, wxyz quaternion). Archival / re-processing format.
- ``.spz``: Niantic's compressed format, written as **version 3** (one gzip stream,
  smallest-three quaternions), which Spark and the Niantic library both read. About
  10x smaller than .ply. Layout follows nianticlabs/spz ``load-spz.cc``.

Coordinates are written as-is. Our world frame is right-handed with +y up, which is
the frame .spz calls RUB (three.js), so no axis flips are needed and the .ply and the
.spz show the same orientation in Spark.
"""

from __future__ import annotations

import gzip
import struct
from dataclasses import dataclass
from pathlib import Path

import numpy as np

SH_C0 = 0.28209479177387814
SPZ_MAGIC = 0x5053474E  # "NGSP"
SPZ_FRACTIONAL_BITS = 12
SPZ_COLOR_SCALE = 0.15
SPZ_FLAG_ANTIALIASED = 0x1
SH_DIM = {0: 0, 1: 3, 2: 8, 3: 15}


def sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-x))


def logit(p: np.ndarray) -> np.ndarray:
    p = np.clip(p, 1e-6, 1 - 1e-6)
    return np.log(p / (1 - p))


@dataclass
class GaussianCloud:
    means: np.ndarray  # [N, 3] float32, world meters
    scales: np.ndarray  # [N, 3] log scale
    quats: np.ndarray  # [N, 4] wxyz (not necessarily normalized)
    opacities: np.ndarray  # [N] logit
    sh0: np.ndarray  # [N, 3] DC coefficient per channel
    shN: np.ndarray  # [N, K, 3] higher-order coefficients, K in {0, 3, 8, 15}
    antialiased: bool = False

    def __len__(self) -> int:
        return len(self.means)

    @property
    def sh_degree(self) -> int:
        k = self.shN.shape[1]
        return {v: d for d, v in SH_DIM.items()}[k]

    def subset(self, mask_or_idx: np.ndarray) -> GaussianCloud:
        return GaussianCloud(
            self.means[mask_or_idx], self.scales[mask_or_idx], self.quats[mask_or_idx],
            self.opacities[mask_or_idx], self.sh0[mask_or_idx], self.shN[mask_or_idx],
            self.antialiased,
        )

    def with_sh_degree(self, degree: int) -> GaussianCloud:
        """Truncate SH bands (bands are orthogonal, so truncation is exact for what remains)."""
        k = SH_DIM[min(degree, self.sh_degree)]
        return GaussianCloud(self.means, self.scales, self.quats, self.opacities, self.sh0,
                             self.shN[:, :k], self.antialiased)

    @staticmethod
    def concat(clouds: list[GaussianCloud]) -> GaussianCloud:
        return GaussianCloud(*(np.concatenate([getattr(c, f) for c in clouds]) for f in
                               ("means", "scales", "quats", "opacities", "sh0", "shN")),
                             antialiased=clouds[0].antialiased)

    def finite(self) -> GaussianCloud:
        ok = np.ones(len(self), bool)
        for arr in (self.means, self.scales, self.quats, self.opacities, self.sh0):
            ok &= np.isfinite(arr.reshape(len(self), -1)).all(axis=1)
        ok &= np.linalg.norm(self.quats, axis=1) > 1e-8
        return self.subset(ok)


# --------------------------------------------------------------------------- PLY


def write_ply(path: Path, cloud: GaussianCloud) -> None:
    n, k = len(cloud), cloud.shN.shape[1]
    names = ["x", "y", "z", "nx", "ny", "nz", "f_dc_0", "f_dc_1", "f_dc_2"]
    names += [f"f_rest_{i}" for i in range(3 * k)]
    names += ["opacity", "scale_0", "scale_1", "scale_2", "rot_0", "rot_1", "rot_2", "rot_3"]
    # f_rest is channel-major: all R coefficients, then G, then B.
    rest = cloud.shN.transpose(0, 2, 1).reshape(n, 3 * k)
    data = np.hstack([
        cloud.means, np.zeros((n, 3)), cloud.sh0, rest, cloud.opacities[:, None],
        cloud.scales, cloud.quats,
    ]).astype("<f4")
    header = "ply\nformat binary_little_endian 1.0\n"
    header += f"element vertex {n}\n" + "".join(f"property float {nm}\n" for nm in names)
    header += "end_header\n"
    with open(path, "wb") as f:
        f.write(header.encode("ascii"))
        f.write(data.tobytes())


_PLY_TYPES = {"float": "f4", "float32": "f4", "double": "f8", "uchar": "u1", "uint8": "u1",
              "int": "i4", "int32": "i4", "uint": "u4", "short": "i2", "ushort": "u2"}


def read_ply(path: Path) -> GaussianCloud:
    with open(path, "rb") as f:
        if f.readline().strip() != b"ply":
            raise ValueError("not a PLY file")
        props, n, endian = [], 0, "<"
        while True:
            line = f.readline().decode("ascii").strip()
            if line.startswith("format"):
                if "ascii" in line:
                    raise ValueError("ASCII PLY is not supported")
                endian = ">" if "big_endian" in line else "<"
            elif line.startswith("element vertex"):
                n = int(line.split()[-1])
            elif line.startswith("property"):
                _, typ, name = line.split()
                props.append((name, endian + _PLY_TYPES[typ]))
            elif line == "end_header":
                break
        arr = np.frombuffer(f.read(n * np.dtype(props).itemsize), dtype=np.dtype(props), count=n)

    def cols(names: list[str]) -> np.ndarray:
        return np.stack([arr[nm].astype(np.float32) for nm in names], axis=1)

    rest_names = sorted((nm for nm, _ in props if nm.startswith("f_rest_")),
                        key=lambda s: int(s.split("_")[-1]))
    k = len(rest_names) // 3
    shN = cols(rest_names).reshape(n, 3, k).transpose(0, 2, 1) if k else np.zeros((n, 0, 3), np.float32)
    return GaussianCloud(
        means=cols(["x", "y", "z"]),
        scales=cols(["scale_0", "scale_1", "scale_2"]),
        quats=cols(["rot_0", "rot_1", "rot_2", "rot_3"]),
        opacities=arr["opacity"].astype(np.float32),
        sh0=cols(["f_dc_0", "f_dc_1", "f_dc_2"]),
        shN=np.ascontiguousarray(shN),
    )


# --------------------------------------------------------------------------- SPZ


def _u8(x: np.ndarray) -> np.ndarray:
    return np.clip(np.round(x), 0, 255).astype(np.uint8)


def _pack_smallest_three(q_xyzw: np.ndarray) -> np.ndarray:
    """[N,4] xyzw -> [N] uint32: 2-bit index of largest |c|, then 3 x (sign bit + 9 bits)."""
    q = q_xyzw / np.linalg.norm(q_xyzw, axis=1, keepdims=True)
    largest = np.argmax(np.abs(q), axis=1)
    neg_largest = q[np.arange(len(q)), largest] < 0
    comp = largest.astype(np.uint32)
    for i in range(4):
        # The three non-largest components, in ascending index order.
        mask = largest != i
        mag = np.minimum(np.floor(511 * np.abs(q[:, i]) / np.sqrt(0.5) + 0.5), 511).astype(np.uint32)
        neg = ((q[:, i] < 0) ^ neg_largest).astype(np.uint32)
        comp = np.where(mask, (comp << 10) | (neg << 9) | mag, comp)
    return comp


def _unpack_smallest_three(comp: np.ndarray) -> np.ndarray:
    comp = comp.astype(np.uint32)
    largest = (comp >> 30).astype(int)
    q = np.zeros((len(comp), 4))
    remaining = comp.copy()
    for i in range(3, -1, -1):  # unpack in reverse order of packing
        mask = largest != i
        mag = (remaining & 511).astype(np.float64)
        neg = ((remaining >> 9) & 1).astype(bool)
        val = np.sqrt(0.5) * mag / 511
        q[:, i] = np.where(mask, np.where(neg, -val, val), q[:, i])
        remaining = np.where(mask, remaining >> 10, remaining)
    rows = np.arange(len(comp))
    q[rows, largest] = np.sqrt(np.maximum(0.0, 1 - (q**2).sum(axis=1)))
    return q


def _quantize_sh(x: np.ndarray, bits: int) -> np.ndarray:
    bucket = 1 << (8 - bits)
    q = np.round(x * 128.0).astype(np.int32) + 128
    q = (q + bucket // 2) // bucket * bucket
    return np.clip(q, 0, 255).astype(np.uint8)


def encode_spz(cloud: GaussianCloud) -> bytes:
    """Serialize to .spz v3 (gzip)."""
    n, deg = len(cloud), cloud.sh_degree
    flags = SPZ_FLAG_ANTIALIASED if cloud.antialiased else 0
    header = struct.pack("<IIIBBBB", SPZ_MAGIC, 3, n, deg, SPZ_FRACTIONAL_BITS, flags, 0)

    limit = (1 << 23) - 1
    fixed = np.clip(np.round(cloud.means.astype(np.float64) * (1 << SPZ_FRACTIONAL_BITS)), -limit, limit)
    pos = fixed.astype("<i4").reshape(-1, 1).view(np.uint8)[:, :3]  # low 3 bytes, two's complement

    alphas = _u8(sigmoid(cloud.opacities.astype(np.float64)) * 255)
    colors = _u8(cloud.sh0 * (SPZ_COLOR_SCALE * 255) + 0.5 * 255)
    scales = _u8((cloud.scales + 10.0) * 16.0)
    wxyz = cloud.quats.astype(np.float64)
    rots = _pack_smallest_three(wxyz[:, [1, 2, 3, 0]]).astype("<u4").view(np.uint8)

    parts = [header, pos.tobytes(), alphas.tobytes(), colors.tobytes(), scales.tobytes(), rots.tobytes()]
    if deg > 0:
        sh = cloud.shN.reshape(n, -1, 3)  # [point][coeff][channel], channel fastest
        packed = np.empty(sh.shape, np.uint8)
        packed[:, :3] = _quantize_sh(sh[:, :3], bits=5)
        packed[:, 3:] = _quantize_sh(sh[:, 3:], bits=4)
        parts.append(packed.tobytes())
    return gzip.compress(b"".join(parts), compresslevel=9, mtime=0)


def write_spz(path: Path, cloud: GaussianCloud) -> int:
    data = encode_spz(cloud)
    Path(path).write_bytes(data)
    return len(data)


def decode_spz(data: bytes) -> GaussianCloud:
    """Read .spz v2 or v3 (gzip). Used for tests and inspection."""
    raw = gzip.decompress(data)
    magic, version, n, deg, frac, flags, _ = struct.unpack_from("<IIIBBBB", raw, 0)
    if magic != SPZ_MAGIC or version not in (2, 3):
        raise ValueError(f"unsupported spz (magic={magic:#x}, version={version})")
    off = 16

    def take(nbytes: int) -> np.ndarray:
        nonlocal off
        out = np.frombuffer(raw, np.uint8, nbytes, off)
        off += nbytes
        return out

    p = take(n * 9).reshape(n * 3, 3).astype(np.int32)
    fixed = p[:, 0] | (p[:, 1] << 8) | (p[:, 2] << 16)
    fixed = np.where(fixed & 0x800000, fixed - (1 << 24), fixed)
    means = (fixed / (1 << frac)).reshape(n, 3).astype(np.float32)
    alpha = take(n).astype(np.float64) / 255
    colors = (take(n * 3).reshape(n, 3) - 0.5 * 255) / (SPZ_COLOR_SCALE * 255)
    scales = take(n * 3).reshape(n, 3) / 16.0 - 10.0
    if version >= 3:
        xyzw = _unpack_smallest_three(take(n * 4).view("<u4"))
    else:
        xyz = take(n * 3).reshape(n, 3) / 127.5 - 1.0
        w = np.sqrt(np.maximum(0.0, 1 - (xyz**2).sum(axis=1)))
        xyzw = np.hstack([xyz, w[:, None]])
    k = SH_DIM[deg]
    shN = (take(n * k * 3).reshape(n, k, 3).astype(np.float64) - 128) / 128 if k else np.zeros((n, 0, 3))
    return GaussianCloud(
        means=means,
        scales=scales.astype(np.float32),
        quats=xyzw[:, [3, 0, 1, 2]].astype(np.float32),
        opacities=logit(alpha).astype(np.float32),
        sh0=colors.astype(np.float32),
        shN=shN.astype(np.float32),
        antialiased=bool(flags & SPZ_FLAG_ANTIALIASED),
    )


def read_spz(path: Path) -> GaussianCloud:
    return decode_spz(Path(path).read_bytes())
