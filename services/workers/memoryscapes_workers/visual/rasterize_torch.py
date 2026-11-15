"""Pure-PyTorch Gaussian splat rasterizer (Apple Silicon MPS, or CPU).

A drop-in for ``gsplat.rasterization`` (classic mode, unpacked, RGB) so the same
training loop runs on a Mac for testing and on an NVIDIA GPU with gsplat's CUDA
kernels in production. It reproduces gsplat 1.5's forward pass (checked against
gsplat's reference code in ``tests/test_rasterize_torch.py``):

- EWA projection of each 3D covariance to screen space with a clamped perspective
  Jacobian, plus a 0.3 px low-pass (``eps2d``) on the 2D covariance.
- Opacity-aware screen-space bounding boxes (gaussians below 1/255 opacity are culled),
  binned into tiles.
- Per tile, gaussians sorted front to back; sigma = 0.5 d^T S^-1 d,
  alpha = min(0.999, o * exp(-sigma)), skipped when below 1/255 (or sigma < 0); colors
  composited with transmittance T, and a pixel stops at the gaussian that would take T
  to 1e-4 or below (that gaussian is excluded, as in gsplat's kernel).
- View-dependent color from real spherical harmonics up to degree 3.
- ``rasterize_mode="antialiased"`` (Mip-Splatting style): opacities are scaled by
  sqrt(det(cov2d) / det(cov2d + eps2d I)) so small gaussians don't brighten when blurred.

Speed: tiles are grouped by how many gaussians overlap them, and each group is
evaluated as one dense [tiles, pixels, K] block with K the group's power-of-two
capacity, so padding costs at most 2x. Autograd does the backward pass. It is still far
slower than gsplat's fused CUDA kernels; use it to test the pipeline at reduced
resolution and step counts.

``info`` carries what gsplat's densification strategies read (``means2d`` with
gradients, per-axis ``radii``, ...).
"""

from __future__ import annotations

import math
from typing import NamedTuple

import torch
from torch import Tensor

TILE = 16
ALPHA_MIN = 1.0 / 255.0
ALPHA_MAX = 0.999
T_MIN = 1e-4  # a pixel is done once its transmittance would reach this
EXTEND = 3.33  # max bounding-box half-size in standard deviations

# Real SH basis constants (same as 3DGS / gsplat).
C0 = 0.28209479177387814
C1 = 0.4886025119029199
C2 = (1.0925484305920792, -1.0925484305920792, 0.31539156525252005, -1.0925484305920792,
      0.5462742152960396)
C3 = (-0.5900435899266435, 2.890611442640554, -0.4570457994644658, 0.3731763325901154,
      -0.4570457994644658, 1.445305721320277, -0.5900435899266435)


def spherical_harmonics(degree: int, dirs: Tensor, coeffs: Tensor) -> Tensor:
    """Evaluate SH colors. dirs [N,3] (normalized), coeffs [N,K,3] -> [N,3]."""
    x, y, z = dirs.unbind(-1)
    out = C0 * coeffs[:, 0]
    if degree >= 1:
        out = out - C1 * y[:, None] * coeffs[:, 1] + C1 * z[:, None] * coeffs[:, 2] \
            - C1 * x[:, None] * coeffs[:, 3]
    if degree >= 2:
        xx, yy, zz, xy, yz, xz = x * x, y * y, z * z, x * y, y * z, x * z
        basis2 = (C2[0] * xy, C2[1] * yz, C2[2] * (2 * zz - xx - yy), C2[3] * xz, C2[4] * (xx - yy))
        for i, b in enumerate(basis2):
            out = out + b[:, None] * coeffs[:, 4 + i]
    if degree >= 3:
        basis3 = (
            C3[0] * y * (3 * xx - yy),
            C3[1] * xy * z,
            C3[2] * y * (4 * zz - xx - yy),
            C3[3] * z * (2 * zz - 3 * xx - 3 * yy),
            C3[4] * x * (4 * zz - xx - yy),
            C3[5] * z * (xx - yy),
            C3[6] * x * (xx - 3 * yy),
        )
        for i, b in enumerate(basis3):
            out = out + b[:, None] * coeffs[:, 9 + i]
    return out


def quat_to_rotmat(q: Tensor) -> Tensor:
    """wxyz (any norm) -> [N,3,3]."""
    w, x, y, z = torch.nn.functional.normalize(q, dim=-1).unbind(-1)
    return torch.stack([
        1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y),
        2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x),
        2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y),
    ], dim=-1).reshape(-1, 3, 3)


class Projection(NamedTuple):
    means2d: Tensor  # [N, 2] pixels
    conics: Tensor  # [N, 3] upper triangle of the inverse 2D covariance
    depths: Tensor  # [N] camera-space z
    radii: Tensor  # [N, 2] int32 bounding-box half sizes, (0, 0) when culled
    visible: Tensor  # [N] bool
    compensations: Tensor | None  # [N] antialiasing opacity scale, if requested


def project(means, quats, scales, viewmat, K, width, height, near=0.01, far=1e10, eps2d=0.3,
            opacities=None, antialiased=False) -> Projection:
    """Project gaussians into one camera.

    With ``opacities`` the bounding box shrinks to where the gaussian's alpha stays
    above 1/255 (gsplat passes opacities for this too); without, it is 3.33 sigma.
    With ``antialiased`` the box uses the compensated opacity, as gsplat does.
    """
    R, t = viewmat[:3, :3], viewmat[:3, 3]
    p = means @ R.T + t
    x, y, z = p.unbind(-1)
    fx, fy, cx, cy = K[0, 0], K[1, 1], K[0, 2], K[1, 2]
    z_safe = torch.where(z > near, z, torch.ones_like(z))

    # Clamp the Jacobian for gaussians far outside the view (as gsplat does).
    tan_x, tan_y = 0.5 * width / fx, 0.5 * height / fy
    lim_xp, lim_xn = (width - cx) / fx + 0.3 * tan_x, cx / fx + 0.3 * tan_x
    lim_yp, lim_yn = (height - cy) / fy + 0.3 * tan_y, cy / fy + 0.3 * tan_y
    tx = z_safe * torch.clamp(x / z_safe, -lim_xn, lim_xp)
    ty = z_safe * torch.clamp(y / z_safe, -lim_yn, lim_yp)
    zeros = torch.zeros_like(z)
    J = torch.stack([fx / z_safe, zeros, -fx * tx / z_safe**2,
                     zeros, fy / z_safe, -fy * ty / z_safe**2], dim=-1).reshape(-1, 2, 3)

    M = quat_to_rotmat(quats) * scales[:, None, :]
    cov_world = M @ M.transpose(1, 2)
    cov_cam = R @ cov_world @ R.T
    cov2d = J @ cov_cam @ J.transpose(1, 2)
    a0, c0 = cov2d[:, 0, 0], cov2d[:, 1, 1]
    b = 0.5 * (cov2d[:, 0, 1] + cov2d[:, 1, 0])
    a, c = a0 + eps2d, c0 + eps2d
    det_raw = a * c - b * b
    det = det_raw.clamp_min(1e-10)
    conics = torch.stack([c / det, -b / det, a / det], dim=-1)
    comp = torch.sqrt(torch.clamp((a0 * c0 - b * b) / det, min=0.0)) if antialiased else None

    means2d = torch.stack([fx * x / z_safe + cx, fy * y / z_safe + cy], dim=-1)
    with torch.no_grad():
        visible = (z > near) & (z < far) & (det_raw > 0)
        if opacities is None:
            extend = torch.full_like(z, EXTEND)
        else:
            opa = opacities.detach() if comp is None else opacities.detach() * comp.detach()
            visible &= opa >= ALPHA_MIN
            extend = torch.sqrt(2.0 * torch.log(opa.clamp_min(ALPHA_MIN) / ALPHA_MIN))
            extend = extend.clamp(max=EXTEND)
        rx = torch.ceil(extend * torch.sqrt(a.clamp_min(0)))
        ry = torch.ceil(extend * torch.sqrt(c.clamp_min(0)))
        mx, my = means2d[:, 0], means2d[:, 1]
        visible &= ((rx > 0) | (ry > 0)) & (mx + rx > 0) & (mx - rx < width) \
            & (my + ry > 0) & (my - ry < height)
        radii = torch.stack([rx, ry], dim=-1) * visible[:, None]
    return Projection(means2d, conics, z, radii.to(torch.int32), visible, comp)


def intersect_tiles(means2d, radii, depths, visible, width, height, tile=TILE):
    """(tile, gaussian) pairs for one camera, sorted by tile and then front to back.

    Tile ranges follow gsplat: [floor((m - r) / tile), ceil((m + r) / tile)).
    Returns (tile_ids [M], gaussian_ids [M], tiles_x, tiles_y), on the CPU.
    """
    m = means2d.detach().float().cpu()
    r = radii.cpu().float()
    d = depths.detach().float().cpu()
    tw, th = math.ceil(width / tile), math.ceil(height / tile)
    ids = torch.nonzero(visible.cpu()).squeeze(1)
    empty = torch.zeros(0, dtype=torch.long)
    if len(ids) == 0:
        return empty, empty, tw, th
    tm, tr = m[ids] / tile, r[ids] / tile
    x0 = (tm[:, 0] - tr[:, 0]).floor().clamp(0, tw).long()
    x1 = (tm[:, 0] + tr[:, 0]).ceil().clamp(0, tw).long()
    y0 = (tm[:, 1] - tr[:, 1]).floor().clamp(0, th).long()
    y1 = (tm[:, 1] + tr[:, 1]).ceil().clamp(0, th).long()
    wx = x1 - x0
    counts = wx * (y1 - y0)
    total = int(counts.sum())
    if total == 0:
        return empty, empty, tw, th
    g = torch.repeat_interleave(torch.arange(len(ids)), counts)
    start = torch.cumsum(counts, 0) - counts
    k = torch.arange(total) - start[g]
    wxg = wx[g]
    tile_ids = (y0[g] + k // wxg) * tw + (x0[g] + k % wxg)
    rank = torch.empty(len(ids), dtype=torch.long)
    rank[torch.argsort(d[ids], stable=True)] = torch.arange(len(ids))
    order = torch.argsort(tile_ids * len(ids) + rank[g])
    return tile_ids[order], ids[g[order]], tw, th


def _ceil_pow2(x: int) -> int:
    return 1 << max(0, (x - 1).bit_length())


def _floor_pow2(x: int) -> int:
    return 1 << max(0, x.bit_length() - 1)


def bucket_tiles(tile_ids: Tensor, gauss_ids: Tensor, n_tiles: int, pad: int):
    """Group non-empty tiles by load into dense [tiles, K] tables (power-of-two K).

    Yields (tiles [T], table [T, K]) with unused slots set to ``pad``.
    """
    if len(tile_ids) == 0:
        return
    per_tile = torch.bincount(tile_ids, minlength=n_tiles)
    tile_start = torch.cumsum(per_tile, 0) - per_tile
    slot = torch.arange(len(tile_ids)) - tile_start[tile_ids]
    cap = torch.where(per_tile > 0, 2 ** torch.ceil(torch.log2(per_tile.clamp_min(1).double())).long(),
                      torch.zeros_like(per_tile))
    for k in torch.unique(cap[cap > 0]).tolist():
        tiles = torch.nonzero(cap == k).squeeze(1)
        row = torch.full((n_tiles,), -1, dtype=torch.long)
        row[tiles] = torch.arange(len(tiles))
        sel = row[tile_ids] >= 0
        table = torch.full((len(tiles), k), pad, dtype=torch.long)
        table[row[tile_ids[sel]], slot[sel]] = gauss_ids[sel]
        yield tiles, table


def composite(pix, idx, means2d, conics, opacities, colors):
    """Front-to-back alpha compositing for a block of tiles.

    pix [T, P, 2] pixel centers; idx [T, K] gaussian ids sorted front to back (the
    padding id must have zero opacity). Returns (rgb [T, P, C], transmittance [T, P]).
    """
    d = means2d[idx][:, None] - pix[:, :, None]  # [T, P, K, 2]
    cn = conics[idx][:, None]  # [T, 1, K, 3]
    sigma = 0.5 * (cn[..., 0] * d[..., 0] ** 2 + cn[..., 2] * d[..., 1] ** 2) \
        + cn[..., 1] * d[..., 0] * d[..., 1]
    alpha = torch.clamp(opacities[idx][:, None] * torch.exp(-sigma.clamp_min(0.0)), max=ALPHA_MAX)
    with torch.no_grad():
        valid = (sigma >= 0) & (alpha >= ALPHA_MIN)
        log_t = torch.cumsum(torch.log1p(-alpha * valid), dim=-1)
        valid &= log_t > math.log(T_MIN)  # stop before the gaussian that ends the pixel
    alpha = torch.where(valid, alpha, torch.zeros_like(alpha))
    log1m = torch.log1p(-alpha)
    log_t = torch.cumsum(log1m, dim=-1)
    weights = alpha * torch.exp(log_t - log1m)  # alpha * T before this gaussian
    rgb = torch.einsum("tpk,tkc->tpc", weights, colors[idx])
    return rgb, torch.exp(log_t[..., -1])


def rasterize_one(means2d, conics, colors, opacities, tile_ids, gauss_ids, tw, th, width, height,
                  background, tile: int = TILE, budget: int = 1 << 24):
    """Render one camera from its sorted tile intersections -> (rgb [H,W,C], alpha [H,W,1])."""
    dev, dtype = means2d.device, means2d.dtype
    n, ch = means2d.shape[0], colors.shape[-1]
    n_tiles, n_pix = tw * th, tile * tile
    # Index n is a padding gaussian with zero opacity.
    m2 = torch.cat([means2d, means2d.new_zeros(1, 2)])
    con = torch.cat([conics, conics.new_tensor([[1.0, 0.0, 1.0]])])
    opa = torch.cat([opacities, opacities.new_zeros(1)])
    col = torch.cat([colors, colors.new_zeros(1, ch)])

    py, px = torch.meshgrid(torch.arange(tile, device=dev), torch.arange(tile, device=dev),
                            indexing="ij")
    local = torch.stack([px.reshape(-1), py.reshape(-1)], dim=-1).to(dtype) + 0.5  # [P, 2]

    # Row n_tiles is a scratch tile that absorbs the padding rows of each block.
    rgb_tiles = background.to(dtype).expand(n_tiles + 1, n_pix, ch)
    t_tiles = means2d.new_ones(n_tiles + 1, n_pix)
    for tiles, table in bucket_tiles(tile_ids, gauss_ids, n_tiles, pad=n):
        k = table.shape[1]
        chunk = _floor_pow2(max(1, budget // (n_pix * k)))
        for s in range(0, len(tiles), chunk):
            # Pad each block to a power-of-two row count: MPS compiles a kernel per
            # tensor shape, and a small fixed set of shapes keeps those cached.
            rows = tiles[s : s + chunk]
            idx = table[s : s + chunk]
            padded = min(chunk, _ceil_pow2(len(rows)))
            if padded > len(rows):
                extra = padded - len(rows)
                rows = torch.cat([rows, rows.new_full((extra,), n_tiles)])
                idx = torch.cat([idx, idx.new_full((extra, k), n)])
            rows, idx = rows.to(dev), idx.to(dev)
            origin = torch.stack([rows % tw, rows // tw], dim=-1).to(dtype) * tile
            rgb, trans = composite(origin[:, None, :] + local[None], idx, m2, con, opa, col)
            rgb_tiles = rgb_tiles.index_put((rows,), rgb + trans[..., None] * background)
            t_tiles = t_tiles.index_put((rows,), trans)

    rgb = rgb_tiles[:n_tiles].reshape(th, tw, tile, tile, ch).permute(0, 2, 1, 3, 4)
    alpha = (1.0 - t_tiles[:n_tiles]).reshape(th, tw, tile, tile).permute(0, 2, 1, 3)
    rgb = rgb.reshape(th * tile, tw * tile, ch)[:height, :width]
    alpha = alpha.reshape(th * tile, tw * tile)[:height, :width, None]
    return rgb, alpha


def rasterization(
    means: Tensor, quats: Tensor, scales: Tensor, opacities: Tensor, colors: Tensor,
    viewmats: Tensor, Ks: Tensor, width: int, height: int, *, sh_degree: int | None = None,
    backgrounds: Tensor | None = None, near_plane: float = 0.01, far_plane: float = 1e10,
    eps2d: float = 0.3, packed: bool = False, rasterize_mode: str = "classic",
    render_mode: str = "RGB", tile_size: int = TILE, **_ignored,
):
    """Subset of gsplat.rasterization: returns (colors [C,H,W,3], alphas [C,H,W,1], info).
    render_mode "RGB+ED" adds a 4th channel, the expected depth (camera z composited with
    the same weights as colour, divided by the accumulated alpha), like gsplat's."""
    if rasterize_mode not in ("classic", "antialiased"):
        raise ValueError(f"Unknown rasterize_mode {rasterize_mode!r}")
    if render_mode not in ("RGB", "RGB+ED"):
        raise ValueError(f"Unknown render_mode {render_mode!r}")
    with_depth = render_mode == "RGB+ED"
    aa = rasterize_mode == "antialiased"
    n_cams = viewmats.shape[0]
    proj = [project(means, quats, scales, viewmats[ci], Ks[ci], width, height, near_plane,
                    far_plane, eps2d, opacities=opacities, antialiased=aa) for ci in range(n_cams)]
    # Render from slices of the stacked means2d so it sits upstream of the loss and
    # strategies can read its gradient after backward().
    means2d = torch.stack([p.means2d for p in proj])
    radii = torch.stack([p.radii for p in proj])
    renders, alphas = [], []
    for ci, (_, conics, depths, _, visible, comp) in enumerate(proj):
        if sh_degree is not None:
            campos = -viewmats[ci, :3, :3].T @ viewmats[ci, :3, 3]
            dirs = torch.nn.functional.normalize(means - campos, dim=-1)
            k = (sh_degree + 1) ** 2
            rgb = torch.clamp_min(spherical_harmonics(sh_degree, dirs, colors[:, :k]) + 0.5, 0.0)
        else:
            rgb = colors
        tile_ids, gauss_ids, tw, th = intersect_tiles(means2d[ci], radii[ci], depths, visible,
                                                      width, height, tile_size)
        bg = backgrounds[ci] if backgrounds is not None else rgb.new_zeros(rgb.shape[-1])
        if with_depth:  # composite z as one more channel (no background depth)
            rgb = torch.cat([rgb, depths[:, None].to(rgb.dtype)], dim=-1)
            bg = torch.cat([bg, bg.new_zeros(1)])
        opa = opacities if comp is None else opacities * comp
        img, alpha = rasterize_one(means2d[ci], conics, rgb, opa, tile_ids, gauss_ids, tw,
                                   th, width, height, bg, tile=tile_size)
        if with_depth:
            img = torch.cat([img[..., :3], img[..., 3:] / alpha.clamp_min(1e-6)], dim=-1)
        renders.append(img)
        alphas.append(alpha)
    info = {
        "means2d": means2d,  # [C, N, 2]; strategies call retain_grad() on it
        "radii": radii,  # [C, N, 2]
        "depths": torch.stack([p.depths for p in proj]),
        "gaussian_ids": None,
        "width": width,
        "height": height,
        "n_cameras": n_cams,
        "tile_size": tile_size,
        "tile_width": math.ceil(width / tile_size),
        "tile_height": math.ceil(height / tile_size),
    }
    return torch.stack(renders), torch.stack(alphas), info
