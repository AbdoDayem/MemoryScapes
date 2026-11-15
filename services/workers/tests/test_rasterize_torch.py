"""The pure-PyTorch rasterizer must match gsplat, so a splat trained on a Mac behaves like
one trained with gsplat's CUDA kernels.

- Projection and SH are compared with gsplat's own PyTorch reference implementation.
- Compositing is compared with a per-pixel loop that transcribes gsplat's CUDA forward
  kernel (RasterizeToPixels3DGSFwd.cu), since gsplat's reference compositor needs CUDA.
- Gradients are checked against finite differences.
"""

import math

import numpy as np
import pytest

torch = pytest.importorskip("torch")
rt = pytest.importorskip("memoryscapes_workers.visual.rasterize_torch")

W, H = 45, 37  # deliberately not a multiple of the tile size


def make_gaussians(n=60, seed=0, dtype=torch.float32):
    g = torch.Generator().manual_seed(seed)

    def rand(*shape):
        return torch.rand(*shape, generator=g, dtype=dtype)

    means = torch.stack([(rand(n) - 0.5) * 2.4, (rand(n) - 0.5) * 2.0, 1.5 + rand(n) * 3.0], -1)
    return {
        "means": means,
        "quats": rand(n, 4) * 2 - 1,
        "scales": 0.03 + rand(n, 3) * 0.25,
        "opacities": rand(n),
        "colors": rand(n, 3),
    }


def camera(dtype=torch.float32):
    K = torch.tensor([[40.0, 0, W / 2 - 1.3], [0, 42.0, H / 2 + 0.7], [0, 0, 1]], dtype=dtype)
    viewmat = torch.eye(4, dtype=dtype)
    return viewmat, K


def reference_composite(means2d, conics, opacities, colors, radii, depths, bg, tile=16):
    """gsplat 1.5 CUDA forward, one pixel at a time, in float64."""
    m, con, opa, col = (np.asarray(t.detach(), np.float64) for t in (means2d, conics, opacities, colors))
    r = np.asarray(radii, np.float64)
    depth = np.asarray(depths.detach(), np.float64)
    tw, th = math.ceil(W / tile), math.ceil(H / tile)
    # IntersectTile.cu: tile_min inclusive, tile_max exclusive; radii == 0 means culled.
    tmin = np.floor((m - r) / tile)
    tmax = np.ceil((m + r) / tile)
    tmin[:, 0], tmax[:, 0] = tmin[:, 0].clip(0, tw), tmax[:, 0].clip(0, tw)
    tmin[:, 1], tmax[:, 1] = tmin[:, 1].clip(0, th), tmax[:, 1].clip(0, th)
    order = np.argsort(depth, kind="stable")
    rgb = np.zeros((H, W, 3))
    alpha_out = np.zeros((H, W, 1))
    for py in range(H):
        for px in range(W):
            tx, ty = px // tile, py // tile
            T, c = 1.0, np.zeros(3)
            for gi in order:
                if r[gi, 0] <= 0 and r[gi, 1] <= 0:
                    continue
                if not (tmin[gi, 0] <= tx < tmax[gi, 0] and tmin[gi, 1] <= ty < tmax[gi, 1]):
                    continue
                dx, dy = m[gi, 0] - (px + 0.5), m[gi, 1] - (py + 0.5)
                a, b, cc = con[gi]
                sigma = 0.5 * (a * dx * dx + cc * dy * dy) + b * dx * dy
                alpha = min(0.999, opa[gi] * math.exp(-sigma))
                if sigma < 0 or alpha < 1.0 / 255.0:
                    continue
                next_t = T * (1 - alpha)
                if next_t <= 1e-4:
                    break
                c += col[gi] * alpha * T
                T = next_t
            rgb[py, px] = c + T * np.asarray(bg, np.float64)
            alpha_out[py, px] = 1 - T
    return rgb, alpha_out


def test_projection_matches_gsplat_reference():
    ref = pytest.importorskip("gsplat.cuda._torch_impl")
    g = make_gaussians(200)
    viewmat, K = camera()
    # A few gaussians behind the camera or far outside the view.
    g["means"][:5, 2] = -1.0
    g["means"][5:10, 0] = 40.0

    covars, _ = ref._quat_scale_to_covar_preci(g["quats"], g["scales"], True, False, triu=False)
    radii_ref, m2_ref, depth_ref, con_ref, _ = ref._fully_fused_projection(
        g["means"], covars, viewmat[None], K[None], W, H, eps2d=0.3)
    m2, con, depth, radii, visible, _ = rt.project(g["means"], g["quats"], g["scales"], viewmat, K, W, H)

    assert torch.equal(radii, radii_ref[0])
    vis = radii_ref[0].all(dim=-1) > 0
    assert torch.equal(visible, vis) and 100 < int(vis.sum()) < 200
    torch.testing.assert_close(m2[vis], m2_ref[0][vis], rtol=1e-5, atol=1e-4)
    torch.testing.assert_close(con[vis], con_ref[0][vis], rtol=1e-4, atol=1e-6)
    torch.testing.assert_close(depth, depth_ref[0])


def test_opacity_aware_radii():
    """gsplat's CUDA projection shrinks boxes to where alpha >= 1/255 (and culls below)."""
    g = make_gaussians(200)
    viewmat, K = camera()
    g["opacities"][:20] = 0.5 / 255
    r_full = rt.project(g["means"], g["quats"], g["scales"], viewmat, K, W, H).radii
    _, _, _, r_opa, vis, _ = rt.project(g["means"], g["quats"], g["scales"], viewmat, K, W, H,
                                        opacities=g["opacities"])
    assert not vis[:20].any() and (r_opa[:20] == 0).all()
    assert (r_opa <= r_full).all() and (r_opa < r_full).any()


@pytest.mark.parametrize("degree", [0, 1, 2, 3])
def test_spherical_harmonics_match_gsplat_reference(degree):
    ref = pytest.importorskip("gsplat.cuda._torch_impl")
    g = torch.Generator().manual_seed(degree)
    dirs = torch.nn.functional.normalize(torch.randn(500, 3, generator=g), dim=-1)
    coeffs = torch.randn(500, 16, 3, generator=g)
    torch.testing.assert_close(rt.spherical_harmonics(degree, dirs, coeffs),
                               ref._spherical_harmonics(degree, dirs, coeffs), rtol=1e-5, atol=1e-5)


def test_antialiased_compensation_matches_gsplat_reference():
    ref = pytest.importorskip("gsplat.cuda._torch_impl")
    g = make_gaussians(200)
    g["scales"] = g["scales"] * 0.1  # small on screen, where compensation matters
    viewmat, K = camera()
    covars, _ = ref._quat_scale_to_covar_preci(g["quats"], g["scales"], True, False, triu=False)
    radii_ref, *_, comp_ref = ref._fully_fused_projection(
        g["means"], covars, viewmat[None], K[None], W, H, eps2d=0.3, calc_compensations=True)
    p = rt.project(g["means"], g["quats"], g["scales"], viewmat, K, W, H, antialiased=True)
    vis = radii_ref[0].all(dim=-1) > 0
    torch.testing.assert_close(p.compensations[vis], comp_ref[0][vis], rtol=1e-4, atol=1e-6)
    assert p.compensations[vis].min() < 0.5  # the test exercises real compensation


@pytest.mark.parametrize("case", ["random", "opaque_stack", "faint", "antialiased"])
def test_compositing_matches_gsplat_kernel(case):
    g = make_gaussians(80, seed=1)
    if case == "opaque_stack":
        # Several large, nearly opaque gaussians in front: pixels terminate early.
        g["opacities"][:6] = 1.0
        g["scales"][:6] = 0.6
        g["means"][:6] = torch.tensor([0.0, 0.0, 1.2]) + 0.05 * torch.arange(6.0)[:, None]
    elif case == "faint":
        g["opacities"] = g["opacities"] * 0.02  # many below the 1/255 threshold
    elif case == "antialiased":
        g["scales"] = g["scales"] * 0.2
    aa = case == "antialiased"
    viewmat, K = camera()
    bg = torch.tensor([0.2, 0.5, 0.9])
    rgb, alpha, info = rt.rasterization(g["means"], g["quats"], g["scales"], g["opacities"],
                                        g["colors"], viewmat[None], K[None], W, H,
                                        backgrounds=bg[None],
                                        rasterize_mode="antialiased" if aa else "classic")
    p = rt.project(g["means"], g["quats"], g["scales"], viewmat, K, W, H,
                   opacities=g["opacities"], antialiased=aa)
    opa = g["opacities"] * p.compensations if aa else g["opacities"]
    ref_rgb, ref_alpha = reference_composite(p.means2d, p.conics, opa, g["colors"], p.radii,
                                             p.depths, bg)
    np.testing.assert_allclose(rgb[0].numpy(), ref_rgb, atol=2e-5)
    np.testing.assert_allclose(alpha[0].numpy(), ref_alpha, atol=2e-5)
    assert rgb.shape == (1, H, W, 3) and info["radii"].shape == (1, 80, 2)
    if case == "opaque_stack":
        assert ref_alpha.max() > 0.999


def test_gradients_match_finite_differences():
    g = make_gaussians(10, seed=3, dtype=torch.float64)
    g["opacities"] = 0.3 + 0.6 * g["opacities"]
    viewmat, K = camera(torch.float64)
    bg = torch.tensor([[0.3, 0.1, 0.7]], dtype=torch.float64)

    def render(means, quats, scales, opacities, colors):
        rgb, alpha, _ = rt.rasterization(means, quats, scales, opacities, colors, viewmat[None],
                                         K[None], W, H, backgrounds=bg)
        return rgb, alpha

    inputs = tuple(g[k].clone().requires_grad_(True)
                   for k in ("means", "quats", "scales", "opacities", "colors"))
    assert torch.autograd.gradcheck(render, inputs, eps=1e-6, atol=1e-5, rtol=1e-3, fast_mode=True)


def test_info_feeds_gsplat_default_strategy():
    """DefaultStrategy reads means2d gradients and radii from info; check it accumulates."""
    strategy_mod = pytest.importorskip("gsplat.strategy")
    g = make_gaussians(50, seed=4)
    params = torch.nn.ParameterDict({
        "means": torch.nn.Parameter(g["means"]), "scales": torch.nn.Parameter(g["scales"].log()),
        "quats": torch.nn.Parameter(g["quats"]),
        "opacities": torch.nn.Parameter(torch.logit(g["opacities"].clamp(0.05, 0.95))),
        "colors": torch.nn.Parameter(g["colors"]),
    })
    optimizers = {k: torch.optim.Adam([{"params": p, "lr": 1e-3, "name": k}]) for k, p in params.items()}
    strategy = strategy_mod.DefaultStrategy(refine_start_iter=0, refine_every=1000)
    state = strategy.initialize_state(scene_scale=3.0)
    viewmat, K = camera()
    rgb, _, info = rt.rasterization(params["means"], params["quats"], params["scales"].exp(),
                                    params["opacities"].sigmoid(), params["colors"], viewmat[None],
                                    K[None], W, H)
    strategy.step_pre_backward(params, optimizers, state, 1, info)
    rgb.sum().backward()
    strategy.step_post_backward(params, optimizers, state, 1, info, packed=False)
    seen = (info["radii"][0] > 0).all(dim=-1)
    assert torch.equal(state["count"] > 0, seen)
    assert (state["grad2d"][seen] > 0).any()


def test_expected_depth_render_mode():
    """RGB+ED: a 4th channel with the alpha-weighted camera depth, like gsplat's."""
    viewmat, K = camera()
    one = {"means": torch.tensor([[0.0, 0.0, 3.0]]), "quats": torch.tensor([[1.0, 0, 0, 0]]),
           "scales": torch.full((1, 3), 0.6), "opacities": torch.tensor([0.9]), "colors": torch.ones(1, 3)}
    out, alpha, _ = rt.rasterization(**one, viewmats=viewmat[None], Ks=K[None], width=W, height=H,
                                     render_mode="RGB+ED")
    assert out.shape == (1, H, W, 4)
    covered = alpha[0, ..., 0] > 0.05
    assert torch.allclose(out[0, ..., 3][covered], torch.tensor(3.0), atol=1e-4)  # one surface at z = 3
    rgb_only, _, _ = rt.rasterization(**one, viewmats=viewmat[None], Ks=K[None], width=W, height=H)
    assert torch.allclose(out[0, ..., :3], rgb_only[0])  # the colours are unchanged
    # Two overlapping surfaces: the depth lies between them, nearer the front one.
    two = {k: torch.cat([v, v]) for k, v in one.items()}
    two["means"] = torch.tensor([[0.0, 0.0, 2.0], [0.0, 0.0, 6.0]])
    out2, _, _ = rt.rasterization(**two, viewmats=viewmat[None], Ks=K[None], width=W, height=H,
                                  render_mode="RGB+ED")
    centre = out2[0, H // 2, W // 2, 3]
    assert 2.0 < centre < 4.0
    # and it carries gradients back to the positions
    g = {k: v.clone().requires_grad_(k == "means") for k, v in one.items()}
    o, _, _ = rt.rasterization(**g, viewmats=viewmat[None], Ks=K[None], width=W, height=H, render_mode="RGB+ED")
    o[0, ..., 3].sum().backward()
    assert g["means"].grad is not None and g["means"].grad[0, 2] > 0
