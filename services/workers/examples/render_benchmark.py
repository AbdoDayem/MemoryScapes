"""A synthetic street scene of 1.5M gaussians for measuring the viewer's frame rate.

    .venv/Scripts/python examples/render_benchmark.py                 # -> data/storage/render_bench
    .venv/Scripts/python examples/render_benchmark.py --count 3000000

No training: the gaussians are placed on a ground plane, two facades, boxes and a distant
shell, with sizes like a trained outdoor capture's. The LODs come from the splat stage's
own `DEFAULT_LODS` and `make_lod`, and a publish manifest is written, so the API and viewer
load it like a real memory: open /memories/render_bench?perf (docs/RENDER_SPEED.md). When
Spark's build-lod is installed, the streaming LoD trees of lod0 (desktops) and lod1 (phones)
are built too, as publish does (add &stream=0 to the address to compare without them, and
&device=mobile to try the phone path on a desktop).
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from memoryscapes_workers.visual import lod_tree, splat_io, splat_post
from memoryscapes_workers.visual.splat_io import GaussianCloud, logit

C0 = 0.28209479177387814  # SH degree-0 constant


def scene(n_total: int, seed: int = 0) -> GaussianCloud:
    rng = np.random.default_rng(seed)

    def surface(n, origin, u, v, size_m, color):
        a = rng.random((n, 2))
        pts = origin + a[:, :1] * u + a[:, 1:] * v + rng.normal(0, 0.02, (n, 3))
        col = color + rng.normal(0, 0.12, (n, 3)) + 0.15 * np.sin(pts[:, :1] * 3 + pts[:, 2:] * 2)
        sc = np.log(np.full((n, 3), size_m) * rng.uniform(0.5, 1.6, (n, 3)))
        sc[:, 2] -= 1.5  # flattish, like trained surface splats
        return pts, col, sc

    shares = {"ground": 0.35, "left": 0.2, "right": 0.2, "boxes": 0.1, "far": 0.15}
    n = {k: int(s * n_total) for k, s in shares.items()}
    parts = [
        surface(n["ground"], np.array([-6, 0, -40.0]), np.array([12, 0, 0.0]), np.array([0, 0, 45.0]), 0.03,
                np.array([0.35, 0.33, 0.3])),
        surface(n["left"], np.array([-6, 0, -40.0]), np.array([0, 12, 0.0]), np.array([0, 0, 45.0]), 0.035,
                np.array([0.6, 0.45, 0.35])),
        surface(n["right"], np.array([6, 0, -40.0]), np.array([0, 12, 0.0]), np.array([0, 0, 45.0]), 0.035,
                np.array([0.4, 0.45, 0.55])),
    ]
    m = n["boxes"]  # clutter along the street
    centers = np.stack([rng.uniform(-5, 5, 60), np.zeros(60), rng.uniform(-38, 3, 60)], 1)
    idx = rng.integers(0, 60, m)
    parts.append((centers[idx] + rng.uniform([-0.4, 0, -0.4], [0.4, 1.2, 0.4], (m, 3)),
                  rng.random((60, 3))[idx] + rng.normal(0, 0.05, (m, 3)),
                  np.log(np.full((m, 3), 0.02) * rng.uniform(0.5, 1.5, (m, 3)))))
    m = n["far"]  # sky and distant buildings: big splats on a shell
    d = rng.normal(size=(m, 3))
    d[:, 1] = np.abs(d[:, 1])
    d /= np.linalg.norm(d, axis=1, keepdims=True)
    parts.append((d * rng.uniform(60, 120, (m, 1)), np.array([0.45, 0.55, 0.75]) + rng.normal(0, 0.05, (m, 3)),
                  np.log(np.full((m, 3), 1.2) * rng.uniform(0.5, 1.5, (m, 3)))))

    means = np.concatenate([p[0] for p in parts]).astype(np.float32)
    cols = np.clip(np.concatenate([p[1] for p in parts]), 0, 1).astype(np.float32)
    k = len(means)
    q = rng.normal(size=(k, 4)).astype(np.float32)
    q /= np.linalg.norm(q, axis=1, keepdims=True)
    return GaussianCloud(
        means=means,
        scales=np.concatenate([p[2] for p in parts]).astype(np.float32),
        quats=q,
        opacities=logit(rng.uniform(0.6, 0.98, k)).astype(np.float32),
        sh0=((cols - 0.5) / C0).astype(np.float32),
        shN=rng.normal(0, 0.03, (k, 15, 3)).astype(np.float32),
    )


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--root", type=Path, default=Path("data/storage"), help="local storage root (STORAGE_ROOT)")
    ap.add_argument("--id", default="render_bench", help="memory id")
    ap.add_argument("--count", type=int, default=1_500_000, help="gaussians before the LODs")
    args = ap.parse_args()

    out = args.root / args.id
    (out / "splat" / "lods").mkdir(parents=True, exist_ok=True)
    (out / "publish").mkdir(parents=True, exist_ok=True)
    cloud = scene(args.count)
    lods = []
    for i, spec in enumerate(splat_post.DEFAULT_LODS):
        lod = splat_post.make_lod(cloud, spec)
        size = splat_io.write_spz(out / "splat" / "lods" / f"lod{i}.spz", lod)
        lods.append({"url": f"splat/lods/lod{i}.spz", "count": len(lod)})
        print(f"lod{i}: {len(lod):,} gaussians, SH{lod.sh_degree}, {size / 1e6:.1f} MB")

    tool = lod_tree.find_tool()
    if tool:
        for n in (0, 1):  # as publish's default lod_tree_levels
            tree = lod_tree.build(tool, out / "splat" / "lods" / f"lod{n}.spz", out / "publish" / "lod",
                                  name=f"lod{n}")
            lods[n]["rad"] = {"url": f"publish/lod/{tree.header}", "chunks": [f"publish/lod/{c}" for c in tree.chunks],
                              **({"count": tree.count} if tree.count else {})}
            print(f"lod{n} LoD tree: {len(tree.chunks)} chunks, {tree.bytes / 1e6:.1f} MB")
    else:
        print("no LoD trees: build-lod isn't installed (deploy/setup.py)")

    manifest = {
        "id": args.id,
        "version": 7,
        "coord_frame": {"units": "meters", "up": "+y"},
        "splat": {"lods": lods},
        "navigation": {
            "spawn": {"position": [0, 1.6, 2], "look_at": [0, 1.6, -10]},
            "walkable_bounds": {"type": "aabb", "min": [-5, 0, -38], "max": [5, 4, 4]},
            "floor_y": 0.0,
        },
        "audio": {"sources": []},
        "meta": {"title": f"Render benchmark ({args.count:,} synthetic gaussians)", "visibility": "private"},
    }
    (out / "publish" / "manifest.json").write_text(json.dumps(manifest, indent=1))
    # Stage markers, so the API reports the memory as ready. Not real stage runs: running
    # the pipeline on this id redoes the stages.
    for stage in ("splat", "publish"):
        (out / stage / "_stage.json").write_text(json.dumps({"stage": stage, "version": "benchmark"}))
    print(f"wrote {out}; open /memories/{args.id}?perf")


if __name__ == "__main__":
    main()
