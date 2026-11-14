import { describe, expect, it } from "vitest";

import { compareClouds, runChecks } from "./checks";
import type { GaussianView } from "./ply";
import type { Camera, MemoryBundle } from "./types";

function grid(): Float32Array {
  const pts: number[] = [];
  for (let x = -5; x <= 5; x += 0.5) for (let z = 0; z <= 10; z += 0.5) pts.push(x, 0, z);
  for (let x = -5; x <= 5; x += 0.5) for (let y = 0; y <= 4; y += 0.5) pts.push(x, y, 10);
  return new Float32Array(pts);
}

function camera(i: number, t: number): Camera {
  // upright OpenCV camera looking along +Z, walking along +X
  return { frame_id: `f${i}`, t, width: 640, height: 480, K: [[500, 0, 320], [0, 500, 240], [0, 0, 1]], c2w: [[-1, 0, 0, -2 + i * 0.5], [0, -1, 0, 1.5], [0, 0, 1, 0], [0, 0, 0, 1]], image_key: null };
}

function splatFrom(points: Float32Array, scale = 1, offset = [0, 0, 0]): GaussianView {
  const positions = points.map((v, i) => v * scale + offset[i % 3]);
  const n = points.length / 3;
  return { count: n, positions, colors: new Float32Array(n * 3), opacity: new Float32Array(n).fill(0.9), scale: new Float32Array(n).fill(0.05) };
}

function bundle(over: Partial<MemoryBundle> = {}): MemoryBundle {
  const files = ["ingest/audio.wav", "ingest/meta.json", "poses/cameras.json", "poses/points3d.npy", "separate/stems.json", "separate/ambient.wav", "separate/stems/s1_water.wav", "localize/sources.json", "soundscape/scene.json", "splat/meta.json", "splat/metrics.json", "splat/point_cloud.ply"];
  return {
    id: "m",
    files: files.map((key) => ({ key, size: 1 })),
    cameras: Array.from({ length: 9 }, (_, i) => camera(i, i * 1.5)),
    points: grid(),
    stems: { sample_rate: 16000, channels: 2, layout: "stereo", duration_s: 12, window_times: [], stems: [{ id: "s1_water", label: "water", kind: "event", key: "m/separate/stems/s1_water.wav", level_dbfs: -30, activity: [], tag_score: 0.6 }], ambient_key: "m/separate/ambient.wav", tags: { water: 0.6 } },
    localized: { bounds: { min: [-6, -1, -1], max: [6, 5, 11] }, sources: [{ id: "s1_water", label: "water", kind: "event", stem_key: "m/separate/stems/s1_water.wav", position: [1, 0.5, 9.8], spread_m: 0.3, confidence: 0.9, method: "av+doa", level_dbfs: -30, level_db_at_1m: -12 }] },
    scene: { provider: { name: "stub", notes: [] }, ambient_bed: { key: "x", format: "stereo" }, sources: [{ id: "s1_water", label: "water", key: "k", position: [1, 0.5, 9.8], ref_distance: 8, rolloff: 1, gain: 1, loop: true, confidence: 0.9, method: "av+doa" }], satie: null },
    splatMeta: { units: "meters", up: "+y" },
    errors: [],
    ...over,
  };
}

const levelOf = (checks: ReturnType<typeof runChecks>, id: string) => checks.find((c) => c.id === id)?.level;

/** Deterministic uniform numbers in [0, 1). */
function rng(seed: number): () => number {
  let s = seed >>> 0;
  return () => ((s = (Math.imul(s, 1664525) + 1013904223) >>> 0) / 2 ** 32);
}

/** A non-repeating scene: random points on a ground plane, a back wall and a sphere. */
function scene(n: number, seed = 1): Float32Array {
  const r = rng(seed);
  const pts: number[] = [];
  for (let i = 0; i < n; i++) {
    const kind = i % 3;
    if (kind === 0) pts.push(-5 + 10 * r(), 0, 10 * r());
    else if (kind === 1) pts.push(-5 + 10 * r(), 4 * r(), 10);
    else {
      const u = 2 * r() - 1, th = 2 * Math.PI * r(), q = Math.sqrt(1 - u * u);
      pts.push(2 + 1.2 * q * Math.cos(th), 1.2 + 1.2 * u, 4 + 1.2 * q * Math.sin(th));
    }
  }
  return new Float32Array(pts);
}

/** What a trained splat looks like: dense on the surfaces, filling between them, plus sky. */
function trainedSplat(seed = 2): Float32Array {
  const r = rng(seed);
  const surface = scene(30000, seed);
  const between = Array.from({ length: 6000 * 3 }, (_, i) => [-5 + 10 * r(), 4 * r(), 10 * r()][i % 3]);
  const sky = Array.from({ length: 6000 * 3 }, (_, i) => [-40 + 80 * r(), 8 + 30 * r(), -30 + 80 * r()][i % 3]);
  return new Float32Array([...surface, ...between, ...sky]);
}

const transform = (xyz: Float32Array, scale: number, offset: number[]) => xyz.map((v, i) => v * scale + offset[i % 3]);

describe("compareClouds", () => {
  it("recognises the same cloud", () => {
    const g = grid();
    const a = compareClouds(g, splatFrom(g).positions);
    expect(a.medianGapM).toBeLessThan(0.01);
    expect(a.scaleRatio).toBeCloseTo(1, 1);
  });
  it("reports scale and offset of a misaligned splat", () => {
    const g = grid();
    const a = compareClouds(g, splatFrom(g, 0.5).positions);
    expect(a.scaleRatio).toBeCloseTo(0.5, 1);
    // A regular grid repeats under a shift, so use a scene that doesn't.
    const pts = scene(3000);
    const b = compareClouds(pts, transform(pts, 1, [0, 0, 3]));
    expect(b.scaleRatio).toBeCloseTo(1, 1);
    expect(b.offset[2]).toBeCloseTo(3, 0);
  });
  it("ignores sky and fill gaussians that reach past the pose points", () => {
    const a = compareClouds(scene(3000), trainedSplat());
    expect(a.scaleRatio).toBeCloseTo(1, 1);
    expect(Math.hypot(...a.offset)).toBeLessThan(0.3);
  });
  it("finds the scale of a dense splat, where every pose point still has a gaussian nearby", () => {
    const a = compareClouds(scene(3000), transform(trainedSplat(), 0.5, [0, 0, 0]));
    expect(a.scaleRatio).toBeCloseTo(0.5, 1);
  });
});

describe("runChecks", () => {
  it("passes a consistent memory", () => {
    const checks = runChecks(bundle(), splatFrom(grid()));
    expect(checks.filter((c) => c.level === "fail" || c.level === "warn")).toEqual([]);
    expect(levelOf(checks, "splat.align")).toBe("ok");
  });

  it("passes a trained splat that reaches into the sky", () => {
    const view: GaussianView = { ...splatFrom(new Float32Array(0)), count: 0, positions: trainedSplat() };
    expect(levelOf(runChecks(bundle({ points: scene(3000) }), view), "splat.align")).toBe("ok");
  });

  it("fails a splat at half scale and names the scale", () => {
    const align = runChecks(bundle(), splatFrom(grid(), 0.5)).find((c) => c.id === "splat.align")!;
    expect(align.level).toBe("fail");
    expect(align.detail).toMatch(/0\.5\d?×/);
  });

  it("flags missing contract files and dangling stem keys", () => {
    const b = bundle();
    b.files = b.files.filter((f) => f.key !== "splat/metrics.json" && f.key !== "separate/stems/s1_water.wav");
    const checks = runChecks(b);
    expect(checks.find((c) => c.id === "contract.splat")?.title).toMatch(/missing metrics.json/);
    expect(levelOf(checks, "contract.stems")).toBe("fail");
  });

  it("flags camera times outside the recording, and partial coverage", () => {
    const late = bundle({ cameras: Array.from({ length: 9 }, (_, i) => camera(i, 5 + i * 1.5)) });
    expect(levelOf(runChecks(late), "timing.range")).toBe("fail");
    const short = bundle({ cameras: Array.from({ length: 9 }, (_, i) => camera(i, i * 0.5)) });
    expect(levelOf(runChecks(short), "timing.range")).toBe("warn");
  });

  it("flags a world that isn't +Y up and non-metric camera speed", () => {
    const yDown = bundle({ cameras: bundle().cameras!.map((c) => ({ ...c, c2w: [[-1, 0, 0, c.c2w[0][3] * 20], [0, 1, 0, 1.5], [0, 0, -1, 0], [0, 0, 0, 1]] })) });
    const checks = runChecks(yDown);
    expect(levelOf(checks, "frame.up")).toBe("warn");
    expect(levelOf(checks, "frame.scale")).toBe("warn");
  });

  it("flags sounds floating away from the scene and soundscape mismatches", () => {
    const b = bundle();
    b.localized!.sources[0].position = [0, 4, 4];
    b.scene!.sources = [];
    const checks = runChecks(b);
    expect(levelOf(checks, "audio.float.s1_water")).toBe("warn");
    expect(levelOf(checks, "audio.scene")).toBe("fail");
  });

  it("checks the splat contract units and training health", () => {
    const b = bundle({ splatMeta: { units: "centimeters", up: "+z" }, splatMetrics: { iteration: [0, 1, 2], loss: [1, NaN, 0.5], psnr: [10, 12, 15] } });
    const checks = runChecks(b);
    expect(levelOf(checks, "splat.units")).toBe("fail");
    expect(levelOf(checks, "splat.up")).toBe("fail");
    expect(levelOf(checks, "splat.nan.loss")).toBe("fail");
    expect(levelOf(checks, "splat.psnr")).toBe("warn");
  });

  it("compares against ground truth when present", () => {
    const b = bundle({ truth: { sources: [{ label: "water", position: [1, 0.5, 10] }, { label: "dog", position: [0, 0, 5] }] } });
    const checks = runChecks(b);
    expect(levelOf(checks, "truth.water")).toBe("ok");
    expect(levelOf(checks, "truth.dog")).toBe("warn");
  });
});
