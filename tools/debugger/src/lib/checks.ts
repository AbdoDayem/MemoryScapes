/**
 * Integration checks over one memory's stage outputs. Pure functions of the bundle (and
 * the parsed splat), so they're unit-tested and the UI only renders their results.
 *
 * The checks that matter most are the ones neither pipeline can run alone: the splat
 * and the poses share one world frame, the camera clock matches the audio, and every
 * placed sound sits near some surface of the scene.
 */
import type { GaussianView } from "./ply";
import type { MemoryBundle, Vec3 } from "./types";

export type Level = "fail" | "warn" | "ok" | "info";
export type Area = "contract" | "timing" | "frame" | "audio" | "splat" | "truth";

export interface Check {
  id: string;
  area: Area;
  level: Level;
  title: string;
  detail?: string;
}

export const EXPECTED: Record<string, string[]> = {
  ingest: ["audio.wav", "meta.json"],
  poses: ["cameras.json", "points3d.npy"],
  splat: ["meta.json", "metrics.json", "point_cloud.ply"],
  separate: ["stems.json", "ambient.wav"],
  localize: ["sources.json"],
  soundscape: ["scene.json"],
};

const fmt = (v: number, d = 2) => (Number.isFinite(v) ? v.toFixed(d) : String(v));
const dist = (a: ArrayLike<number>, b: ArrayLike<number>) => Math.hypot(a[0] - b[0], a[1] - b[1], a[2] - b[2]);

export function percentile(values: ArrayLike<number>, p: number): number {
  const s = Float64Array.from(values).sort();
  if (!s.length) return NaN;
  const i = Math.min(s.length - 1, Math.max(0, Math.round((p / 100) * (s.length - 1))));
  return s[i];
}

function sample(xyz: Float32Array, max: number): Float32Array {
  const n = xyz.length / 3;
  if (n <= max) return xyz;
  const out = new Float32Array(max * 3);
  const step = n / max;
  for (let i = 0; i < max; i++) {
    const j = Math.floor(i * step);
    out.set(xyz.subarray(j * 3, j * 3 + 3), i * 3);
  }
  return out;
}

function axis(xyz: Float32Array, k: number): Float32Array {
  const out = new Float32Array(xyz.length / 3);
  for (let i = 0; i < out.length; i++) out[i] = xyz[i * 3 + k];
  return out;
}

export interface Alignment {
  medianGapM: number; // median distance from a pose point to the nearest splat centre
  // The similarity that best lines the splat up with the pose points: the splat looks like
  // the poses scaled by scaleRatio about their median point, then shifted by offset.
  scaleRatio: number;
  offset: Vec3;
  fitGapM: number; // median gap after that fit
  extentM: number; // poses extent, for judging the gap
}

/**
 * Distance from any point to the nearest of a set of seeds, precomputed on a voxel grid
 * (exact squared EDT, Felzenszwalb & Huttenlocher) so the fit below can score thousands of
 * candidate transforms cheaply. Near the seeds (where the median gap is decided) the
 * distance is exact, from the seeds bucketed in the surrounding voxels; further out it's
 * the grid's estimate, accurate to about one voxel.
 */
class DistanceField {
  private readonly d: Float32Array;
  private readonly dims: [number, number, number];
  private readonly start: Int32Array; // seeds of voxel v: sorted[3 * start[v] .. 3 * start[v + 1])
  private readonly sorted: Float32Array; // seed coordinates, grouped by voxel in index order
  constructor(private readonly lo: Vec3, private readonly h: number, hi: Vec3, seeds: Float32Array) {
    this.dims = [0, 1, 2].map((k) => Math.max(1, Math.ceil((hi[k] - lo[k]) / h) + 1)) as [number, number, number];
    const [nx, ny, nz] = this.dims;
    const f = new Float64Array(nx * ny * nz).fill(1e20);
    const cellOf = new Int32Array(seeds.length / 3).fill(-1);
    this.start = new Int32Array(nx * ny * nz + 1);
    for (let j = 0; j < seeds.length; j += 3) {
      const ix = Math.round((seeds[j] - lo[0]) / h), iy = Math.round((seeds[j + 1] - lo[1]) / h), iz = Math.round((seeds[j + 2] - lo[2]) / h);
      if (ix >= 0 && iy >= 0 && iz >= 0 && ix < nx && iy < ny && iz < nz) {
        const v = (iz * ny + iy) * nx + ix;
        f[v] = 0;
        cellOf[j / 3] = v;
        this.start[v + 1]++;
      }
    }
    for (let v = 0; v < nx * ny * nz; v++) this.start[v + 1] += this.start[v];
    this.sorted = new Float32Array(this.start[nx * ny * nz] * 3);
    const fill = this.start.slice(0, -1);
    cellOf.forEach((v, s) => {
      if (v >= 0) this.sorted.set(seeds.subarray(s * 3, s * 3 + 3), 3 * fill[v]++);
    });
    const n = Math.max(nx, ny, nz);
    const line = new Float64Array(n), out = new Float64Array(n), v = new Int32Array(n), z = new Float64Array(n + 1);
    const pass = (len: number, count: number, index: (line: number, i: number) => number) => {
      for (let l = 0; l < count; l++) {
        for (let i = 0; i < len; i++) line[i] = f[index(l, i)];
        edt1d(line, len, out, v, z);
        for (let i = 0; i < len; i++) f[index(l, i)] = out[i];
      }
    };
    pass(nx, ny * nz, (l, i) => l * nx + i);
    pass(ny, nx * nz, (l, i) => (Math.floor(l / nx) * ny + i) * nx + (l % nx));
    pass(nz, nx * ny, (l, i) => (i * ny + Math.floor(l / nx)) * nx + (l % nx));
    this.d = Float32Array.from(f, (x) => Math.sqrt(x) * h);
  }

  /** Distance to the nearest seed; outside the grid, plus the distance to the grid. */
  at(x: number, y: number, z: number): number {
    const { h, lo } = this;
    const [nx, ny, nz] = this.dims;
    const rx = Math.round((x - lo[0]) / h), ry = Math.round((y - lo[1]) / h), rz = Math.round((z - lo[2]) / h);
    const cx = rx < 0 ? 0 : rx >= nx ? nx - 1 : rx;
    const cy = ry < 0 ? 0 : ry >= ny ? ny - 1 : ry;
    const cz = rz < 0 ? 0 : rz >= nz ? nz - 1 : rz;
    const approx = this.d[(cz * ny + cy) * nx + cx];
    if (cx !== rx || cy !== ry || cz !== rz) return approx + h * Math.hypot(rx - cx, ry - cy, rz - cz);
    if (approx > 3 * h) return approx;
    // Exact: the nearest seed is within about approx + one voxel of this voxel's centre.
    const r = Math.ceil(approx / h) + 1;
    const pts = this.sorted, start = this.start;
    let best = Infinity;
    for (let iz = Math.max(0, cz - r); iz <= Math.min(nz - 1, cz + r); iz++)
      for (let iy = Math.max(0, cy - r); iy <= Math.min(ny - 1, cy + r); iy++) {
        const row = (iz * ny + iy) * nx;
        const end = start[row + Math.min(nx - 1, cx + r) + 1];
        for (let q = start[row + Math.max(0, cx - r)] * 3; q < end * 3; q += 3) {
          const dx = x - pts[q], dy = y - pts[q + 1], dz = z - pts[q + 2];
          const d = dx * dx + dy * dy + dz * dz;
          if (d < best) best = d;
        }
      }
    return best < Infinity ? Math.sqrt(best) : approx;
  }
}

function edt1d(f: Float64Array, n: number, out: Float64Array, v: Int32Array, z: Float64Array): void {
  let k = 0;
  v[0] = 0;
  z[0] = -1e30;
  z[1] = 1e30;
  const cross = (q: number, r: number) => (f[q] + q * q - (f[r] + r * r)) / (2 * q - 2 * r);
  for (let q = 1; q < n; q++) {
    let s = cross(q, v[k]);
    while (s <= z[k]) s = cross(q, v[--k]);
    k++;
    v[k] = q;
    z[k] = s;
    z[k + 1] = 1e30;
  }
  k = 0;
  for (let q = 0; q < n; q++) {
    while (z[k + 1] < q) k++;
    out[q] = (q - v[k]) ** 2 + f[v[k]];
  }
}

/**
 * How well two point sets (pose points and splat centres) occupy the same space.
 *
 * Measured from the pose points: each should have a gaussian close by. The other direction
 * is meaningless, because a trained splat legitimately reaches far past the SfM points (the
 * sky, distant walls) and fills the space between them. For the same reason, comparing
 * extents or centroids reports scale errors on a correctly aligned splat. So scale and
 * offset come from a fit instead: the scale and shift of the pose points that minimise
 * their median gap to the splat, with the gap in pose metres.
 *
 * The fit takes about a second on a real splat, so results are cached per pair of arrays
 * (the Checks and Splat tabs both ask for the same one).
 */
export function compareClouds(poses: Float32Array, splat: Float32Array): Alignment {
  let byPoses = alignmentCache.get(splat);
  if (!byPoses) alignmentCache.set(splat, (byPoses = new WeakMap()));
  let al = byPoses.get(poses);
  if (!al) byPoses.set(poses, (al = computeAlignment(poses, splat)));
  return al;
}

const alignmentCache = new WeakMap<Float32Array, WeakMap<Float32Array, Alignment>>();

function computeAlignment(poses: Float32Array, splat: Float32Array): Alignment {
  const a = sample(poses, 600);
  const b = sample(splat, 25000);
  const gaps = new Float32Array(a.length / 3);
  for (let i = 0; i < gaps.length; i++) {
    let best = Infinity;
    const ax = a[i * 3], ay = a[i * 3 + 1], az = a[i * 3 + 2];
    for (let j = 0; j < b.length; j += 3) {
      const dx = ax - b[j], dy = ay - b[j + 1], dz = az - b[j + 2];
      const d = dx * dx + dy * dy + dz * dz;
      if (d < best) best = d;
    }
    gaps[i] = Math.sqrt(best);
  }
  const ea = [0, 1, 2].map((k) => percentile(axis(a, k), 95) - percentile(axis(a, k), 5));
  const extentM = Math.max(...ea);
  const fit = fitSimilarity(a, b, extentM);
  return { medianGapM: percentile(gaps, 50), ...fit, extentM };
}

function fitSimilarity(a: Float32Array, b: Float32Array, extentM: number): Pick<Alignment, "scaleRatio" | "offset" | "fitGapM"> {
  const lo = [0, 1, 2].map((k) => percentile(axis(b, k), 1)) as Vec3;
  const hi = [0, 1, 2].map((k) => percentile(axis(b, k), 99)) as Vec3;
  const span = Math.max(...hi.map((v, k) => v - lo[k]), 1e-3);
  const volume = hi.reduce((acc, v, k) => acc * Math.max(v - lo[k], 1e-3), 1);
  const h = Math.max(span / 160, Math.cbrt(volume / 1e6));
  const field = new DistanceField(lo, h, hi, b);
  const c = [0, 1, 2].map((k) => percentile(axis(a, k), 50));
  const scorer = (pts: Float32Array) => {
    const n = pts.length / 3;
    const g = new Float32Array(n);
    return (s: number, t: number[]) => {
      for (let i = 0; i < n; i++)
        g[i] = field.at((pts[i * 3] - c[0]) * s + c[0] + t[0], (pts[i * 3 + 1] - c[1]) * s + c[1] + t[1], (pts[i * 3 + 2] - c[2]) * s + c[2] + t[2]) / s;
      return percentile(g, 50);
    };
  };
  const quick = scorer(sample(a, 200)), score = scorer(a);
  const ext = Math.max(extentM, 1e-3);
  type Fit = { gap: number; s: number; t: number[] };
  // Coordinate search over the offset at one scale, one step size at a time.
  const refine = (f: Fit, steps: number[], sc: typeof score): Fit => {
    let { gap, t } = f;
    for (const step of steps)
      for (let round = 0; round < 2; round++)
        for (let k = 0; k < 3; k++)
          for (let d = -6; d <= 6; d++) {
            if (d === 0) continue;
            const tt = [...t];
            tt[k] += d * step;
            const gg = sc(f.s, tt);
            if (gg < gap) [gap, t] = [gg, tt];
          }
    return { gap, s: f.s, t };
  };
  // Coarse: every scale from 0.25x to 4x. Fine: the five best.
  const coarse = Array.from({ length: 17 }, (_, si) => 0.25 * 16 ** (si / 16))
    .map((s) => refine({ gap: quick(s, [0, 0, 0]), s, t: [0, 0, 0] }, [ext * s / 8, ext * s / 32], quick))
    .sort((x, y) => x.gap - y.gap);
  let best = coarse
    .slice(0, 5)
    .map((f) => refine({ ...f, gap: score(f.s, f.t) }, [ext * f.s / 32, ext * f.s / 128], score))
    .reduce((x, y) => (y.gap < x.gap ? y : x));
  for (let fi = -10; fi <= 10; fi++) {
    const s = best.s * Math.exp(fi / 100);
    const gap = score(s, best.t);
    if (gap < best.gap) best = { ...best, gap, s };
  }
  // Distances near the surfaces are exact, so only a near-tie with the identity snaps to it.
  const identity = score(1, [0, 0, 0]);
  if (identity <= best.gap * 1.05) best = { gap: identity, s: 1, t: [0, 0, 0] };
  return { scaleRatio: best.s, offset: best.t as Vec3, fitGapM: best.gap };
}

export function runChecks(b: MemoryBundle, splat?: GaussianView): Check[] {
  const out: Check[] = [];
  const add = (c: Check) => out.push(c);
  const keys = new Set(b.files.map((f) => f.key));
  const rel = (k: string) => (k.startsWith(`${b.id}/`) ? k.slice(b.id.length + 1) : k);

  // ---------- contract: files each stage promised
  for (const [stage, names] of Object.entries(EXPECTED)) {
    const present = [...keys].some((k) => k.startsWith(`${stage}/`));
    const missing = names.filter((n) => !keys.has(`${stage}/${n}`));
    if (!present) add({ id: `contract.${stage}`, area: "contract", level: "info", title: `${stage}: not run yet` });
    else if (missing.length) add({ id: `contract.${stage}`, area: "contract", level: "fail", title: `${stage} is missing ${missing.join(", ")}` });
    else add({ id: `contract.${stage}`, area: "contract", level: "ok", title: `${stage}: all expected files present` });
  }
  for (const e of b.errors) add({ id: `contract.parse.${e.key}`, area: "contract", level: "fail", title: `Couldn't read ${e.key}`, detail: e.message });
  if (b.stems) {
    const missing = [...b.stems.stems.map((s) => s.key), b.stems.ambient_key].map(rel).filter((k) => !keys.has(k));
    if (missing.length) add({ id: "contract.stems", area: "contract", level: "fail", title: `stems.json points at ${missing.length} missing audio file(s)`, detail: missing.join(", ") });
  }
  if (b.cameras) {
    const missing = b.cameras.filter((c) => c.image_key && !keys.has(rel(c.image_key)));
    if (missing.length) add({ id: "contract.frames", area: "contract", level: "warn", title: `${missing.length} camera(s) point at missing frame images` });
  }

  // ---------- timing: camera clock vs audio clock
  const duration = b.stems?.duration_s;
  if (b.cameras?.length) {
    const ts = b.cameras.map((c) => c.t);
    const sorted = ts.every((t, i) => i === 0 || t >= ts[i - 1]);
    if (!sorted) add({ id: "timing.order", area: "timing", level: "warn", title: "Camera times are not in order" });
    if (duration) {
      const lo = Math.min(...ts), hi = Math.max(...ts);
      if (lo < -0.05 || hi > duration + 0.05)
        add({ id: "timing.range", area: "timing", level: "fail", title: "Camera times fall outside the recording", detail: `cameras ${fmt(lo)}–${fmt(hi)} s, audio 0–${fmt(duration)} s. Are both on the recording's clock?` });
      else if (hi - lo < 0.7 * duration)
        add({ id: "timing.range", area: "timing", level: "warn", title: `Keyframes cover only ${Math.round(((hi - lo) / duration) * 100)}% of the audio`, detail: `cameras ${fmt(lo)}–${fmt(hi)} s of ${fmt(duration)} s. Localization can only use frames inside that span.` });
      else add({ id: "timing.range", area: "timing", level: "ok", title: `Keyframes span the recording (${fmt(lo, 1)}–${fmt(hi, 1)} s of ${fmt(duration, 1)} s)` });
    }
  }

  // ---------- frame: +Y up, metric scale
  if (b.cameras?.length) {
    const downDotUp = b.cameras.reduce((s, c) => s + c.c2w[1][1], 0) / b.cameras.length; // camera +y (down) · world +y
    if (downDotUp > -0.7)
      add({ id: "frame.up", area: "frame", level: "warn", title: "World may not be +Y up", detail: `Cameras' down axis averages ${fmt(downDotUp)} along world +Y (expected about -1 for an upright phone).` });
    else add({ id: "frame.up", area: "frame", level: "ok", title: "Cameras are upright in a +Y-up world" });
    let path = 0;
    for (let i = 1; i < b.cameras.length; i++) path += dist(col3(b.cameras[i].c2w), col3(b.cameras[i - 1].c2w));
    const span = b.cameras[b.cameras.length - 1].t - b.cameras[0].t;
    const speed = span > 0 ? path / span : 0;
    if (speed > 4) add({ id: "frame.scale", area: "frame", level: "warn", title: `Camera moves ${fmt(speed, 1)} m/s`, detail: "Faster than walking. Poses may not be in metres." });
    else add({ id: "frame.scale", area: "frame", level: "ok", title: `Camera path ${fmt(path, 1)} m at ${fmt(speed, 2)} m/s (walking pace, so metric)` });
    if (path < 0.2) add({ id: "frame.static", area: "frame", level: "info", title: "Camera barely moves", detail: "Stereo DOA can't judge distance without movement; expect far sounds to stay diffuse." });
  }
  if (b.points && b.points.length === 0) add({ id: "frame.points", area: "frame", level: "warn", title: "points3d.npy is empty", detail: "Audio-visual lifting needs scene points." });

  // ---------- audio stages
  if (b.localized) {
    const { min, max } = b.localized.bounds;
    for (const s of b.localized.sources) {
      if (!s.position) continue;
      const outside = s.position.some((v, k) => v < min[k] - 0.5 || v > max[k] + 0.5);
      if (outside) add({ id: `audio.bounds.${s.id}`, area: "audio", level: "warn", title: `${s.label} is placed outside the scene bounds`, detail: `at (${s.position.map((v) => fmt(v, 1)).join(", ")})` });
      if (b.points?.length) {
        let best = Infinity;
        for (let i = 0; i < b.points.length; i += 3) best = Math.min(best, dist(s.position, b.points.subarray(i, i + 3)));
        if (best > 3) add({ id: `audio.float.${s.id}`, area: "audio", level: "warn", title: `${s.label} floats ${fmt(best, 1)} m from any scene point`, detail: "Sounds usually sit on surfaces. Check its detections, or whether the point cloud is too sparse there." });
      }
      if (s.confidence < 0.4) add({ id: `audio.conf.${s.id}`, area: "audio", level: "info", title: `${s.label}: low confidence ${fmt(s.confidence)} (${s.method})` });
    }
    const placed = b.localized.sources.filter((s) => s.position).length;
    add({ id: "audio.summary", area: "audio", level: "ok", title: `${placed} of ${b.localized.sources.length} sounds placed` });
  }
  for (const d of b.stems?.dropped ?? []) add({ id: `audio.dropped.${d.label}`, area: "audio", level: "info", title: `Dropped stem: ${d.label}`, detail: d.reason });
  if (b.scene) {
    if (b.scene.provider.fallback_reason) add({ id: "audio.fallback", area: "audio", level: "warn", title: "Soundscape fell back to the stub", detail: b.scene.provider.fallback_reason });
    if (b.localized && b.scene.sources.length !== b.localized.sources.length)
      add({ id: "audio.scene", area: "audio", level: "fail", title: "soundscape/scene.json and localize/sources.json disagree", detail: `${b.scene.sources.length} vs ${b.localized.sources.length} sources` });
    for (const n of b.scene.provider.notes ?? []) add({ id: `audio.note.${n}`, area: "audio", level: "info", title: n });
  }

  // ---------- splat
  if (b.splatMeta) {
    if (b.splatMeta.units && b.splatMeta.units !== "meters") add({ id: "splat.units", area: "splat", level: "fail", title: `splat/meta.json says units = ${b.splatMeta.units}`, detail: "The contract is metres." });
    if (b.splatMeta.up && b.splatMeta.up !== "+y") add({ id: "splat.up", area: "splat", level: "fail", title: `splat/meta.json says up = ${b.splatMeta.up}`, detail: "The contract is +Y up." });
  }
  if (splat && b.points?.length) {
    const al = compareClouds(b.points, splat.positions);
    const tol = Math.max(0.3, 0.03 * al.extentM);
    const hint: string[] = [];
    if (al.fitGapM > tol) hint.push(`no scale or shift lines the splat up with the pose points (best median gap ${fmt(al.fitGapM)} m)`);
    else {
      if (Math.abs(al.scaleRatio - 1) > 0.1) hint.push(`splat is about ${fmt(al.scaleRatio)}× the size of the poses`);
      if (Math.hypot(...al.offset) > tol) hint.push(`shifted by (${al.offset.map((v) => fmt(v, 1)).join(", ")}) m`);
    }
    if (al.medianGapM > tol || hint.length)
      add({ id: "splat.align", area: "splat", level: "fail", title: "Splat doesn't line up with the camera poses", detail: `Median gap ${fmt(al.medianGapM)} m (tolerance ${fmt(tol)} m). ${hint.join("; ")}. Audio positions use the poses' frame, so sounds will be misplaced in the splat.` });
    else add({ id: "splat.align", area: "splat", level: "ok", title: `Splat lines up with the poses (median gap ${fmt(al.medianGapM)} m, scale ${fmt(al.scaleRatio)}×)` });
  }
  const m = b.splatMetrics;
  if (m?.iteration?.length) {
    for (const [name, values] of Object.entries(m)) {
      if (values.some((v) => !Number.isFinite(v))) add({ id: `splat.nan.${name}`, area: "splat", level: "fail", title: `splat metrics: ${name} has NaN/Inf values` });
    }
    const psnr = m.psnr?.[m.psnr.length - 1];
    if (psnr !== undefined && psnr < 20) add({ id: "splat.psnr", area: "splat", level: "warn", title: `Final PSNR ${fmt(psnr, 1)} dB is low`, detail: "Under 20 dB usually means bad poses or too few iterations." });
    else if (psnr !== undefined) add({ id: "splat.psnr", area: "splat", level: "ok", title: `Final PSNR ${fmt(psnr, 1)} dB` });
    const loss = m.loss;
    if (loss && loss.length > 10) {
      const tail = loss.slice(-Math.ceil(loss.length / 10));
      const mid = loss.slice(Math.floor(loss.length / 2), Math.floor(loss.length * 0.6));
      const avg = (a: number[]) => a.reduce((x, y) => x + y, 0) / a.length;
      if (avg(tail) > avg(mid) * 1.1) add({ id: "splat.diverge", area: "splat", level: "warn", title: "Training loss rose at the end", detail: "Possible divergence or a learning-rate problem." });
    }
  }
  if (b.cameras && keys.has("splat/point_cloud.ply")) {
    const renders = b.cameras.filter((c) => keys.has(`splat/renders/${c.frame_id}.png`)).length;
    if (renders < b.cameras.length) add({ id: "splat.renders", area: "splat", level: "info", title: `Splat renders for ${renders} of ${b.cameras.length} keyframes` });
  }

  // ---------- ground truth (synthetic examples only)
  if (b.truth && b.localized) {
    for (const t of b.truth.sources) {
      const est = b.localized.sources.find((s) => s.label === t.label);
      if (!est) add({ id: `truth.${t.label}`, area: "truth", level: "warn", title: `${t.label}: in the scene but never separated` });
      else if (!est.position) add({ id: `truth.${t.label}`, area: "truth", level: "info", title: `${t.label}: left diffuse` });
      else {
        const e = dist(est.position, t.position);
        add({ id: `truth.${t.label}`, area: "truth", level: e > 2 ? "warn" : "ok", title: `${t.label}: ${fmt(e)} m from the true position`, detail: `${est.method}, confidence ${fmt(est.confidence)}` });
      }
    }
  }

  const order: Record<Level, number> = { fail: 0, warn: 1, info: 2, ok: 3 };
  return out.sort((x, y) => order[x.level] - order[y.level]);
}

function col3(m: number[][]): number[] {
  return [m[0][3], m[1][3], m[2][3]];
}
