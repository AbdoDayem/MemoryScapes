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

export function trimmedMean(values: Float32Array, trim = 0.1): number {
  const s = Float64Array.from(values).sort();
  const lo = Math.floor(s.length * trim), hi = Math.max(lo + 1, Math.ceil(s.length * (1 - trim)));
  let sum = 0;
  for (let i = lo; i < hi; i++) sum += s[i];
  return sum / (hi - lo);
}

function axis(xyz: Float32Array, k: number): Float32Array {
  const out = new Float32Array(xyz.length / 3);
  for (let i = 0; i < out.length; i++) out[i] = xyz[i * 3 + k];
  return out;
}

export interface Alignment {
  medianGapM: number; // median distance from a pose point to the nearest splat centre
  scaleRatio: number; // splat extent / poses extent
  offset: Vec3; // splat median - poses median
  extentM: number; // poses extent, for judging the gap
}

/** How well two point sets (pose points and splat centres) occupy the same space. */
export function compareClouds(poses: Float32Array, splat: Float32Array): Alignment {
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
  const ext = (xyz: Float32Array) => [0, 1, 2].map((k) => percentile(axis(xyz, k), 95) - percentile(axis(xyz, k), 5));
  const ea = ext(a), eb = ext(b);
  const ratios = ea.map((e, k) => (e > 1e-3 ? eb[k] / e : NaN)).filter(Number.isFinite);
  // Trimmed means over the full clouds: medians of structured (grid-like) clouds jump by
  // whole grid steps between subsamples, and plain means chase splat floaters.
  const offset = [0, 1, 2].map((k) => trimmedMean(axis(splat, k)) - trimmedMean(axis(poses, k))) as Vec3;
  return {
    medianGapM: percentile(gaps, 50),
    scaleRatio: ratios.length ? ratios.reduce((x, y) => x + y, 0) / ratios.length : NaN,
    offset,
    extentM: Math.max(...ea),
  };
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
    if (Math.abs(al.scaleRatio - 1) > 0.1) hint.push(`splat is about ${fmt(al.scaleRatio)}× the size of the poses`);
    if (Math.hypot(...al.offset) > tol) hint.push(`shifted by (${al.offset.map((v) => fmt(v, 1)).join(", ")}) m`);
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
