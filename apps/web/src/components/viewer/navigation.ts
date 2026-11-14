import type { Vec3 } from "@memoryscapes/manifest";

/** poses/navigation.json `walkable_bounds` (type "aabb+corridor"). */
export interface Walkable {
  min: Vec3;
  max: Vec3;
  corridor?: { radius_m: number; points: Vec3[] };
}

/** One sample of poses/path.json: a three.js camera pose (looks along -z, +y up). */
export interface PathSample {
  t: number;
  position: Vec3;
  quaternion_xyzw: [number, number, number, number];
}

export interface CapturePath {
  fov_y_deg?: number;
  samples: PathSample[];
}

export function asWalkable(raw: unknown): Walkable | undefined {
  const w = raw as Partial<Walkable> | undefined;
  return w && Array.isArray(w.min) && Array.isArray(w.max) ? (w as Walkable) : undefined;
}

/**
 * Keep the viewer inside the space that was actually captured: inside the box, within
 * the corridor around the capture path (beyond it the splat is mostly floaters), and
 * above the floor.
 */
export function clampToWalkable(p: Vec3, w: Walkable | undefined, minY?: number): Vec3 {
  let [x, y, z] = p;
  if (w?.corridor && w.corridor.points.length > 0) {
    const { radius_m: r, points } = w.corridor;
    let best = points[0];
    let bestD = Infinity;
    for (const q of points) {
      const d = (x - q[0]) ** 2 + (y - q[1]) ** 2 + (z - q[2]) ** 2;
      if (d < bestD) [best, bestD] = [q, d];
    }
    const d = Math.sqrt(bestD);
    if (d > r) {
      const k = r / d;
      x = best[0] + (x - best[0]) * k;
      y = best[1] + (y - best[1]) * k;
      z = best[2] + (z - best[2]) * k;
    }
  }
  if (w) {
    x = Math.min(Math.max(x, w.min[0]), w.max[0]);
    y = Math.min(Math.max(y, w.min[1]), w.max[1]);
    z = Math.min(Math.max(z, w.min[2]), w.max[2]);
  }
  if (minY !== undefined) y = Math.max(y, minY);
  return [x, y, z];
}

/** Pose on the capture path at time t (seconds), interpolated; wraps around. */
export function samplePath(path: CapturePath, t: number): { position: Vec3; quaternion: [number, number, number, number] } | null {
  const s = path.samples;
  if (s.length === 0) return null;
  if (s.length === 1) return { position: s[0].position, quaternion: s[0].quaternion_xyzw };
  const t0 = s[0].t;
  const span = s[s.length - 1].t - t0;
  const tt = span > 0 ? t0 + (((t - t0) % span) + span) % span : t0;
  let i = 0;
  while (i < s.length - 2 && s[i + 1].t < tt) i++;
  const a = s[i];
  const b = s[i + 1];
  const f = b.t > a.t ? Math.min(1, Math.max(0, (tt - a.t) / (b.t - a.t))) : 0;
  const position: Vec3 = [0, 1, 2].map((k) => a.position[k] + (b.position[k] - a.position[k]) * f) as Vec3;
  return { position, quaternion: slerp(a.quaternion_xyzw, b.quaternion_xyzw, f) };
}

function slerp(a: number[], b: number[], f: number): [number, number, number, number] {
  let dot = a[0] * b[0] + a[1] * b[1] + a[2] * b[2] + a[3] * b[3];
  const bb = dot < 0 ? b.map((v) => -v) : b; // shortest arc
  dot = Math.abs(dot);
  if (dot > 0.9995) {
    const q = a.map((v, k) => v + (bb[k] - v) * f);
    const n = Math.hypot(...q);
    return q.map((v) => v / n) as [number, number, number, number];
  }
  const th = Math.acos(dot);
  const sa = Math.sin((1 - f) * th) / Math.sin(th);
  const sb = Math.sin(f * th) / Math.sin(th);
  return a.map((v, k) => v * sa + bb[k] * sb) as [number, number, number, number];
}
