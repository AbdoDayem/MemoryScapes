/**
 * Grouping nearby memories into clusters (docs/webapp-plan/frontend.md § Page `/`).
 *
 * Memories merge only when they'd overlap on screen: the merge distance shrinks with the
 * camera's altitude, so as you zoom in they come apart exactly when you can pick them.
 * Plain functions, no three.js, so they're easy to test.
 */

export interface Located {
  id: string;
  lat: number;
  lon: number;
}

export type Shown<T extends Located> =
  | { kind: "single"; id: string; lat: number; lon: number; memory: T }
  | { kind: "cluster"; id: string; lat: number; lon: number; members: T[] };

/**
 * How close (in degrees of arc) two memories can be before they merge, at a camera altitude
 * (in globe radii). About a marker's width on screen: the visible arc shrinks with altitude.
 */
export function mergeDistance(altitude: number): number {
  return Math.max(0.002, altitude * 5);
}

/**
 * The camera altitude at which every member of a group is shown separately, or null if
 * they're so close they never would be (then the UI offers a list to pick from).
 */
export function altitudeToSeparate(members: Located[], minAltitude = 0.02): number | null {
  let closest = Infinity;
  for (let i = 0; i < members.length; i++) {
    for (let j = i + 1; j < members.length; j++) closest = Math.min(closest, arcDegrees(members[i], members[j]));
  }
  if (!Number.isFinite(closest)) return null;
  const altitude = (closest / 5) * 0.8; // a little closer than strictly needed
  return altitude >= minAltitude ? altitude : null;
}

/**
 * Greedy clustering: each memory joins the first group whose first member is within
 * `mergeDeg`, else starts a new group. Input order doesn't change which ids end up together
 * for well-separated places, and cluster ids are stable (sorted member ids).
 */
export function clusterPoints<T extends Located>(points: T[], mergeDeg: number): Shown<T>[] {
  const sorted = [...points].sort((a, b) => (a.id < b.id ? -1 : a.id > b.id ? 1 : 0));
  const groups: T[][] = [];
  for (const p of sorted) {
    const g = groups.find((group) => arcDegrees(group[0], p) <= mergeDeg);
    if (g) g.push(p);
    else groups.push([p]);
  }
  return groups.map((g) => {
    if (g.length === 1) return { kind: "single", id: g[0].id, lat: g[0].lat, lon: g[0].lon, memory: g[0] };
    const { lat, lon } = centre(g);
    return { kind: "cluster", id: "cluster:" + g.map((m) => m.id).join("+"), lat, lon, members: g };
  });
}

/** Great-circle distance between two points, in degrees of arc. */
export function arcDegrees(a: { lat: number; lon: number }, b: { lat: number; lon: number }): number {
  const r = Math.PI / 180;
  const dLat = (b.lat - a.lat) * r, dLon = (b.lon - a.lon) * r;
  const h = Math.sin(dLat / 2) ** 2 + Math.cos(a.lat * r) * Math.cos(b.lat * r) * Math.sin(dLon / 2) ** 2;
  return (2 * Math.asin(Math.min(1, Math.sqrt(h)))) / r;
}

/** Average position on the sphere (so a group near ±180° doesn't land on the wrong side). */
export function centre(points: Located[]): { lat: number; lon: number } {
  let x = 0, y = 0, z = 0;
  for (const p of points) {
    const la = (p.lat * Math.PI) / 180, lo = (p.lon * Math.PI) / 180;
    x += Math.cos(la) * Math.cos(lo);
    y += Math.cos(la) * Math.sin(lo);
    z += Math.sin(la);
  }
  const lon = (Math.atan2(y, x) * 180) / Math.PI;
  const lat = (Math.atan2(z, Math.hypot(x, y)) * 180) / Math.PI;
  return { lat, lon };
}
