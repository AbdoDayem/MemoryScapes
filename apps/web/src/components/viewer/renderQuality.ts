/**
 * Frame-rate controls for the splat viewer. Pure logic (no three.js); MemoryViewer applies it.
 *
 * Spark's cost per frame grows with how many splats it draws (each one is evaluated and
 * sorted) and how many pixels they cover. So the levers are splat count and pixels:
 *
 * 1. `pickLods`: a phone stops at the most detailed LOD that fits its budget, so it never
 *    downloads or draws the 1.5M-gaussian lod0.
 * 2. The most detailed LOD loads as a Spark LoD tree (`lod: true`). Spark then draws at
 *    most a per-device splat budget, merging distant detail into splats no smaller than
 *    `lodRenderScale` pixels.
 * 3. `QualityGovernor` watches frame times and steps through `TIERS` (pixel ratio, splat
 *    budget, smallest splat, SH degree) while frames are slow, and back up when the
 *    viewer keeps up with the display.
 *
 * `?quality=high|medium|low|lowest` fixes a tier, `?lod=N` forces a LOD, and `?perf`
 * shows the frame rate (for measuring on real devices).
 */

export interface QualityTier {
  name: string;
  /** Cap on window.devicePixelRatio. */
  maxPixelRatio: number;
  /** SparkRenderer.lodSplatScale: multiplies Spark's per-device splat budget. */
  lodSplatScale: number;
  /** SparkRenderer.lodRenderScale: smallest splat the LoD tree draws, in pixels. */
  lodRenderScale: number;
  /** Highest spherical-harmonic degree evaluated (view-dependent colour). */
  maxSh: number;
  /** SparkRenderer.maxStdDev: how far out each Gaussian is drawn. */
  maxStdDev: number;
}

/**
 * Pixels cost the most: on a 1.5M-gaussian scene (M1 Pro, 3024x1658 canvas) halving the
 * pixel ratio took a frame from 47 to 25 ms, cutting drawn splats 1.5M -> 400k only to
 * 32 ms, and SH3 -> SH0 to 40 ms. So each tier lowers the pixel ratio first. Spark's splat
 * budget is per device (2.5M desktop, 1.5M iOS, 1M Android) times `lodSplatScale`.
 */
export const TIERS: readonly QualityTier[] = [
  { name: "high", maxPixelRatio: 2, lodSplatScale: 1, lodRenderScale: 1.5, maxSh: 3, maxStdDev: Math.sqrt(8) },
  { name: "medium", maxPixelRatio: 1.5, lodSplatScale: 0.4, lodRenderScale: 2, maxSh: 3, maxStdDev: Math.sqrt(8) },
  { name: "low", maxPixelRatio: 1, lodSplatScale: 0.25, lodRenderScale: 3, maxSh: 1, maxStdDev: Math.sqrt(7) },
  { name: "lowest", maxPixelRatio: 0.75, lodSplatScale: 0.15, lodRenderScale: 4, maxSh: 0, maxStdDev: Math.sqrt(6) },
];

export type DeviceClass = "desktop" | "mobile";

/** Phones and tablets (iPadOS reports itself as a Mac, but with touch). */
export function deviceClass(nav: { userAgent: string; maxTouchPoints?: number }): DeviceClass {
  if (/Android|iPhone|iPad|iPod|Mobile/i.test(nav.userAgent)) return "mobile";
  if (/Macintosh/.test(nav.userAgent) && (nav.maxTouchPoints ?? 0) > 1) return "mobile";
  return "desktop";
}

/** The most gaussians a device downloads and draws (lods.json caps lod0 at 1.5M). */
export const MAX_LOD_SPLATS: Record<DeviceClass, number> = { desktop: Infinity, mobile: 600_000 };

/** The tier a device starts at, before the governor has measured anything. */
export const START_TIER: Record<DeviceClass, number> = { desktop: 0, mobile: 1 };

/**
 * The LODs to load, coarsest first, each replacing the previous: from the coarsest (a quick
 * first view) up to the most detailed one the device takes. `lods` is most detailed first,
 * as in the manifest. `force` (an index into `lods`) picks the top one instead.
 */
export function pickLods<T extends { count?: number }>(lods: readonly T[], device: DeviceClass, force?: number): T[] {
  if (lods.length === 0) return [];
  let top: number;
  if (force !== undefined && Number.isInteger(force) && force >= 0 && force < lods.length) top = force;
  else {
    top = lods.findIndex((l) => (l.count ?? 0) <= MAX_LOD_SPLATS[device]);
    if (top < 0) top = lods.length - 1;
  }
  return lods.slice(top).reverse();
}

export interface ViewerOverrides {
  /** A fixed tier (the governor is off). */
  tier?: number;
  /** A forced top LOD index. */
  lod?: number;
  /** Show the frame-rate readout. */
  perf: boolean;
}

export function parseOverrides(search: string): ViewerOverrides {
  const q = new URLSearchParams(search);
  const out: ViewerOverrides = { perf: q.has("perf") };
  const tier = TIERS.findIndex((t) => t.name === q.get("quality"));
  if (tier >= 0) out.tier = tier;
  const lod = q.get("lod");
  if (lod !== null && /^\d+$/.test(lod)) out.lod = Number(lod);
  return out;
}

export interface GovernorOptions {
  /** Frames judged together, by their mean without the slowest 10% (one hitch doesn't count;
   * a mean, not a median, because vsync rounds each frame to a whole refresh interval). */
  windowMs: number;
  /** A frame time above this steps quality down (45 fps)... */
  slowMs: number;
  /** ...or above this times the display's own frame time, when the display is slow (a
   * browser's 30 fps battery saver). */
  slowVsDisplay: number;
  /** Time keeping up with the display before stepping back up; x4 for each time that tier failed. */
  upHoldMs: number;
  /** Frames are ignored this long after a change (shader rebuild, LoD refresh). */
  settleMs: number;
  /** A tier that had to be left this many times isn't retried. */
  maxFailures: number;
}

export const GOVERNOR_DEFAULTS: GovernorOptions = {
  windowMs: 1000,
  slowMs: 1000 / 45,
  slowVsDisplay: 1.15,
  upHoldMs: 5000,
  settleMs: 1500,
  maxFailures: 2,
};

/** Steps through TIERS from frame times. Feed it every frame; it says when to change. */
export class QualityGovernor {
  tier: number;
  private readonly opts: GovernorOptions;
  private readonly failures: number[];
  private samples: number[] = [];
  private windowStart: number | null = null;
  private settleUntil = 0;
  private keptUpSince: number | null = null;
  /** The display's frame time: the fastest window seen, first while loading (cheap frames:
   * a browser capped at 30 fps shows here), then any faster one. 60 Hz until measured. */
  private displayMs = 1000 / 60;
  private displayMeasured = false;

  constructor(tier: number, opts: Partial<GovernorOptions> = {}) {
    this.tier = Math.max(0, Math.min(TIERS.length - 1, tier));
    this.opts = { ...GOVERNOR_DEFAULTS, ...opts };
    this.failures = TIERS.map(() => 0);
  }

  /** One frame: `now` and the time since the previous frame, in ms. With `act` false (the
   * splat is still loading) it only measures the display. Returns the new tier when it
   * changes, else null. */
  frame(now: number, dtMs: number, act = true): number | null {
    if (now < this.settleUntil) return null;
    if (!(dtMs > 0) || dtMs > 250) {
      // A hidden tab or a loading stall says nothing about rendering: start over.
      this.samples = [];
      this.windowStart = null;
      return null;
    }
    this.windowStart ??= now;
    this.samples.push(dtMs);
    if (now - this.windowStart < this.opts.windowMs) return null;
    const frameMs = trimmedMean(this.samples);
    this.samples = [];
    this.windowStart = now;
    if (!act) {
      this.displayMs = this.displayMeasured ? Math.min(this.displayMs, frameMs) : frameMs;
      this.displayMeasured = true;
      return null;
    }
    this.displayMs = Math.min(this.displayMs, frameMs);

    const slow = Math.max(this.opts.slowMs, this.displayMs * this.opts.slowVsDisplay);
    if (frameMs > slow) {
      this.keptUpSince = null;
      if (this.tier >= TIERS.length - 1) return null;
      this.failures[this.tier]++;
      return this.change(now, this.tier + 1);
    }
    const keepingUp = frameMs <= this.displayMs * 1.1 + 0.5;
    if (!keepingUp) {
      this.keptUpSince = null;
      return null;
    }
    this.keptUpSince ??= now;
    const up = this.tier - 1;
    if (up < 0 || this.failures[up] >= this.opts.maxFailures) return null;
    if (now - this.keptUpSince < this.opts.upHoldMs * 4 ** this.failures[up]) return null;
    return this.change(now, up);
  }

  private change(now: number, tier: number): number {
    this.tier = tier;
    this.settleUntil = now + this.opts.settleMs;
    this.keptUpSince = null;
    this.samples = [];
    this.windowStart = null;
    return tier;
  }
}

/** Mean without the slowest 10%. */
export function trimmedMean(xs: readonly number[]): number {
  const s = [...xs].sort((a, b) => a - b).slice(0, Math.max(1, Math.ceil(xs.length * 0.9)));
  return s.reduce((a, b) => a + b, 0) / s.length;
}
