import type { TrainReport } from "./types";

/** Scores first, in this order; then every other scalar field, alphabetically. */
const FIRST = ["psnr", "psnr_split", "psnr_cc", "depth_error_vs_sfm", "train_images", "right_eye_images",
  "depth_supervised_images", "exposure_comp", "gaussians", "steps", "seconds"];

/** The report's scalar fields as [name, text] rows (lists and objects are left out). */
export function reportRows(r: TrainReport): [string, string][] {
  const scalar = (v: unknown) => v === null || ["number", "string", "boolean"].includes(typeof v);
  const keys = Object.keys(r).filter((k) => scalar(r[k]));
  keys.sort((a, b) => {
    const [ia, ib] = [FIRST.indexOf(a), FIRST.indexOf(b)];
    if (ia >= 0 || ib >= 0) return (ia < 0 ? Infinity : ia) - (ib < 0 ? Infinity : ib);
    return a.localeCompare(b);
  });
  return keys.map((k) => {
    const v = r[k];
    const text = v === null ? "–" : typeof v === "number" && !Number.isInteger(v) ? v.toFixed(Math.abs(v) < 1 ? 4 : 2) : String(v);
    return [k, text];
  });
}

/** The learned exposure per training frame, in frame order, or null without exposure_comp. */
export function exposureSeries(r: TrainReport): { frames: string[]; gains: number[] } | null {
  const g = r.exposure_gains;
  if (!g || Object.keys(g).length === 0) return null;
  const frames = Object.keys(g).sort((a, b) => a.localeCompare(b, undefined, { numeric: true }));
  return { frames, gains: frames.map((f) => g[f]) };
}
