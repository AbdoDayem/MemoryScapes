import { useEffect, useRef, useState } from "react";

import { hasFile, loadSplat } from "./api";
import type { GaussianView } from "./ply";
import type { MemoryBundle } from "./types";

export function cssVar(name: string): string {
  return getComputedStyle(document.documentElement).getPropertyValue(name).trim() || "#888";
}

export const METHOD_VAR: Record<string, string> = {
  "av+doa": "--fused",
  av: "--av",
  doa: "--doa",
  diffuse: "--diffuse",
};

export const methodColor = (m: string) => cssVar(METHOD_VAR[m] ?? "--diffuse");

/** Diffuse classes in services/workers/memoryscapes_workers/audio/vocabulary.py. */
export const DIFFUSE_CLASSES = new Set(["wind", "rain", "insects", "crowd"]);
export const TAG_THRESHOLD = 0.2;
export const TAG_STRONG = 0.35;

export function useSplat(bundle: MemoryBundle | null) {
  const [state, setState] = useState<{ splat?: GaussianView; error?: string; loading: boolean }>({ loading: false });
  useEffect(() => {
    if (!bundle || !hasFile(bundle, "splat/point_cloud.ply")) {
      setState({ loading: false });
      return;
    }
    let live = true;
    setState({ loading: true });
    loadSplat(bundle.id)
      .then((splat) => live && setState({ splat, loading: false }))
      .catch((e) => live && setState({ error: String(e), loading: false }));
    return () => {
      live = false;
    };
  }, [bundle]);
  return state;
}

/** Canvas sized to its CSS width at device pixel ratio; redraws on resize. */
export function useCanvas(draw: (ctx: CanvasRenderingContext2D, w: number, h: number) => void, height: number, deps: unknown[]) {
  const ref = useRef<HTMLCanvasElement>(null);
  const [width, setWidth] = useState(0);
  useEffect(() => {
    const el = ref.current;
    if (!el) return;
    const ro = new ResizeObserver(() => setWidth(el.clientWidth));
    ro.observe(el);
    setWidth(el.clientWidth);
    return () => ro.disconnect();
  }, []);
  useEffect(() => {
    const el = ref.current;
    if (!el || !width) return;
    const dpr = window.devicePixelRatio || 1;
    el.width = Math.round(width * dpr);
    el.height = Math.round(height * dpr);
    el.style.height = `${height}px`;
    const ctx = el.getContext("2d")!;
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    ctx.clearRect(0, 0, width, height);
    draw(ctx, width, height);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [width, height, ...deps]);
  return ref;
}

export const fmt = (v: number | null | undefined, d = 2) => (v === null || v === undefined || !Number.isFinite(v) ? "–" : v.toFixed(d));
export const fmtVec = (v: number[] | null | undefined) => (v ? `(${v.map((x) => x.toFixed(1)).join(", ")})` : "–");
export const bytes = (n: number) => (n > 1e6 ? `${(n / 1e6).toFixed(1)} MB` : n > 1e3 ? `${(n / 1e3).toFixed(0)} kB` : `${n} B`);
