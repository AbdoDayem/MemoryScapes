import type { MemoryManifest } from "@memoryscapes/manifest";

const API_URL = process.env.NEXT_PUBLIC_API_URL ?? "http://localhost:8000";

/**
 * Memory states. Today the API sends `ready` or `processing`; `draft`, `uploading` and
 * `failed` come with uploads (docs/webapp-plan/backend.md).
 */
export type MemoryStatus = "draft" | "uploading" | "processing" | "ready" | "failed";

/** One stage of a pipeline run (from the runner's job/status.json). */
export interface StageDetail {
  status: "pending" | "running" | "done" | "failed" | "skipped" | "blocked";
  seconds?: number;
  error?: string;
}

/** GET /memories/:id (services/api/app/routers/memories.py). */
export interface MemoryResponse {
  id: string;
  status: MemoryStatus;
  /** pipeline stage -> finished */
  stages: Record<string, boolean>;
  manifest: MemoryManifest | null;
  /** Base URL the manifest's relative asset URLs resolve against. */
  asset_base: string;
  /** Planned: the pipeline run's per-stage progress. Absent for CLI-made memories. */
  stage_detail?: { state?: string; detail?: string; stages: Record<string, StageDetail> } | null;
}

export async function getMemory(id: string): Promise<MemoryResponse> {
  const res = await fetch(`${API_URL}/memories/${encodeURIComponent(id)}`);
  if (!res.ok) throw new Error(`Failed to load memory ${id}: ${res.status}`);
  return res.json();
}

export async function queryGlobe(bbox: [number, number, number, number], zoom: number) {
  const res = await fetch(`${API_URL}/globe?bbox=${bbox.join(",")}&zoom=${zoom}`);
  if (!res.ok) throw new Error(`Globe query failed: ${res.status}`);
  return res.json();
}
