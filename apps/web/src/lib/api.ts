import type { MemoryManifest } from "@memoryscapes/manifest";

const API_URL = process.env.NEXT_PUBLIC_API_URL ?? "http://localhost:8000";

/** GET /memories/:id (services/api/app/routers/memories.py). */
export interface MemoryResponse {
  id: string;
  status: "ready" | "processing";
  /** pipeline stage -> finished */
  stages: Record<string, boolean>;
  manifest: MemoryManifest | null;
  /** Base URL the manifest's relative asset URLs resolve against. */
  asset_base: string;
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
