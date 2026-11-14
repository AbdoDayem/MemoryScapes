import type { MemoryManifest } from "@memoryscapes/manifest";

const API_URL = process.env.NEXT_PUBLIC_API_URL ?? "http://localhost:8000";

export async function getMemory(id: string): Promise<{ status: string; manifest?: MemoryManifest }> {
  const res = await fetch(`${API_URL}/memories/${id}`);
  if (!res.ok) throw new Error(`Failed to load memory ${id}: ${res.status}`);
  return res.json();
}

export async function queryGlobe(bbox: [number, number, number, number], zoom: number) {
  const res = await fetch(`${API_URL}/globe?bbox=${bbox.join(",")}&zoom=${zoom}`);
  if (!res.ok) throw new Error(`Globe query failed: ${res.status}`);
  return res.json();
}
