import { parseNpy } from "./npy";
import { gaussians, parsePly, type GaussianView } from "./ply";
import type { FileEntry, MemoryBundle, MemorySummary } from "./types";

export const dataUrl = (memoryId: string, key: string) => `/data/${encodeURIComponent(memoryId)}/${key.split("/").map(encodeURIComponent).join("/")}`;

/** Stage outputs store keys with the memory id prefix ("mem_1/separate/x.wav"); strip it. */
export function relKey(memoryId: string, key: string): string {
  return key.startsWith(`${memoryId}/`) ? key.slice(memoryId.length + 1) : key;
}

export async function listMemories(): Promise<MemorySummary[]> {
  const res = await fetch("/api/memories");
  if (!res.ok) throw new Error(`Couldn't list memories (${res.status})`);
  return res.json();
}

const JSON_FILES = {
  ingestMeta: "ingest/meta.json",
  cameras: "poses/cameras.json",
  stems: "separate/stems.json",
  localized: "localize/sources.json",
  scene: "soundscape/scene.json",
  splatMeta: "splat/meta.json",
  splatMetrics: "splat/metrics.json",
  truth: "debug/truth.json",
} as const;

export async function loadMemory(id: string): Promise<MemoryBundle> {
  const filesRes = await fetch(`/api/memories/${encodeURIComponent(id)}/files`);
  if (!filesRes.ok) throw new Error(`Couldn't list files for ${id} (${filesRes.status})`);
  const files: FileEntry[] = await filesRes.json();
  const have = new Set(files.map((f) => f.key));
  const bundle: MemoryBundle = { id, files, errors: [] };

  await Promise.all(
    Object.entries(JSON_FILES).map(async ([field, key]) => {
      if (!have.has(key)) return;
      try {
        const res = await fetch(dataUrl(id, key));
        (bundle as unknown as Record<string, unknown>)[field] = await res.json();
      } catch (e) {
        bundle.errors.push({ key, message: String(e) });
      }
    }),
  );

  for (const [key, set] of [
    ["poses/points3d.npy", (a: ReturnType<typeof parseNpy>) => (bundle.points = Float32Array.from(a.data))],
    ["poses/points3d_rgb.npy", (a: ReturnType<typeof parseNpy>) => (bundle.pointColors = Uint8Array.from(a.data))],
  ] as const) {
    if (!have.has(key)) continue;
    try {
      set(parseNpy(await (await fetch(dataUrl(id, key))).arrayBuffer()));
    } catch (e) {
      bundle.errors.push({ key, message: String(e) });
    }
  }
  return bundle;
}

const plyCache = new Map<string, Promise<GaussianView>>();

export function loadSplat(id: string, key = "splat/point_cloud.ply"): Promise<GaussianView> {
  const cacheKey = `${id}/${key}`;
  if (!plyCache.has(cacheKey)) {
    plyCache.set(
      cacheKey,
      fetch(dataUrl(id, key))
        .then((r) => {
          if (!r.ok) throw new Error(`${key}: ${r.status}`);
          return r.arrayBuffer();
        })
        .then((b) => gaussians(parsePly(b))),
    );
  }
  return plyCache.get(cacheKey)!;
}

let audioCtx: AudioContext | null = null;
export function audioContext(): AudioContext {
  audioCtx ??= new AudioContext();
  return audioCtx;
}

const audioCache = new Map<string, Promise<AudioBuffer>>();
export function loadAudio(id: string, key: string): Promise<AudioBuffer> {
  const k = `${id}/${key}`;
  if (!audioCache.has(k)) {
    audioCache.set(
      k,
      fetch(dataUrl(id, key))
        .then((r) => {
          if (!r.ok) throw new Error(`${key}: ${r.status}`);
          return r.arrayBuffer();
        })
        .then((b) => audioContext().decodeAudioData(b)),
    );
  }
  return audioCache.get(k)!;
}

export function hasFile(bundle: MemoryBundle, key: string): boolean {
  return bundle.files.some((f) => f.key === key);
}
