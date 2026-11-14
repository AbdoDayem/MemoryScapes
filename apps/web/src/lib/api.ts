/**
 * The web app's client for the API (services/api). One function per endpoint; components
 * never build URLs themselves. The contract is docs/webapp-plan/backend.md.
 */
import type { MemoryManifest } from "@memoryscapes/manifest";

export const API_URL = process.env.NEXT_PUBLIC_API_URL ?? "http://localhost:8000";

/** Memory states (services/api/app/models.py MemoryStatus). */
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
  /** The pipeline run's per-stage progress. Null before processing and for CLI-made memories. */
  stage_detail?: {
    state?: string | null;
    current?: string | null;
    detail?: string | null;
    stages: Record<string, StageDetail>;
  } | null;
  // Details from the database. Null for memories made with the worker CLI.
  title?: string | null;
  description?: string | null;
  visibility?: "private" | "unlisted" | "public" | null;
  lat?: number | null;
  lon?: number | null;
  location_source?: "video_gps" | "user_pin" | null;
  captured_at?: string | null;
  created_at?: string | null;
  error?: string | null;
  /** The video's own GPS, rounded, when the memory has no location yet. */
  suggested_location?: { lat: number; lon: number } | null;
}

/** GET /globe: one memory that has a location. */
export interface GlobeMemory {
  id: string;
  title: string | null;
  lat: number;
  lon: number;
  status: MemoryStatus;
  captured_at: string | null;
  created_at: string;
  thumbnail_url: string | null;
}

/** GET /memories: one of the dev user's memories (with or without a location). */
export interface MemorySummary {
  id: string;
  title: string | null;
  status: MemoryStatus;
  visibility: "private" | "unlisted" | "public";
  thumbnail_url: string | null;
  captured_at: string | null;
}

export interface UploadTarget {
  file_name: string;
  /** Where to PUT the file's bytes: this API now, a presigned S3 URL later. */
  url: string;
}

export interface UpdateMemory {
  title?: string | null;
  description?: string | null;
  lat?: number | null;
  lon?: number | null;
  location_source?: "video_gps" | "user_pin";
}

/** An error the API answered with: its status code and its `detail` message. */
export class ApiError extends Error {
  constructor(public status: number, message: string) {
    super(message);
    this.name = "ApiError";
  }
}

async function request<T>(path: string, init?: RequestInit): Promise<T> {
  let res: Response;
  try {
    res = await fetch(`${API_URL}${path}`, init);
  } catch {
    throw new ApiError(0, `Can't reach the API at ${API_URL}. Is it running?`);
  }
  if (!res.ok) throw new ApiError(res.status, await detail(res));
  return res.json() as Promise<T>;
}

async function detail(res: Response): Promise<string> {
  try {
    return detailMessage(await res.json(), res.status);
  } catch {
    return `Request failed (${res.status})`; // not JSON
  }
}

/** FastAPI's `detail`: a string, or a list of validation errors. */
function detailMessage(body: { detail?: unknown }, status: number): string {
  if (typeof body.detail === "string") return body.detail;
  if (Array.isArray(body.detail)) return body.detail.map((d: { msg?: string }) => d.msg).filter(Boolean).join("; ");
  return `Request failed (${status})`;
}

const json = (method: string, body: unknown): RequestInit => ({
  method,
  headers: { "Content-Type": "application/json" },
  body: JSON.stringify(body),
});

export function getMemory(id: string): Promise<MemoryResponse> {
  return request(`/memories/${encodeURIComponent(id)}`);
}

export function listMemories(): Promise<MemorySummary[]> {
  return request("/memories");
}

/** Memories that have a location. Without a bbox: the whole world. */
export async function queryGlobe(bbox?: [number, number, number, number]): Promise<GlobeMemory[]> {
  const query = bbox ? `?bbox=${bbox.join(",")}` : "";
  const body = await request<{ memories: GlobeMemory[] }>(`/globe${query}`);
  return body.memories;
}

export function createMemory(req: { title?: string | null; file_names: string[] }): Promise<{ id: string; uploads: UploadTarget[] }> {
  return request("/memories", json("POST", { ...req, input_kind: "video" }));
}

export function updateMemory(id: string, patch: UpdateMemory): Promise<MemoryResponse> {
  return request(`/memories/${encodeURIComponent(id)}`, json("PATCH", patch));
}

export function startProcessing(id: string): Promise<{ id: string; status: MemoryStatus }> {
  return request(`/memories/${encodeURIComponent(id)}/process`, { method: "POST" });
}

/**
 * PUT one file's bytes to its upload URL. Uses XMLHttpRequest because fetch can't report
 * upload progress. Returns the request so the caller can abort() it.
 */
export function uploadFile(url: string, file: File, onProgress: (fraction: number) => void):
    { done: Promise<void>; abort: () => void } {
  const xhr = new XMLHttpRequest();
  const done = new Promise<void>((resolve, reject) => {
    xhr.upload.onprogress = (e) => e.lengthComputable && onProgress(e.loaded / e.total);
    xhr.onload = () => {
      if (xhr.status >= 200 && xhr.status < 300) return resolve();
      let message = `Upload failed (${xhr.status})`;
      try {
        message = detailMessage(JSON.parse(xhr.responseText), xhr.status);
      } catch {
        /* not JSON */
      }
      reject(new ApiError(xhr.status, message));
    };
    xhr.onerror = () => reject(new ApiError(0, `Can't reach the API at ${API_URL}. Is it running?`));
    xhr.onabort = () => reject(new ApiError(-1, "Upload cancelled."));
  });
  xhr.open("PUT", url);
  xhr.setRequestHeader("Content-Type", file.type || "application/octet-stream");
  xhr.send(file);
  return { done, abort: () => xhr.abort() };
}
