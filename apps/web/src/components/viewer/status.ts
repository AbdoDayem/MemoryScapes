import type { MemoryResponse, StageDetail } from "@/lib/api";

/** Pipeline stages in DAG order (services/workers/memoryscapes_workers/pipeline.py). */
export const STAGES = ["ingest", "poses", "splat", "separate", "localize", "soundscape", "publish"] as const;

export type PollAction = "show" | "poll" | "stop";

/**
 * What the viewer does with a memory response: show the scene, keep polling, or stop
 * (the run failed: polling would never end).
 */
export function pollAction(m: MemoryResponse): PollAction {
  if (m.status === "ready" && m.manifest) return "show";
  if (m.status === "failed") return "stop";
  return "poll";
}

export interface StageRow {
  name: string;
  status: StageDetail["status"];
  seconds?: number;
  error?: string;
}

/** Per-stage rows: the detailed job status when the API has it, else the done flags. */
export function stageRows(m: MemoryResponse): StageRow[] {
  const detail = m.stage_detail?.stages;
  const names = [...STAGES, ...Object.keys(m.stages ?? {}).filter((s) => !(STAGES as readonly string[]).includes(s))];
  return names
    .filter((name) => (detail && name in detail) || (m.stages && name in m.stages))
    .map((name) => {
      const d = detail?.[name];
      if (d) return { name, status: d.status, seconds: d.seconds, error: d.error };
      return { name, status: m.stages?.[name] ? "done" : "pending" };
    });
}

/** One line saying why a run failed, for the failed screen. */
export function failureSummary(m: MemoryResponse): string {
  const failed = stageRows(m).find((r) => r.status === "failed");
  if (failed) return `The ${failed.name} stage failed${failed.error ? `: ${failed.error}` : "."}`;
  return m.stage_detail?.detail ? `Processing failed: ${m.stage_detail.detail}` : "Processing failed.";
}
