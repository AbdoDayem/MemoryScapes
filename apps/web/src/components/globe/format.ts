/** Small display helpers shared by the globe, card, list and upload panel (no three.js). */
import type { MemoryStatus } from "@/lib/api";

export const STATUS_COLORS: Record<MemoryStatus, string> = {
  ready: "#ff7f8f",
  processing: "#6f8dff",
  uploading: "#6f8dff",
  draft: "#6f8dff",
  failed: "#b4b7c9",
};
export const CLUSTER_COLOR = "#ffc65c";

export function statusLabel(s: string): string {
  return ({ ready: "Ready", processing: "Processing", uploading: "Uploading", draft: "Draft", failed: "Failed" } as Record<string, string>)[s] ?? s;
}

export function formatDate(iso: string | null | undefined): string {
  if (!iso) return "";
  const d = new Date(iso);
  return Number.isNaN(d.getTime()) ? "" : d.toLocaleDateString(undefined, { month: "short", day: "numeric", year: "numeric" });
}
