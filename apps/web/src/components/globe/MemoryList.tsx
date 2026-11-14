"use client";

/**
 * "My memories": every memory, including ones without a location. Also the keyboard,
 * screen-reader and no-3D way in.
 */
import type { MemorySummary } from "@/lib/api";

import { STATUS_COLORS, formatDate } from "./format";

export interface MemoryListProps {
  memories: MemorySummary[];
  located: Set<string>;
  onPick(id: string): void;
  /** Inside another card (the "memories here" picker): no own box or heading. */
  embedded?: boolean;
}

export default function MemoryList({ memories, located, onPick, embedded = false }: MemoryListProps) {
  return (
    <nav className={embedded ? undefined : "ms-list ms-glass"} aria-label={embedded ? "Memories here" : "My memories"}>
      {!embedded && <h3>My memories</h3>}
      {memories.length === 0 && <div className="ms-empty">No memories yet. Upload your first one.</div>}
      {memories.map((m) => (
        <button key={m.id} className="ms-row" onClick={() => onPick(m.id)}>
          <span className="ms-dot" style={{ background: STATUS_COLORS[m.status] ?? STATUS_COLORS.processing }} />
          <span className="t">{m.title || "Untitled memory"}</span>
          <span className="d">{located.has(m.id) ? formatDate(m.captured_at) : "no location"}</span>
        </button>
      ))}
    </nav>
  );
}
