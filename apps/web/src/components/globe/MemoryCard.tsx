"use client";

/** The preview card for one memory, opened by clicking its marker or picking it from the list. */
import Link from "next/link";
import { useEffect, useState } from "react";

import { getMemory, startProcessing, updateMemory, type MemoryResponse } from "@/lib/api";

import { STATUS_COLORS, formatDate, statusLabel } from "./format";

export interface MemoryCardProps {
  id: string;
  /** Bumped by the parent when the memory may have changed (after an edit or a poll). */
  version: number;
  moving: boolean;
  onClose(): void;
  /** Start or stop "click the globe to move this memory's pin". */
  onMoveRequest(moving: boolean): void;
  /** The memory changed (title, location, re-processing): reload the globe. */
  onChanged(): void;
}

export default function MemoryCard({ id, version, moving, onClose, onMoveRequest, onChanged }: MemoryCardProps) {
  const [m, setM] = useState<MemoryResponse | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [editing, setEditing] = useState(false);
  const [title, setTitle] = useState("");
  const [busy, setBusy] = useState(false);

  useEffect(() => {
    let cancelled = false;
    getMemory(id)
      .then((r) => !cancelled && (setM(r), setError(null)))
      .catch((e) => !cancelled && setError(String(e.message ?? e)));
    return () => { cancelled = true; };
  }, [id, version]);

  const act = async (fn: () => Promise<unknown>) => {
    setBusy(true);
    setError(null);
    try {
      await fn();
      onChanged();
    } catch (e) {
      setError(e instanceof Error ? e.message : String(e));
    } finally {
      setBusy(false);
    }
  };

  const status = m?.status ?? "processing";
  const color = STATUS_COLORS[status] ?? STATUS_COLORS.processing;
  const name = m?.title || "Untitled memory";

  return (
    <section className="ms-card ms-glass" aria-label={`Memory: ${name}`}>
      <button className="ms-close" onClick={onClose} aria-label="Close">×</button>
      {!m && !error && <div className="ms-meta">Loading…</div>}
      {m && (
        <>
          <h2>{name}</h2>
          <div className="ms-meta">
            {m.captured_at ? `Filmed ${formatDate(m.captured_at)}` : m.created_at ? `Added ${formatDate(m.created_at)}` : ""}
            {m.lat != null && m.lon != null && ` · ${m.lat.toFixed(3)}, ${m.lon.toFixed(3)}`}
          </div>
          <div className="ms-status"><span className="ms-dot" style={{ background: color }} />{statusLabel(status)}</div>
          {m.status === "failed" && m.error && <div className="ms-err">{m.error}</div>}

          {editing ? (
            <div className="ms-edit">
              <input className="ms-input" value={title} maxLength={120} placeholder="Title" autoFocus
                     onChange={(e) => setTitle(e.target.value)} />
              <div className="ms-actions">
                <button className="ms-btn" disabled={busy}
                        onClick={() => act(async () => { await updateMemory(id, { title: title.trim() || null }); setEditing(false); })}>
                  Save
                </button>
                <button className="ms-ghost" onClick={() => setEditing(false)}>Cancel</button>
              </div>
            </div>
          ) : (
            <div className="ms-actions">
              {m.status === "ready" && <Link className="ms-btn" href={`/memories/${id}`}>Enter memory</Link>}
              {(m.status === "processing" || m.status === "uploading") && (
                <Link className="ms-btn" href={`/memories/${id}`}>View progress</Link>
              )}
              {m.status === "failed" && (
                <button className="ms-btn" disabled={busy} onClick={() => act(() => startProcessing(id))}>Try again</button>
              )}
              <button className="ms-ghost" onClick={() => { setTitle(m.title ?? ""); setEditing(true); }}>Edit title</button>
              <button className="ms-ghost" onClick={() => onMoveRequest(!moving)}>
                {moving ? "Cancel moving" : m.lat == null ? "Place on globe" : "Move pin"}
              </button>
            </div>
          )}

          {m.lat == null && m.suggested_location && (
            <div className="ms-actions" style={{ marginTop: 10 }}>
              <button className="ms-ghost" disabled={busy}
                      onClick={() => act(() => updateMemory(id, { ...m.suggested_location!, location_source: "video_gps" }))}>
                📍 Use the video&apos;s location
              </button>
            </div>
          )}
          {moving && <div className="ms-note" style={{ color: "#4057c7", fontWeight: 700 }}>Tap the globe where this was filmed.</div>}
          <div className="ms-note">Locations are rounded to about 100 m.</div>
        </>
      )}
      {error && <div className="ms-err">{error}</div>}
    </section>
  );
}
