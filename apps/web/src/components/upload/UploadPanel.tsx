"use client";

/**
 * "+ New memory": a panel over the globe (docs/webapp-plan/frontend.md § Upload panel).
 *   1 choose files → 2 title → 3 tap the globe to place it → 4 upload → 5 processing
 * The globe stays live behind it; while placing, the parent passes taps through as `pin`.
 */
import Link from "next/link";
import { useEffect, useRef, useState } from "react";

import {
  ApiError, createMemory, getMemory, startProcessing, updateMemory, uploadFile, type MemoryResponse,
} from "@/lib/api";
import { stageRows } from "@/components/viewer/status";

// What the pipeline accepts (services/api/app/routers/memories.py, from the workers' ingest).
const VIDEO_EXTS = [".mp4", ".mov", ".m4v", ".mkv", ".webm", ".avi", ".3gp"];
const AUDIO_EXTS = [".wav", ".flac", ".m4a", ".mp3", ".aac", ".ogg", ".opus", ".caf", ".aiff", ".aif"];
const MAX_MB = 500;
const MAX_SECONDS = 120; // the API's MAX_VIDEO_SECONDS (services/api/app/settings.py); keep the two equal
// The API's rule (Python's \w includes letters and digits in any script).
const FILE_NAME = /^[\p{L}\p{N}_.\- ()]{1,128}$/u;

export interface UploadPanelProps {
  pin: { lat: number; lon: number } | null;
  /** Ask the parent to pass globe taps to us as `pin` (step 3), or stop. */
  onPlacingChange(placing: boolean): void;
  onClose(): void;
  /** A memory was created or changed: reload the globe. */
  onChanged(id: string): void;
}

type Step = 0 | 1 | 2 | 3 | 4;

interface Picked {
  file: File;
  problem: string | null;
  seconds: number | null;
}

const ext = (name: string) => (name.match(/\.[^.]+$/)?.[0] ?? "").toLowerCase();
const mb = (bytes: number) => `${(bytes / 1024 / 1024).toFixed(bytes > 10 * 1024 * 1024 ? 0 : 1)} MB`;
const mmss = (s: number) => `${Math.floor(s / 60)}:${String(Math.round(s % 60)).padStart(2, "0")}`;

export default function UploadPanel({ pin, onPlacingChange, onClose, onChanged }: UploadPanelProps) {
  const [step, setStep] = useState<Step>(0);
  const [video, setVideo] = useState<Picked | null>(null);
  const [audio, setAudio] = useState<Picked | null>(null);
  const [title, setTitle] = useState("");
  const [progress, setProgress] = useState<Record<string, number>>({});
  const [error, setError] = useState<string | null>(null);
  const [memoryId, setMemoryId] = useState<string | null>(null);
  const [memory, setMemory] = useState<MemoryResponse | null>(null);
  const aborts = useRef<(() => void)[]>([]);
  const cancelled = useRef(false); // set by Cancel and by closing the panel; checked after every await
  // The created memory and its upload URLs, reused if an upload is retried (no duplicate drafts),
  // and which files already finished uploading to it.
  const created = useRef<{ id: string; uploads: { file_name: string; url: string }[]; done: Set<string> } | null>(null);

  useEffect(() => onPlacingChange(step === 2), [step, onPlacingChange]);
  useEffect(() => () => {
    cancelled.current = true;
    aborts.current.forEach((a) => a());
    onPlacingChange(false);
  }, [onPlacingChange]);

  // Step 5: poll the run's progress until it ends.
  useEffect(() => {
    if (step !== 4 || !memoryId) return;
    let timer: ReturnType<typeof setTimeout>;
    let cancelled = false;
    const poll = async () => {
      try {
        const m = await getMemory(memoryId);
        if (cancelled) return;
        setMemory(m);
        if (m.status === "ready" || m.status === "failed") return onChanged(memoryId);
      } catch {
        /* the API restarting: keep trying */
      }
      if (!cancelled) timer = setTimeout(poll, 3000);
    };
    poll();
    return () => { cancelled = true; clearTimeout(timer); };
  }, [step, memoryId, onChanged]);

  const pickVideo = async (file: File | undefined) => {
    if (!file) return;
    setVideo({ file, problem: checkName(file, VIDEO_EXTS, "video"), seconds: null });
    const seconds = await videoDuration(file);
    setVideo((v) => v && v.file === file ? {
      ...v, seconds,
      problem: v.problem ?? (seconds != null && seconds > MAX_SECONDS ? `This video is ${mmss(seconds)}. Keep it under 2 minutes.` : null),
    } : v);
  };
  const pickAudio = (file: File | undefined) => {
    if (file) setAudio({ file, problem: checkName(file, AUDIO_EXTS, "audio file"), seconds: null });
  };

  const CANCELLED = "Upload cancelled. The memory is saved as a draft in “My memories”.";

  /** Create (once) → PUT each file not yet uploaded → title and location → start processing. */
  const upload = async (withPin: boolean) => {
    if (!video) return;
    cancelled.current = false;
    setStep(3);
    setError(null);
    const files = [video.file, ...(audio ? [audio.file] : [])];
    const stop = () => { if (cancelled.current) throw new ApiError(-1, CANCELLED); };
    try {
      // Reuse the draft while every file was listed when it was created (the API only accepts
      // those names). Picking a different video makes a new draft.
      const listed = created.current && files.every((f) => created.current!.uploads.some((u) => u.file_name === f.name));
      const isRetry = !!listed;
      if (!listed) {
        const c = await createMemory({ title: title.trim() || null, file_names: files.map((f) => f.name) });
        created.current = { ...c, done: new Set() };
        setMemoryId(c.id);
        setProgress({});
        onChanged(c.id); // a draft now exists (shows in "My memories")
        stop();
      }
      const { id, uploads, done } = created.current!;
      for (const file of files) {
        if (done.has(file.name)) continue; // already uploaded on an earlier try
        const target = uploads.find((u) => u.file_name === file.name)!;
        const req = uploadFile(target.url, file, (f) => setProgress((p) => ({ ...p, [file.name]: f })));
        aborts.current = [req.abort];
        await req.done;
        aborts.current = [];
        done.add(file.name);
        setProgress((p) => ({ ...p, [file.name]: 1 }));
        stop();
      }
      if (isRetry) { await updateMemory(id, { title: title.trim() || null }); stop(); }
      if (withPin && pin) { await updateMemory(id, { lat: pin.lat, lon: pin.lon }); stop(); }
      await startProcessing(id);
      onChanged(id);
      setStep(4);
    } catch (e) {
      aborts.current = [];
      if (cancelled.current && !(e instanceof ApiError && e.status === -1)) return; // panel closed
      setError(e instanceof ApiError && e.status === -1 ? CANCELLED : e instanceof Error ? e.message : String(e));
    }
  };

  const cancelUpload = () => {
    cancelled.current = true;
    aborts.current.forEach((a) => a());
    if (!aborts.current.length) setError(CANCELLED); // cancelled between requests
  };

  const canNext = video && !video.problem && !(audio && audio.problem);
  const uploading = step === 3 && !error;

  return (
    <aside className="ms-panel ms-glass" aria-label="New memory">
      <button className="ms-close" onClick={onClose} aria-label="Close">×</button>
      <h2>New memory</h2>
      <div className="ms-small">A video of a place, and the sound around you.</div>
      <div className="ms-steps" aria-hidden>{[0, 1, 2, 3].map((i) => <div key={i} className={i <= Math.min(step, 3) ? "on" : ""} />)}</div>

      {step === 0 && (
        <>
          <div className="ms-field">
            <span className="ms-label">1 · Video</span>
            <DropZone accept="video/*,.mov,.mkv" picked={video} onPick={pickVideo}
                      empty={<>Drop a video here, or click to choose<br /><span className="ms-small">Up to 2 minutes, {MAX_MB} MB</span></>} />
          </div>
          <div className="ms-field">
            <span className="ms-label">Separate audio recording (optional)</span>
            <DropZone accept="audio/*" picked={audio} onPick={pickAudio} small
                      empty={<>Add a separate recording<br /><span className="ms-small">If you recorded sound on another device</span></>} />
            {audio && <button className="ms-ghost" style={{ marginTop: 8 }} onClick={() => setAudio(null)}>Remove audio</button>}
          </div>
          <ul className="ms-tips">
            <li>Move slowly and walk around the scene</li>
            <li>Don&apos;t just turn on the spot</li>
            <li>Keep sound sources in view</li>
            <li>Shield the mic from wind</li>
          </ul>
          <div className="ms-small" style={{ marginTop: 10 }}>The first 60 seconds are processed.</div>
          <div className="ms-panel-actions"><span /><button className="ms-btn" disabled={!canNext} onClick={() => setStep(1)}>Next</button></div>
        </>
      )}

      {step === 1 && (
        <>
          <div className="ms-field">
            <label htmlFor="ms-title">2 · Title (optional)</label>
            <input id="ms-title" className="ms-input" value={title} maxLength={120} autoFocus
                   placeholder="e.g. Golden Gate walk" onChange={(e) => setTitle(e.target.value)}
                   onKeyDown={(e) => e.key === "Enter" && setStep(2)} />
          </div>
          <div className="ms-panel-actions">
            <button className="ms-ghost" onClick={() => setStep(0)}>Back</button>
            <button className="ms-btn" onClick={() => setStep(2)}>Next</button>
          </div>
        </>
      )}

      {step === 2 && (
        <>
          <div className="ms-field">
            <span className="ms-label">3 · Where was this filmed?</span>
            <div className="ms-hint" aria-live="polite">
              {pin ? `📍 ${pin.lat.toFixed(3)}, ${pin.lon.toFixed(3)}. Tap the globe again to move it.` : "Tap the globe where this was filmed."}
            </div>
            <div className="ms-small" style={{ marginTop: 8 }}>
              Rounded to about 100 m. You can also skip, and use the video&apos;s GPS later from its card.
            </div>
          </div>
          <div className="ms-panel-actions">
            <button className="ms-ghost" onClick={() => setStep(1)}>Back</button>
            <span style={{ display: "flex", gap: 8 }}>
              <button className="ms-ghost" onClick={() => upload(false)}>Skip</button>
              <button className="ms-btn" disabled={!pin} onClick={() => upload(true)}>Upload</button>
            </span>
          </div>
        </>
      )}

      {step === 3 && (
        <>
          <div className="ms-field">
            <span className="ms-label">4 · Uploading</span>
            {[video?.file, audio?.file].filter((f): f is File => !!f).map((f) => (
              <div key={f.name} style={{ marginBottom: 10 }}>
                <div style={{ fontWeight: 700, fontSize: 14 }}>{f.name}</div>
                <div className="ms-bar"><div style={{ width: `${Math.round((progress[f.name] ?? 0) * 100)}%` }} /></div>
                <div className="ms-small">{Math.round((progress[f.name] ?? 0) * 100)}% of {mb(f.size)}</div>
              </div>
            ))}
          </div>
          {error && <div className="ms-err" role="alert">{error}</div>}
          <div className="ms-panel-actions">
            {error ? <button className="ms-ghost" onClick={() => setStep(2)}>Back</button> : <span />}
            {uploading
              ? <button className="ms-ghost" onClick={cancelUpload}>Cancel</button>
              : <button className="ms-btn" onClick={() => upload(!!pin)}>Try again</button>}
          </div>
        </>
      )}

      {step === 4 && memoryId && (
        <>
          <div className="ms-field">
            <span className="ms-label">5 · Processing</span>
            {(memory ? stageRows(memory) : []).map((r) => (
              <div key={r.name} className={`ms-stage ${r.status}`}>
                <span>{r.name}</span>
                <span>{r.status === "done" ? "✓" : r.status === "running" ? "…" : r.status === "failed" ? "✕" : ""}</span>
              </div>
            ))}
            {memory?.stage_detail?.detail && memory.status === "processing" && (
              <div className="ms-small" style={{ marginTop: 6 }}>{memory.stage_detail.detail}</div>
            )}
            {memory?.status === "failed" && <div className="ms-err" style={{ marginTop: 8 }}>{memory.error}</div>}
            {memory?.status === "ready" && <div className="ms-hint" style={{ marginTop: 8 }}>Your memory is ready 🎉</div>}
            {memory?.status !== "ready" && memory?.status !== "failed" && (
              <div className="ms-small" style={{ marginTop: 10 }}>
                This takes several minutes. You can close this and come back: it&apos;s already in “My memories”.
              </div>
            )}
          </div>
          <div className="ms-panel-actions">
            <button className="ms-ghost" onClick={onClose}>Close</button>
            <Link className="ms-btn" href={`/memories/${memoryId}`}>{memory?.status === "ready" ? "Enter memory" : "View progress"}</Link>
          </div>
        </>
      )}
    </aside>
  );
}

function DropZone({ accept, picked, onPick, empty, small }: {
  accept: string; picked: Picked | null; onPick(f: File | undefined): void; empty: React.ReactNode; small?: boolean;
}) {
  const [drag, setDrag] = useState(false);
  return (
    <label className={`ms-drop ${picked && !picked.problem ? "ok" : ""} ${drag ? "drag" : ""}`}
           style={small ? { padding: 14 } : undefined}
           onDragOver={(e) => { e.preventDefault(); setDrag(true); }}
           onDragLeave={() => setDrag(false)}
           onDrop={(e) => { e.preventDefault(); setDrag(false); onPick(e.dataTransfer.files[0]); }}>
      <input type="file" accept={accept} onChange={(e) => onPick(e.target.files?.[0])} />
      {picked ? (
        <>
          🎞 {picked.file.name}<br />
          <span className="ms-small">{[picked.seconds != null ? mmss(picked.seconds) : null, mb(picked.file.size)].filter(Boolean).join(" · ")}</span>
          {picked.problem && <div className="ms-err" style={{ margin: "6px 0 0" }}>{picked.problem}</div>}
        </>
      ) : empty}
    </label>
  );
}

function checkName(file: File, exts: string[], what: string): string | null {
  if (!exts.includes(ext(file.name))) return `That isn't a supported ${what} (${exts.join(" ")}).`;
  if (!FILE_NAME.test(file.name)) return "Rename the file: use letters, numbers, spaces and . - _ ( ) only.";
  if (file.size > MAX_MB * 1024 * 1024) return `This file is ${mb(file.size)}. The limit is ${MAX_MB} MB.`;
  if (file.size === 0) return "This file is empty.";
  return null;
}

/** The video's length from its metadata, or null if the browser can't read it (e.g. some .mkv). */
function videoDuration(file: File): Promise<number | null> {
  return new Promise((resolve) => {
    const v = document.createElement("video");
    const url = URL.createObjectURL(file);
    let settled = false;
    const done = (s: number | null) => {
      if (settled) return;
      settled = true;
      clearTimeout(timer);
      URL.revokeObjectURL(url);
      resolve(s);
    };
    const timer = setTimeout(() => done(null), 8000); // some files never report either way
    v.preload = "metadata";
    v.onloadedmetadata = () => done(Number.isFinite(v.duration) ? v.duration : null);
    v.onerror = () => done(null);
    v.src = url;
  });
}
