import { useState } from "react";

const MEMORY_ID = /^[a-z0-9][a-z0-9_-]{0,63}$/;

export function slugify(name: string): string {
  const base = name
    .replace(/\.[^.]+$/, "")
    .toLowerCase()
    .replace(/[^a-z0-9]+/g, "_")
    .replace(/^_+|_+$/g, "")
    .slice(0, 48);
  return base || "upload";
}

function put(url: string, file: File, onProgress: (f: number) => void): Promise<void> {
  return new Promise((resolve, reject) => {
    const xhr = new XMLHttpRequest();
    xhr.open("PUT", url);
    xhr.setRequestHeader("Content-Type", file.type || "application/octet-stream");
    xhr.upload.onprogress = (e) => e.lengthComputable && onProgress(e.loaded / e.total);
    xhr.onload = () => {
      if (xhr.status >= 200 && xhr.status < 300) resolve();
      else {
        let msg = `Upload failed (${xhr.status})`;
        try {
          msg = JSON.parse(xhr.responseText).error ?? msg;
        } catch {
          /* keep status text */
        }
        reject(new Error(msg));
      }
    };
    xhr.onerror = () => reject(new Error("Upload failed: the debugger server didn't respond."));
    xhr.send(file);
  });
}

export function UploadView({ onStarted }: { onStarted: (memoryId: string) => void }) {
  const [video, setVideo] = useState<File | null>(null);
  const [audio, setAudio] = useState<File | null>(null);
  const [name, setName] = useState("");
  const [maxSeconds, setMaxSeconds] = useState(30);
  const [detector, setDetector] = useState(true);
  const [progress, setProgress] = useState<string | null>(null);
  const [error, setError] = useState<string | null>(null);
  const memoryId = name || (video ? slugify(video.name) : "");
  const valid = !!video && MEMORY_ID.test(memoryId);

  const submit = async (e: React.FormEvent) => {
    e.preventDefault();
    if (!video || !valid) return;
    setError(null);
    try {
      const q = (kind: string, f: File) => `/api/upload?memory=${encodeURIComponent(memoryId)}&kind=${kind}&name=${encodeURIComponent(f.name)}`;
      await put(q("video", video), video, (f) => setProgress(`Uploading video… ${Math.round(f * 100)}%`));
      if (audio) await put(q("audio", audio), audio, (f) => setProgress(`Uploading audio… ${Math.round(f * 100)}%`));
      setProgress("Starting the pipeline…");
      const res = await fetch(`/api/memories/${memoryId}/process`, { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify({ maxSeconds, detector }) });
      if (!res.ok) throw new Error((await res.json().catch(() => ({}))).error ?? `Couldn't start processing (${res.status})`);
      setProgress(null);
      onStarted(memoryId);
    } catch (err) {
      setProgress(null);
      setError(String(err instanceof Error ? err.message : err));
    }
  };

  return (
    <section className="panel" style={{ maxWidth: 720 }}>
      <h2>Upload a video</h2>
      <p className="muted" style={{ margin: 0 }}>
        Runs the pipeline on this machine: ingest (ffmpeg) → camera poses (COLMAP) → separation (CLAP, AudioSep, Demucs) → localization (Grounding DINO + stereo) → soundscape (stub). The
        splat stage is skipped until it exists. On a CPU, expect several minutes for 30 seconds of video.
      </p>
      <form className="form" onSubmit={submit}>
        <label className="field">
          <span>Video, with its sound</span>
          <input
            id="video"
            type="file"
            accept="video/*"
            onChange={(e) => {
              const f = e.target.files?.[0] ?? null;
              setVideo(f);
              if (f && !name) setName(slugify(f.name));
            }}
          />
          <small className="muted">Walk slowly and pan a little. Movement helps both COLMAP and stereo localization.</small>
        </label>
        <label className="field">
          <span>Separate audio recording (optional)</span>
          <input id="audio" type="file" accept="audio/*" onChange={(e) => setAudio(e.target.files?.[0] ?? null)} />
          <small className="muted">Used instead of the video's own sound. It must start when the video starts.</small>
        </label>
        <label className="field">
          <span>Memory name</span>
          <input id="name" type="text" value={name} placeholder="e.g. park_walk_june" onChange={(e) => setName(e.target.value)} />
          {name && !MEMORY_ID.test(name) && <small style={{ color: "var(--fail)" }}>Use lowercase letters, digits, - and _ only.</small>}
          <small className="muted">Re-using a name replaces that memory's outputs.</small>
        </label>
        <div className="row">
          <label className="field" style={{ flex: "0 0 auto" }}>
            <span>Process the first</span>
            <span className="row" style={{ gap: 6 }}>
              <input id="maxSeconds" type="number" min={5} max={120} value={maxSeconds} onChange={(e) => setMaxSeconds(Number(e.target.value))} style={{ width: 80 }} />
              seconds
            </span>
          </label>
          <label className="row" style={{ gap: 6, cursor: "pointer" }}>
            <input id="detector" type="checkbox" checked={detector} onChange={(e) => setDetector(e.target.checked)} />
            Visual grounding (Grounding DINO). Slowest step on a CPU; off means stereo only.
          </label>
        </div>
        <div className="row">
          <button className="btn primary" type="submit" disabled={!valid || !!progress}>
            Upload and process
          </button>
          {progress && <span className="muted">{progress}</span>}
          {error && <span style={{ color: "var(--fail)" }}>{error}</span>}
        </div>
      </form>
    </section>
  );
}
