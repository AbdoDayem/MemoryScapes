import { useEffect, useRef, useState } from "react";

interface StageStatus {
  status: "pending" | "running" | "done" | "failed" | "skipped" | "blocked" | "not requested";
  seconds?: number;
  error?: string;
  started_at?: number;
}

interface JobStatus {
  state: "running" | "done" | "failed";
  current: string | null;
  detail: string;
  started_at?: number;
  finished_at?: number | null;
  options?: { max_seconds: number; detector: boolean };
  stages: Record<string, StageStatus>;
}

const PILL: Record<string, string> = { done: "ok", running: "info", failed: "fail", blocked: "fail", skipped: "warn", pending: "", "not requested": "" };

export function JobView({ memoryId, onFinished }: { memoryId: string; onFinished: () => void }) {
  const [job, setJob] = useState<{ running: boolean; status: JobStatus | null; log: string[] } | null>(null);
  const [now, setNow] = useState(Date.now() / 1000);
  const wasRunning = useRef(false);
  const logRef = useRef<HTMLPreElement>(null);

  useEffect(() => {
    let live = true;
    let timer: ReturnType<typeof setTimeout>;
    const poll = async () => {
      try {
        const res = await fetch(`/api/memories/${encodeURIComponent(memoryId)}/job`);
        const data = await res.json();
        if (!live) return;
        setJob(data);
        setNow(Date.now() / 1000);
        const active = data.running || data.status?.state === "running";
        if (active) wasRunning.current = true;
        else if (wasRunning.current) {
          wasRunning.current = false;
          onFinished();
        }
        timer = setTimeout(poll, active ? 1500 : 5000);
      } catch {
        timer = setTimeout(poll, 3000);
      }
    };
    poll();
    return () => {
      live = false;
      clearTimeout(timer);
    };
  }, [memoryId, onFinished]);

  useEffect(() => {
    logRef.current?.scrollTo({ top: logRef.current.scrollHeight });
  }, [job?.log.length]);

  const rerun = async (detector: boolean) => {
    await fetch(`/api/memories/${encodeURIComponent(memoryId)}/process`, { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify({ maxSeconds: job?.status?.options?.max_seconds ?? 30, detector }) });
    wasRunning.current = true;
    setJob((j) => (j ? { ...j, running: true } : j));
  };

  if (!job) return <p className="empty">Loading…</p>;
  const s = job.status;
  if (!s && !job.running) return <p className="empty">This memory hasn't been processed here. Upload a video to run the pipeline.</p>;
  const active = job.running || s?.state === "running";
  const elapsed = s?.started_at ? (s.finished_at ?? now) - s.started_at : 0;

  return (
    <>
      <section className="panel">
        <div className="row">
          <h2>Processing</h2>
          <span className={`badge ${active ? "info" : s?.state === "done" ? "ok" : "fail"}`}>{active ? "running" : s?.state}</span>
          <span className="muted">
            {Math.floor(elapsed / 60)}m {Math.round(elapsed % 60)}s
            {s?.options && ` · first ${s.options.max_seconds} s · visual grounding ${s.options.detector ? "on" : "off"}`}
          </span>
          {!active && (
            <>
              <button className="btn" onClick={() => rerun(true)}>
                Run again
              </button>
              <button className="btn" onClick={() => rerun(false)}>
                Run again without visual grounding
              </button>
            </>
          )}
        </div>
        {active && s?.current && (
          <p style={{ margin: 0 }}>
            <strong>{s.current}</strong> <span className="muted">{s.detail}</span>
          </p>
        )}
        <table>
          <thead>
            <tr>
              <th>Stage</th>
              <th>Status</th>
              <th className="num">Time</th>
              <th>Notes</th>
            </tr>
          </thead>
          <tbody>
            {Object.entries(s?.stages ?? {}).map(([name, st]) => (
              <tr key={name}>
                <td>{name}</td>
                <td>
                  <span className={`badge ${PILL[st.status] ?? ""}`} style={PILL[st.status] ? undefined : { color: "var(--muted)", border: "1px solid var(--line)" }}>
                    {st.status}
                  </span>
                </td>
                <td className="num">{st.seconds !== undefined ? `${st.seconds.toFixed(1)} s` : st.status === "running" && st.started_at ? `${Math.round(now - st.started_at)} s…` : ""}</td>
                <td className="muted">{st.error ?? ""}</td>
              </tr>
            ))}
          </tbody>
        </table>
      </section>
      <section className="panel">
        <h2>Log</h2>
        <pre className="code" ref={logRef}>
          {job.log.join("\n") || "(no output yet)"}
        </pre>
      </section>
    </>
  );
}
