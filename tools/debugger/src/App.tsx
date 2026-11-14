import { useCallback, useEffect, useMemo, useState } from "react";

import { listMemories, loadMemory } from "./lib/api";
import { runChecks } from "./lib/checks";
import type { MemoryBundle, MemorySummary } from "./lib/types";
import { useSplat } from "./lib/ui";
import { AudioView } from "./views/AudioView";
import { ChecksView } from "./views/ChecksView";
import { FilesView } from "./views/FilesView";
import { JobView } from "./views/JobView";
import { LocalizationView } from "./views/LocalizationView";
import { SceneView } from "./views/SceneView";
import { SplatView } from "./views/SplatView";
import { UploadView } from "./views/UploadView";

const STAGE_DOTS = ["ingest", "poses", "splat", "separate", "localize", "soundscape"];
const TABS = ["job", "checks", "audio", "localization", "splat", "scene", "files"] as const;
type Tab = (typeof TABS)[number] | "upload";
const TAB_LABEL: Record<(typeof TABS)[number], string> = { job: "Processing", checks: "Checks", audio: "Audio", localization: "Localization", splat: "Splat", scene: "3D scene", files: "Files" };

function readHash(): { memory?: string; tab: Tab } {
  const raw = decodeURIComponent(location.hash.slice(1));
  if (raw === "upload") return { tab: "upload" };
  const [memory, tab] = raw.split("/");
  return { memory: memory || undefined, tab: (TABS as readonly string[]).includes(tab) ? (tab as Tab) : "checks" };
}

export function App() {
  const [memories, setMemories] = useState<MemorySummary[] | null>(null);
  const [listError, setListError] = useState<string>();
  const [{ memory, tab }, setRoute] = useState(readHash);
  const [bundle, setBundle] = useState<MemoryBundle | null>(null);
  const [loadError, setLoadError] = useState<string>();
  const [version, setVersion] = useState(0); // bump to reload after a pipeline run

  useEffect(() => {
    listMemories().then(setMemories).catch((e) => setListError(String(e)));
  }, [version]);
  useEffect(() => {
    const onHash = () => setRoute(readHash());
    window.addEventListener("hashchange", onHash);
    return () => window.removeEventListener("hashchange", onHash);
  }, []);

  const go = (m: string | undefined, t: Tab) => {
    location.hash = t === "upload" ? "upload" : `${m ?? ""}/${t}`;
  };

  const selected = tab === "upload" ? undefined : memory ?? memories?.[0]?.id;
  useEffect(() => {
    if (!selected) return;
    let live = true;
    setLoadError(undefined);
    loadMemory(selected)
      .then((b) => live && setBundle(b))
      .catch((e) => live && setLoadError(String(e)));
    return () => {
      live = false;
    };
  }, [selected, version]);
  useEffect(() => {
    if (bundle && bundle.id !== selected) setBundle(null);
  }, [selected, bundle]);

  const reload = useCallback(() => setVersion((v) => v + 1), []);
  const { splat, error: splatError, loading: splatLoading } = useSplat(bundle);
  const checks = useMemo(() => (bundle ? runChecks(bundle, splat) : []), [bundle, splat]);
  const counts = { fail: checks.filter((c) => c.level === "fail").length, warn: checks.filter((c) => c.level === "warn").length };
  const hasJob = !!bundle?.files.some((f) => f.key === "job/status.json" || f.key === "upload/meta.json");
  const visibleTabs = TABS.filter((t) => t !== "job" || hasJob);

  return (
    <div className="app">
      <aside className="sidebar">
        <div className="brand">
          <h1>MemoryScapes debugger</h1>
          <p>Stage outputs on disk</p>
        </div>
        <div style={{ padding: "10px 16px", borderBottom: "1px solid var(--line)" }}>
          <button className="btn primary" style={{ width: "100%" }} onClick={() => go(undefined, "upload")}>
            Upload a video
          </button>
        </div>
        {listError && <p className="empty">{listError}</p>}
        {memories?.length === 0 && (
          <p className="empty">
            No memories yet. Upload a video, or run <code>examples/run_example.py</code>.
          </p>
        )}
        {memories?.map((m) => (
          <button key={m.id} className="mem" aria-current={m.id === selected} onClick={() => go(m.id, tab === "upload" ? "checks" : tab)}>
            <div className="name">{m.id}</div>
            <div className="dots" title={STAGE_DOTS.map((s) => `${s}: ${m.stages[s] ?? 0} files`).join("\n")}>
              {STAGE_DOTS.map((s) => (
                <span key={s} className={`dot ${m.stages[s] ? "on" : ""}`} />
              ))}
            </div>
          </button>
        ))}
      </aside>

      <main className="main">
        {tab === "upload" ? (
          <>
            <div className="mainhead">
              <h2>New memory</h2>
              <p>Upload a video to run the pipeline on it.</p>
            </div>
            <div className="content">
              <UploadView
                onStarted={(id) => {
                  reload();
                  go(id, "job");
                }}
              />
            </div>
          </>
        ) : (
          <>
            <div className="mainhead">
              <h2>{selected ?? "No memory selected"}</h2>
              <p>{bundle?.truth?.description ?? (bundle ? `${bundle.files.length} files` : loadError ?? "Loading…")}</p>
              <div className="tabs" role="tablist">
                {visibleTabs.map((t) => (
                  <button key={t} role="tab" className="tab" aria-selected={t === tab} onClick={() => go(selected, t)}>
                    {TAB_LABEL[t]}
                    {t === "checks" && counts.fail > 0 && <span className="badge fail">{counts.fail}</span>}
                    {t === "checks" && counts.warn > 0 && <span className="badge warn">{counts.warn}</span>}
                  </button>
                ))}
              </div>
            </div>
            <div className="content">
              {selected && tab === "job" && <JobView memoryId={selected} onFinished={reload} />}
              {tab !== "job" && !bundle && !loadError && <p className="empty">Loading…</p>}
              {loadError && <p className="empty">{loadError}</p>}
              {bundle && tab === "checks" && <ChecksView checks={checks} splatLoading={splatLoading} splatError={splatError} />}
              {bundle && tab === "audio" && <AudioView bundle={bundle} />}
              {bundle && tab === "localization" && <LocalizationView bundle={bundle} />}
              {bundle && tab === "splat" && <SplatView bundle={bundle} splat={splat} splatError={splatError} />}
              {bundle && tab === "scene" && <SceneView bundle={bundle} splat={splat} />}
              {bundle && tab === "files" && <FilesView bundle={bundle} />}
            </div>
          </>
        )}
      </main>
    </div>
  );
}
