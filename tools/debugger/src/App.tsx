import { useEffect, useMemo, useState } from "react";

import { listMemories, loadMemory } from "./lib/api";
import { runChecks } from "./lib/checks";
import type { MemoryBundle, MemorySummary } from "./lib/types";
import { useSplat } from "./lib/ui";
import { AudioView } from "./views/AudioView";
import { ChecksView } from "./views/ChecksView";
import { FilesView } from "./views/FilesView";
import { LocalizationView } from "./views/LocalizationView";
import { SceneView } from "./views/SceneView";
import { SplatView } from "./views/SplatView";

const STAGE_DOTS = ["ingest", "poses", "splat", "separate", "localize", "soundscape"];
const TABS = ["checks", "audio", "localization", "splat", "scene", "files"] as const;
type Tab = (typeof TABS)[number];
const TAB_LABEL: Record<Tab, string> = { checks: "Checks", audio: "Audio", localization: "Localization", splat: "Splat", scene: "3D scene", files: "Files" };

function readHash(): { memory?: string; tab: Tab } {
  const [memory, tab] = decodeURIComponent(location.hash.slice(1)).split("/");
  return { memory: memory || undefined, tab: (TABS as readonly string[]).includes(tab) ? (tab as Tab) : "checks" };
}

export function App() {
  const [memories, setMemories] = useState<MemorySummary[] | null>(null);
  const [listError, setListError] = useState<string>();
  const [{ memory, tab }, setRoute] = useState(readHash);
  const [bundle, setBundle] = useState<MemoryBundle | null>(null);
  const [loadError, setLoadError] = useState<string>();

  useEffect(() => {
    listMemories().then(setMemories).catch((e) => setListError(String(e)));
    const onHash = () => setRoute(readHash());
    window.addEventListener("hashchange", onHash);
    return () => window.removeEventListener("hashchange", onHash);
  }, []);

  const go = (m: string | undefined, t: Tab) => {
    location.hash = `${m ?? ""}/${t}`;
  };

  const selected = memory ?? memories?.[0]?.id;
  useEffect(() => {
    if (!selected) return;
    let live = true;
    setBundle(null);
    setLoadError(undefined);
    loadMemory(selected)
      .then((b) => live && setBundle(b))
      .catch((e) => live && setLoadError(String(e)));
    return () => {
      live = false;
    };
  }, [selected]);

  const { splat, error: splatError, loading: splatLoading } = useSplat(bundle);
  const checks = useMemo(() => (bundle ? runChecks(bundle, splat) : []), [bundle, splat]);
  const counts = { fail: checks.filter((c) => c.level === "fail").length, warn: checks.filter((c) => c.level === "warn").length };

  return (
    <div className="app">
      <aside className="sidebar">
        <div className="brand">
          <h1>MemoryScapes debugger</h1>
          <p>Stage outputs on disk</p>
        </div>
        {listError && <p className="empty">{listError}</p>}
        {memories?.length === 0 && (
          <p className="empty">
            No memories found. Run <code>examples/run_example.py</code> or set <code>MEMORYSCAPES_DATA</code>.
          </p>
        )}
        {memories?.map((m) => (
          <button key={m.id} className="mem" aria-current={m.id === selected} onClick={() => go(m.id, tab)}>
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
        <div className="mainhead">
          <h2>{selected ?? "No memory selected"}</h2>
          <p>{bundle?.truth?.description ?? (bundle ? `${bundle.files.length} files` : loadError ?? "Loading…")}</p>
          <div className="tabs" role="tablist">
            {TABS.map((t) => (
              <button key={t} role="tab" className="tab" aria-selected={t === tab} onClick={() => go(selected, t)}>
                {TAB_LABEL[t]}
                {t === "checks" && counts.fail > 0 && <span className="badge fail">{counts.fail}</span>}
                {t === "checks" && counts.warn > 0 && <span className="badge warn">{counts.warn}</span>}
              </button>
            ))}
          </div>
        </div>
        <div className="content">
          {!bundle && !loadError && <p className="empty">Loading…</p>}
          {loadError && <p className="empty">{loadError}</p>}
          {bundle && tab === "checks" && <ChecksView checks={checks} splatLoading={splatLoading} splatError={splatError} />}
          {bundle && tab === "audio" && <AudioView bundle={bundle} />}
          {bundle && tab === "localization" && <LocalizationView bundle={bundle} />}
          {bundle && tab === "splat" && <SplatView bundle={bundle} splat={splat} splatError={splatError} />}
          {bundle && tab === "scene" && <SceneView bundle={bundle} splat={splat} />}
          {bundle && tab === "files" && <FilesView bundle={bundle} />}
        </div>
      </main>
    </div>
  );
}
