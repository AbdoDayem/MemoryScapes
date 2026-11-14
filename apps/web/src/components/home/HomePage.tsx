"use client";

/**
 * Home: the low-poly globe of every memory, with "+ New memory", "My memories" and the
 * preview card on top (docs/webapp-plan/frontend.md § Page `/`).
 */
import "@fontsource/nunito/500.css";
import "@fontsource/nunito/700.css";
import "@fontsource/nunito/800.css";
import "./home.css";

import { useCallback, useEffect, useMemo, useRef, useState } from "react";

import MemoryCard from "@/components/globe/MemoryCard";
import MemoryGlobe from "@/components/globe/MemoryGlobe";
import MemoryList from "@/components/globe/MemoryList";
import UploadPanel from "@/components/upload/UploadPanel";
import { listMemories, queryGlobe, updateMemory, type GlobeMemory, type MemorySummary } from "@/lib/api";

// Only real processing changes by itself; drafts wait for the user.
const BUSY = new Set(["processing"]);

export default function HomePage({ startWithUpload = false }: { startWithUpload?: boolean }) {
  const [onGlobe, setOnGlobe] = useState<GlobeMemory[]>([]);
  const [all, setAll] = useState<MemorySummary[]>([]);
  const [loadError, setLoadError] = useState<string | null>(null);
  const [selected, setSelected] = useState<string | null>(null);
  const [cardVersion, setCardVersion] = useState(0);
  const [focus, setFocus] = useState<{ lat: number; lon: number; key: number } | null>(null);
  const [listOpen, setListOpen] = useState(false);
  const [uploadOpen, setUploadOpen] = useState(startWithUpload);
  const [placing, setPlacing] = useState(false); // the upload panel wants taps on the globe
  const [uploadPin, setUploadPin] = useState<{ lat: number; lon: number } | null>(null);
  const [moving, setMoving] = useState(false); // the card's "Move pin"
  const [picking, setPicking] = useState<string[] | null>(null); // memories at the same spot

  const loadSeq = useRef(0);
  const load = useCallback(async () => {
    const seq = ++loadSeq.current;
    try {
      const [globe, list] = await Promise.all([queryGlobe(), listMemories()]);
      if (seq !== loadSeq.current) return; // a newer load already finished
      setOnGlobe(globe);
      setAll(list);
      setLoadError(null);
    } catch (e) {
      if (seq === loadSeq.current) setLoadError(e instanceof Error ? e.message : String(e));
    }
  }, []);

  useEffect(() => { load(); }, [load]);

  // While something is processing, refresh now and then so pins change colour by themselves.
  const anyBusy = all.some((m) => BUSY.has(m.status));
  const selectedBusy = !!selected && all.some((m) => m.id === selected && BUSY.has(m.status));
  useEffect(() => {
    if (!anyBusy) return;
    const timer = setInterval(() => {
      load();
      if (selectedBusy) setCardVersion((v) => v + 1);
    }, 8000);
    return () => clearInterval(timer);
  }, [anyBusy, selectedBusy, load]);

  const located = useMemo(() => new Set(onGlobe.map((m) => m.id)), [onGlobe]);

  const select = useCallback((id: string) => {
    setPicking(null);
    setSelected(id);
    setMoving(false);
    const m = onGlobe.find((x) => x.id === id);
    if (m) setFocus({ lat: m.lat, lon: m.lon, key: Date.now() });
  }, [onGlobe]);

  const onGlobeClick = useCallback(async (lat: number, lon: number) => {
    if (placing) return setUploadPin({ lat, lon });
    if (moving && selected) {
      setMoving(false);
      try {
        await updateMemory(selected, { lat, lon });
        await load();
        setCardVersion((v) => v + 1);
      } catch (e) {
        setLoadError(e instanceof Error ? e.message : String(e));
      }
      return;
    }
    setSelected(null);
    setPicking(null);
  }, [placing, moving, selected, load]);
  const onClusterPick = useCallback((ids: string[]) => { setSelected(null); setPicking(ids); }, []);

  const changed = useCallback(() => { load(); setCardVersion((v) => v + 1); }, [load]);
  const onPlacingChange = useCallback((p: boolean) => setPlacing(p), []);

  const openUpload = () => {
    setSelected(null);
    setListOpen(false);
    setUploadPin(null);
    setUploadOpen(true);
    if (window.location.pathname !== "/upload") window.history.replaceState(null, "", "/upload");
  };
  const closeUpload = () => {
    setUploadOpen(false);
    setSelected(null); // a marker clicked while the panel was open shouldn't pop up now
    setPlacing(false);
    setUploadPin(null);
    if (window.location.pathname === "/upload") window.history.replaceState(null, "", "/");
  };

  return (
    <div className="ms-home">
      <MemoryGlobe memories={onGlobe} pin={uploadOpen ? uploadPin : null} focus={focus}
                   onSelect={select} onGlobeClick={onGlobeClick} onClusterPick={onClusterPick} />

      <header className="ms-header">
        <div className="ms-brand">MemoryScapes <span>immerse in your memory</span></div>
        {!uploadOpen && <button className="ms-btn" onClick={openUpload}>+ New memory</button>}
      </header>

      {loadError && <div className="ms-banner ms-glass" role="alert">{loadError}</div>}
      {(placing || moving) && !loadError && (
        <div className="ms-placing ms-glass">📍 Tap the globe where it was filmed</div>
      )}

      <button className="ms-ghost ms-glass ms-list-toggle" onClick={() => setListOpen((o) => !o)} aria-expanded={listOpen}>
        ≡ My memories{all.length ? ` (${all.length})` : ""}
      </button>
      {listOpen && <MemoryList memories={all} located={located} onPick={(id) => { setListOpen(false); select(id); }} />}

      {picking && !uploadOpen && (
        <section className="ms-card ms-glass" aria-label="Memories here">
          <button className="ms-close" onClick={() => setPicking(null)} aria-label="Close">×</button>
          <h2>{picking.length} memories here</h2>
          <div className="ms-meta">They were filmed at almost the same spot. Pick one:</div>
          <MemoryList memories={all.filter((m) => picking.includes(m.id))} located={located} onPick={select} embedded />
        </section>
      )}

      {selected && !uploadOpen && (
        <MemoryCard key={selected} id={selected} version={cardVersion} moving={moving}
                    onClose={() => { setSelected(null); setMoving(false); }}
                    onMoveRequest={setMoving} onChanged={changed} />
      )}

      {uploadOpen && (
        <UploadPanel pin={uploadPin} onPlacingChange={onPlacingChange} onClose={closeUpload} onChanged={changed} />
      )}

      {!uploadOpen && !selected && (
        <div className="ms-help">
          {all.length === 0 && !loadError ? "No memories yet. Tap “+ New memory” to add your first one." : "Drag to spin · scroll to zoom · poke a memory"}
        </div>
      )}
    </div>
  );
}
