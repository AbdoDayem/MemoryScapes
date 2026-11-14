"use client";

/**
 * The homepage globe as a React component. All the three.js work lives in scene.ts; this
 * only mounts it in the browser and passes props through.
 */
import { useEffect, useRef } from "react";

import type { GlobeMemory } from "@/lib/api";

import type { GlobeScene } from "./scene";

export interface MemoryGlobeProps {
  memories: GlobeMemory[];
  /** The pin shown while placing a memory. */
  pin: { lat: number; lon: number } | null;
  /** Fly the camera here whenever `key` changes. */
  focus: { lat: number; lon: number; key: number } | null;
  onSelect(id: string): void;
  onGlobeClick(lat: number, lon: number): void;
  /** Memories at the same spot: show a list to pick from. */
  onClusterPick(ids: string[]): void;
}

export default function MemoryGlobe({ memories, pin, focus, onSelect, onGlobeClick, onClusterPick }: MemoryGlobeProps) {
  const el = useRef<HTMLDivElement>(null);
  const scene = useRef<GlobeScene | null>(null);
  // Latest callbacks and data, so the scene (made once) always uses current ones.
  const latest = useRef({ onSelect, onGlobeClick, onClusterPick, memories, pin, focus });
  latest.current = { onSelect, onGlobeClick, onClusterPick, memories, pin, focus };

  useEffect(() => {
    let cancelled = false;
    // globe.gl touches `window`, so load it only in the browser.
    import("./scene").then(({ createGlobeScene }) => {
      if (cancelled || !el.current) return;
      scene.current = createGlobeScene(el.current, {
        onSelect: (id) => latest.current.onSelect(id),
        onGlobeClick: (lat, lon) => latest.current.onGlobeClick(lat, lon),
        onClusterPick: (ids) => latest.current.onClusterPick(ids),
      });
      scene.current.setMemories(latest.current.memories);
      scene.current.setPin(latest.current.pin);
      const f = latest.current.focus;
      if (f) scene.current.flyTo(f.lat, f.lon);
    });
    return () => {
      cancelled = true;
      scene.current?.dispose();
      scene.current = null;
    };
  }, []);

  useEffect(() => scene.current?.setMemories(memories), [memories]);
  useEffect(() => scene.current?.setPin(pin), [pin]);
  useEffect(() => {
    if (focus) scene.current?.flyTo(focus.lat, focus.lon);
  }, [focus]);

  return <div ref={el} className="ms-globe" aria-label="Globe of memories" />;
}
