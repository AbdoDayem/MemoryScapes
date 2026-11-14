"use client";

/**
 * Walk through one memory: the splat (three.js + Spark) and its soundscape.
 *
 * - Loads the manifest from the API; while the pipeline runs, shows stage progress.
 * - Splat LODs load coarsest first, each replacing the previous one when ready.
 * - Walk with WASD/arrows (Shift: faster, Q/E: down/up), drag to look. Movement stays
 *   inside navigation.walkable_bounds: the box and corridor around the capture path,
 *   where the splat is reliable.
 * - "Replay capture" follows poses/path.json, synced to the audio when it's playing.
 * - Audio starts from a click (browser autoplay rules): placed sources are spatialized
 *   with HRTF panners, the ambient bed (or the original recording) plays as is.
 *
 * Splat, audio positions and camera share one metric frame (+y up), so nothing is
 * transformed here.
 */

import type { MemoryManifest, Vec3 } from "@memoryscapes/manifest";
import { useEffect, useRef, useState } from "react";

import { SpatialAudioEngine } from "@/components/audio/SpatialAudioEngine";
import { getMemory, type MemoryResponse } from "@/lib/api";

import { asWalkable, clampToWalkable, samplePath, type CapturePath } from "./navigation";

type Phase = "loading" | "processing" | "ready" | "error";
type AudioState = "none" | "loading" | "ready" | "playing" | "error";

const WALK_SPEED = 1.4; // m/s
const RUN_SPEED = 3.5;
const LOOK_SPEED = 0.004; // rad per pixel

export default function MemoryViewer({ memoryId }: { memoryId: string }) {
  const container = useRef<HTMLDivElement>(null);
  const [phase, setPhase] = useState<Phase>("loading");
  const [memory, setMemory] = useState<MemoryResponse | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [lodStatus, setLodStatus] = useState("");
  const [audioState, setAudioState] = useState<AudioState>("none");
  const [replay, setReplay] = useState(false);
  const replayRef = useRef(false);
  const audioRef = useRef<SpatialAudioEngine | null>(null);

  replayRef.current = replay;

  // ---- manifest (poll while processing)
  useEffect(() => {
    let cancelled = false;
    let timer: ReturnType<typeof setTimeout> | undefined;
    const poll = async () => {
      try {
        const m = await getMemory(memoryId);
        if (cancelled) return;
        setMemory(m);
        if (m.status === "ready" && m.manifest) setPhase("ready");
        else {
          setPhase("processing");
          timer = setTimeout(poll, 5000);
        }
      } catch (err) {
        if (!cancelled) {
          setError(String(err));
          setPhase("error");
        }
      }
    };
    poll();
    return () => {
      cancelled = true;
      if (timer) clearTimeout(timer);
    };
  }, [memoryId]);

  // ---- scene
  const manifest = phase === "ready" ? memory?.manifest : undefined;
  const assetBase = memory?.asset_base;
  useEffect(() => {
    if (!manifest || !assetBase || !container.current) return;
    let disposed = false;
    let cleanup = () => {};
    mountScene(container.current, manifest, assetBase, {
      isDisposed: () => disposed,
      setLodStatus,
      replay: () => replayRef.current,
      audio: () => audioRef.current,
    })
      .then((c) => {
        if (disposed) c();
        else cleanup = c;
      })
      .catch((err) => {
        setError(`Could not load the splat: ${err}`);
        setPhase("error");
      });

    // Decode the audio up front; it starts on the user's click.
    const resolve = (u: string) => new URL(u, assetBase).toString();
    const engine = new SpatialAudioEngine(resolve);
    audioRef.current = engine;
    const hasAudio = manifest.audio.sources.length > 0 || !!manifest.audio.ambient_bed;
    if (hasAudio) {
      setAudioState("loading");
      engine
        .load(manifest)
        .then(() => !disposed && setAudioState("ready"))
        .catch((err) => {
          console.error(err);
          if (!disposed) setAudioState("error");
        });
    }
    return () => {
      disposed = true;
      cleanup();
      audioRef.current = null;
      void engine.dispose();
    };
  }, [manifest, assetBase]);

  const startAudio = async () => {
    const engine = audioRef.current;
    if (!engine || audioState !== "ready") return;
    await engine.start();
    setAudioState("playing");
  };

  return (
    <div style={{ position: "fixed", inset: 0, background: "#101014", color: "#eee", font: "14px system-ui" }}>
      <div ref={container} style={{ position: "absolute", inset: 0, touchAction: "none" }} />
      <div style={{ position: "absolute", top: 16, left: 16, right: 16, display: "flex", gap: 12, alignItems: "flex-start", pointerEvents: "none" }}>
        <div style={{ background: "rgba(0,0,0,.55)", padding: "10px 14px", borderRadius: 8, maxWidth: 420, pointerEvents: "auto" }}>
          <div style={{ fontWeight: 600 }}>{manifest?.meta.title ?? memoryId}</div>
          {phase === "loading" && <div>Loading…</div>}
          {phase === "processing" && <StageList stages={memory?.stages} />}
          {phase === "error" && <div style={{ color: "#f88" }}>{error}</div>}
          {phase === "ready" && (
            <>
              <div style={{ opacity: 0.8 }}>{lodStatus}</div>
              <div style={{ opacity: 0.6, marginTop: 4 }}>WASD to walk · drag to look · Shift run · Q/E down/up</div>
              <div style={{ display: "flex", gap: 8, marginTop: 8 }}>
                <AudioButton state={audioState} onStart={startAudio} />
                <button onClick={() => setReplay((r) => !r)}>{replay ? "Walk freely" : "Replay capture"}</button>
              </div>
              {manifest?.audio.soundscape_provider?.fallback_reason && (
                <div style={{ opacity: 0.6, marginTop: 6 }}>{manifest.audio.soundscape_provider.fallback_reason}</div>
              )}
            </>
          )}
        </div>
      </div>
    </div>
  );
}

function StageList({ stages }: { stages?: Record<string, boolean> }) {
  if (!stages) return <div>Processing…</div>;
  return (
    <div>
      Processing:
      <ul style={{ margin: "4px 0 0", paddingLeft: 18 }}>
        {Object.entries(stages).map(([name, done]) => (
          <li key={name} style={{ opacity: done ? 1 : 0.5 }}>
            {done ? "✓" : "…"} {name}
          </li>
        ))}
      </ul>
    </div>
  );
}

function AudioButton({ state, onStart }: { state: AudioState; onStart: () => void }) {
  const label = { none: "No audio", loading: "Loading sound…", ready: "Start sound", playing: "Sound on", error: "Sound failed" }[state];
  return (
    <button onClick={onStart} disabled={state !== "ready"}>
      {label}
    </button>
  );
}

interface SceneHooks {
  isDisposed: () => boolean;
  setLodStatus: (s: string) => void;
  replay: () => boolean;
  audio: () => SpatialAudioEngine | null;
}

/** Builds the three.js scene; returns a cleanup function. */
async function mountScene(el: HTMLElement, manifest: MemoryManifest, assetBase: string, hooks: SceneHooks): Promise<() => void> {
  const THREE = await import("three");
  const { SparkRenderer, SplatMesh } = await import("@sparkjsdev/spark");
  const resolve = (u: string) => new URL(u, assetBase).toString();

  const renderer = new THREE.WebGLRenderer({ antialias: false });
  renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
  renderer.setSize(el.clientWidth, el.clientHeight);
  el.appendChild(renderer.domElement);
  const scene = new THREE.Scene();
  scene.background = new THREE.Color(0x101014);
  scene.add(new SparkRenderer({ renderer }));
  const camera = new THREE.PerspectiveCamera(60, el.clientWidth / el.clientHeight, 0.05, 20000);

  // ---- spawn
  const nav = manifest.navigation ?? {};
  const walkable = asWalkable(nav.walkable_bounds);
  const minY = nav.floor_y !== undefined ? nav.floor_y + 0.3 : undefined;
  const spawn = nav.spawn as { position?: Vec3; look_at?: Vec3 } | undefined;
  camera.position.fromArray(spawn?.position ?? [0, 1.6, 0]);
  camera.lookAt(new THREE.Vector3().fromArray(spawn?.look_at ?? [0, 1.6, -1]));
  const euler = new THREE.Euler().setFromQuaternion(camera.quaternion, "YXZ");

  // ---- capture path for replay
  let path: CapturePath | null = null;
  if (nav.original_path) {
    fetch(resolve(nav.original_path))
      .then((r) => (r.ok ? r.json() : null))
      .then((p) => (path = p))
      .catch(() => (path = null));
  }

  // ---- splat LODs: coarsest first, each replaces the previous when loaded
  const lods = [...manifest.splat.lods].reverse();
  let current: InstanceType<typeof SplatMesh> | null = null;
  (async () => {
    for (const [i, lod] of lods.entries()) {
      hooks.setLodStatus(`Loading splat ${i + 1}/${lods.length}${lod.count ? ` (${lod.count.toLocaleString()} gaussians)` : ""}…`);
      const mesh = new SplatMesh({ url: resolve(lod.url) });
      await mesh.initialized;
      if (hooks.isDisposed()) {
        mesh.dispose();
        return;
      }
      scene.add(mesh);
      if (current) {
        scene.remove(current);
        current.dispose();
      }
      current = mesh;
    }
    const top = lods[lods.length - 1];
    hooks.setLodStatus(`${top?.count ? top.count.toLocaleString() + " gaussians" : "Splat loaded"}`);
  })().catch((err) => hooks.setLodStatus(`Splat failed to load: ${err}`));

  // ---- input
  const keys = new Set<string>();
  const onKey = (e: KeyboardEvent) => {
    if ((e.target as HTMLElement)?.tagName === "INPUT") return;
    if (e.type === "keydown") keys.add(e.code);
    else keys.delete(e.code);
  };
  let dragging = false;
  const onDown = (e: PointerEvent) => {
    dragging = true;
    el.setPointerCapture(e.pointerId);
  };
  const onUp = () => (dragging = false);
  const onMove = (e: PointerEvent) => {
    if (!dragging || hooks.replay()) return;
    euler.y -= e.movementX * LOOK_SPEED;
    euler.x = Math.max(-1.45, Math.min(1.45, euler.x - e.movementY * LOOK_SPEED));
    camera.quaternion.setFromEuler(euler);
  };
  window.addEventListener("keydown", onKey);
  window.addEventListener("keyup", onKey);
  el.addEventListener("pointerdown", onDown);
  el.addEventListener("pointerup", onUp);
  el.addEventListener("pointermove", onMove);
  const onResize = () => {
    camera.aspect = el.clientWidth / el.clientHeight;
    camera.updateProjectionMatrix();
    renderer.setSize(el.clientWidth, el.clientHeight);
  };
  const resizer = new ResizeObserver(onResize);
  resizer.observe(el);

  // ---- frame loop
  const clock = new THREE.Clock();
  const forward = new THREE.Vector3();
  const right = new THREE.Vector3();
  const up = new THREE.Vector3(0, 1, 0);
  let replayStart = 0;
  let wasReplay = false;
  renderer.setAnimationLoop(() => {
    const dt = Math.min(clock.getDelta(), 0.1);
    const audio = hooks.audio();
    if (hooks.replay() && path) {
      if (!wasReplay) replayStart = clock.elapsedTime;
      const t = audio?.playbackTime ?? clock.elapsedTime - replayStart + (path.samples[0]?.t ?? 0);
      const pose = samplePath(path, t);
      if (pose) {
        camera.position.fromArray(pose.position);
        camera.quaternion.set(...pose.quaternion);
        euler.setFromQuaternion(camera.quaternion, "YXZ");
      }
    } else {
      const speed = keys.has("ShiftLeft") || keys.has("ShiftRight") ? RUN_SPEED : WALK_SPEED;
      camera.getWorldDirection(forward);
      forward.y = 0;
      forward.normalize();
      right.crossVectors(forward, up);
      const move = new THREE.Vector3();
      if (keys.has("KeyW") || keys.has("ArrowUp")) move.add(forward);
      if (keys.has("KeyS") || keys.has("ArrowDown")) move.sub(forward);
      if (keys.has("KeyD") || keys.has("ArrowRight")) move.add(right);
      if (keys.has("KeyA") || keys.has("ArrowLeft")) move.sub(right);
      if (keys.has("KeyE")) move.y += 1;
      if (keys.has("KeyQ")) move.y -= 1;
      if (move.lengthSq() > 0) {
        move.normalize().multiplyScalar(speed * dt);
        const p = camera.position.clone().add(move);
        camera.position.fromArray(clampToWalkable([p.x, p.y, p.z], walkable, minY));
      }
    }
    wasReplay = hooks.replay() && !!path;
    if (audio) {
      camera.getWorldDirection(forward);
      const camUp = up.clone().applyQuaternion(camera.quaternion);
      audio.updateListener(camera.position.toArray() as Vec3, forward.toArray() as Vec3, camUp.toArray() as Vec3);
    }
    renderer.render(scene, camera);
  });

  return () => {
    renderer.setAnimationLoop(null);
    resizer.disconnect();
    window.removeEventListener("keydown", onKey);
    window.removeEventListener("keyup", onKey);
    el.removeEventListener("pointerdown", onDown);
    el.removeEventListener("pointerup", onUp);
    el.removeEventListener("pointermove", onMove);
    if (current) current.dispose();
    renderer.dispose();
    renderer.domElement.remove();
  };
}
