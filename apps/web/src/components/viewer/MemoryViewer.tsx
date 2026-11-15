"use client";

/**
 * Walk through one memory: the splat (three.js + Spark) and its soundscape.
 *
 * - Loads the manifest from the API; while the pipeline runs, shows stage progress.
 * - Splat LODs load coarsest first, each replacing the previous one when ready, up to the
 *   most detailed one the device takes (phones stop below lod0). That one loads as a Spark
 *   LoD tree, and frame times pick the render quality (see renderQuality.ts).
 * - Walk with WASD/arrows (Shift: faster, Q/E: down/up), drag to look. Movement stays
 *   inside navigation.walkable_bounds: the box and corridor around the capture path,
 *   where the splat is reliable.
 * - "Replay capture" follows poses/path.json.
 * - Sound is always SATIE's: its runtime plays the memory's SATIE soundscape, with each
 *   localized source a host at its position and the camera as the listener. It starts
 *   from a click (browser autoplay rules). The "Sound by SATIE" badge shows, and SATIE's
 *   mixer in development. A memory without a SATIE soundscape plays no sound and says why.
 *
 * Splat, audio positions and camera share one metric frame (+y up), so nothing is
 * transformed here.
 */

import type { MemoryManifest, Vec3 } from "@memoryscapes/manifest";
import { useEffect, useRef, useState } from "react";

import { SatieAudioEngine } from "@/components/audio/SatieAudioEngine";
import { getMemory, type MemoryResponse } from "@/lib/api";

import { assetResolver } from "./assets";
import { createGpuTimer } from "./gpuTimer";
import { asWalkable, clampToWalkable, samplePath, type CapturePath } from "./navigation";
import { blurFor, deviceClass, parseOverrides, pickLods, pickTree, QualityGovernor, START_TIER, TIERS } from "./renderQuality";
import { failureSummary, pollAction, stageRows } from "./status";

type Phase = "loading" | "processing" | "ready" | "failed" | "error";
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
  const [perf, setPerf] = useState("");
  const [audioState, setAudioState] = useState<AudioState>("none");
  const [replay, setReplay] = useState(false);
  const replayRef = useRef(false);
  const [audioNote, setAudioNote] = useState<string | null>(null);
  const audioRef = useRef<SatieAudioEngine | null>(null);

  replayRef.current = replay;

  // ---- manifest (poll while processing)
  useEffect(() => {
    let cancelled = false;
    let timer: ReturnType<typeof setTimeout> | undefined;
    let failures = 0;
    let seenOnce = false;
    const poll = async () => {
      try {
        const m = await getMemory(memoryId);
        if (cancelled) return;
        failures = 0;
        setMemory(m);
        const action = pollAction(m);
        if (action === "show") setPhase("ready");
        else if (action === "stop") setPhase("failed"); // a failed run: polling would never end
        else {
          setPhase("processing");
          timer = setTimeout(poll, 5000);
        }
      } catch (err) {
        if (cancelled) return;
        // A blip (the API restarting) shouldn't end a long wait; give up after a few.
        if (++failures < 4 && seenOnce) {
          timer = setTimeout(poll, 5000 * failures);
          return;
        }
        setError(String(err));
        setPhase("error");
      } finally {
        seenOnce = true;
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
    let resolveThree: (t: ThreeHandles) => void = () => {};
    const three = new Promise<ThreeHandles>((r) => (resolveThree = r));
    mountScene(container.current, manifest, assetBase, {
      isDisposed: () => disposed,
      setLodStatus,
      setPerf,
      replay: () => replayRef.current,
      audio: () => audioRef.current,
      onReady: (t) => resolveThree(t),
    })
      .then((c) => {
        if (disposed) c();
        else cleanup = c;
      })
      .catch((err) => {
        setError(`Could not load the splat: ${err}`);
        setPhase("error");
      });

    // Load SATIE's soundscape up front; it starts on the user's click.
    const resolve = assetResolver(assetBase, window.location.href);
    const satieOut = manifest.audio.satie;
    const engine = new SatieAudioEngine(resolve);
    let badge: { destroy(): void; setPanelOpen(open: boolean): void } | null = null;
    let mixer: { destroy(): void; open(): void } | null = null;
    if (!satieOut) {
      const why = manifest.audio.soundscape_provider?.fallback_reason;
      setAudioNote(why ? `No SATIE soundscape: ${why}` : "No SATIE soundscape for this memory yet.");
    } else {
      setAudioState("loading");
      const dev = process.env.NODE_ENV !== "production";
      const loaded = engine.load(manifest);
      // The credit SATIE's licence asks for; its "loading sounds" line shows until `ready`.
      import("satie-runtime/badge")
        .then(({ badge: mount }) => {
          if (!disposed) badge = mount({ dev, ready: loaded, onOpen: () => mixer?.open() });
        })
        .catch((err) => console.warn("SATIE badge:", err));
      loaded
        .then(async () => {
          if (disposed) return;
          if (engine.warnings.length) console.warn("SATIE:", engine.warnings);
          audioRef.current = engine;
          setAudioState("ready");
          const runtime = engine.runtime;
          if (!dev || !runtime) return;
          // SATIE's mixer (development only): every voice as a row and a dot in the world.
          const { scene, camera, renderer, look, teleport } = await three;
          const { mountMixer } = await import("satie-runtime/three");
          if (disposed) return;
          mixer = mountMixer(runtime, {
            scene, camera, renderer, look, teleport,
            script: resolve(satieOut.script_url),
            collapsed: true,
            // These go through SATIE's Vite dev plugin, which a Next.js app doesn't have.
            capture: false,
            eventUrl: null,
            historyUrl: null,
            nowUrl: null,
            watchUrl: null,
            context: () => runtime.authoringContext(),
            onOpenChange: (open: boolean) => badge?.setPanelOpen(open),
          });
        })
        .catch((err) => {
          console.error("SATIE soundscape:", err);
          if (disposed) return;
          setAudioState("error");
          setAudioNote(`SATIE soundscape didn't load: ${err instanceof Error ? err.message : err}`);
        });
    }
    return () => {
      disposed = true;
      cleanup();
      audioRef.current = null;
      mixer?.destroy();
      badge?.destroy();
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
          {phase === "processing" && memory && <StageList memory={memory} />}
          {phase === "failed" && memory && (
            <>
              <div style={{ color: "#f88", marginTop: 4 }}>{failureSummary(memory)}</div>
              <StageList memory={memory} />
            </>
          )}
          {phase === "error" && <div style={{ color: "#f88" }}>{error}</div>}
          {phase === "ready" && (
            <>
              <div style={{ opacity: 0.8 }}>{lodStatus}</div>
              {perf && <div style={{ opacity: 0.8, fontVariantNumeric: "tabular-nums" }}>{perf}</div>}
              <div style={{ opacity: 0.6, marginTop: 4 }}>WASD to walk · drag to look · Shift run · Q/E down/up</div>
              <div style={{ display: "flex", gap: 8, marginTop: 8 }}>
                <AudioButton state={audioState} onStart={startAudio} />
                <button onClick={() => setReplay((r) => !r)}>{replay ? "Walk freely" : "Replay capture"}</button>
              </div>
              {audioNote && <div style={{ opacity: 0.6, marginTop: 6 }}>{audioNote}</div>}
            </>
          )}
        </div>
      </div>
    </div>
  );
}

const STAGE_ICON = { pending: "·", running: "…", done: "✓", failed: "✗", skipped: "–", blocked: "–" };

function StageList({ memory }: { memory: MemoryResponse }) {
  const rows = stageRows(memory);
  if (rows.length === 0) return <div>Processing…</div>;
  return (
    <div>
      {memory.status !== "failed" && "Processing:"}
      <ul style={{ margin: "4px 0 0", paddingLeft: 18, listStyle: "none" }}>
        {rows.map((r) => (
          <li key={r.name} style={{ opacity: r.status === "done" || r.status === "failed" ? 1 : 0.55 }}>
            {STAGE_ICON[r.status]} {r.name}
            {r.seconds !== undefined && <span style={{ opacity: 0.6 }}> · {r.seconds.toFixed(0)} s</span>}
          </li>
        ))}
      </ul>
      {memory.stage_detail?.detail && memory.status !== "failed" && (
        <div style={{ opacity: 0.6, marginTop: 4, fontSize: 12 }}>{memory.stage_detail.detail}</div>
      )}
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

/** What SATIE's mixer needs from the scene (see satie-runtime/three mountMixer). */
interface ThreeHandles {
  scene: import("three").Scene;
  camera: import("three").PerspectiveCamera;
  renderer: import("three").WebGLRenderer;
  look: { get(): { yaw: number; pitch: number }; set(yaw: number, pitch: number): void };
  teleport: (x: number, z: number, yaw: number) => void;
}

interface SceneHooks {
  isDisposed: () => boolean;
  setLodStatus: (s: string) => void;
  setPerf: (s: string) => void;
  replay: () => boolean;
  audio: () => SatieAudioEngine | null;
  onReady?: (three: ThreeHandles) => void;
}

/** Builds the three.js scene; returns a cleanup function. */
async function mountScene(el: HTMLElement, manifest: MemoryManifest, assetBase: string, hooks: SceneHooks): Promise<() => void> {
  const THREE = await import("three");
  const { SparkRenderer, SplatMesh } = await import("@sparkjsdev/spark");
  const resolve = assetResolver(assetBase, window.location.href);

  const overrides = parseOverrides(window.location.search);
  const device = overrides.device ?? deviceClass(navigator);

  const renderer = new THREE.WebGLRenderer({ antialias: false });
  renderer.setSize(el.clientWidth, el.clientHeight);
  el.appendChild(renderer.domElement);
  const scene = new THREE.Scene();
  scene.background = new THREE.Color(0x101014);
  // Blur as in training (classic or anti-aliased), or fine detail looks thinner than trained.
  const spark = new SparkRenderer({ renderer, ...blurFor(manifest.splat.antialiased) });
  scene.add(spark);
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

  // ---- render quality: a fixed ?quality tier, or the governor picks one from frame times
  let current: InstanceType<typeof SplatMesh> | null = null;
  let tier = overrides.tier ?? START_TIER[device];
  const governor = overrides.tier === undefined ? new QualityGovernor(tier) : null;
  let measuring = false; // the governor acts once the last LOD is in
  const applyTier = () => {
    const t = TIERS[tier];
    renderer.setPixelRatio(Math.min(window.devicePixelRatio, t.maxPixelRatio));
    spark.lodSplatScale = t.lodSplatScale;
    spark.lodRenderScale = t.lodRenderScale;
    spark.maxStdDev = t.maxStdDev;
    if (current && current.maxSh !== t.maxSh) {
      current.maxSh = t.maxSh;
      current.updateGenerator();
    }
  };
  applyTier();
  // ?perf: devtools handles for timing frames on a real device (frame rate may be vsync-capped)
  const debug = window as unknown as { memoryscapesViewer?: object };
  if (overrides.perf) {
    const setTier = (i: number) => {
      tier = Math.max(0, Math.min(TIERS.length - 1, i));
      applyTier();
    };
    debug.memoryscapesViewer = { renderer, scene, camera, spark, setTier };
  }

  // ---- the splat. Streamed from a LoD tree (.rad, built by the publish stage) when the
  // most detailed level this device takes has one (lod0 on desktops, lod1 on phones): the
  // first chunk is a coarse view of everything, and Spark fetches the detail the view
  // needs. Otherwise the .spz LODs, coarsest first, each replacing the previous when
  // loaded; the last becomes a LoD tree in the browser. Either way Spark draws a splat budget.
  const picked = pickTree(manifest.splat, device, overrides);
  const streamTree = async ({ tree, level }: NonNullable<typeof picked>): Promise<boolean> => {
    const url = resolve(tree.url);
    hooks.setLodStatus("Streaming splat…");
    // A streamed mesh is "initialized" before fetching anything: check the header first.
    const head = await fetch(url, { headers: { Range: "bytes=0-0" } }).catch(() => null);
    if (!head?.ok) return false;
    const mesh = new SplatMesh({ url, paged: true });
    mesh.maxSh = TIERS[tier].maxSh;
    await mesh.initialized;
    if (hooks.isDisposed()) {
      mesh.dispose();
      return true;
    }
    scene.add(mesh);
    const drawing = () => (mesh.paged?.getNumSplats() ?? 0) > 0; // Spark has paged splats in
    await waitFor(() => drawing() || hooks.isDisposed(), 15_000);
    if (hooks.isDisposed()) {
      mesh.dispose();
      return true;
    }
    if (!drawing()) {
      scene.remove(mesh);
      mesh.dispose();
      return false;
    }
    current = mesh;
    measuring = true;
    const count = manifest.splat.lods[level]?.count; // the tree's own count includes interior nodes
    hooks.setLodStatus(`${count ? count.toLocaleString() + " gaussians" : "Splat"}, streamed`);
    return true;
  };

  const lods = pickLods(manifest.splat.lods, device, overrides.lod);
  (async () => {
    if (picked && (await streamTree(picked).catch(() => false))) return;
    if (picked) console.warn("Couldn't stream the splat's LoD tree; loading its .spz levels instead.");
    for (const [i, lod] of lods.entries()) {
      hooks.setLodStatus(`Loading splat ${i + 1}/${lods.length}${lod.count ? ` (${lod.count.toLocaleString()} gaussians)` : ""}…`);
      const isTree = i === lods.length - 1;
      const mesh = new SplatMesh({ url: resolve(lod.url), lod: isTree });
      mesh.maxSh = TIERS[tier].maxSh;
      await mesh.initialized;
      if (hooks.isDisposed()) {
        mesh.dispose();
        return;
      }
      scene.add(mesh);
      // A LoD tree draws nothing until Spark has picked its first splats: keep the previous
      // LOD on screen until then, or the scene flashes empty.
      if (isTree && current) await waitFor(() => spark.lodInstances.has(mesh) || hooks.isDisposed(), 10_000);
      if (hooks.isDisposed()) {
        mesh.dispose(); // the cleanup already disposed `current`
        return;
      }
      if (current) {
        scene.remove(current);
        current.dispose();
      }
      current = mesh;
    }
    measuring = true;
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
  let lastFrame = performance.now();
  let perfFrames = 0;
  let perfStart = lastFrame;
  let firstSplatAt: number | null = null; // ms since the page started loading
  let loadedAt: number | null = null;
  const gpu = overrides.perf ? createGpuTimer(renderer.getContext() as WebGL2RenderingContext) : null;
  renderer.setAnimationLoop((now: number) => {
    const dt = Math.min(clock.getDelta(), 0.1);
    const next = governor?.frame(now, now - lastFrame, measuring);
    lastFrame = now;
    if (next != null) {
      tier = next;
      applyTier();
    }
    if (overrides.perf) {
      perfFrames++;
      if (firstSplatAt === null && spark.activeSplats > 0) firstSplatAt = now;
      if (loadedAt === null && measuring) loadedAt = now;
    }
    if (overrides.perf && now - perfStart >= 1000) {
      const fps = (perfFrames * 1000) / (now - perfStart);
      const gpuMs = gpu?.take() ?? NaN;
      const secs = (t: number | null) => (t === null ? "…" : `${(t / 1000).toFixed(1)} s`);
      hooks.setPerf(
        `${fps.toFixed(0)} fps${Number.isFinite(gpuMs) ? ` (GPU ${gpuMs.toFixed(1)} ms)` : ""} · ` +
          `${TIERS[tier].name}${governor ? " (auto)" : ""} · ` +
          `${spark.activeSplats.toLocaleString()} splats drawn · ${renderer.getPixelRatio()}x pixels · ` +
          `first splat ${secs(firstSplatAt)}, loaded ${secs(loadedAt)} after page start`,
      );
      perfFrames = 0;
      perfStart = now;
    }
    const audio = hooks.audio();
    if (hooks.replay() && path) {
      if (!wasReplay) replayStart = clock.elapsedTime;
      const t = clock.elapsedTime - replayStart + (path.samples[0]?.t ?? 0);
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
      camera.updateMatrixWorld(); // SATIE reads the listener from the camera's world matrix
      audio.update(camera);
    }
    gpu?.begin();
    renderer.render(scene, camera);
    gpu?.end();
  });

  hooks.onReady?.({
    scene,
    camera,
    renderer,
    look: {
      get: () => ({ yaw: euler.y, pitch: euler.x }),
      set: (yaw, pitch) => {
        euler.set(pitch, yaw, 0, "YXZ");
        camera.quaternion.setFromEuler(euler);
      },
    },
    teleport: (x, z, yaw) => {
      camera.position.fromArray(clampToWalkable([x, camera.position.y, z], walkable, minY));
      euler.set(euler.x, yaw, 0, "YXZ");
      camera.quaternion.setFromEuler(euler);
    },
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
    if (overrides.perf) delete debug.memoryscapesViewer;
    renderer.dispose();
    renderer.domElement.remove();
  };
}

/** Resolves once `cond()` holds, or after `timeoutMs`. */
async function waitFor(cond: () => boolean, timeoutMs: number): Promise<void> {
  const end = performance.now() + timeoutMs;
  while (!cond() && performance.now() < end) await new Promise((r) => setTimeout(r, 50));
}
