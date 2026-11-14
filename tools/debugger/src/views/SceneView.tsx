import { useEffect, useRef, useState } from "react";
import * as THREE from "three";
import { OrbitControls } from "three/addons/controls/OrbitControls.js";

import { audioContext, loadAudio, relKey } from "../lib/api";
import type { GaussianView } from "../lib/ply";
import type { MemoryBundle } from "../lib/types";
import { METHOD_VAR, cssVar, fmt } from "../lib/ui";

const LAYERS = ["poses points", "splat", "cameras", "sounds", "ground truth", "bounds"] as const;
type Layer = (typeof LAYERS)[number];

function labelSprite(text: string, color: string): THREE.Sprite {
  const c = document.createElement("canvas");
  const ctx = c.getContext("2d")!;
  ctx.font = "600 28px Inter, system-ui, sans-serif";
  c.width = Math.ceil(ctx.measureText(text).width) + 16;
  c.height = 40;
  ctx.font = "600 28px Inter, system-ui, sans-serif";
  ctx.fillStyle = "rgba(10,14,18,0.75)";
  ctx.fillRect(0, 0, c.width, c.height);
  ctx.fillStyle = color;
  ctx.fillText(text, 8, 29);
  const sprite = new THREE.Sprite(new THREE.SpriteMaterial({ map: new THREE.CanvasTexture(c), depthTest: false }));
  sprite.scale.set(c.width / 80, c.height / 80, 1);
  sprite.renderOrder = 10;
  return sprite;
}

function frustum(c2w: number[][], K: number[][], width: number, height: number, depth = 0.5): THREE.BufferGeometry {
  const m = new THREE.Matrix4().set(...(c2w.flat() as [number, number, number, number, number, number, number, number, number, number, number, number, number, number, number, number]));
  const corner = (u: number, v: number) => new THREE.Vector3(((u - K[0][2]) / K[0][0]) * depth, ((v - K[1][2]) / K[1][1]) * depth, depth).applyMatrix4(m);
  const o = new THREE.Vector3(0, 0, 0).applyMatrix4(m);
  const cs = [corner(0, 0), corner(width, 0), corner(width, height), corner(0, height)];
  const pts: THREE.Vector3[] = [];
  for (let i = 0; i < 4; i++) pts.push(o, cs[i], cs[i], cs[(i + 1) % 4]);
  return new THREE.BufferGeometry().setFromPoints(pts);
}

export function SceneView({ bundle, splat }: { bundle: MemoryBundle; splat?: GaussianView }) {
  const mount = useRef<HTMLDivElement>(null);
  const groups = useRef<Partial<Record<Layer, THREE.Object3D>>>({});
  const cameraRef = useRef<THREE.PerspectiveCamera>();
  const [visible, setVisible] = useState<Record<Layer, boolean>>({ "poses points": true, splat: true, cameras: true, sounds: true, "ground truth": true, bounds: false });
  const [listening, setListening] = useState(false);
  const [listenError, setListenError] = useState<string>();

  // ---------- build the scene
  useEffect(() => {
    const el = mount.current!;
    const renderer = new THREE.WebGLRenderer({ antialias: true });
    renderer.setPixelRatio(window.devicePixelRatio);
    renderer.setSize(el.clientWidth, el.clientHeight);
    el.appendChild(renderer.domElement);
    const scene = new THREE.Scene();
    scene.background = new THREE.Color("#0b1014");
    const camera = new THREE.PerspectiveCamera(55, el.clientWidth / el.clientHeight, 0.05, 500);
    cameraRef.current = camera;
    const controls = new OrbitControls(camera, renderer.domElement);
    controls.enableDamping = true;

    const g: Partial<Record<Layer, THREE.Object3D>> = {};
    const add = (layer: Layer, obj: THREE.Object3D) => {
      g[layer] ??= new THREE.Group();
      g[layer]!.add(obj);
    };

    if (bundle.points?.length) {
      const geo = new THREE.BufferGeometry();
      geo.setAttribute("position", new THREE.BufferAttribute(bundle.points, 3));
      if (bundle.pointColors) geo.setAttribute("color", new THREE.BufferAttribute(Float32Array.from(bundle.pointColors, (v) => v / 255), 3));
      add("poses points", new THREE.Points(geo, new THREE.PointsMaterial({ size: 0.06, vertexColors: !!bundle.pointColors, color: bundle.pointColors ? 0xffffff : 0x8aa0b0 })));
    }
    if (splat) {
      const keep: number[] = [];
      for (let i = 0; i < splat.count; i++) if (splat.opacity[i] > 0.1) keep.push(i);
      const pos = new Float32Array(keep.length * 3);
      const col = new Float32Array(keep.length * 3);
      keep.forEach((j, i) => {
        pos.set(splat.positions.subarray(j * 3, j * 3 + 3), i * 3);
        col.set(splat.colors.subarray(j * 3, j * 3 + 3), i * 3);
      });
      const geo = new THREE.BufferGeometry();
      geo.setAttribute("position", new THREE.BufferAttribute(pos, 3));
      geo.setAttribute("color", new THREE.BufferAttribute(col, 3));
      add("splat", new THREE.Points(geo, new THREE.PointsMaterial({ size: 0.05, vertexColors: true, transparent: true, opacity: 0.8 })));
    }
    const cams = bundle.cameras ?? [];
    if (cams.length) {
      const mat = new THREE.LineBasicMaterial({ color: 0xd0d8e0, transparent: true, opacity: 0.7 });
      cams.forEach((c, i) => i % 2 === 0 && add("cameras", new THREE.LineSegments(frustum(c.c2w, c.K, c.width, c.height), mat)));
      const path = new THREE.BufferGeometry().setFromPoints(cams.map((c) => new THREE.Vector3(c.c2w[0][3], c.c2w[1][3], c.c2w[2][3])));
      add("cameras", new THREE.Line(path, new THREE.LineBasicMaterial({ color: 0xffffff })));
    }
    for (const s of bundle.localized?.sources ?? []) {
      if (!s.position) continue;
      const color = new THREE.Color(cssVar(METHOD_VAR[s.method]));
      const p = new THREE.Vector3(...s.position);
      const ball = new THREE.Mesh(new THREE.SphereGeometry(0.18, 20, 14), new THREE.MeshBasicMaterial({ color }));
      ball.position.copy(p);
      add("sounds", ball);
      if (s.spread_m) {
        const shell = new THREE.Mesh(new THREE.SphereGeometry(Math.max(0.2, s.spread_m), 16, 10), new THREE.MeshBasicMaterial({ color, wireframe: true, transparent: true, opacity: 0.25 }));
        shell.position.copy(p);
        add("sounds", shell);
      }
      const label = labelSprite(`${s.label} · ${s.method} ${fmt(s.confidence)}`, `#${color.getHexString()}`);
      label.position.copy(p).add(new THREE.Vector3(0, 0.55, 0));
      add("sounds", label);
    }
    for (const t of bundle.truth?.sources ?? []) {
      const m = new THREE.Mesh(new THREE.OctahedronGeometry(0.22), new THREE.MeshBasicMaterial({ color: cssVar("--truth"), wireframe: true }));
      m.position.set(...t.position);
      add("ground truth", m);
    }
    if (bundle.localized) {
      const { min, max } = bundle.localized.bounds;
      const box = new THREE.Box3(new THREE.Vector3(...min), new THREE.Vector3(...max));
      add("bounds", new THREE.Box3Helper(box, new THREE.Color(0x52606d)));
    }
    add("bounds", new THREE.AxesHelper(1));
    for (const obj of Object.values(g)) scene.add(obj!);
    groups.current = g;
    scene.add(new THREE.GridHelper(40, 40, 0x2a3742, 0x1a242c));

    // frame the cameras and sounds
    const focus = new THREE.Box3();
    cams.forEach((c) => focus.expandByPoint(new THREE.Vector3(c.c2w[0][3], c.c2w[1][3], c.c2w[2][3])));
    bundle.localized?.sources.forEach((s) => s.position && focus.expandByPoint(new THREE.Vector3(...s.position)));
    if (focus.isEmpty()) focus.setFromCenterAndSize(new THREE.Vector3(), new THREE.Vector3(10, 10, 10));
    const centre = focus.getCenter(new THREE.Vector3());
    const size = focus.getSize(new THREE.Vector3()).length();
    camera.position.copy(centre).add(new THREE.Vector3(0, size * 0.55, -size * 0.75));
    controls.target.copy(centre);
    controls.update();

    let raf = 0;
    const loop = () => {
      raf = requestAnimationFrame(loop);
      controls.update();
      renderer.render(scene, camera);
    };
    loop();
    const ro = new ResizeObserver(() => {
      renderer.setSize(el.clientWidth, el.clientHeight);
      camera.aspect = el.clientWidth / el.clientHeight;
      camera.updateProjectionMatrix();
    });
    ro.observe(el);
    return () => {
      cancelAnimationFrame(raf);
      ro.disconnect();
      controls.dispose();
      renderer.dispose();
      el.removeChild(renderer.domElement);
    };
  }, [bundle, splat]);

  useEffect(() => {
    for (const layer of LAYERS) {
      const obj = groups.current[layer];
      if (obj) obj.visible = visible[layer];
    }
  }, [visible, bundle, splat]);

  // ---------- spatial listening from the orbit camera
  useEffect(() => {
    if (!listening) return;
    const ctx = audioContext();
    const nodes: AudioScheduledSourceNode[] = [];
    let raf = 0;
    let cancelled = false;
    (async () => {
      try {
        await ctx.resume();
        const sceneSources = bundle.scene?.sources ?? [];
        const refDist = new Map(sceneSources.map((s) => [s.id, s.ref_distance]));
        const out = ctx.createGain();
        out.gain.value = 0.9;
        out.connect(ctx.destination);
        const start = ctx.currentTime + 0.1;
        for (const s of bundle.localized?.sources ?? []) {
          const buf = await loadAudio(bundle.id, relKey(bundle.id, s.stem_key));
          if (cancelled) return;
          const node = ctx.createBufferSource();
          node.buffer = buf;
          node.loop = true;
          if (s.position) {
            const panner = new PannerNode(ctx, {
              panningModel: "HRTF",
              distanceModel: "inverse",
              refDistance: refDist.get(s.id) ?? 1,
              positionX: s.position[0],
              positionY: s.position[1],
              positionZ: s.position[2],
            });
            node.connect(panner).connect(out);
          } else node.connect(out);
          node.start(start);
          nodes.push(node);
        }
        if (bundle.stems) {
          const amb = ctx.createBufferSource();
          amb.buffer = await loadAudio(bundle.id, relKey(bundle.id, bundle.stems.ambient_key));
          amb.loop = true;
          amb.connect(out);
          amb.start(start);
          nodes.push(amb);
        }
        const fwd = new THREE.Vector3();
        const tick = () => {
          raf = requestAnimationFrame(tick);
          const cam = cameraRef.current;
          if (!cam) return;
          const l = ctx.listener;
          cam.getWorldDirection(fwd);
          l.positionX.value = cam.position.x;
          l.positionY.value = cam.position.y;
          l.positionZ.value = cam.position.z;
          l.forwardX.value = fwd.x;
          l.forwardY.value = fwd.y;
          l.forwardZ.value = fwd.z;
          l.upX.value = cam.up.x;
          l.upY.value = cam.up.y;
          l.upZ.value = cam.up.z;
        };
        tick();
      } catch (e) {
        setListenError(String(e));
        setListening(false);
      }
    })();
    return () => {
      cancelled = true;
      cancelAnimationFrame(raf);
      nodes.forEach((n) => {
        try {
          n.stop();
        } catch {
          /* not started yet */
        }
      });
    };
  }, [listening, bundle]);

  return (
    <section className="panel">
      <div className="toolbar">
        {LAYERS.map((l) => (
          <label key={l}>
            <input type="checkbox" checked={visible[l]} onChange={(e) => setVisible({ ...visible, [l]: e.target.checked })} />
            {l}
          </label>
        ))}
        <button className={`btn ${listening ? "primary" : ""}`} onClick={() => setListening(!listening)} disabled={!bundle.localized}>
          {listening ? "Stop listening" : "Listen from this viewpoint"}
        </button>
        {listenError && <span className="muted">{listenError}</span>}
      </div>
      <div className="viewer" ref={mount} />
      <p className="muted">
        Drag to orbit, right-drag to pan, scroll to zoom. Everything is drawn in the shared world frame (metres, +Y up), so a splat that doesn't overlap the pose points is in
        the wrong frame. "Listen" plays every stem looped through HRTF panners, with you as the listener at the orbit camera.
      </p>
    </section>
  );
}
