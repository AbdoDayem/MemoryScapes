/**
 * The memory marker: a small low-poly pin that blows low-poly soap bubbles
 * (the "Pin + bubbles" style chosen in the mockups). Browser-only (three.js).
 *
 *   ready       coral pin, gently bobbing, 6 bubbles
 *   processing  blue pin, hopping and spinning, 3 slower bubbles
 *   failed      grey pin, drooping, no bubbles
 *   cluster     bigger golden pin, 9 bubbles (plus a count badge, drawn by the scene)
 */
import * as THREE from "three";

import type { MemoryStatus } from "@/lib/api";

import { CLUSTER_COLOR, STATUS_COLORS } from "./format";

export interface MarkerData {
  id: string;
  lat: number;
  lon: number;
  kind: "single" | "cluster";
  status: MemoryStatus;
  count: number;
  title: string | null;
  captured_at: string | null;
  /** For a cluster: its memories. */
  members?: { id: string; lat: number; lon: number; title: string | null; status: MemoryStatus }[];
}

export interface Marker {
  object: THREE.Group;
  data: MarkerData;
  /** Call every frame. `zoomK` keeps the marker the same size on screen. */
  update(t: number, zoomK: number, hovered: boolean): void;
  /** A jelly squash, e.g. on click. */
  squish(t: number): void;
}
// No dispose(): three-globe frees a marker's geometries and materials when it's removed.

// Proportions (in units of the globe's radius = 100), tuned in the mockups.
const PIN_SCALE = 0.45;     // the pin is small: the bubbles are the star
const STICK = 2.4;          // short stick: the head sits near the ground
const HEAD_Y = STICK + 1.0;
const BUBBLE_SOLID = 0.25;  // 0 = glassy, 1 = nearly opaque

const isBusy = (s: MemoryStatus) => s === "processing" || s === "uploading" || s === "draft";

let glowTexture: THREE.Texture | null = null;
function glow(): THREE.Texture {
  if (glowTexture) return glowTexture;
  const c = document.createElement("canvas");
  c.width = c.height = 128;
  const g = c.getContext("2d")!;
  const r = g.createRadialGradient(64, 64, 0, 64, 64, 64);
  r.addColorStop(0, "rgba(255,255,255,1)");
  r.addColorStop(0.35, "rgba(255,255,255,.55)");
  r.addColorStop(1, "rgba(255,255,255,0)");
  g.fillStyle = r;
  g.fillRect(0, 0, 128, 128);
  return (glowTexture = new THREE.CanvasTexture(c));
}

const flat = (color: THREE.ColorRepresentation) => new THREE.MeshLambertMaterial({ color, flatShading: true });
const HIT = new THREE.MeshBasicMaterial({ transparent: true, opacity: 0, depthWrite: false, colorWrite: false });

// Low-poly bubbles: one flat pastel colour per face (like the land and clouds) and a blocky
// white "shine" facet instead of a realistic highlight.
const BUBBLE_GEO = new THREE.IcosahedronGeometry(1, 1);
const BUBBLE_VERT = /* glsl */ `varying vec3 vPos;
  void main() { vec4 mv = modelViewMatrix * vec4(position, 1.0); vPos = mv.xyz; gl_Position = projectionMatrix * mv; }`;
const BUBBLE_FRAG = /* glsl */ `uniform vec3 tint; uniform float opacity; uniform float time; uniform float solid; varying vec3 vPos;
  void main() {
    vec3 n = normalize(cross(dFdx(vPos), dFdy(vPos)));           // the face's own normal: flat shading
    float f = 1.0 - abs(dot(n, normalize(-vPos)));              // faces at the rim are more coloured
    vec3 rainbow = 0.5 + 0.5 * cos(6.2831 * (f * 0.9 + time * 0.08 + vec3(0.0, 0.33, 0.67)));
    vec3 col = mix(mix(rainbow, vec3(1.0), 0.45), tint, 0.25);
    float shine = step(0.9, dot(n, normalize(vec3(-0.45, 0.6, 0.65))));
    col = mix(col, vec3(1.0), shine * 0.85);
    float a = mix(0.2 + f * 0.55, 0.88 + f * 0.12, solid) + shine * 0.5;
    gl_FragColor = vec4(col, min(a, 1.0) * opacity);
  }`;

export function makeMarker(d: MarkerData): Marker {
  const cluster = d.kind === "cluster";
  const big = cluster ? 1.35 + Math.min(d.count, 6) * 0.08 : 1;
  const color = cluster ? CLUSTER_COLOR : STATUS_COLORS[d.status];
  const failed = !cluster && d.status === "failed";
  const busy = !cluster && isBusy(d.status);
  const phase = Math.random() * Math.PI * 2;

  const object = new THREE.Group();

  // ---- the small pin
  const pin = new THREE.Group();
  pin.scale.setScalar(PIN_SCALE);
  object.add(pin);
  const shadow = new THREE.Mesh(new THREE.CircleGeometry(1.3 * big, 14),
    new THREE.MeshBasicMaterial({ color: 0x2f3354, transparent: true, opacity: 0.15, depthWrite: false }));
  shadow.rotation.x = -Math.PI / 2;
  shadow.position.y = 0.05;
  pin.add(shadow);
  const bob = new THREE.Group();
  const stick = new THREE.Mesh(new THREE.CylinderGeometry(0.3 * big, 0.16 * big, STICK * big, 6), flat("#ffffff"));
  stick.position.y = (STICK / 2) * big;
  const head = new THREE.Mesh(new THREE.IcosahedronGeometry(2.1 * big, 0), flat(color));
  head.position.y = HEAD_Y * big;
  const outline = new THREE.Mesh(new THREE.IcosahedronGeometry(2.55 * big, 0),
    new THREE.MeshBasicMaterial({ color: 0xffffff, side: THREE.BackSide }));
  outline.position.copy(head.position);
  const shine = new THREE.Mesh(new THREE.IcosahedronGeometry(0.5 * big, 0), flat("#ffffff"));
  shine.position.set(-0.75 * big, (HEAD_Y + 0.75) * big, 1.25 * big);
  const halo = new THREE.Sprite(new THREE.SpriteMaterial({ map: glow(), color, transparent: true, depthWrite: false }));
  halo.scale.setScalar(9 * big);
  halo.position.copy(head.position);
  bob.add(halo, stick, outline, head, shine);
  pin.add(bob);
  if (failed) bob.rotation.z = 0.55; // droopy

  // ---- the bubbles, floating up from just above the pin's head
  const startY = (HEAD_Y + 2.7) * PIN_SCALE * big;
  const count = failed ? 0 : busy ? 3 : cluster ? 9 : 6;
  const bubbles = Array.from({ length: count }, (_, i) => {
    const mat = new THREE.ShaderMaterial({
      uniforms: { tint: { value: new THREE.Color(color) }, opacity: { value: 0 }, time: { value: 0 }, solid: { value: BUBBLE_SOLID } },
      vertexShader: BUBBLE_VERT, fragmentShader: BUBBLE_FRAG, transparent: true, depthWrite: false,
    });
    const mesh = new THREE.Mesh(BUBBLE_GEO, mat);
    object.add(mesh);
    return { mesh, mat, i, seed: Math.random() };
  });

  // ---- a generous invisible hit area, so the moving marker is easy to hover and click
  const hit = new THREE.Mesh(new THREE.CylinderGeometry(3.2 * big, 3.2 * big, 12 * big, 8), HIT);
  hit.position.y = 6 * big;
  object.add(hit);

  let scale = 1;
  let squishAt = -1;

  return {
    object,
    data: d,
    squish(t) { squishAt = t; },
    update(t, zoomK, hovered) {
      // pin: bob (ready), hop and spin (processing), sulk (failed)
      halo.material.opacity = failed ? 0.35 : 0.6 + 0.3 * Math.sin(t * 2.4 + phase);
      const y = failed ? 0 : busy ? Math.abs(Math.sin(t * 3.2 + phase)) * 1.4 * big : Math.sin(t * 1.6 + phase) * 0.35 * big;
      bob.position.y = y;
      shadow.scale.setScalar(1 - y / (8 * big));
      head.rotation.y = busy ? t * 2 : Math.sin(t * 0.8 + phase) * 0.4;

      // bubbles: blown out, rising and drifting, wobbling, then popping
      const period = busy ? 6.5 : 4.2;
      const rise = (busy ? 13 : 22) * big;
      for (const b of bubbles) {
        const p = ((t + (b.i * period) / count + b.seed * period) % period) / period;
        b.mesh.position.set(Math.sin(t * 1.1 + b.i * 1.7) * 1.8 * p * big, startY + p * rise, Math.cos(t * 0.9 + b.i) * 1.3 * p * big);
        let s = (0.8 + b.seed * 0.9) * big * (0.5 + 0.7 * p);
        let a = 1;
        if (p < 0.07) a = p / 0.07;
        if (p > 0.9) { const q = (p - 0.9) / 0.1; s *= 1 + q * 0.5; a = 1 - q; }
        const w = 0.07 * Math.sin(t * 7 + b.i * 2);
        b.mesh.scale.set(s * (1 + w), s * (1 - w), s * (1 + w));
        b.mat.uniforms.opacity.value = a;
        b.mat.uniforms.time.value = t + b.seed * 10;
      }

      // hover grows it (springy); a click squashes it like jelly
      scale += ((hovered ? 1.25 : 1) - scale) * 0.18;
      let sx = 1, sy = 1;
      const since = t - squishAt;
      if (squishAt >= 0 && since < 0.6) {
        const k = Math.sin(since * 22) * Math.exp(-since * 7);
        sx = 1 + k * 0.35;
        sy = 1 - k * 0.35;
      }
      const k = scale * zoomK;
      object.scale.set(k * sx, k * sy, k * sx);
    },
  };
}
