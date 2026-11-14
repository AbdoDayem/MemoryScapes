/**
 * The homepage globe: globe.gl (three.js) with the low-poly world, memory markers, clusters,
 * name tags, ripples and the "place your memory" pin. React talks to it only through the
 * returned GlobeScene. Browser-only: import it dynamically (it touches `window`).
 */
import Globe, { type GlobeInstance } from "globe.gl";
import * as THREE from "three";

import type { GlobeMemory } from "@/lib/api";

import { altitudeToSeparate, clusterPoints, mergeDistance } from "./clusters";
import { CLUSTER_COLOR, STATUS_COLORS, formatDate, statusLabel } from "./format";
import { makeMarker, type Marker, type MarkerData } from "./markers";
import {
  type Country, landAltitude, landCapMaterial, landSideMaterial, makeClouds, makeLights, makeOcean,
} from "./world";

export interface GlobeCallbacks {
  /** A memory's marker was clicked. */
  onSelect(id: string): void;
  /** The globe (land or sea) was clicked, away from any marker. */
  onGlobeClick(lat: number, lon: number): void;
  /** A cluster whose memories are at (almost) the same spot: let the user pick one. */
  onClusterPick(ids: string[]): void;
}

export interface GlobeScene {
  setMemories(memories: GlobeMemory[]): void;
  /** The pin shown while placing a new memory (null hides it). */
  setPin(pin: { lat: number; lon: number } | null): void;
  flyTo(lat: number, lon: number, altitude?: number): void;
  stopAutoRotate(): void;
  dispose(): void;
}

const UP = new THREE.Vector3(0, 1, 0);
const MARKER_ALT = 0.022;  // just above the tallest land tiles
const TAG_ALT = 0.05;      // name tags float just above the small pins
const BADGE_ALT = 0.08;
const TAGS_BELOW = 1.3;      // show name tags when the camera is lower than this
const MIN_ALTITUDE = 0.02;   // how close the camera may get (globe radii)

type HtmlItem =
  | { kind: "badge"; lat: number; lon: number; count: number }
  | { kind: "tag"; lat: number; lon: number; title: string; color: string }
  | { kind: "pin"; lat: number; lon: number };
type RingItem = { lat: number; lon: number; color: string; splash: boolean };

export function createGlobeScene(el: HTMLElement, cb: GlobeCallbacks): GlobeScene {
  let memories: GlobeMemory[] = [];
  let shown: MarkerData[] = [];
  let lastSignature = "";
  let altitude = 2.5;
  let tagsShown = false;
  let markers = new Map<MarkerData, Marker>();
  let hovered: MarkerData | null = null;
  let pin: { lat: number; lon: number } | null = null;
  let splashes: RingItem[] = [];
  let frame = 0;
  let disposed = false;

  const globe: GlobeInstance = new Globe(el, { animateIn: true })
    .width(el.clientWidth)
    .height(el.clientHeight)
    .backgroundColor("rgba(0,0,0,0)")
    // The built-in sphere stays (it catches clicks) but is invisible; the low-poly ocean replaces it.
    .globeMaterial(new THREE.MeshBasicMaterial({ transparent: true, opacity: 0, depthWrite: false }))
    .showAtmosphere(true)
    .atmosphereColor("#ffffff")
    .atmosphereAltitude(0.13)
    // land
    .polygonCapCurvatureResolution(6)
    .polygonAltitude((c) => landAltitude(c as Country))
    .polygonCapMaterial((c) => landCapMaterial(c as Country))
    .polygonSideMaterial((c) => landSideMaterial(c as Country))
    .polygonStrokeColor(() => false)
    .polygonsTransitionDuration(0)
    .onPolygonClick((_c, _e, at) => globeClicked(at.lat, at.lng))
    .onGlobeClick((at) => globeClicked(at.lat, at.lng))
    // memories
    .customLayerData([])
    .customThreeObject((d) => markerFor(d as MarkerData).object)
    .customThreeObjectUpdate((obj, d) => {
      const m = d as MarkerData;
      const c = globe.getCoords(m.lat, m.lon, MARKER_ALT);
      obj.position.set(c.x, c.y, c.z);
      obj.quaternion.setFromUnitVectors(UP, obj.position.clone().normalize());
    })
    .customLayerLabel((d) => label(d as MarkerData))
    .onCustomLayerHover((d) => {
      hovered = (d as MarkerData | null) ?? null;
      el.style.cursor = hovered ? "pointer" : "";
    })
    .onCustomLayerClick((d) => {
      const m = d as MarkerData;
      markers.get(m)?.squish(performance.now() / 1000);
      if (m.kind === "cluster") openCluster(m);
      else cb.onSelect(m.id);
    })
    // ripples under memories, and a splash wherever you tap
    .ringLat("lat").ringLng("lon")
    .ringColor((d: object) => {
      const r = d as RingItem;
      return (t: number) => rgba(r.color, (r.splash ? 0.9 : 0.8) * (1 - t));
    })
    .ringMaxRadius((d) => ((d as RingItem).splash ? 4 : 3.2))
    .ringPropagationSpeed((d) => ((d as RingItem).splash ? 3 : 0.7))
    .ringRepeatPeriod((d) => ((d as RingItem).splash ? 0 : 2200))
    // cluster badges, name tags, the placing pin
    .htmlLat("lat").htmlLng("lon")
    .htmlAltitude((d) => ({ badge: BADGE_ALT, tag: TAG_ALT, pin: 0.02 })[(d as HtmlItem).kind])
    .htmlElement((d) => htmlFor(d as HtmlItem))
    .onZoom(() => refresh());

  globe.lights(makeLights());
  globe.scene().add(makeOcean(globe.getGlobeRadius()));
  const clouds = makeClouds((lat, lon, alt) => globe.getCoords(lat, lon, alt));
  globe.scene().add(clouds);
  globe.pointOfView({ lat: 25, lng: -40, altitude: 2.5 }, 0);

  const controls = globe.controls() as unknown as {
    autoRotate: boolean; autoRotateSpeed: number;
    addEventListener(type: string, fn: () => void): void;
  };
  controls.autoRotate = true;
  controls.autoRotateSpeed = 0.5;
  (controls as unknown as { minDistance: number }).minDistance = globe.getGlobeRadius() * (1 + MIN_ALTITUDE);
  controls.addEventListener("start", () => (controls.autoRotate = false));

  fetch("/globe/countries-110m.geojson")
    .then((r) => r.json())
    .then((geo: { features: object[] }) => !disposed && globe.polygonsData(geo.features))
    .catch((err) => console.error("Could not load the country shapes", err));

  const onResize = () => globe.width(el.clientWidth).height(el.clientHeight);
  window.addEventListener("resize", onResize);

  // ---- animation
  const animate = () => {
    const t = performance.now() / 1000;
    clouds.rotation.y = t * 0.012;
    clouds.rotation.x = Math.sin(t * 0.05) * 0.04;
    // Markers keep about the same size on screen: bigger in the world when the camera is far.
    const zoomK = Math.min(2.4, Math.max(0.55, globe.pointOfView().altitude / 1.1));
    for (const [data, marker] of markers) marker.update(t, zoomK, hovered === data);
    frame = requestAnimationFrame(animate);
  };
  frame = requestAnimationFrame(animate);

  // ---- helpers
  function markerFor(d: MarkerData): Marker {
    let m = markers.get(d);
    if (!m) markers.set(d, (m = makeMarker(d)));
    return m;
  }

  function refresh() {
    altitude = globe.pointOfView().altitude;
    const next = clusterPoints(memories, mergeDistance(altitude)).map((s): MarkerData => s.kind === "single"
      ? { id: s.id, lat: s.lat, lon: s.lon, kind: "single", status: s.memory.status, count: 1,
          title: s.memory.title, captured_at: s.memory.captured_at }
      : { id: s.id, lat: s.lat, lon: s.lon, kind: "cluster", status: "ready", count: s.members.length,
          title: null, captured_at: null,
          members: s.members.map((m) => ({ id: m.id, lat: m.lat, lon: m.lon, title: m.title, status: m.status })) });
    // Polling hands us fresh objects every few seconds; only rebuild when something visible
    // changed, so bubbles and ripples don't restart. (three-globe frees removed markers.)
    const tags = altitude < TAGS_BELOW;
    const signature = next.map((m) => [m.id, m.kind, m.status, m.count, m.title, m.lat, m.lon].join(",")).join(";");
    if (signature === lastSignature) {
      if (tags !== tagsShown) updateHtml(); // only the name tags change; markers keep their bubbles
      return;
    }
    lastSignature = signature;
    shown = next;
    markers = new Map();
    globe.customLayerData(shown); // makes the new markers
    updateRings();
    updateHtml();
  }

  function updateRings() {
    const under = shown
      .filter((m) => m.kind === "cluster" || m.status !== "failed")
      .map((m): RingItem => ({ lat: m.lat, lon: m.lon, color: m.kind === "cluster" ? CLUSTER_COLOR : STATUS_COLORS[m.status], splash: false }));
    globe.ringsData([...under, ...splashes]);
  }

  function updateHtml() {
    const items: HtmlItem[] = shown
      .filter((m) => m.kind === "cluster")
      .map((m) => ({ kind: "badge", lat: m.lat, lon: m.lon, count: m.count }));
    // Name tags once zoomed in far enough that the pins are spread out.
    tagsShown = altitude < TAGS_BELOW;
    if (tagsShown) {
      for (const m of shown) {
        if (m.kind === "single") {
          items.push({ kind: "tag", lat: m.lat, lon: m.lon, title: m.title || "Untitled memory", color: STATUS_COLORS[m.status] });
        }
      }
    }
    if (pin) items.push({ kind: "pin", ...pin });
    globe.htmlElementsData(items);
  }

  function globeClicked(lat: number, lon: number) {
    const s: RingItem = { lat, lon, color: "#ffffff", splash: true };
    splashes.push(s);
    updateRings();
    setTimeout(() => {
      splashes = splashes.filter((x) => x !== s);
      if (!disposed) updateRings();
    }, 1400);
    cb.onGlobeClick(lat, lon);
  }

  /** Zoom in just far enough for a cluster's memories to come apart, or offer a list. */
  function openCluster(m: MarkerData) {
    const members = m.members ?? [];
    const target = altitudeToSeparate(members, MIN_ALTITUDE);
    if (target === null) return cb.onClusterPick(members.map((x) => x.id));
    controls.autoRotate = false;
    globe.pointOfView({ lat: m.lat, lng: m.lon, altitude: Math.min(target, globe.pointOfView().altitude * 0.8) }, 1200);
  }

  return {
    setMemories(list) {
      memories = list;
      refresh();
    },
    setPin(p) {
      pin = p;
      updateHtml();
    },
    flyTo(lat, lon, altitude = 0.9) {
      controls.autoRotate = false;
      globe.pointOfView({ lat: lat - 4, lng: lon, altitude }, 1400);
    },
    stopAutoRotate() {
      controls.autoRotate = false;
    },
    dispose() {
      disposed = true;
      cancelAnimationFrame(frame);
      window.removeEventListener("resize", onResize);
      const renderer = globe.renderer();
      (globe as unknown as { _destructor?: () => void })._destructor?.();
      renderer.forceContextLoss(); // browsers allow only a few live WebGL contexts
      el.innerHTML = "";
    },
  };
}

// ---- DOM bits (text goes in with textContent: titles are user input)

function label(m: MarkerData): string {
  const box = document.createElement("div");
  box.className = "globe-tooltip";
  const b = document.createElement("b");
  if (m.kind === "cluster") {
    b.textContent = `${m.count} memories`;
    box.append(b, "poke to zoom in");
  } else {
    b.textContent = m.title || "Untitled memory";
    box.append(b, [formatDate(m.captured_at), statusLabel(m.status)].filter(Boolean).join(" · "));
  }
  return box.outerHTML;
}

function htmlFor(d: HtmlItem): HTMLElement {
  const el = document.createElement("div");
  const inner = document.createElement("div");
  if (d.kind === "badge") {
    inner.className = "globe-badge";
    inner.textContent = String(d.count);
  } else if (d.kind === "tag") {
    inner.className = "globe-tag";
    inner.style.setProperty("--c", d.color);
    inner.textContent = d.title;
  } else {
    inner.className = "globe-drop-pin";
    inner.textContent = "📍";
  }
  el.append(inner);
  return el;
}

function rgba(hex: string, a: number): string {
  const n = parseInt(hex.slice(1), 16);
  return `rgba(${n >> 16},${(n >> 8) & 255},${n & 255},${Math.max(0, a).toFixed(3)})`;
}
