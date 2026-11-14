/**
 * The low-poly world around the memories: a faceted ocean, pastel country tiles, drifting
 * clouds and soft lighting. Browser-only (three.js).
 */
import * as THREE from "three";

export const OCEAN = "#7cc6ec";
/** Country tile colours. No pinks or peaches: they'd blend with the coral memory pins. */
export const LAND = ["#b5e3b0", "#cdeaa8", "#a3d9b1", "#f3e4a8", "#d6cdee", "#c4e6cf", "#e3efb0"];

export interface Country {
  properties: { name?: string | null };
}

const hash = (s: string) => [...s].reduce((h, c) => (h * 31 + c.charCodeAt(0)) >>> 0, 7);
const nameOf = (c: Country) => c.properties.name ?? "";

/** Each country always gets the same colour and height. */
export const landColor = (c: Country) => LAND[hash(nameOf(c)) % LAND.length];
export const landAltitude = (c: Country) => 0.01 + (hash(nameOf(c)) % 4) * 0.0035;

const capMaterials = new Map<string, THREE.Material>();
const sideMaterials = new Map<string, THREE.Material>();
export function landCapMaterial(c: Country): THREE.Material {
  const color = landColor(c);
  let m = capMaterials.get(color);
  if (!m) capMaterials.set(color, (m = new THREE.MeshLambertMaterial({ color, flatShading: true })));
  return m;
}
export function landSideMaterial(c: Country): THREE.Material {
  const color = landColor(c);
  let m = sideMaterials.get(color);
  if (!m) {
    m = new THREE.MeshLambertMaterial({ color: new THREE.Color(color).multiplyScalar(0.78), flatShading: true });
    sideMaterials.set(color, m);
  }
  return m;
}

export function makeLights(): THREE.Light[] {
  const sun = new THREE.DirectionalLight(0xfff4e0, 2.2);
  sun.position.set(1, 1.2, 1.5);
  return [new THREE.AmbientLight(0xffffff, 1.6), sun];
}

/** A faceted sphere just under the land tiles, with a little wobble so facets catch the light. */
export function makeOcean(radius: number): THREE.Mesh {
  const geo = new THREE.IcosahedronGeometry(radius * 0.995, 5);
  const pos = geo.attributes.position;
  const v = new THREE.Vector3();
  for (let i = 0; i < pos.count; i++) {
    v.fromBufferAttribute(pos, i);
    v.multiplyScalar(1 - 0.004 * (0.5 + 0.5 * Math.sin(v.x * 0.21) * Math.cos(v.y * 0.17 + v.z * 0.13)));
    pos.setXYZ(i, v.x, v.y, v.z);
  }
  geo.computeVertexNormals();
  return new THREE.Mesh(geo, new THREE.MeshLambertMaterial({ color: OCEAN, flatShading: true }));
}

/** Little low-poly clouds around the planet. Rotate the returned group to make them drift. */
export function makeClouds(coords: (lat: number, lon: number, alt: number) => { x: number; y: number; z: number },
                           count = 14): THREE.Group {
  const clouds = new THREE.Group();
  const mat = new THREE.MeshLambertMaterial({ color: 0xffffff, flatShading: true, transparent: true, opacity: 0.95 });
  const rand = seeded(42); // the same sky every time
  for (let i = 0; i < count; i++) {
    const cloud = new THREE.Group();
    const puffs = 3 + (i % 3);
    for (let j = 0; j < puffs; j++) {
      const puff = new THREE.Mesh(new THREE.IcosahedronGeometry(2.2 + rand() * 1.8, 0), mat);
      puff.position.set((j - puffs / 2) * 2.6, rand() * 1.2, rand() * 1.5);
      puff.rotation.set(rand() * 3, rand() * 3, 0);
      cloud.add(puff);
    }
    const c = coords((rand() - 0.5) * 130, rand() * 360 - 180, 0.16 + rand() * 0.08);
    cloud.position.set(c.x, c.y, c.z);
    cloud.lookAt(0, 0, 0);
    clouds.add(cloud);
  }
  return clouds;
}

function seeded(seed: number): () => number {
  let s = seed;
  return () => ((s = (s * 16807) % 2147483647) - 1) / 2147483646;
}
