// Run with: npm test --workspace @memoryscapes/web (node --test; Node >= 23.6 runs TS directly)
import assert from "node:assert/strict";
import { test } from "node:test";

import { asWalkable, clampToWalkable, samplePath, type CapturePath, type Walkable } from "./navigation.ts";

const close = (a: number[], b: number[], eps = 1e-6) =>
  assert.ok(a.every((v, i) => Math.abs(v - b[i]) < eps), `${a} != ${b}`);

const box: Walkable = { min: [-5, 0, -5], max: [5, 3, 5] };

test("inside the box and corridor nothing moves", () => {
  const w = { ...box, corridor: { radius_m: 1.5, points: [[0, 1.7, 0], [0, 1.7, -2]] as [number, number, number][] } };
  close(clampToWalkable([0.5, 1.7, -1], w), [0.5, 1.7, -1]);
});

test("box clamps each axis", () => {
  close(clampToWalkable([9, -1, 2], box), [5, 0, 2]);
});

test("corridor pulls back to radius around the nearest path point", () => {
  const w = { ...box, corridor: { radius_m: 1, points: [[0, 1, 0], [0, 1, -4]] as [number, number, number][] } };
  close(clampToWalkable([3, 1, -4], w), [1, 1, -4]);
});

test("floor height wins over the box", () => {
  close(clampToWalkable([0, 0.1, 0], box, 0.3), [0, 0.3, 0]);
});

test("asWalkable ignores malformed bounds", () => {
  assert.equal(asWalkable(undefined), undefined);
  assert.equal(asWalkable({ type: "aabb" }), undefined);
  assert.ok(asWalkable(box));
});

const path: CapturePath = {
  samples: [
    { t: 0, position: [0, 1, 0], quaternion_xyzw: [0, 0, 0, 1] },
    { t: 2, position: [2, 1, -4], quaternion_xyzw: [0, Math.SQRT1_2, 0, Math.SQRT1_2] }, // 90° yaw
  ],
};

test("path interpolates position and rotation", () => {
  const p = samplePath(path, 1)!;
  close(p.position, [1, 1, -2]);
  close(p.quaternion, [0, Math.sin(Math.PI / 8), 0, Math.cos(Math.PI / 8)]); // 45° yaw
});

test("path wraps around for looping replay", () => {
  close(samplePath(path, 3)!.position, [1, 1, -2]);
  close(samplePath(path, 0)!.position, [0, 1, 0]);
});

test("empty and single-sample paths", () => {
  assert.equal(samplePath({ samples: [] }, 1), null);
  close(samplePath({ samples: [path.samples[0]] }, 5)!.position, [0, 1, 0]);
});
