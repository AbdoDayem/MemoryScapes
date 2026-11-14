import assert from "node:assert/strict";
import { test } from "node:test";

import { altitudeToSeparate, arcDegrees, centre, clusterPoints, mergeDistance } from "./clusters.ts";

const sf = { id: "sf", lat: 37.82, lon: -122.48 };
const stanford = { id: "stanford", lat: 37.43, lon: -122.17 }; // ~50 km from sf
const paris = { id: "paris", lat: 48.86, lon: 2.29 };
const sfAgain = { id: "sf2", lat: 37.8201, lon: -122.4801 };   // ~15 m from sf

const kinds = (shown: { kind: string }[]) => shown.map((s) => s.kind).sort();

test("arc distance", () => {
  assert.ok(Math.abs(arcDegrees(sf, stanford) - 0.45) < 0.05);
  assert.equal(arcDegrees(sf, sf), 0);
  assert.ok(Math.abs(arcDegrees({ lat: 0, lon: 179 }, { lat: 0, lon: -179 }) - 2) < 1e-9); // across ±180°
});

test("the merge distance shrinks as the camera gets closer", () => {
  assert.ok(mergeDistance(2.5) > mergeDistance(1) && mergeDistance(1) > mergeDistance(0.1));
});

test("zoomed out: nearby memories merge, far ones don't", () => {
  const shown = clusterPoints([sf, stanford, paris], mergeDistance(2.5));
  assert.deepEqual(kinds(shown), ["cluster", "single"]);
  const c = shown.find((s) => s.kind === "cluster")!;
  assert.ok(c.kind === "cluster");
  assert.deepEqual(c.members.map((m) => m.id), ["sf", "stanford"]);
  assert.equal(c.id, "cluster:sf+stanford");
  assert.ok(c.lat > 37.4 && c.lat < 37.9 && c.lon > -122.5 && c.lon < -122.1);
});

test("zoomed in over the Bay Area: San Francisco and Stanford come apart", () => {
  const shown = clusterPoints([sf, stanford, paris], mergeDistance(0.05));
  assert.deepEqual(kinds(shown), ["single", "single", "single"]);
});

test("the altitude a cluster needs to come apart actually separates it", () => {
  const alt = altitudeToSeparate([sf, stanford]);
  assert.ok(alt !== null && alt > 0.02);
  assert.deepEqual(kinds(clusterPoints([sf, stanford], mergeDistance(alt!))), ["single", "single"]);
});

test("memories at the same spot never come apart: offer a list instead", () => {
  assert.equal(altitudeToSeparate([sf, sfAgain]), null);
  assert.deepEqual(kinds(clusterPoints([sf, sfAgain], mergeDistance(0.02))), ["cluster"]);
});

test("cluster ids don't depend on order", () => {
  assert.equal(clusterPoints([sf, stanford], 5)[0].id, clusterPoints([stanford, sf], 5)[0].id);
});

test("the centre of points either side of ±180° stays near ±180°", () => {
  const c = centre([{ id: "a", lat: -17, lon: 179 }, { id: "b", lat: -17, lon: -179 }]);
  assert.ok(Math.abs(Math.abs(c.lon) - 180) < 0.01, `lon ${c.lon}`);
  assert.ok(Math.abs(c.lat + 17) < 0.01);
});

test("empty input", () => {
  assert.deepEqual(clusterPoints([], 5), []);
  assert.equal(altitudeToSeparate([]), null);
});
