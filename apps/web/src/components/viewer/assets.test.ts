import assert from "node:assert/strict";
import { test } from "node:test";

import { assetResolver } from "./assets.ts";

test("absolute asset base (development)", () => {
  const resolve = assetResolver("http://localhost:8000/memories/mem_1/files/", "http://localhost:3000/memories/mem_1");
  assert.equal(resolve("splat/lods/lod0.spz"), "http://localhost:8000/memories/mem_1/files/splat/lods/lod0.spz");
});

test("root-relative asset base (deployed behind /api) resolves against the page", () => {
  const resolve = assetResolver("/api/memories/mem_1/files/", "https://memories.example.org/memories/mem_1");
  assert.equal(resolve("splat/lods/lod0.spz"), "https://memories.example.org/api/memories/mem_1/files/splat/lods/lod0.spz");
  assert.equal(resolve("soundscape/satie/scene.satie"),
    "https://memories.example.org/api/memories/mem_1/files/soundscape/satie/scene.satie");
});

test("already absolute asset URLs are kept", () => {
  const resolve = assetResolver("/api/memories/mem_1/files/", "http://192.168.1.20:3000/memories/mem_1");
  assert.equal(resolve("https://cdn.example/t.jpg"), "https://cdn.example/t.jpg");
});
