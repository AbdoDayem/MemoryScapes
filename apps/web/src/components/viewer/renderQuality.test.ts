import assert from "node:assert/strict";
import { test } from "node:test";

import {
  deviceClass,
  parseOverrides,
  pickLods,
  QualityGovernor,
  streamsTree,
  TIERS,
  trimmedMean,
  type GovernorOptions,
} from "./renderQuality.ts";

// The splat stage's default LODs, most detailed first (as in the manifest).
const LODS = [
  { url: "lod0.spz", count: 1_500_000 },
  { url: "lod1.spz", count: 500_000 },
  { url: "lod2.spz", count: 150_000 },
];

test("desktop loads every LOD, coarsest first, ending on lod0", () => {
  assert.deepEqual(pickLods(LODS, "desktop").map((l) => l.url), ["lod2.spz", "lod1.spz", "lod0.spz"]);
});

test("a phone never downloads the 1.5M-gaussian lod0", () => {
  assert.deepEqual(pickLods(LODS, "mobile").map((l) => l.url), ["lod2.spz", "lod1.spz"]);
});

test("a phone with only big LODs still gets the coarsest", () => {
  const big = [{ url: "a", count: 3_000_000 }, { url: "b", count: 2_000_000 }];
  assert.deepEqual(pickLods(big, "mobile").map((l) => l.url), ["b"]);
});

test("?lod forces the top LOD; out-of-range values are ignored", () => {
  assert.deepEqual(pickLods(LODS, "mobile", 0).map((l) => l.url), ["lod2.spz", "lod1.spz", "lod0.spz"]);
  assert.deepEqual(pickLods(LODS, "desktop", 2).map((l) => l.url), ["lod2.spz"]);
  assert.equal(pickLods(LODS, "desktop", 7).length, 3);
  assert.deepEqual(pickLods([], "desktop"), []);
});

test("LODs without a count are treated as fitting", () => {
  const unknown: { url: string; count?: number }[] = [{ url: "x" }, { url: "y" }];
  assert.equal(pickLods(unknown, "mobile").length, 2);
});

test("device class from the user agent", () => {
  const iphone = "Mozilla/5.0 (iPhone; CPU iPhone OS 18_0 like Mac OS X) AppleWebKit/605.1.15 Mobile/15E148";
  const android = "Mozilla/5.0 (Linux; Android 14; Pixel 8) AppleWebKit/537.36 Chrome/130 Mobile Safari/537.36";
  const mac = "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 Chrome/130 Safari/537.36";
  const windows = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 Chrome/130 Safari/537.36";
  assert.equal(deviceClass({ userAgent: iphone }), "mobile");
  assert.equal(deviceClass({ userAgent: android }), "mobile");
  assert.equal(deviceClass({ userAgent: mac, maxTouchPoints: 0 }), "desktop");
  assert.equal(deviceClass({ userAgent: mac, maxTouchPoints: 5 }), "mobile"); // iPadOS
  assert.equal(deviceClass({ userAgent: windows }), "desktop");
});

test("URL overrides", () => {
  assert.deepEqual(parseOverrides(""), { perf: false });
  assert.deepEqual(parseOverrides("?quality=low&perf&lod=1"), { perf: true, tier: 2, lod: 1 });
  assert.deepEqual(parseOverrides("?quality=ultra&lod=-1&stream=yes"), { perf: false });
  assert.deepEqual(parseOverrides("?stream=0"), { perf: false, stream: false });
});

test("desktops stream the LoD tree, phones load the smaller .spz levels", () => {
  const none = parseOverrides("");
  assert.equal(streamsTree("desktop", true, none), true);
  assert.equal(streamsTree("mobile", true, none), false);
  assert.equal(streamsTree("desktop", false, none), false); // memories published before trees
  assert.equal(streamsTree("mobile", true, parseOverrides("?stream=1")), true);
  assert.equal(streamsTree("desktop", true, parseOverrides("?stream=0")), false);
  assert.equal(streamsTree("desktop", true, parseOverrides("?lod=0&stream=1")), false); // ?lod means .spz
});

test("tiers get cheaper on every lever", () => {
  for (let i = 1; i < TIERS.length; i++) {
    const [a, b] = [TIERS[i - 1], TIERS[i]];
    assert.ok(b.maxPixelRatio < a.maxPixelRatio, b.name);
    assert.ok(b.lodSplatScale < a.lodSplatScale, b.name);
    assert.ok(b.lodRenderScale >= a.lodRenderScale, b.name);
    assert.ok(b.maxSh <= a.maxSh && b.maxStdDev <= a.maxStdDev, b.name);
  }
});

test("trimmed mean ignores the slowest 10%", () => {
  assert.equal(trimmedMean([10, 10, 10, 10, 10, 10, 10, 10, 10, 500]), 10);
  assert.equal(trimmedMean([7]), 7);
});

/** Feeds `seconds` of frames of `ms` each; returns [time, tier] for every change. */
function run(g: QualityGovernor, clock: { t: number }, ms: number, seconds: number, act = true): [number, number][] {
  const changes: [number, number][] = [];
  const end = clock.t + seconds * 1000;
  while (clock.t < end) {
    clock.t += ms;
    const r = g.frame(clock.t, ms, act);
    if (r !== null) changes.push([clock.t, r]);
  }
  return changes;
}

const FAST: Partial<GovernorOptions> = {}; // the defaults

test("slow frames step quality down, one tier per settle period", () => {
  const g = new QualityGovernor(0, FAST);
  const clock = { t: 0 };
  run(g, clock, 1000 / 60, 2, false); // loading: a 60 Hz display
  const changes = run(g, clock, 40, 3); // 25 fps once the splat is in
  assert.deepEqual(changes.map(([, tier]) => tier), [1]);
  assert.equal(run(g, clock, 40, 3).length, 1); // still slow: down again after settling
  assert.equal(g.tier, 2);
});

test("measured from the start even when the splat is slow right away", () => {
  // No loading frames at all: assume 60 Hz rather than learning 25 fps as the display.
  const g = new QualityGovernor(0);
  run(g, { t: 0 }, 40, 3);
  assert.ok(g.tier > 0);
});

test("a browser capped at 30 fps isn't mistaken for a slow GPU", () => {
  const g = new QualityGovernor(0);
  const clock = { t: 0 };
  run(g, clock, 1000 / 30, 2, false); // loading frames already at 30 fps: the cap
  assert.deepEqual(run(g, clock, 1000 / 30, 10), []);
  assert.equal(run(g, clock, 1000 / 20, 3).length > 0, true); // 20 fps is still slow
});

test("one hitch doesn't change the tier", () => {
  const g = new QualityGovernor(0);
  const clock = { t: 0 };
  run(g, clock, 1000 / 60, 2, false);
  for (let i = 0; i < 600; i++) {
    clock.t += i === 300 ? 200 : 1000 / 60;
    assert.equal(g.frame(clock.t, i === 300 ? 200 : 1000 / 60), null);
  }
});

test("hidden-tab gaps are ignored", () => {
  const g = new QualityGovernor(0);
  const clock = { t: 0 };
  run(g, clock, 1000 / 60, 2, false);
  for (let i = 0; i < 20; i++) {
    clock.t += 1000;
    assert.equal(g.frame(clock.t, 1000), null);
  }
  assert.equal(g.tier, 0);
});

test("keeping up with the display steps back up; a tier that keeps failing isn't retried", () => {
  const g = new QualityGovernor(1, { upHoldMs: 2000 });
  const clock = { t: 0 };
  run(g, clock, 1000 / 60, 2, false);
  assert.deepEqual(run(g, clock, 1000 / 60, 4).map(([, tier]) => tier), [0]); // up to high
  run(g, clock, 40, 2); // too slow at high: back to medium (high failed once)
  assert.equal(g.tier, 1);
  assert.equal(run(g, clock, 1000 / 60, 5).length, 0); // retry waits 4x longer
  run(g, clock, 1000 / 60, 5);
  assert.equal(g.tier, 0); // retried
  run(g, clock, 40, 2); // failed twice
  assert.equal(g.tier, 1);
  assert.equal(run(g, clock, 1000 / 60, 120).length, 0); // never again
});

test("never past the lowest tier or above the highest", () => {
  const g = new QualityGovernor(TIERS.length - 1);
  const clock = { t: 0 };
  assert.equal(run(g, clock, 100, 10).length, 0);
  assert.equal(new QualityGovernor(99).tier, TIERS.length - 1);
  assert.equal(new QualityGovernor(-3).tier, 0);
});
