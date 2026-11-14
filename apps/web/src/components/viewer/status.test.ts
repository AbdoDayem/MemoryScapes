import assert from "node:assert/strict";
import { test } from "node:test";

import { failureSummary, pollAction, stageRows } from "./status.ts";

const base = {
  id: "mem_1",
  status: "processing" as const,
  stages: { ingest: true, poses: false },
  manifest: null,
  asset_base: "http://localhost:8000/memories/mem_1/files/",
};

test("keeps polling while processing, shows when ready, stops when failed", () => {
  assert.equal(pollAction(base), "poll");
  assert.equal(pollAction({ ...base, status: "ready" }), "poll"); // ready but no manifest yet
  assert.equal(pollAction({ ...base, status: "ready", manifest: {} as never }), "show");
  assert.equal(pollAction({ ...base, status: "failed" }), "stop");
  assert.equal(pollAction({ ...base, status: "uploading" }), "poll");
});

test("stage rows fall back to the done flags (CLI-made memories)", () => {
  assert.deepEqual(stageRows(base), [
    { name: "ingest", status: "done" },
    { name: "poses", status: "pending" },
  ]);
});

test("stage rows prefer the run's detail, in pipeline order", () => {
  const m = {
    ...base,
    status: "failed" as const,
    stage_detail: {
      stages: {
        poses: { status: "failed" as const, error: "Could not recover the camera path" },
        ingest: { status: "done" as const, seconds: 12.3 },
        splat: { status: "blocked" as const },
      },
    },
  };
  assert.deepEqual(stageRows(m).map((r) => [r.name, r.status]), [
    ["ingest", "done"], ["poses", "failed"], ["splat", "blocked"],
  ]);
  assert.equal(failureSummary(m), "The poses stage failed: Could not recover the camera path");
});

test("failure summary without stage detail", () => {
  assert.equal(failureSummary({ ...base, status: "failed" }), "Processing failed.");
});
