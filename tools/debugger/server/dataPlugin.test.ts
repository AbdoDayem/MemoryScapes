import fs from "node:fs";
import os from "node:os";
import path from "node:path";

import { afterAll, beforeAll, describe, expect, it } from "vitest";

import { listMemories, safeJoin, walk } from "./dataPlugin";

let root: string;
beforeAll(() => {
  root = fs.mkdtempSync(path.join(os.tmpdir(), "msdbg-"));
  for (const key of ["mem_a/separate/stems.json", "mem_a/splat/renders/f0.png", "mem_b/ingest/audio.wav"]) {
    fs.mkdirSync(path.dirname(path.join(root, key)), { recursive: true });
    fs.writeFileSync(path.join(root, key), "x");
  }
  fs.mkdirSync(path.join(root, "not_a_memory"));
});
afterAll(() => fs.rmSync(root, { recursive: true, force: true }));

describe("data server", () => {
  it("refuses paths that escape the data root", () => {
    expect(safeJoin(root, "mem_a/separate/stems.json")).toBe(path.join(root, "mem_a", "separate", "stems.json"));
    expect(safeJoin(root, "../etc/passwd")).toBeNull();
    expect(safeJoin(root, "mem_a/../../secret")).toBeNull();
  });
  it("lists memories with per-stage file counts, skipping empty folders", () => {
    expect(listMemories(root)).toEqual([
      { id: "mem_a", stages: { splat: 1, separate: 1 } },
      { id: "mem_b", stages: { ingest: 1 } },
    ]);
  });
  it("walks files with forward-slash keys", () => {
    expect(walk(path.join(root, "mem_a")).map((f) => f.key).sort()).toEqual(["separate/stems.json", "splat/renders/f0.png"]);
  });
});
