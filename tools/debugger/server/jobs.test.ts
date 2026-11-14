import fs from "node:fs";
import os from "node:os";
import path from "node:path";
import { Readable } from "node:stream";
import type { IncomingMessage, ServerResponse } from "node:http";

import { afterAll, beforeAll, describe, expect, it } from "vitest";

import { jobHandler } from "./jobs";

let root: string;
beforeAll(() => {
  root = fs.mkdtempSync(path.join(os.tmpdir(), "msjobs-"));
});
afterAll(() => fs.rmSync(root, { recursive: true, force: true }));

/** Drive the middleware with a fake request/response; resolves with status and JSON body. */
function call(method: string, url: string, body = ""): Promise<{ status: number; json: any; nextCalled: boolean }> {
  const handler = jobHandler({ root, python: "python-that-does-not-exist", workersDir: root });
  return new Promise((resolve) => {
    const req = Object.assign(Readable.from(body ? [Buffer.from(body)] : []), { method, url }) as unknown as IncomingMessage;
    const res = {
      statusCode: 200,
      setHeader() {},
      end(this: { statusCode: number }, data: string) {
        resolve({ status: this.statusCode, json: JSON.parse(data), nextCalled: false });
      },
    } as unknown as ServerResponse;
    handler(req, res, () => resolve({ status: 0, json: null, nextCalled: true }));
  });
}

describe("upload + job API", () => {
  it("stores an upload and records it in upload/meta.json", async () => {
    const r = await call("PUT", "/api/upload?memory=park_1&kind=video&name=My%20Walk.MOV", "fake-video-bytes");
    expect(r.status).toBe(200);
    expect(r.json.file).toBe("video.mov");
    const meta = JSON.parse(fs.readFileSync(path.join(root, "park_1", "upload", "meta.json"), "utf8"));
    expect(meta).toMatchObject({ video: "video.mov", video_original_name: "My Walk.MOV", audio: null });
    expect(fs.readFileSync(path.join(root, "park_1", "upload", "video.mov"), "utf8")).toBe("fake-video-bytes");
  });

  it("rejects bad memory names, kinds and file names", async () => {
    expect((await call("PUT", "/api/upload?memory=../evil&kind=video&name=a.mp4")).status).toBe(400);
    expect((await call("PUT", "/api/upload?memory=Park&kind=video&name=a.mp4")).status).toBe(400);
    expect((await call("PUT", "/api/upload?memory=ok&kind=exe&name=a.mp4")).status).toBe(400);
    expect((await call("PUT", "/api/upload?memory=ok&kind=video&name=..%2F..%2Fx.mp4")).status).toBe(400);
  });

  it("won't process a memory with nothing uploaded", async () => {
    const r = await call("POST", "/api/memories/empty_one/process", "{}");
    expect(r.status).toBe(400);
    expect(r.json.error).toMatch(/Upload a video first/);
  });

  it("reports job status and log", async () => {
    const dir = path.join(root, "done_one", "job");
    fs.mkdirSync(dir, { recursive: true });
    fs.writeFileSync(path.join(dir, "status.json"), JSON.stringify({ state: "done", stages: { ingest: { status: "done" } } }));
    fs.writeFileSync(path.join(dir, "log.txt"), "line 1\nline 2\n");
    const r = await call("GET", "/api/memories/done_one/job");
    expect(r.json).toMatchObject({ running: false, status: { state: "done" }, log: ["line 1", "line 2"] });
  });

  it("passes other routes through", async () => {
    expect((await call("GET", "/api/memories")).nextCalled).toBe(true);
  });
});
