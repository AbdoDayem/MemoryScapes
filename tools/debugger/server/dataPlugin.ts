/**
 * Serves pipeline outputs from a local folder to the debugger, in dev and preview.
 *
 *   GET /api/memories               -> [{ id, stages: { ingest: 12, splat: 30, ... } }]
 *   GET /api/memories/:id/files     -> [{ key, size }]   (key relative to the memory)
 *   GET /data/:id/<key>             -> the raw file
 *
 * Read-only. Paths are resolved inside the data root; anything escaping it is refused.
 */
import fs from "node:fs";
import path from "node:path";
import type { IncomingMessage, ServerResponse } from "node:http";

import type { Plugin } from "vite";

export const STAGES = ["ingest", "poses", "splat", "separate", "localize", "soundscape", "publish", "debug"] as const;

const TYPES: Record<string, string> = {
  ".json": "application/json",
  ".wav": "audio/wav",
  ".png": "image/png",
  ".jpg": "image/jpeg",
  ".jpeg": "image/jpeg",
  ".svg": "image/svg+xml",
  ".md": "text/markdown; charset=utf-8",
  ".satie": "text/plain; charset=utf-8",
};

export function safeJoin(root: string, ...parts: string[]): string | null {
  const full = path.resolve(root, ...parts);
  const rel = path.relative(root, full);
  if (rel.startsWith("..") || path.isAbsolute(rel)) return null;
  return full;
}

export function walk(dir: string, base = dir): { key: string; size: number }[] {
  if (!fs.existsSync(dir)) return [];
  const out: { key: string; size: number }[] = [];
  for (const entry of fs.readdirSync(dir, { withFileTypes: true })) {
    const full = path.join(dir, entry.name);
    if (entry.isDirectory()) out.push(...walk(full, base));
    else out.push({ key: path.relative(base, full).split(path.sep).join("/"), size: fs.statSync(full).size });
  }
  return out;
}

export function listMemories(root: string) {
  if (!fs.existsSync(root)) return [];
  return fs
    .readdirSync(root, { withFileTypes: true })
    .filter((e) => e.isDirectory())
    .map((e) => {
      const stages: Record<string, number> = {};
      for (const s of STAGES) {
        const n = walk(path.join(root, e.name, s)).length;
        if (n) stages[s] = n;
      }
      return { id: e.name, stages };
    })
    .filter((m) => Object.keys(m.stages).length > 0)
    .sort((a, b) => a.id.localeCompare(b.id));
}

function send(res: ServerResponse, status: number, body: string, type = "application/json") {
  res.statusCode = status;
  res.setHeader("Content-Type", type);
  res.setHeader("Cache-Control", "no-store");
  res.end(body);
}

export function handler(root: string) {
  return (req: IncomingMessage, res: ServerResponse, next: () => void) => {
    const url = decodeURIComponent((req.url ?? "").split("?")[0]);
    if (url === "/api/memories") return send(res, 200, JSON.stringify(listMemories(root)));

    const files = url.match(/^\/api\/memories\/([^/]+)\/files$/);
    if (files) {
      const dir = safeJoin(root, files[1]);
      if (!dir) return send(res, 400, '{"error":"bad memory id"}');
      return send(res, 200, JSON.stringify(walk(dir)));
    }

    if (url.startsWith("/data/")) {
      const full = safeJoin(root, url.slice("/data/".length));
      if (!full) return send(res, 400, '{"error":"path escapes the data root"}');
      if (!fs.existsSync(full) || !fs.statSync(full).isFile()) return send(res, 404, '{"error":"not found"}');
      res.statusCode = 200;
      res.setHeader("Content-Type", TYPES[path.extname(full).toLowerCase()] ?? "application/octet-stream");
      res.setHeader("Cache-Control", "no-store");
      return fs.createReadStream(full).pipe(res);
    }
    next();
  };
}

export function memoryDataPlugin(root: string): Plugin {
  return {
    name: "memoryscapes-data",
    configureServer(server) {
      server.config.logger.info(`  memoryscapes data: ${root}`);
      server.middlewares.use(handler(root));
    },
    configurePreviewServer(server) {
      server.middlewares.use(handler(root));
    },
  };
}
