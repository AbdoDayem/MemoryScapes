/**
 * Uploads and local pipeline runs for the debugger.
 *
 *   PUT  /api/upload?memory=<id>&kind=video|audio&name=<file>  raw file body -> <root>/<id>/upload/
 *   POST /api/memories/:id/process  {"maxSeconds", "detector"}  starts `python -m memoryscapes_workers.run`
 *   GET  /api/memories/:id/job      -> { status: <job/status.json>, log: [last lines], running }
 *
 * One run per memory at a time. The Python side writes job/status.json as it goes.
 */
import { spawn, type ChildProcess } from "node:child_process";
import fs from "node:fs";
import path from "node:path";
import type { IncomingMessage, ServerResponse } from "node:http";

import type { Plugin } from "vite";

import { safeJoin } from "./dataPlugin";

export function jobsPlugin(cfg: JobConfig): Plugin {
  return {
    name: "memoryscapes-jobs",
    configureServer(server) {
      server.config.logger.info(`  pipeline python: ${cfg.python}\n  workers: ${cfg.workersDir}`);
      server.middlewares.use(jobHandler(cfg));
    },
    configurePreviewServer(server) {
      server.middlewares.use(jobHandler(cfg));
    },
  };
}

export interface JobConfig {
  root: string;
  python: string;
  workersDir: string;
  env?: Record<string, string>;
}

const running = new Map<string, ChildProcess>();

export const MEMORY_ID = /^[a-z0-9][a-z0-9_-]{0,63}$/;
const FILE_NAME = /^[\w.\- ()]{1,128}$/;

function send(res: ServerResponse, status: number, body: unknown) {
  res.statusCode = status;
  res.setHeader("Content-Type", "application/json");
  res.setHeader("Cache-Control", "no-store");
  res.end(JSON.stringify(body));
}

function readBody(req: IncomingMessage): Promise<string> {
  return new Promise((resolve, reject) => {
    let s = "";
    req.on("data", (c) => (s += c));
    req.on("end", () => resolve(s));
    req.on("error", reject);
  });
}

function tail(file: string, lines = 80): string[] {
  if (!fs.existsSync(file)) return [];
  const text = fs.readFileSync(file, "utf8");
  return text.split(/\r?\n/).filter(Boolean).slice(-lines);
}

/**
 * PATH for the pipeline, plus where ffmpeg lands when installed after this server started
 * (a winget install only updates PATH for new shells): FFMPEG_DIR, and the bin folders of
 * winget's FFmpeg packages.
 */
export function childPath(env: NodeJS.ProcessEnv = process.env): string {
  const extra: string[] = [];
  if (env.FFMPEG_DIR) extra.push(env.FFMPEG_DIR);
  const local = env.LOCALAPPDATA;
  if (local) {
    const packages = path.join(local, "Microsoft", "WinGet", "Packages");
    try {
      for (const pkg of fs.readdirSync(packages).filter((d) => /ffmpeg/i.test(d))) {
        for (const build of fs.readdirSync(path.join(packages, pkg))) {
          const bin = path.join(packages, pkg, build, "bin");
          if (fs.existsSync(path.join(bin, "ffmpeg.exe")) || fs.existsSync(path.join(bin, "ffmpeg"))) extra.push(bin);
        }
      }
    } catch {
      /* no winget packages folder */
    }
    extra.push(path.join(local, "Microsoft", "WinGet", "Links"));
  }
  return [...extra, env.PATH ?? ""].join(path.delimiter);
}

export function jobHandler(cfg: JobConfig) {
  return async (req: IncomingMessage, res: ServerResponse, next: () => void) => {
    const url = new URL(req.url ?? "", "http://local");

    if (url.pathname === "/api/upload" && req.method === "PUT") {
      const memory = url.searchParams.get("memory") ?? "";
      const kind = url.searchParams.get("kind") ?? "video";
      const name = url.searchParams.get("name") ?? "";
      if (!MEMORY_ID.test(memory)) return send(res, 400, { error: "Memory name: use lowercase letters, digits, - and _ (up to 64)." });
      if (!["video", "audio"].includes(kind)) return send(res, 400, { error: "kind must be video or audio" });
      if (!FILE_NAME.test(name)) return send(res, 400, { error: "That file name has characters the uploader doesn't accept. Rename it and try again." });
      if (running.has(memory)) return send(res, 409, { error: `${memory} is being processed. Wait for it to finish.` });
      const dir = safeJoin(cfg.root, memory, "upload");
      if (!dir) return send(res, 400, { error: "bad memory id" });
      fs.mkdirSync(dir, { recursive: true });
      const ext = path.extname(name).toLowerCase() || (kind === "video" ? ".mp4" : ".wav");
      const fileName = `${kind}${ext}`;
      const dest = path.join(dir, fileName);
      await new Promise<void>((resolve, reject) => {
        const out = fs.createWriteStream(dest);
        req.pipe(out);
        out.on("finish", () => resolve());
        out.on("error", reject);
        req.on("error", reject);
      });
      const metaPath = path.join(dir, "meta.json");
      const meta = fs.existsSync(metaPath) ? JSON.parse(fs.readFileSync(metaPath, "utf8")) : { video: null, audio: null };
      meta[kind] = fileName;
      meta[`${kind}_original_name`] = name;
      fs.writeFileSync(metaPath, JSON.stringify(meta, null, 2));
      return send(res, 200, { memory, kind, file: fileName, bytes: fs.statSync(dest).size });
    }

    const process_ = url.pathname.match(/^\/api\/memories\/([^/]+)\/process$/);
    if (process_ && req.method === "POST") {
      const memory = process_[1];
      if (!MEMORY_ID.test(memory)) return send(res, 400, { error: "bad memory id" });
      if (running.has(memory)) return send(res, 409, { error: "already running" });
      const uploadMeta = safeJoin(cfg.root, memory, "upload", "meta.json");
      if (!uploadMeta || !fs.existsSync(uploadMeta)) return send(res, 400, { error: "Upload a video first." });
      const opts = JSON.parse((await readBody(req)) || "{}");
      const args = ["-m", "memoryscapes_workers.run", "--root", cfg.root, "--memory", memory, "--max-seconds", String(Number(opts.maxSeconds) || 30)];
      if (opts.detector === false) args.push("--no-detector");
      const jobDir = path.join(cfg.root, memory, "job");
      fs.mkdirSync(jobDir, { recursive: true });
      // clear the previous run's status so the UI doesn't show stale results
      fs.rmSync(path.join(jobDir, "status.json"), { force: true });
      const child = spawn(cfg.python, args, {
        cwd: cfg.workersDir,
        env: { ...process.env, ...cfg.env, PYTHONPATH: cfg.workersDir, PYTHONIOENCODING: "utf-8", PATH: childPath() },
        stdio: ["ignore", "pipe", "pipe"],
        windowsHide: true,
      });
      const stderr = fs.createWriteStream(path.join(jobDir, "process.txt"));
      child.stdout?.pipe(stderr);
      child.stderr?.pipe(stderr);
      running.set(memory, child);
      child.on("exit", (code) => {
        running.delete(memory);
        fs.appendFileSync(path.join(jobDir, "process.txt"), `\n[exit ${code}]\n`);
      });
      child.on("error", (e) => {
        running.delete(memory);
        fs.writeFileSync(path.join(jobDir, "status.json"), JSON.stringify({ memory_id: memory, state: "failed", detail: `Couldn't start Python (${cfg.python}): ${e.message}`, stages: {} }));
      });
      return send(res, 202, { memory, started: true });
    }

    const job = url.pathname.match(/^\/api\/memories\/([^/]+)\/job$/);
    if (job) {
      const memory = job[1];
      const dir = safeJoin(cfg.root, memory, "job");
      if (!dir) return send(res, 400, { error: "bad memory id" });
      const statusPath = path.join(dir, "status.json");
      let status = null;
      try {
        status = fs.existsSync(statusPath) ? JSON.parse(fs.readFileSync(statusPath, "utf8")) : null;
      } catch {
        /* being rewritten; the next poll gets it */
      }
      const log = tail(path.join(dir, "log.txt"));
      const proc = running.has(memory) ? [] : tail(path.join(dir, "process.txt"), 30);
      return send(res, 200, { running: running.has(memory), status, log: log.length ? log : proc });
    }
    next();
  };
}
