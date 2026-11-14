import fs from "node:fs";
import path from "node:path";
import { fileURLToPath } from "node:url";

import react from "@vitejs/plugin-react";
import { defineConfig } from "vite";

import { memoryDataPlugin } from "./server/dataPlugin";
import { jobsPlugin } from "./server/jobs";

const here = path.dirname(fileURLToPath(import.meta.url));

// Folder holding one sub-folder per memory, in the pipeline's storage layout.
// Defaults to the worker examples' output; point it at a LocalStorage root or a
// synced bucket with MEMORYSCAPES_DATA=/path/to/root.
const dataRoot = path.resolve(here, process.env.MEMORYSCAPES_DATA ?? "../../services/workers/examples/out");

// Uploads are processed by the workers package in this checkout, with this Python.
const workersDir = path.resolve(here, process.env.MEMORYSCAPES_WORKERS ?? "../../services/workers");
const venvPython = path.join(workersDir, ".venv", process.platform === "win32" ? "Scripts/python.exe" : "bin/python");
const python = process.env.MEMORYSCAPES_PYTHON ?? (fs.existsSync(venvPython) ? venvPython : "python");

// Model locations passed through to the pipeline (see services/workers/memoryscapes_workers/backends.py).
const passEnv: Record<string, string> = Object.fromEntries(
  ["AUDIOSEP_DIR", "AUDIO_TAGGER", "SEPARATOR", "SEPARATOR_SPEECH", "DETECTOR", "FFMPEG_DIR", "HF_HOME"].filter((k) => process.env[k]).map((k) => [k, process.env[k]!]),
);
// Default AudioSep location: <repo>/.models/AudioSep (see services/workers/README.md).
const defaultAudioSep = path.resolve(here, "../../.models/AudioSep");
if (!passEnv.AUDIOSEP_DIR && fs.existsSync(defaultAudioSep)) passEnv.AUDIOSEP_DIR = defaultAudioSep;

export default defineConfig({
  plugins: [react(), memoryDataPlugin(dataRoot), jobsPlugin({ root: dataRoot, python, workersDir, env: passEnv })],
  build: { chunkSizeWarningLimit: 1200 }, // three.js is most of the bundle; fine for a local tool
  server: { port: 5178 },
  preview: { port: 5178 },
});
