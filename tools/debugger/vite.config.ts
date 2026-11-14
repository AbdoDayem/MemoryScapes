import path from "node:path";
import { fileURLToPath } from "node:url";

import react from "@vitejs/plugin-react";
import { defineConfig } from "vite";

import { memoryDataPlugin } from "./server/dataPlugin";

const here = path.dirname(fileURLToPath(import.meta.url));

// Folder holding one sub-folder per memory, in the pipeline's storage layout.
// Defaults to the worker examples' output; point it at a LocalStorage root or a
// synced bucket with MEMORYSCAPES_DATA=/path/to/root.
const dataRoot = path.resolve(here, process.env.MEMORYSCAPES_DATA ?? "../../services/workers/examples/out");

export default defineConfig({
  plugins: [react(), memoryDataPlugin(dataRoot)],
  build: { chunkSizeWarningLimit: 1200 }, // three.js is most of the bundle; fine for a local tool
  server: { port: 5178 },
  preview: { port: 5178 },
});
