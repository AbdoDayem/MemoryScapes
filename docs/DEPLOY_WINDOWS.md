# Deploying MemoryScapes on the Windows PC

One Windows PC with an NVIDIA GPU runs everything: the web app, the API, the processing
pipeline, the SQLite database and the memories' files. People open the site in a browser.

```
browser ──► web app :3000 ──/api/*──► API 127.0.0.1:8000 ──► pipeline (workers .venv, GPU)
            (Next.js)                 (FastAPI)            ──► services/workers/data/storage/  files
                                                           ──► services/api/data/memoryscapes.db  database
```

The browser only ever talks to the web app. `/api/*` is passed to the API, which listens
only on the PC itself. The links the API and SATIE hand out start with `/api`, so a memory
works at whatever address people use: `localhost`, the PC's network address, or a tunnel later.

## 1. Install the tools (once)

From a terminal (PowerShell or cmd):

```bat
winget install Python.Python.3.12
winget install OpenJS.NodeJS.LTS
winget install Git.Git
winget install Gyan.FFmpeg
```

For splats trained on the GPU, also install the CUDA toolkit and the C++ build tools. These
versions are the ones tested in `services/workers/README.md`, "GPU splat training on Windows":

```bat
winget install Nvidia.CUDA -v 12.8
winget install Microsoft.VisualStudio.2022.BuildTools --override "--quiet --wait --norestart --add Microsoft.VisualStudio.Workload.VCTools --includeRecommended"
```

For faster-loading splats (streamed in the browser), also install Rust. Setup builds Spark's
`build-lod` with it, which also needs the C++ build tools above:

```bat
winget install Rustlang.Rustup
```

Open a new terminal afterwards so the tools are on `PATH`. You also need a recent NVIDIA driver.

## 2. Get the code and set it up

```bat
git clone https://github.com/AbdoDayem/MemoryScapes.git
cd MemoryScapes
deploy\windows\setup.cmd --build-gsplat
```

`setup.cmd` runs `deploy/setup.py`. Each step is safe to repeat:
- **`.env`:** created from `.env.example`, with `PUBLIC_API_URL=/api`, `NEXT_PUBLIC_API_URL=/api`
  and `SOUNDSCAPE_PROVIDER=satie`. An existing `.env` is never changed.
- **Workers:** `services\workers\.venv` gets CUDA torch from PyTorch's cu128 index (PyPI's
  Windows torch is CPU-only), the audio, visual and SATIE extras, and, with
  `--build-gsplat`, gsplat's CUDA kernels built through `cuda_env.cmd`.
- **Streaming LoD tool:** with Rust installed, Spark's `build-lod` is built into
  `services\workers\.tools\` (about 2 minutes, once per Spark version). The publish stage
  then gives each memory a streaming LoD tree. Without Rust this step is skipped, and memories
  load the slower way.
- **Models:** about 6 GB, the first time only.
- **API:** `services\api\.venv`.
- **Web app:** `npm ci`, then the production build.
- **Checks:** at the end it prints whether torch sees the GPU, which splat trainer will be
  used (`gsplat` is the fast one), whether `build-lod` is installed, and whether SATIE is
  signed in.

## 3. Sign in to SATIE (once)

The server uses its own SATIE account:

```bat
services\workers\.venv\Scripts\memoryscapes-worker satie-login
```

A browser opens on satie.world. The sign-in is saved to `.satie\worker-token.json`, which is
git-ignored; keep it private. `satie-login --status` shows the plan and balance. Leave
`SATIE_MAX_CENTS=0` and `SATIE_ALLOW_GENERATION=false` in `.env`: SATIE only arranges the
recorded sounds. On the Free plan each memory uses one of 3 compositions a month.

## 4. Start the site

```bat
deploy\windows\start.cmd
```

Open http://localhost:3000. The API and the web app restart by themselves if one of them
stops. Ctrl+C stops both. Logs are in `logs\api.log` and `logs\web.log`.

To start the site at every Windows login:

```bat
deploy\windows\autostart.cmd          :: add the task (Task Scheduler, "MemoryScapes")
deploy\windows\autostart.cmd remove   :: remove it
```

## 5. After a `git pull`

```bat
git pull
deploy\windows\setup.cmd --skip-models
```

Then restart `start.cmd`. Add `--build-gsplat` when torch or gsplat changed version.

## Reaching the site from other machines

- **Same network:** `deploy\windows\start.cmd --host 0.0.0.0`, then allow port 3000 in
  Windows Firewall when it asks. People open `http://<the PC's address>:3000`.
- **From anywhere:** not set up yet. A free tunnel will do it without opening router ports:
  Cloudflare Tunnel (a public https address, with Cloudflare Access to ask visitors to log
  in) or Tailscale (only invited devices).

**There's no login yet.** Anyone who can reach port 3000 can see every memory, and can also
upload videos and start processing (GPU time and SATIE compositions). Keep it to this PC or
a trusted network until there's a login. Also, nothing should be public before the privacy
pass (face and number-plate blur) exists.

## Where things live

| What | Where |
|---|---|
| Settings | `.env` (git-ignored) |
| Memories' files | `services\workers\data\storage\<id>\` (`STORAGE_ROOT` moves it) |
| Database | `services\api\data\memoryscapes.db` (`DATABASE_URL` moves it). Finished memories are also stored here, files included |
| SATIE sign-in | `.satie\worker-token.json` (`SATIE_TOKEN_FILE` moves it) |
| Models | `.models\` and the Hugging Face cache |
| Logs | `logs\` |

## If something's wrong

- **`setup.cmd` stops at a step:** it prints the command that failed. Fix the cause and run
  it again; finished steps are quick.
- **"Port 3000 (web) is in use":** the site or a dev server is already running.
- **The splat trains slowly:** run `deploy\windows\setup.cmd --check`. If the trainer isn't
  `gsplat`, run `setup.cmd --build-gsplat --skip-models`.
- **A memory loads slowly or doesn't stream:** add `?perf` to its address. The status says
  "streamed" when it uses its LoD tree. Memories published before `build-lod` was installed have
  none. Rebuild one with
  `services\workers\.venv\Scripts\memoryscapes-worker run <id> --stages publish`, then
  `services\api\.venv\Scripts\python -m app.memory_files store <id>` (run from `services\api`).
- **A memory has no sound:** the viewer says why. Usually SATIE isn't signed in, or the month's
  compositions are used up. Re-run the soundscape stage when that's fixed:
  `services\workers\.venv\Scripts\memoryscapes-worker run <id> --stages soundscape,publish`.
- **Pages load but nothing else does:** check `.env` has `PUBLIC_API_URL=/api` and
  `NEXT_PUBLIC_API_URL=/api`, then re-run `setup.cmd --skip-workers --skip-models` to rebuild
  the web app.
