"""Set up this machine to run MemoryScapes (the deploy PC: Windows + NVIDIA). Re-run after
every `git pull`; finished steps are quick.

    py -3 deploy\\setup.py                  # everything (first time: ~6 GB of models)
    py -3 deploy\\setup.py --build-gsplat   # also build gsplat's CUDA kernels (once per torch/gsplat version)
    py -3 deploy\\setup.py --skip-models    # after a pull: code and the web build only

Steps (docs/DEPLOY_WINDOWS.md):
1. .env: created from .env.example with the deploy values (PUBLIC_API_URL=/api and
   NEXT_PUBLIC_API_URL=/api: the browser reaches the API through the web app's /api proxy).
   An existing .env is never edited; missing deploy values are reported.
2. Workers: services/workers/.venv with CUDA torch (from PyTorch's index: PyPI's Windows
   torch is CPU-only), the audio/visual/satie extras, and optionally gsplat built from source.
3. Models: memoryscapes-worker setup-models (AudioSep, CLAP, Grounding DINO, Demucs).
4. API: services/api/.venv.
5. Web: npm ci, then the production build with NEXT_PUBLIC_API_URL from .env.
6. Checks: GPU, splat trainer, SATIE sign-in.

Only the standard library: this runs before any virtualenv exists.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import re
import shutil
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
WORKERS = REPO / "services" / "workers"
API = REPO / "services" / "api"
WEB = REPO / "apps" / "web"
ENV_FILE = REPO / ".env"
WINDOWS = os.name == "nt"
# AbdoDayem's tested setup (services/workers/README.md "GPU splat training on Windows"):
# CUDA 12.8 toolkit, torch from the cu128 index, gsplat 1.5.3 built from source.
DEFAULT_TORCH_INDEX = "https://download.pytorch.org/whl/cu128"
GSPLAT_VERSION = "1.5.3"
# What the deployed site needs in .env, and why.
DEPLOY_ENV = {
    "PUBLIC_API_URL": ("/api", "links the API and SATIE give browsers go through the web app's /api proxy"),
    "NEXT_PUBLIC_API_URL": ("/api", "the web app calls the API through its own /api proxy (read at build time)"),
    "SOUNDSCAPE_PROVIDER": ("satie", "sound is always SATIE's"),
}
BUILD_INFO = WEB / ".next" / "memoryscapes-build.json"


def say(msg: str) -> None:
    print(f"\n== {msg}", flush=True)


def run(cmd: list[str], cwd: Path = REPO, env: dict[str, str] | None = None) -> None:
    print("  > " + " ".join(str(c) for c in cmd), flush=True)
    subprocess.run([str(c) for c in cmd], cwd=cwd, env=env, check=True)


def venv_python(project: Path) -> Path:
    return project / ".venv" / ("Scripts/python.exe" if WINDOWS else "bin/python")


def venv_tool(project: Path, name: str) -> Path:
    return project / ".venv" / ("Scripts" if WINDOWS else "bin") / (name + (".exe" if WINDOWS else ""))


def tool(name: str) -> str | None:
    return shutil.which(name)


def read_env(path: Path = ENV_FILE) -> dict[str, str]:
    """KEY=VALUE lines (the same rules as services/workers/memoryscapes_workers/envfile.py)."""
    values: dict[str, str] = {}
    if not path.is_file():
        return values
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        if line.startswith("export "):
            line = line[len("export "):].lstrip()
        key, _, value = line.partition("=")
        key, value = key.strip(), value.strip()
        if len(value) >= 2 and value[0] == value[-1] and value[0] in "'\"":
            value = value[1:-1]
        elif " #" in value:
            value = value.split(" #", 1)[0].rstrip()
        if key:
            values[key] = value
    return values


# ---------------------------------------------------------------- steps


def check_tools() -> None:
    say("Checking tools")
    if sys.version_info < (3, 11):
        sys.exit(f"Python 3.11 or newer is needed (this is {platform.python_version()}).")
    print(f"  Python {platform.python_version()} ({sys.executable})")
    for name, needed, hint in (
        ("node", True, "winget install OpenJS.NodeJS.LTS"),
        ("npm", True, "comes with Node.js"),
        ("ffmpeg", True, "winget install Gyan.FFmpeg (then open a new terminal)"),
        ("git", False, "winget install Git.Git"),
        ("nvidia-smi", False, "the NVIDIA driver; without it splats train on the CPU (very slow)"),
    ):
        found = tool(name)
        print(f"  {name:11s} {'ok' if found else 'MISSING'}{'' if found else f'  ({hint})'}")
        if needed and not found:
            sys.exit(f"{name} is needed: {hint}")
    out = subprocess.run([tool("node"), "--version"], capture_output=True, text=True).stdout.strip()
    if int(re.sub(r"^v(\d+).*", r"\1", out) or 0) < 20:
        sys.exit(f"Node.js 20 or newer is needed (found {out}).")


def write_env() -> dict[str, str]:
    say("Configuration (.env)")
    if not ENV_FILE.exists():
        text = (REPO / ".env.example").read_text(encoding="utf-8")
        for key, (value, _why) in DEPLOY_ENV.items():
            if re.search(rf"^{key}=", text, re.M):
                text = re.sub(rf"^{key}=.*$", f"{key}={value}", text, flags=re.M)
            else:
                text += f"\n{key}={value}\n"
        ENV_FILE.write_text(text, encoding="utf-8")
        print(f"  created {ENV_FILE} from .env.example with the deploy values")
    env = read_env()
    for key, (value, why) in DEPLOY_ENV.items():
        if env.get(key) != value:
            print(f"  note: .env has {key}={env.get(key, '(unset)')}; the deployed site expects {key}={value} ({why})")
    return env


def setup_workers(torch_index: str | None, build_gsplat: bool) -> None:
    say("Workers (services/workers/.venv)")
    py = venv_python(WORKERS)
    if not py.exists():
        run([sys.executable, "-m", "venv", WORKERS / ".venv"])
    run([py, "-m", "pip", "install", "--upgrade", "pip"])
    if torch_index:
        # Before the extras, so pip keeps this CUDA build instead of pulling PyPI's CPU one.
        run([py, "-m", "pip", "install", "torch", "torchvision", "torchaudio", "--index-url", torch_index])
    run([py, "-m", "pip", "install", "-e", ".[audio,visual,satie]"], cwd=WORKERS)
    if build_gsplat:
        if not WINDOWS:
            print("  --build-gsplat is for Windows; elsewhere pip's gsplat builds on first use")
        else:
            run([py, "-m", "pip", "install", "wheel", "setuptools", "ninja"])
            run(["cmd", "/c", "cuda_env.cmd", r".venv\Scripts\python", "-m", "pip", "install",
                 "--no-build-isolation", "--no-binary", "gsplat", "--force-reinstall", "--no-deps",
                 f"gsplat=={GSPLAT_VERSION}"], cwd=WORKERS)


def setup_models() -> None:
    say("Models (memoryscapes-worker setup-models; about 6 GB the first time)")
    run([venv_tool(WORKERS, "memoryscapes-worker"), "setup-models"], cwd=WORKERS)


def setup_api() -> None:
    say("API (services/api/.venv)")
    py = venv_python(API)
    if not py.exists():
        run([sys.executable, "-m", "venv", API / ".venv"])
    run([py, "-m", "pip", "install", "--upgrade", "pip"])
    run([py, "-m", "pip", "install", "-e", "."], cwd=API)


def setup_web(env: dict[str, str]) -> None:
    say("Web app (npm ci, production build)")
    npm = tool("npm")
    # npm ci reinstalls everything; skip it when package-lock.json hasn't changed since.
    lock_sha = hashlib.sha256((REPO / "package-lock.json").read_bytes()).hexdigest()
    marker = REPO / "node_modules" / ".memoryscapes-lock-sha256"
    if marker.exists() and marker.read_text(encoding="utf-8").strip() == lock_sha:
        print("  node_modules is up to date with package-lock.json")
    else:
        run([npm, "ci"])
        marker.write_text(lock_sha + "\n", encoding="utf-8")
    public_api = os.environ.get("NEXT_PUBLIC_API_URL") or env.get("NEXT_PUBLIC_API_URL") or "/api"
    build_env = {**os.environ, "NEXT_PUBLIC_API_URL": public_api}
    if env.get("API_ORIGIN"):
        build_env.setdefault("API_ORIGIN", env["API_ORIGIN"])
    run([npm, "run", "build", "--workspace", "@memoryscapes/web"], env=build_env)
    BUILD_INFO.write_text(json.dumps({"NEXT_PUBLIC_API_URL": public_api,
                                      "API_ORIGIN": build_env.get("API_ORIGIN", "http://127.0.0.1:8000")}),
                          encoding="utf-8")


def checks() -> None:
    say("Checks")
    py = venv_python(WORKERS)
    if not py.exists():
        print("  workers not installed")
        return
    probe = ("import json, torch\n"
             "from memoryscapes_workers.visual.train import detect_trainer\n"
             "print(json.dumps({'torch': torch.__version__, 'cuda': torch.cuda.is_available(),\n"
             "  'gpu': torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,\n"
             "  'trainer': detect_trainer()}))")
    out = subprocess.run([py, "-c", probe], cwd=WORKERS, capture_output=True, text=True)
    if out.returncode == 0:
        info = json.loads(out.stdout.strip().splitlines()[-1])
        print(f"  torch {info['torch']}, CUDA {'yes: ' + info['gpu'] if info['cuda'] else 'NO'}, "
              f"splat trainer: {info['trainer']}")
        if info["cuda"] and info["trainer"] != "gsplat":
            print("  note: splats won't train on the GPU until gsplat's CUDA kernels are built: "
                  "re-run with --build-gsplat (needs the CUDA 12.8 toolkit and VS Build Tools)")
        elif not info["cuda"] and tool("nvidia-smi"):
            print("  note: there's an NVIDIA GPU but torch can't use it: reinstall torch from "
                  f"{DEFAULT_TORCH_INDEX} (re-run without --torch-index none)")
    else:
        print("  could not check torch: " + (out.stderr.strip().splitlines() or ["?"])[-1])
    login = subprocess.run([venv_tool(WORKERS, "memoryscapes-worker"), "satie-login", "--status"],
                           cwd=WORKERS, capture_output=True, text=True)
    try:
        status = json.loads(login.stdout[login.stdout.index("{"):])
        signed_in = status.get("signed_in")
    except (ValueError, json.JSONDecodeError):
        signed_in = None
    print("  SATIE: " + ("signed in" if signed_in else
                         "not signed in: run  services\\workers\\.venv\\Scripts\\memoryscapes-worker satie-login"))


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--skip-workers", action="store_true")
    ap.add_argument("--skip-models", action="store_true")
    ap.add_argument("--skip-api", action="store_true")
    ap.add_argument("--skip-web", action="store_true")
    ap.add_argument("--torch-index", default=None,
                    help=f"PyTorch wheel index for CUDA torch (default {DEFAULT_TORCH_INDEX} where there's "
                         "an NVIDIA GPU; 'none' keeps whatever torch is installed)")
    ap.add_argument("--build-gsplat", action="store_true",
                    help="build gsplat's CUDA kernels from source (Windows; once per torch/gsplat version)")
    ap.add_argument("--check", action="store_true", help="only run the checks")
    args = ap.parse_args(argv)

    if args.check:
        checks()
        return 0
    check_tools()
    env = write_env()
    if not args.skip_workers:
        index = args.torch_index
        if index is None:
            index = DEFAULT_TORCH_INDEX if tool("nvidia-smi") else "none"
        setup_workers(None if index == "none" else index, args.build_gsplat)
    if not args.skip_models:
        setup_models()
    if not args.skip_api:
        setup_api()
    if not args.skip_web:
        setup_web(env)
    checks()
    say("Done")
    print("  Start the site:  deploy\\windows\\start.cmd   (or: python deploy\\serve.py)")
    print("  Then open        http://localhost:3000")
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except subprocess.CalledProcessError as exc:
        sys.exit(f"\nA step failed (exit code {exc.returncode}): {' '.join(map(str, exc.cmd))}")
