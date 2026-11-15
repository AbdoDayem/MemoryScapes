"""Run the deployed site: the API and the web app, restarted if either one stops.

    py -3 deploy\\serve.py                    # http://localhost:3000 on this PC
    py -3 deploy\\serve.py --host 0.0.0.0     # also from other machines on the network

- API: uvicorn on 127.0.0.1:8000, reachable only from this machine. Browsers reach it
  through the web app's /api proxy, which is why .env has PUBLIC_API_URL=/api.
- Web: the production build (deploy/setup.py) with `next start` on --host:--port.
- Settings come from <repo>/.env; real environment variables win.
- Logs: <repo>/logs/api.log and logs/web.log. Ctrl+C stops both.

Only the standard library: it runs with the system Python and starts the venvs' programs.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import signal
import socket
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from setup import API, BUILD_INFO, REPO, WEB, read_env, venv_python  # noqa: E402

LOGS = REPO / "logs"


def port_free(host: str, port: int) -> bool:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        try:
            s.bind(("127.0.0.1" if host in ("0.0.0.0", "") else host, port))
            return True
        except OSError:
            return False


def next_bin() -> Path:
    for base in (WEB, REPO):  # npm workspaces hoist next to the root
        candidate = base / "node_modules" / "next" / "dist" / "bin" / "next"
        if candidate.exists():
            return candidate
    sys.exit("Next.js isn't installed: run deploy\\setup.py first.")


class Service:
    def __init__(self, name: str, cmd: list[str], cwd: Path, env: dict[str, str], health: str) -> None:
        self.name, self.cmd, self.cwd, self.env, self.health = name, cmd, cwd, env, health
        self.proc: subprocess.Popen | None = None
        self.log_path = LOGS / f"{name}.log"
        self.restarts = 0
        self.next_start = 0.0

    def start(self) -> None:
        log = open(self.log_path, "ab")
        log.write(f"\n--- {time.strftime('%Y-%m-%d %H:%M:%S')} starting: {' '.join(self.cmd)}\n".encode())
        log.flush()
        flags = subprocess.CREATE_NEW_PROCESS_GROUP if os.name == "nt" else 0
        self.proc = subprocess.Popen(self.cmd, cwd=self.cwd, env=self.env, stdout=log,
                                     stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL, creationflags=flags)
        log.close()  # the child keeps its own handle

    def ready(self) -> bool:
        try:
            with urllib.request.urlopen(self.health, timeout=3) as r:
                return r.status < 500
        except Exception:  # noqa: BLE001
            return False

    def stop(self) -> None:
        if self.proc and self.proc.poll() is None:
            self.proc.terminate()
            try:
                self.proc.wait(timeout=10)
            except subprocess.TimeoutExpired:
                self.proc.kill()


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--host", default="127.0.0.1",
                    help="where the web app listens (0.0.0.0: other machines too; mind the firewall)")
    ap.add_argument("--port", type=int, default=3000)
    ap.add_argument("--api-port", type=int, default=8000)
    args = ap.parse_args(argv)

    api_python = venv_python(API)
    if not api_python.exists():
        sys.exit("The API isn't installed: run deploy\\setup.py first.")
    if not (WEB / ".next" / "BUILD_ID").exists():
        sys.exit("The web app isn't built: run deploy\\setup.py first.")
    node = shutil.which("node")
    if not node:
        sys.exit("Node.js isn't on PATH.")

    env = {**read_env(), **os.environ}  # .env, then real environment variables win
    env.setdefault("PUBLIC_API_URL", "/api")
    env["PYTHONIOENCODING"] = "utf-8"
    built = json.loads(BUILD_INFO.read_text(encoding="utf-8")) if BUILD_INFO.exists() else {}
    if built.get("NEXT_PUBLIC_API_URL") and built["NEXT_PUBLIC_API_URL"] != env.get("NEXT_PUBLIC_API_URL", "/api"):
        print(f"note: the web app was built with NEXT_PUBLIC_API_URL={built['NEXT_PUBLIC_API_URL']}, but .env "
              f"says {env.get('NEXT_PUBLIC_API_URL')}; re-run deploy\\setup.py --skip-workers --skip-models to rebuild")
    if env["PUBLIC_API_URL"] != "/api":
        print(f"note: PUBLIC_API_URL={env['PUBLIC_API_URL']}; links work only where that address is reachable "
              "(/api works at any address)")

    for name, host, port in (("API", "127.0.0.1", args.api_port), ("web", args.host, args.port)):
        if not port_free(host, port):
            sys.exit(f"Port {port} ({name}) is in use. Is the site already running, or a dev server?")

    LOGS.mkdir(exist_ok=True)
    services = [
        Service("api", [str(api_python), "-m", "uvicorn", "app.main:app", "--host", "127.0.0.1",
                        "--port", str(args.api_port), "--timeout-graceful-shutdown", "5"],
                API, env, f"http://127.0.0.1:{args.api_port}/health"),
        Service("web", [node, str(next_bin()), "start", "--port", str(args.port), "--hostname", args.host],
                WEB, {**env, "PORT": str(args.port)}, f"http://127.0.0.1:{args.port}/"),
    ]
    stopping = False

    def stop(*_args: object) -> None:
        nonlocal stopping
        stopping = True

    signal.signal(signal.SIGINT, stop)
    signal.signal(signal.SIGTERM, stop)
    for s in services:
        s.start()
    print(f"Starting the API and the web app (logs in {LOGS})...", flush=True)
    announced = False
    try:
        while not stopping:
            time.sleep(1)
            if not announced and all(s.ready() for s in services):
                announced = True
                where = "this PC" if args.host == "127.0.0.1" else "this PC and the network"
                print(f"MemoryScapes is up on {where}: http://localhost:{args.port}  (Ctrl+C to stop)", flush=True)
            for s in services:
                code = s.proc.poll() if s.proc else 0
                if code is None or stopping:
                    continue
                now = time.time()
                if s.next_start == 0.0:  # just noticed: schedule a restart, backing off
                    delay = min(60, 2 ** min(s.restarts, 6))
                    print(f"{s.name} stopped (exit code {code}); restarting in {delay}s. See {s.log_path}", flush=True)
                    s.next_start = now + delay
                elif now >= s.next_start:
                    s.restarts += 1
                    s.next_start = 0.0
                    s.start()
                    announced = False
    finally:
        print("Stopping...", flush=True)
        for s in services:
            s.stop()
    return 0


if __name__ == "__main__":
    sys.exit(main())
