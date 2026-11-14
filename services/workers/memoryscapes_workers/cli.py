"""Command line for running pipeline stages locally.

    memoryscapes-worker demo                          # synthetic video -> splat, all local
    memoryscapes-worker import mem_1 clip.mp4 [--audio rec.wav]
    memoryscapes-worker run mem_1 --until splat --set splat.trainer=points
    memoryscapes-worker run mem_1 --stages poses --set poses.mapper=incremental --force
    memoryscapes-worker synth data/synthetic

Storage defaults to a local directory (STORAGE_ROOT, default data/storage); set
STORAGE_BACKEND=s3 for MinIO/S3. trainer=gsplat needs a CUDA GPU; trainer=points runs anywhere.

Each stage runs in its own process (``--in-process`` to debug): pycolmap (poses) and torch
(splat) bundle different OpenMP runtimes, which abort when loaded into one process on macOS
("OMP: Error #15") and can on Windows too. Production workers run one stage per task anyway.
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import logging
import subprocess
import sys
import tempfile
import traceback
from pathlib import Path
from typing import Any

from memoryscapes_workers.context import StageContext
from memoryscapes_workers.pipeline import VISUAL_STAGES, ancestors, run_local, topological_order
from memoryscapes_workers.stagekit import UPLOAD_PREFIX, StageError
from memoryscapes_workers.storage import Storage, get_storage


def _parse_value(raw: str) -> Any:
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        return raw


def build_config(config_file: Path | None, overrides: list[str]) -> dict:
    """Merge a JSON config file with ``stage.key.sub=value`` overrides."""
    cfg: dict = json.loads(config_file.read_text(encoding="utf-8")) if config_file else {}
    for item in overrides:
        key, _, raw = item.partition("=")
        if not raw:
            raise SystemExit(f"--set expects stage.key=value, got {item!r}")
        node = cfg
        *parents, leaf = key.split(".")
        for p in parents:
            node = node.setdefault(p, {})
        node[leaf] = _parse_value(raw)
    return cfg


def _jsonable(value: Any) -> Any:
    if hasattr(value, "to_json"):
        return value.to_json()
    if dataclasses.is_dataclass(value) and not isinstance(value, type):
        return dataclasses.asdict(value)
    return str(value)


def run_isolated(memory_id: str, stages: list[str], config: dict, force: bool = False,
                 work_root: Path | None = None) -> dict[str, Any]:
    """Run stages in DAG order, each in a fresh ``python -m memoryscapes_workers`` process.

    A plain subprocess (not multiprocessing) so it behaves the same with console-script
    launchers on Windows. Storage comes from the environment; logs stream to stderr.
    """
    results: dict[str, Any] = {}
    with tempfile.TemporaryDirectory(prefix="ms-run-", ignore_cleanup_errors=True) as tmp:
        cfg_path, out_path = Path(tmp) / "config.json", Path(tmp) / "result.json"
        cfg_path.write_text(json.dumps(config), encoding="utf-8")
        for stage in [s for s in topological_order() if s in stages]:
            out_path.unlink(missing_ok=True)
            cmd = [sys.executable, "-m", "memoryscapes_workers"]
            if logging.getLogger().level <= logging.DEBUG:
                cmd.append("-v")
            cmd += ["run", memory_id, "--stages", stage, "--in-process", "--config", str(cfg_path),
                    "--result-file", str(out_path)]
            if force:
                cmd.append("--force")
            if work_root is not None:
                cmd += ["--work-root", str(work_root)]
            code = subprocess.run(cmd, check=False).returncode
            out = json.loads(out_path.read_text(encoding="utf-8")) if out_path.exists() else {}
            if "stage_error" in out:
                raise StageError(**out["stage_error"])
            if "result" not in out:
                raise RuntimeError(f"stage {stage!r} failed (exit code {code}): "
                                   f"{out.get('error', 'the process exited without a result')}")
            results[stage] = out["result"]
    return results


def import_upload(storage: Storage, memory_id: str, video: Path, audio: Path | None = None) -> None:
    """Put local files where the API's presigned uploads would land."""
    prefix = f"{memory_id}/{UPLOAD_PREFIX}/"
    storage.delete_prefix(prefix)
    storage.upload(video, prefix + video.name)
    if audio:
        storage.upload(audio, prefix + audio.name)


def _stages(args) -> list[str]:
    if args.stages:
        return [s.strip() for s in args.stages.split(",") if s.strip()]
    target = args.until or VISUAL_STAGES[-1]
    return [*ancestors(target), target]


def _run_for_parent(args, storage: Storage, config: dict) -> int:
    """Child side of run_isolated: run in-process and report through --result-file."""
    ctx = StageContext(memory_id=args.memory_id, storage=storage, config=config,
                       force=args.force, work_root=args.work_root)
    try:
        out: dict[str, Any] = {"result": run_local(ctx, _stages(args))}
        code = 0
    except StageError as err:
        out, code = {"stage_error": {"stage": err.stage, "message": err.message, "hint": err.hint}}, 2
    except Exception as err:  # noqa: BLE001 - reported by the parent
        traceback.print_exc()
        out, code = {"error": f"{type(err).__name__}: {err}"}, 1
    if "result" in out:
        out["result"] = next(iter(out["result"].values()), None) if len(out["result"]) == 1 \
            else out["result"]
    args.result_file.write_text(json.dumps(out, default=_jsonable), encoding="utf-8")
    return code


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="memoryscapes-worker", description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("-v", "--verbose", action="store_true")
    sub = ap.add_subparsers(dest="cmd", required=True)

    p_imp = sub.add_parser("import", help="copy a local video into a memory's upload area")
    p_imp.add_argument("memory_id")
    p_imp.add_argument("video", type=Path)
    p_imp.add_argument("--audio", type=Path)

    p_run = sub.add_parser("run", help="run pipeline stages")
    p_run.add_argument("memory_id")
    g = p_run.add_mutually_exclusive_group()
    g.add_argument("--stages", help="comma-separated stages to run, e.g. poses,splat")
    g.add_argument("--until", help="run this stage and everything upstream (default: splat)")
    p_run.add_argument("--config", type=Path, help="JSON file: {stage: {option: value}}")
    p_run.add_argument("--set", action="append", default=[], metavar="STAGE.KEY=VALUE")
    p_run.add_argument("--force", action="store_true", help="re-run even if up to date")
    p_run.add_argument("--work-root", type=Path)
    p_run.add_argument("--in-process", action="store_true",
                       help="run all stages in this process (debugging; see the OpenMP note)")
    p_run.add_argument("--result-file", type=Path, help=argparse.SUPPRESS)  # for run_isolated

    p_syn = sub.add_parser("synth", help="render the synthetic test video")
    p_syn.add_argument("out_dir", type=Path)
    p_syn.add_argument("--seconds", type=float, default=10.0)

    p_demo = sub.add_parser("demo", help="synthetic video -> ingest -> poses -> splat (points)")
    p_demo.add_argument("--memory-id", default="demo")
    p_demo.add_argument("--out", type=Path, default=Path("data/synthetic"))
    p_demo.add_argument("--trainer", default="points")

    args = ap.parse_args(argv)
    logging.basicConfig(level=logging.DEBUG if args.verbose else logging.INFO,
                        format="%(asctime)s %(message)s", datefmt="%H:%M:%S")

    if args.cmd == "synth":
        from memoryscapes_workers.devtools.synthetic import SyntheticSpec, generate

        print(generate(args.out_dir, SyntheticSpec(seconds=args.seconds)))
        return 0

    storage = get_storage()
    try:
        if args.cmd == "import":
            import_upload(storage, args.memory_id, args.video, args.audio)
            print(f"imported into {args.memory_id}/{UPLOAD_PREFIX}/")
            return 0
        if args.cmd == "run":
            config = build_config(args.config, args.set)
            if args.result_file:
                return _run_for_parent(args, storage, config)
            if args.in_process:
                ctx = StageContext(memory_id=args.memory_id, storage=storage, config=config,
                                   force=args.force, work_root=args.work_root)
                results = run_local(ctx, _stages(args))
            else:
                results = run_isolated(args.memory_id, _stages(args), config, args.force,
                                       args.work_root)
        else:  # demo
            from memoryscapes_workers.devtools.synthetic import generate

            video = args.out / "video.mp4"
            if not video.exists():
                logging.info("rendering synthetic video into %s (takes ~1-2 min)", args.out)
                generate(args.out)
            import_upload(storage, args.memory_id, video)
            results = run_isolated(args.memory_id, VISUAL_STAGES, {"splat": {"trainer": args.trainer}})
    except StageError as err:
        print(f"\nStage {err.stage!r} failed: {err.message}", file=sys.stderr)
        if err.hint:
            print(f"Hint: {err.hint}", file=sys.stderr)
        return 2
    print(json.dumps(results, indent=2, default=_jsonable))
    return 0


if __name__ == "__main__":
    sys.exit(main())
