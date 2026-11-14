"""Store a memory's files in the database (``memory_files``), and get them back.

A memory, as the viewer needs it, is ``publish/manifest.json`` plus every file the manifest
names: the splat levels, the walk path, the sound stems and ambient bed, and the SATIE script
and samples. ``store(id)`` copies exactly those, with their bytes, sizes and SHA-256, and runs
when a pipeline run finishes ready. Files whose checksum hasn't changed aren't rewritten, and
files the manifest no longer names are removed. Pipeline working files (the upload, frames,
training output) aren't part of a memory and stay out.

The files also stay in storage, where the API serves them from, so the frontend is unchanged.

    python -m app.memory_files store canada [--register]   # copy into the database
    python -m app.memory_files verify canada               # database == storage?
    python -m app.memory_files restore canada --to DIR     # write the files back out
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Any

from app import db
from app.storage import BlobStore, get_store

MANIFEST = "publish/manifest.json"
_CONTENT_TYPES = {".spz": "application/octet-stream", ".ply": "application/octet-stream",
                  ".json": "application/json", ".wav": "audio/wav", ".mp3": "audio/mpeg",
                  ".ogg": "audio/ogg", ".satie": "text/plain; charset=utf-8",
                  ".jpg": "image/jpeg", ".png": "image/png", ".webp": "image/webp"}


def content_type(key: str) -> str:
    return _CONTENT_TYPES.get(Path(key).suffix.lower(), "application/octet-stream")


def _is_key(value: Any) -> bool:
    """A path inside the memory (the manifest's relative URLs), not a web address."""
    return (isinstance(value, str) and value and "://" not in value and not value.startswith("/")
            and ".." not in value.split("/") and "." in value.rsplit("/", 1)[-1])


def referenced_files(manifest: dict[str, Any]) -> dict[str, str]:
    """Every file the manifest names -> its kind."""
    files: dict[str, str] = {}

    def add(key: Any, kind: str) -> None:
        if _is_key(key):
            files.setdefault(key, kind)

    for lod in (manifest.get("splat") or {}).get("lods") or []:
        add(lod.get("url"), "splat")
    add((manifest.get("navigation") or {}).get("original_path"), "path")
    audio = manifest.get("audio") or {}
    for source in audio.get("sources") or []:
        add(source.get("url"), "stem")
    add((audio.get("ambient_bed") or {}).get("url"), "ambient")
    satie = audio.get("satie") or {}
    add(satie.get("script_url"), "satie_script")
    add(satie.get("samples_url"), "satie_samples")

    def walk(node: Any) -> None:  # anything else the manifest names (thumbnails, future fields)
        if isinstance(node, dict):
            for name, value in node.items():
                if isinstance(value, str) and (name == "url" or name.endswith(("_url", "_path"))):
                    add(value, "other")
                else:
                    walk(value)
        elif isinstance(node, list):
            for item in node:
                walk(item)

    walk(manifest)
    return files


def store(memory_id: str, blob_store: BlobStore | None = None) -> dict[str, Any]:
    """Copy the memory's files into the database. Needs the memory's database row and its
    manifest. Returns what happened (files, stored, unchanged, removed, missing, bytes)."""
    blobs = blob_store or get_store()
    manifest_bytes = blobs.read_bytes(f"{memory_id}/{MANIFEST}")
    wanted = {MANIFEST: "manifest", **referenced_files(json.loads(manifest_bytes))}
    existing = {f["key"]: f["sha256"] for f in db.list_files(memory_id)}
    summary: dict[str, Any] = {"files": 0, "stored": 0, "unchanged": 0, "missing": [], "bytes": 0}
    for key, kind in wanted.items():
        path = f"{memory_id}/{key}"
        if key != MANIFEST and not blobs.exists(path):
            summary["missing"].append(key)
            continue
        data = manifest_bytes if key == MANIFEST else blobs.read_bytes(path)
        sha = hashlib.sha256(data).hexdigest()
        summary["files"] += 1
        summary["bytes"] += len(data)
        if existing.get(key) == sha:
            summary["unchanged"] += 1
            continue
        db.put_file(memory_id, key, kind, content_type(key), data, sha)
        summary["stored"] += 1
    summary["removed"] = db.delete_files(memory_id, [k for k in existing if k not in wanted])
    return summary


def verify(memory_id: str, blob_store: BlobStore | None = None) -> list[str]:
    """Files whose stored copy differs from storage (empty: they match)."""
    blobs = blob_store or get_store()
    problems = []
    for info in db.list_files(memory_id):
        path = f"{memory_id}/{info['key']}"
        if not blobs.exists(path):
            problems.append(f"{info['key']}: in the database, not in storage")
        elif hashlib.sha256(blobs.read_bytes(path)).hexdigest() != info["sha256"]:
            problems.append(f"{info['key']}: differs from storage")
    return problems


def restore(memory_id: str, dest: Path) -> list[Path]:
    """Write the memory's stored files under ``dest/<memory_id>/``, checking each checksum."""
    written = []
    for info in db.list_files(memory_id):
        row = db.get_file(memory_id, info["key"])
        if hashlib.sha256(row["data"]).hexdigest() != row["sha256"]:
            raise ValueError(f"{info['key']}: stored bytes don't match their checksum")
        target = dest / memory_id / info["key"]
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(row["data"])
        written.append(target)
    return written


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="python -m app.memory_files", description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("action", choices=["store", "verify", "restore"])
    ap.add_argument("memory_id")
    ap.add_argument("--register", action="store_true",
                    help="store: create the database row for a memory made with the worker CLI")
    ap.add_argument("--to", type=Path, help="restore: the folder to write into")
    args = ap.parse_args(argv)
    db.init_db()
    if db.get_memory(args.memory_id) is None:
        if not (args.action == "store" and args.register):
            print(f"{args.memory_id}: no database row (made with the worker CLI?); use store --register")
            return 1
        from app.soundscapes import register

        register(args.memory_id)
    if args.action == "store":
        s = store(args.memory_id)
        print(f"{args.memory_id}: {s['files']} files, {s['bytes'] / 1e6:.1f} MB in the database "
              f"({s['stored']} written, {s['unchanged']} unchanged, {s['removed']} removed)")
        if s["missing"]:
            print("  named by the manifest but not in storage: " + ", ".join(s["missing"]))
        return 1 if s["missing"] else 0
    if args.action == "verify":
        problems = verify(args.memory_id)
        print(f"{args.memory_id}: " + ("database matches storage" if not problems else "; ".join(problems)))
        return 1 if problems else 0
    if args.to is None:
        ap.error("restore needs --to")
    paths = restore(args.memory_id, args.to)
    print(f"{args.memory_id}: restored {len(paths)} files under {args.to / args.memory_id}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
