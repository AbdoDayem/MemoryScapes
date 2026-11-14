"""Install the models the audio stages use, the same way on Windows, macOS and Linux.

    memoryscapes-worker setup-models            # download what's missing (~5.6 GB)
    memoryscapes-worker setup-models --check    # also load each model and run it once

- AudioSep (separation): the code isn't on PyPI, so a pinned commit is unpacked into
  ``<repo>/.models/AudioSep`` (git-ignored; ``AUDIOSEP_DIR`` overrides) and its two
  checkpoints are fetched from the authors' Hugging Face Space, resumable and checked
  against SHA-256.
- CLAP (tagging), Grounding DINO (finding sound sources in frames), HTDemucs (speech)
  and RoBERTa (AudioSep's text encoder): fetched into the Hugging Face cache by their
  own libraries.

Needs the ``audio`` and ``visual`` extras installed first.
"""

from __future__ import annotations

import hashlib
import io
import logging
import os
import shutil
import urllib.request
import zipfile
from pathlib import Path

log = logging.getLogger("memoryscapes.models")

REPO_ROOT = Path(__file__).resolve().parents[4]
DEFAULT_AUDIOSEP_DIR = REPO_ROOT / ".models" / "AudioSep"

AUDIOSEP_COMMIT = "944583f18b84589dc965de3ad77525c945334252"  # main, 2024-11-26
AUDIOSEP_SPACE_REV = "0807a713283035d8b0d0628f1b8e3d6580459f7b"
AUDIOSEP_CHECKPOINTS = {  # name -> (bytes, sha256)
    "audiosep_base_4M_steps.ckpt": (
        1264844076, "f8cda01bfd0ebd141eef45d41db7a3ada23a56568465840d3cff04b8010ce82c"),
    "music_speech_audioset_epoch_15_esc_89.98.pt": (
        2352471003, "51c68f12f9d7ea25fdaaccf741ec7f81e93ee594455410f3bca4f47f88d8e006"),
}
CLAP_MODEL = "laion/clap-htsat-unfused"
# AudioSep's text encoder builds on RoBERTa and fetches it from the Hugging Face hub.
ROBERTA_MODEL = "roberta-base"
GROUNDING_DINO_MODEL = "IDEA-Research/grounding-dino-tiny"
DEMUCS_MODEL = "htdemucs"

_UA = {"User-Agent": "memoryscapes-setup/1.0"}


def audiosep_dir() -> Path:
    return Path(os.environ.get("AUDIOSEP_DIR") or DEFAULT_AUDIOSEP_DIR)


def install_audiosep_code(dest: Path) -> None:
    marker = dest / ".memoryscapes-commit"
    if marker.exists() and marker.read_text(encoding="utf-8").strip() == AUDIOSEP_COMMIT:
        log.info("AudioSep code: up to date (%s)", AUDIOSEP_COMMIT[:7])
        return
    url = f"https://codeload.github.com/Audio-AGI/AudioSep/zip/{AUDIOSEP_COMMIT}"
    log.info("AudioSep code: downloading %s", url)
    with urllib.request.urlopen(urllib.request.Request(url, headers=_UA), timeout=120) as r:
        data = r.read()
    dest.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(io.BytesIO(data)) as z:
        top = z.namelist()[0].split("/")[0]  # AudioSep-<sha>/
        for member in z.infolist():
            rel = member.filename[len(top) + 1:]
            if not rel or member.is_dir():
                continue
            target = (dest / rel).resolve()
            if dest.resolve() not in target.parents:
                raise ValueError(f"unsafe path in archive: {member.filename}")
            target.parent.mkdir(parents=True, exist_ok=True)
            with z.open(member) as src, open(target, "wb") as out:
                shutil.copyfileobj(src, out)
    marker.write_text(AUDIOSEP_COMMIT + "\n", encoding="utf-8")


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 22), b""):
            h.update(block)
    return h.hexdigest()


def download_checked(url: str, dest: Path, size: int, sha256: str) -> None:
    """Resumable download to ``dest`` (via ``dest.part``), verified against size and hash."""
    if dest.exists() and dest.stat().st_size == size:
        log.info("%s: present", dest.name)
        return
    dest.parent.mkdir(parents=True, exist_ok=True)
    part = dest.with_name(dest.name + ".part")
    have = part.stat().st_size if part.exists() else 0
    headers = dict(_UA, **({"Range": f"bytes={have}-"} if have else {}))
    log.info("%s: downloading %.0f MB%s", dest.name, size / 1e6,
             f" (resuming at {have / 1e6:.0f} MB)" if have else "")
    with urllib.request.urlopen(urllib.request.Request(url, headers=headers), timeout=120) as r:
        if have and r.status != 206:  # server ignored the range: start over
            have = 0
        with open(part, "ab" if have else "wb") as out:
            done, next_report = have, have + 250_000_000
            while block := r.read(1 << 22):
                out.write(block)
                done += len(block)
                if done >= next_report:
                    log.info("  %s: %.0f / %.0f MB", dest.name, done / 1e6, size / 1e6)
                    next_report += 250_000_000
    if part.stat().st_size != size:
        raise RuntimeError(f"{dest.name}: got {part.stat().st_size} bytes, expected {size}; re-run to resume")
    if _sha256(part) != sha256:
        part.unlink()
        raise RuntimeError(f"{dest.name}: checksum mismatch; deleted, re-run to download again")
    part.replace(dest)


def install_audiosep(dest: Path) -> None:
    install_audiosep_code(dest)
    for name, (size, sha) in AUDIOSEP_CHECKPOINTS.items():
        url = f"https://huggingface.co/spaces/Audio-AGI/AudioSep/resolve/{AUDIOSEP_SPACE_REV}/checkpoint/{name}"
        download_checked(url, dest / "checkpoint" / name, size, sha)


def prefetch_library_models() -> None:
    from transformers import (AutoModelForZeroShotObjectDetection, AutoProcessor, AutoTokenizer,
                              ClapModel, ClapProcessor, RobertaModel)

    for load, model_id in ((ClapProcessor, CLAP_MODEL), (ClapModel, CLAP_MODEL),
                           (AutoTokenizer, ROBERTA_MODEL), (RobertaModel, ROBERTA_MODEL),
                           (AutoProcessor, GROUNDING_DINO_MODEL),
                           (AutoModelForZeroShotObjectDetection, GROUNDING_DINO_MODEL)):
        log.info("%s: %s", model_id, load.__name__)
        load.from_pretrained(model_id)
    from demucs.pretrained import get_model

    log.info("demucs: %s", DEMUCS_MODEL)
    get_model(DEMUCS_MODEL)


def check_models() -> dict[str, str]:
    """Load every backend and run it once on a second of noise."""
    import numpy as np

    from memoryscapes_workers import backends
    from memoryscapes_workers.audio.vocabulary import BY_LABEL, VOCABULARY

    rng = np.random.default_rng(0)
    sr = 48_000
    noise = (0.05 * rng.standard_normal(2 * sr)).astype(np.float32)
    results = {}
    tags = backends.get_tagger().tag(noise, sr, VOCABULARY)
    results["tagger"] = f"ok ({len(tags.scores)} classes scored)"
    stems = backends.get_separator().separate(noise, sr, [BY_LABEL["birdsong"], BY_LABEL["speech"]])
    results["separator"] = f"ok ({', '.join(f'{k}: {len(v)} samples' for k, v in stems.items())})"
    detector = backends.get_detector()
    if detector is not None:
        buf = io.BytesIO()
        from PIL import Image

        Image.fromarray((rng.uniform(0, 255, (240, 320, 3))).astype(np.uint8)).save(buf, format="JPEG")
        results["detector"] = f"ok ({len(detector.detect(buf.getvalue(), 'bird.'))} boxes on noise)"
    return results


def setup(check: bool = False) -> dict[str, str]:
    install_audiosep(audiosep_dir())
    prefetch_library_models()
    return check_models() if check else {"status": "installed"}
