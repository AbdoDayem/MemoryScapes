"""Source separation backends.

A Separator returns one mono estimate per requested class, at the input sample rate and
length. The separate stage then turns those estimates into consistent multichannel
masks (see dsp.consistent_masks), so backends don't need to be perfect or disjoint.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Protocol

import numpy as np

from memoryscapes_workers.audio.dsp import fit_length, process_in_chunks, resample
from memoryscapes_workers.audio.vocabulary import SoundClass


def _checkpoint_safe_globals() -> list:
    """NumPy types AudioSep's Lightning checkpoints pickle alongside the weights.

    torch >= 2.6 loads checkpoints with weights_only=True, which refuses them. Allowing
    just these types keeps the restricted unpickler on, instead of weights_only=False,
    which would run arbitrary code from the file.
    """
    import numpy as np

    out: list = [np.dtype]
    try:
        from numpy._core.multiarray import scalar
    except ImportError:  # numpy < 2
        from numpy.core.multiarray import scalar
    # The checkpoints were pickled with numpy 1.x, so they name it numpy.core...; numpy 2
    # moved it to numpy._core. Register it under both names.
    out += [scalar, (scalar, "numpy.core.multiarray.scalar")]
    for t in ("float64", "float32", "int64", "int32", "bool"):
        cls = type(np.dtype(t))
        out += [cls, (cls, f"numpy.dtype[{t}]")]
    return out


def _drop_stale_clap_keys() -> None:
    """transformers >= 4.31 no longer stores RoBERTa's position_ids buffer, so AudioSep's
    CLAP checkpoint carries one key the model doesn't have. Strip it when AudioSep loads
    the checkpoint, without editing the AudioSep checkout."""
    import models.CLAP.open_clip.factory as factory  # AudioSep's vendored open_clip

    if getattr(factory.load_state_dict, "_memoryscapes_patched", False):
        return
    original = factory.load_state_dict

    def load_state_dict(*args, **kwargs):
        state = original(*args, **kwargs)
        state.pop("text_branch.embeddings.position_ids", None)
        return state

    load_state_dict._memoryscapes_patched = True  # type: ignore[attr-defined]
    factory.load_state_dict = load_state_dict


class Separator(Protocol):
    def separate(self, mono: np.ndarray, sr: int, classes: Sequence[SoundClass]) -> dict[str, np.ndarray]: ...


class AudioSepSeparator:
    """Text-queried separation with AudioSep (https://github.com/Audio-AGI/AudioSep).

    AudioSep isn't on PyPI: clone the repo, download its checkpoints, and put the repo
    root on PYTHONPATH. The calls below follow the repo's `pipeline.py` inference code;
    check them against the version you install.
    """

    SAMPLE_RATE = 32_000

    def __init__(
        self,
        config_yaml: str = "config/audiosep_base.yaml",
        checkpoint_path: str = "checkpoint/audiosep_base_4M_steps.ckpt",
        device: str | None = None,
        chunk_s: float = 10.0,
        overlap_s: float = 1.0,
        repo_dir: str | None = None,
    ) -> None:
        self.config_yaml = config_yaml
        self.checkpoint_path = checkpoint_path
        self.device = device
        self.chunk_s = chunk_s
        self.overlap_s = overlap_s
        self.repo_dir = repo_dir
        self._model = None

    def _load(self):
        if self._model is None:
            import contextlib
            import os
            import sys

            import torch

            # AudioSep resolves its config and its CLAP text-encoder checkpoint relative
            # to the repo root, so import and build from inside it.
            repo = self.repo_dir
            cwd = contextlib.chdir(repo) if repo else contextlib.nullcontext()
            if repo and repo not in sys.path:
                sys.path.insert(0, repo)
            with cwd, torch.serialization.safe_globals(_checkpoint_safe_globals()):
                # before importing pipeline: models/audiosep.py builds a CLAP encoder as a
                # default argument, i.e. at import time
                _drop_stale_clap_keys()
                from pipeline import build_audiosep  # AudioSep repo root

                self.device = self.device or ("cuda" if torch.cuda.is_available() else "cpu")
                self._model = build_audiosep(
                    config_yaml=self.config_yaml, checkpoint_path=self.checkpoint_path, device=self.device
                )
            os.environ.setdefault("AUDIOSEP_LOADED_FROM", repo or os.getcwd())
        return self._model

    def separate(self, mono: np.ndarray, sr: int, classes: Sequence[SoundClass]) -> dict[str, np.ndarray]:
        import torch

        model = self._load()
        audio = resample(mono, sr, self.SAMPLE_RATE)
        out: dict[str, np.ndarray] = {}
        for c in classes:
            with torch.no_grad():
                condition = model.query_encoder.get_query_embed(modality="text", text=[c.query], device=self.device)

                def run(seg: np.ndarray) -> np.ndarray:
                    mixture = torch.tensor(seg, dtype=torch.float32)[None, None, :].to(self.device)
                    wav = model.ss_model({"mixture": mixture, "condition": condition})["waveform"]
                    return wav.squeeze().cpu().numpy()

                est = process_in_chunks(
                    audio, int(self.chunk_s * self.SAMPLE_RATE), int(self.overlap_s * self.SAMPLE_RATE), run
                )
            out[c.label] = fit_length(resample(est, self.SAMPLE_RATE, sr), len(mono))
        return out


class DemucsVocalsSeparator:
    """Speech via HTDemucs' vocals stem. Better than a text query for voices.

    Requires `demucs` (in the `audio` extra).
    """

    def __init__(self, model_name: str = "htdemucs", device: str | None = None) -> None:
        self.model_name = model_name
        self.device = device
        self._model = None

    def _load(self):
        if self._model is None:
            import torch
            from demucs.pretrained import get_model

            self.device = self.device or ("cuda" if torch.cuda.is_available() else "cpu")
            self._model = get_model(self.model_name).to(self.device).eval()
        return self._model

    def separate(self, mono: np.ndarray, sr: int, classes: Sequence[SoundClass]) -> dict[str, np.ndarray]:
        import torch
        from demucs.apply import apply_model

        model = self._load()
        audio = resample(mono, sr, model.samplerate)
        wav = torch.tensor(np.stack([audio] * model.audio_channels), dtype=torch.float32)
        with torch.no_grad():
            sources = apply_model(model, wav[None], device=self.device, split=True, overlap=0.25)[0]
        vocals = sources[model.sources.index("vocals")].mean(dim=0).cpu().numpy()
        vocals = fit_length(resample(vocals, model.samplerate, sr), len(mono))
        return {c.label: vocals for c in classes}


class RoutingSeparator:
    """Sends each class to a backend by kind (e.g. speech -> Demucs, rest -> AudioSep)."""

    def __init__(self, default: Separator, by_kind: dict[str, Separator] | None = None) -> None:
        self.default = default
        self.by_kind = by_kind or {}

    def separate(self, mono: np.ndarray, sr: int, classes: Sequence[SoundClass]) -> dict[str, np.ndarray]:
        groups: dict[int, tuple[Separator, list[SoundClass]]] = {}
        for c in classes:
            backend = self.by_kind.get(c.kind, self.default)
            groups.setdefault(id(backend), (backend, []))[1].append(c)
        out: dict[str, np.ndarray] = {}
        for backend, group in groups.values():
            out.update(backend.separate(mono, sr, group))
        return out
