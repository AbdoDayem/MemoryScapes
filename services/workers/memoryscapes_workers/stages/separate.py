"""Audio source separation and labeling.

Inputs (from ingest):
    ingest/audio.wav    original recording, all channels, original sample rate
    ingest/meta.json    {"mic": {"channels", "layout", "spacing_m"?}}

Outputs:
    separate/stems/<id>.wav   one stem per detected sound, same channel count as the input
    separate/ambient.wav      everything not assigned to a stem (plus diffuse classes)
    separate/stems.json       StemsIndex: labels, levels, activity envelopes

Steps:
1. Tag the mono mix over sliding windows (CLAP) and keep classes that clear a threshold.
2. Ask the separator for a mono estimate per kept class (AudioSep; Demucs for speech).
3. Turn the estimates into masks that partition the mix, and apply them to every
   channel. Stems + ambient == recording, and inter-channel cues survive for DOA.
4. Drop stems that are too quiet (their energy goes back to the ambient bed).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from memoryscapes_workers.audio import dsp
from memoryscapes_workers.audio.separation import Separator
from memoryscapes_workers.audio.tagging import Tagger
from memoryscapes_workers.audio.vocabulary import VOCABULARY, SoundClass
from memoryscapes_workers.context import StageContext
from memoryscapes_workers.models import MicInfo, StemInfo, StemsIndex
from memoryscapes_workers.storage import read_json, read_wav, write_json, write_wav


@dataclass(frozen=True)
class SeparationConfig:
    tag_threshold: float = 0.2  # min per-window score for a class to count as present
    max_stems: int = 8
    min_stem_dbfs: float = -55.0
    nfft: int = 2048
    hop: int = 512
    mask_power: float = 2.0
    activity_hop_s: float = 0.25


def select_classes(clip_scores: dict[str, float], cfg: SeparationConfig) -> list[SoundClass]:
    present = [c for c in VOCABULARY if clip_scores.get(c.label, 0.0) >= cfg.tag_threshold and not c.diffuse]
    present.sort(key=lambda c: clip_scores[c.label], reverse=True)
    return present[: cfg.max_stems]


def run(
    ctx: StageContext,
    tagger: Tagger | None = None,
    separator: Separator | None = None,
    cfg: SeparationConfig = SeparationConfig(),
) -> StemsIndex:
    from memoryscapes_workers import backends

    tagger = tagger or backends.get_tagger()
    separator = separator or backends.get_separator()
    storage = ctx.storage

    audio, sr = read_wav(storage, ctx.key("ingest", "audio.wav"))  # (C, N)
    mic = MicInfo.from_json(read_json(storage, ctx.key("ingest", "meta.json"))["mic"])
    n = audio.shape[-1]
    mono = dsp.to_mono(audio)

    tags = tagger.tag(mono, sr, VOCABULARY)
    clip_scores = tags.clip_scores()
    classes = select_classes(clip_scores, cfg)

    estimates = separator.separate(mono, sr, classes) if classes else {}
    classes = [c for c in classes if c.label in estimates]

    mix_mag = np.abs(dsp.stft(mono, cfg.nfft, cfg.hop))
    if classes:
        stem_mags = np.stack([np.abs(dsp.stft(dsp.fit_length(estimates[c.label], n), cfg.nfft, cfg.hop)) for c in classes])
    else:
        stem_mags = np.zeros((0, *mix_mag.shape))
    masks, residual = dsp.consistent_masks(mix_mag, stem_mags, cfg.mask_power)
    stems, ambient = dsp.apply_masks(audio, masks, residual, cfg.nfft, cfg.hop)

    window_times: list[float] | None = None
    infos: list[StemInfo] = []
    for i, c in enumerate(classes):
        stem = stems[i]
        level = dsp.dbfs(stem)
        if level < cfg.min_stem_dbfs:
            ambient = ambient + stem
            continue
        times, env = dsp.activity_envelope(dsp.to_mono(stem), sr, hop_s=cfg.activity_hop_s)
        window_times = window_times or times.tolist()
        stem_id = f"s{len(infos) + 1}_{c.label}"
        key = ctx.key("separate", f"stems/{stem_id}.wav")
        write_wav(storage, key, stem, sr)
        infos.append(
            StemInfo(
                id=stem_id,
                label=c.label,
                kind=c.kind,
                key=key,
                level_dbfs=round(level, 2),
                activity=np.round(env, 4).tolist(),
                tag_score=round(clip_scores[c.label], 4),
            )
        )

    if window_times is None:
        window_times = dsp.activity_envelope(mono, sr, hop_s=cfg.activity_hop_s)[0].tolist()

    ambient_key = ctx.key("separate", "ambient.wav")
    write_wav(storage, ambient_key, ambient, sr)

    index = StemsIndex(
        sample_rate=sr,
        channels=audio.shape[0],
        layout=mic.layout,
        duration_s=n / sr,
        window_times=window_times,
        stems=infos,
        ambient_key=ambient_key,
        tags={k: round(v, 4) for k, v in clip_scores.items()},
    )
    write_json(storage, ctx.key("separate", "stems.json"), index.to_json())
    return index
