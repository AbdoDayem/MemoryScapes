"""Audio source separation and labeling.

Inputs (from ingest):
    ingest/audio.wav    original recording, all channels, original sample rate
    ingest/meta.json    {"mic": {"channels", "layout", "spacing_m"?}}

Outputs:
    separate/stems/<id>.wav   one stem per detected sound, same channel count as the input
    separate/ambient.wav      everything not assigned to a stem (plus diffuse classes)
    separate/stems.json       StemsIndex: labels, levels, activity envelopes

Steps:
0. Clean the recording (audio/denoise.py): a 60 Hz high-pass, then, for two-channel
   recordings, wind suppression with the same gain on both channels. Everything below
   works on the cleaned recording. AUDIO_CLEAN=false turns it off.
1. Tag the mono mix over sliding windows (CLAP) and keep classes that clear a threshold.
2. Ask the separator for a mono estimate per kept class (AudioSep; Demucs for speech).
3. Turn the estimates into masks that partition the mix, and apply them to every
   channel. Stems + ambient == the cleaned recording, and inter-channel cues survive for DOA.
4. Drop stems that are too quiet (their energy goes back to the ambient bed).
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field

import numpy as np

from memoryscapes_workers.audio import dsp
from memoryscapes_workers.audio.denoise import CleaningConfig, clean_recording
from memoryscapes_workers.audio.separation import Separator
from memoryscapes_workers.audio.tagging import Tagger, TagResult
from memoryscapes_workers.audio.vocabulary import VOCABULARY, SoundClass
from memoryscapes_workers.context import StageContext
from memoryscapes_workers.models import MicInfo, StemInfo, StemsIndex
from memoryscapes_workers.storage import read_json, read_wav, write_json, write_wav


@dataclass(frozen=True)
class SeparationConfig:
    # A class is present if it clears tag_threshold in at least two windows, or
    # tag_strong in one (a single loud event, like one dog bark). One noisy window
    # just above the threshold is not enough.
    tag_threshold: float = 0.2
    tag_strong: float = 0.35
    max_stems: int = 8
    # A stem is dropped as too quiet when it's more than min_stem_below_mix_db under the
    # recording's own level, or under min_stem_dbfs at all. Relative, because phone clips
    # (at night especially) are quiet overall: a fixed -55 dBFS cut a person's voice at
    # -58 dBFS from a clip whose mix was about -38 dBFS.
    min_stem_below_mix_db: float = 30.0
    min_stem_dbfs: float = -70.0
    # A separator asked for a sound that isn't there returns leakage, which looks like
    # the other stems combined. Real stems stay well below this even with heavy leakage.
    max_redundancy: float = 0.9
    nfft: int = 2048
    hop: int = 512
    mask_power: float = 2.0
    activity_hop_s: float = 0.25
    # None: on unless AUDIO_CLEAN=false. The cleaning itself: audio/denoise.py.
    clean: bool | None = None
    cleaning: CleaningConfig = field(default_factory=CleaningConfig)


def is_present(window_scores: np.ndarray, cfg: SeparationConfig) -> bool:
    s = np.sort(np.asarray(window_scores, float))[::-1]
    if len(s) == 0:
        return False
    if len(s) == 1:
        return bool(s[0] >= cfg.tag_threshold)
    return bool(s[0] >= cfg.tag_strong or s[1] >= cfg.tag_threshold)


def select_classes(tags: TagResult, cfg: SeparationConfig) -> list[SoundClass]:
    clip = tags.clip_scores()
    present = [c for c in VOCABULARY if not c.diffuse and c.label in tags.scores and is_present(tags.scores[c.label], cfg)]
    present.sort(key=lambda c: clip[c.label], reverse=True)
    # Voices first: the people in a memory matter more than any other sound, and on real
    # recordings CLAP scores "people speaking" well below broadband prompts (construction,
    # train, airplane), which would otherwise take every slot.
    return ([c for c in present if c.kind == "speech"] + [c for c in present if c.kind != "speech"])[: cfg.max_stems]


def find_redundant(stem_mags: np.ndarray, tag_scores: list, max_similarity: float) -> dict[int, float]:
    """Stems whose log spectrogram is nearly the same as the sum of the other kept stems.

    Similarity is Pearson correlation (mean removed), so two unrelated dense,
    noise-like stems (water and traffic, say) don't look alike just for being positive.
    Checked weakest-tag first, so of two look-alike stems the better-supported one stays.
    ``tag_scores`` are sortable per stem: run() passes (clip score, mean window score), so
    classes tied at a saturated clip score are told apart by how much of the clip they
    fill. Returns {stem index: similarity} for the stems to drop.
    """
    k = len(stem_mags)
    if k < 2:
        return {}
    logs = np.log1p(100.0 * stem_mags).reshape(k, -1)
    kept = set(range(k))
    out: dict[int, float] = {}
    for i in sorted(range(k), key=lambda j: tag_scores[j]):
        others = [j for j in kept if j != i]
        if not others:
            break
        a = logs[i] - logs[i].mean()
        b = np.log1p(100.0 * stem_mags[others].sum(axis=0)).ravel()
        b = b - b.mean()
        sim = float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-12))
        if sim >= max_similarity:
            out[i] = sim
            kept.discard(i)
    return out


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
    clean = cfg.clean if cfg.clean is not None else os.environ.get("AUDIO_CLEAN", "true").lower() != "false"
    if clean:
        audio, cleaning = clean_recording(audio, sr, cfg.cleaning)
    else:
        cleaning = {"wind": "off", "highpass_hz": None}
    n = audio.shape[-1]
    mono = dsp.to_mono(audio)

    tags = tagger.tag(mono, sr, VOCABULARY)
    clip_scores = tags.clip_scores()
    classes = select_classes(tags, cfg)

    estimates = separator.separate(mono, sr, classes) if classes else {}
    classes = [c for c in classes if c.label in estimates]

    mix_mag = np.abs(dsp.stft(mono, cfg.nfft, cfg.hop))
    if classes:
        stem_mags = np.stack([np.abs(dsp.stft(dsp.fit_length(estimates[c.label], n), cfg.nfft, cfg.hop)) for c in classes])
    else:
        stem_mags = np.zeros((0, *mix_mag.shape))
    masks, residual = dsp.consistent_masks(mix_mag, stem_mags, cfg.mask_power)
    stems, ambient = dsp.apply_masks(audio, masks, residual, cfg.nfft, cfg.hop)
    # Real recordings saturate several broadband classes at 1.0 (train, construction,
    # airplane on a station platform); by clip score alone the real train was checked first
    # and dropped as the others combined. Ties go to the class present for more of the clip.
    support = [(clip_scores[c.label], float(np.mean(tags.scores[c.label]))) for c in classes]
    redundant = find_redundant(masks * mix_mag[None], support, cfg.max_redundancy)

    window_times: list[float] | None = None
    infos: list[StemInfo] = []
    dropped: list[dict] = []
    mix_level = dsp.dbfs(audio)
    for i, c in enumerate(classes):
        stem = stems[i]
        level = dsp.dbfs(stem)
        reason = None
        if i in redundant:
            reason = f"looks like the other stems combined (similarity {redundant[i]:.2f}): likely a false tag"
        elif level < max(cfg.min_stem_dbfs, mix_level - cfg.min_stem_below_mix_db):
            reason = f"too quiet ({level:.1f} dBFS, {mix_level - level:.0f} dB under the recording)"
        if reason:
            ambient = ambient + stem
            dropped.append({"label": c.label, "reason": reason})
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
        dropped=dropped,
        cleaning=cleaning,
    )
    write_json(storage, ctx.key("separate", "stems.json"), index.to_json())
    return index
