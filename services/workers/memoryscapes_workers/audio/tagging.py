"""Sound-event tagging: which vocabulary classes are present, and when."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Protocol

import numpy as np

from memoryscapes_workers.audio.dsp import resample
from memoryscapes_workers.audio.vocabulary import SoundClass


@dataclass
class TagResult:
    window_times: np.ndarray  # (W,) window centres in seconds
    scores: dict[str, np.ndarray]  # label -> (W,) score in 0..1

    def clip_scores(self) -> dict[str, float]:
        return {label: float(s.max()) if len(s) else 0.0 for label, s in self.scores.items()}


class Tagger(Protocol):
    def tag(self, mono: np.ndarray, sr: int, classes: Sequence[SoundClass]) -> TagResult: ...


def cosine_scores(cos: np.ndarray, floor: float = 0.05, ceil: float = 0.30) -> np.ndarray:
    """Multi-label presence from CLAP audio-text cosine similarity, mapped to 0..1.

    Each class is scored on its own, so several sounds can be present at once. A softmax
    across classes would make them compete: one class (often a broadband one like wind)
    takes all the mass and every other sound in the recording falls under the threshold.
    The floor/ceil defaults come from laion/clap-htsat-unfused on our example clips,
    where present classes scored about 0.15-0.3 and absent ones about 0.1 or less.
    Calibrate on real recordings.
    """
    return np.clip((cos - floor) / (ceil - floor), 0.0, 1.0)


class ClapTagger:
    """Zero-shot tagging with CLAP (Hugging Face transformers), over sliding windows.

    Requires the `audio` extra (torch + transformers). The model loads on first use.
    """

    SAMPLE_RATE = 48_000

    def __init__(
        self,
        model_id: str = "laion/clap-htsat-unfused",
        device: str | None = None,
        window_s: float = 5.0,
        hop_s: float = 2.5,
        batch_size: int = 16,
        cos_floor: float = 0.05,
        cos_ceil: float = 0.30,
    ) -> None:
        self.model_id = model_id
        self.device = device
        self.window_s = window_s
        self.hop_s = hop_s
        self.batch_size = batch_size
        self.cos_floor = cos_floor
        self.cos_ceil = cos_ceil
        self._model = None
        self._processor = None

    def _load(self):
        if self._model is None:
            import torch
            from transformers import ClapModel, ClapProcessor

            self.device = self.device or ("cuda" if torch.cuda.is_available() else "cpu")
            self._processor = ClapProcessor.from_pretrained(self.model_id)
            self._model = ClapModel.from_pretrained(self.model_id).to(self.device).eval()
        return self._model, self._processor

    def tag(self, mono: np.ndarray, sr: int, classes: Sequence[SoundClass]) -> TagResult:
        import torch

        model, processor = self._load()
        audio = resample(mono, sr, self.SAMPLE_RATE)
        win, hop = int(self.window_s * self.SAMPLE_RATE), int(self.hop_s * self.SAMPLE_RATE)
        starts = list(range(0, max(1, len(audio) - win + 1), hop))
        windows = [audio[s : s + win] for s in starts]
        times = np.array([(s + win / 2) / self.SAMPLE_RATE for s in starts])

        with torch.no_grad():
            text = processor.tokenizer([c.clap_prompt for c in classes], padding=True, return_tensors="pt")
            t_emb = model.get_text_features(**{k: v.to(self.device) for k, v in text.items()})
            t_emb = torch.nn.functional.normalize(t_emb, dim=-1)
            sims = []
            for i in range(0, len(windows), self.batch_size):
                feats = processor.feature_extractor(
                    windows[i : i + self.batch_size], sampling_rate=self.SAMPLE_RATE, return_tensors="pt"
                )
                a_emb = model.get_audio_features(**{k: v.to(self.device) for k, v in feats.items()})
                a_emb = torch.nn.functional.normalize(a_emb, dim=-1)
                sims.append((a_emb @ t_emb.T).cpu().numpy())
        p = cosine_scores(np.concatenate(sims, axis=0), self.cos_floor, self.cos_ceil)  # (W, K)
        return TagResult(times, {c.label: p[:, k] for k, c in enumerate(classes)})
