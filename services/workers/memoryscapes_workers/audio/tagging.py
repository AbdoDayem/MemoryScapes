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
    ) -> None:
        self.model_id = model_id
        self.device = device
        self.window_s = window_s
        self.hop_s = hop_s
        self.batch_size = batch_size
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
            probs = []
            for i in range(0, len(windows), self.batch_size):
                feats = processor.feature_extractor(
                    windows[i : i + self.batch_size], sampling_rate=self.SAMPLE_RATE, return_tensors="pt"
                )
                a_emb = model.get_audio_features(**{k: v.to(self.device) for k, v in feats.items()})
                a_emb = torch.nn.functional.normalize(a_emb, dim=-1)
                logits = 100.0 * a_emb @ t_emb.T
                probs.append(logits.softmax(dim=-1).cpu().numpy())
        p = np.concatenate(probs, axis=0)  # (W, K)
        return TagResult(times, {c.label: p[:, k] for k, c in enumerate(classes)})
