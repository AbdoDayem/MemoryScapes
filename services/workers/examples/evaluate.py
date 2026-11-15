"""Score a pipeline run against the scenario's ground truth."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from scenes import Rendered


@dataclass
class Row:
    label: str
    truth: list[float] | None  # None = diffuse in the scenario
    estimate: list[float] | None
    method: str
    confidence: float
    spread_m: float | None
    error_m: float | None
    verdict: str


# A placement counts as good within this absolute error, or within this fraction of the
# source's distance from the camera path (far sources are judged relatively).
GOOD_ABS_M = 1.5
GOOD_REL = 0.15
CONFIDENT = 0.5


def _distance_to_path(p: np.ndarray, r: Rendered) -> float:
    return float(min(np.linalg.norm(c.position - p) for c in r.cameras))


def evaluate(r: Rendered, localized: dict) -> list[Row]:
    placed_truth = {s.label: np.array(s.position, float) for s in r.scenario.sources}
    diffuse_truth = {label for label, _ in r.scenario.diffuse}
    rows: list[Row] = []
    seen = set()
    for src in localized["sources"]:
        label = src["label"]
        seen.add(label)
        est = src["position"]
        conf = src["confidence"]
        if label in placed_truth:
            truth = placed_truth[label]
            if est is None:
                verdict = "narration (heard everywhere)" if src["method"] == "narration" else "missed (left diffuse)"
                rows.append(Row(label, truth.tolist(), None, src["method"], conf, None, None, verdict))
                continue
            err = float(np.linalg.norm(np.array(est) - truth))
            tolerance = max(GOOD_ABS_M, GOOD_REL * _distance_to_path(truth, r))
            if err <= tolerance:
                verdict = "placed"
            elif conf >= CONFIDENT:
                verdict = "confidently wrong"
            else:
                verdict = "loosely placed"
            rows.append(Row(label, truth.tolist(), est, src["method"], conf, src["spread_m"], round(err, 2), verdict))
        elif label in diffuse_truth:
            verdict = "diffuse (correct)" if est is None else "placed a diffuse sound"
            rows.append(Row(label, None, est, src["method"], conf, src["spread_m"], None, verdict))
        else:
            verdict = "false stem, diffuse" if est is None else ("false stem, placed" + (" confidently" if conf >= CONFIDENT else ""))
            rows.append(Row(label, None, est, src["method"], conf, src["spread_m"], None, verdict))
    for label, truth in placed_truth.items():
        if label not in seen:
            rows.append(Row(label, truth.tolist(), None, "-", 0.0, None, None, "not separated"))
    return rows


BAD_VERDICTS = {"confidently wrong", "placed a diffuse sound", "false stem, placed confidently"}


def summary(rows: list[Row]) -> dict[str, int]:
    out: dict[str, int] = {}
    for row in rows:
        out[row.verdict] = out.get(row.verdict, 0) + 1
    return out
