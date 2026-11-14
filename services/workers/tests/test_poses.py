"""The poses stage's mapper fallback, with a stand-in pycolmap (the real one can't share a
process with torch on macOS, and mapping real footage takes minutes)."""

import sys
from types import SimpleNamespace

import pytest

from memoryscapes_workers.stagekit import StageError, StageIO
from memoryscapes_workers.stages import poses


class Rec:
    def __init__(self, n):
        self.n = n

    def num_reg_images(self):
        return self.n


def fake_pycolmap(calls, global_result, incremental_result):
    def mapper(kind, result):
        def run(*_a, **_k):
            calls.append(kind)
            if isinstance(result, Exception):
                raise result
            return result
        return run

    return SimpleNamespace(
        Device=SimpleNamespace(auto="auto", cpu="cpu"),
        CameraMode=SimpleNamespace(SINGLE="single"),
        FeatureExtractionOptions=lambda: SimpleNamespace(sift=SimpleNamespace(max_num_features=0)),
        ImageReaderOptions=lambda **k: SimpleNamespace(**k),
        extract_features=lambda *_a, **_k: None,
        SequentialPairingOptions=lambda: SimpleNamespace(overlap=0, quadratic_overlap=False),
        match_sequential=lambda *_a, **_k: None,
        global_mapping=mapper("global", global_result),
        incremental_mapping=mapper("incremental", incremental_result),
    )


def run_sfm(monkeypatch, ctx, tmp_path, global_result, incremental_result, n_images=50):
    calls = []
    monkeypatch.setitem(sys.modules, "pycolmap", fake_pycolmap(calls, global_result, incremental_result))
    io = StageIO(ctx=ctx, stage="poses", work_dir=tmp_path, out_dir=tmp_path / "out")
    return calls, poses._sfm(io, poses.PosesConfig(), tmp_path, n_images, 0.8)


def test_focal_prior_prefers_video_metadata():
    cfg = poses.PosesConfig()
    assert poses.focal_prior({"width": 1600, "height": 900, "focal_px": None}, cfg) == \
        {"source": "default", "factor": 0.8}
    # iPhone main camera: 26 mm equivalent -> 26 * hypot(1600, 900) / 43.27 = 1103 px.
    got = poses.focal_prior({"width": 1600, "height": 900, "focal_px": 1103.0}, cfg)
    assert got["source"] == "video metadata" and got["factor"] == pytest.approx(0.689, abs=1e-3)


def test_global_success_needs_no_fallback(monkeypatch, ctx, tmp_path):
    calls, (rec, _, used) = run_sfm(monkeypatch, ctx, tmp_path, {0: Rec(48)}, {0: Rec(50)})
    assert calls == ["global"] and used == "global" and rec.n == 48


@pytest.mark.parametrize("global_result", [{}, {0: Rec(3)}, RuntimeError("no 3D points")])
def test_failed_global_mapping_falls_back_to_incremental(monkeypatch, ctx, tmp_path, global_result):
    calls, (rec, _, used) = run_sfm(monkeypatch, ctx, tmp_path, global_result, {0: Rec(12), 1: Rec(45)})
    assert calls == ["global", "incremental"] and used == "incremental"
    assert rec.n == 45  # the largest piece


def test_both_mappers_failing_is_a_stage_error_with_a_capture_hint(monkeypatch, ctx, tmp_path):
    with pytest.raises(StageError) as err:
        run_sfm(monkeypatch, ctx, tmp_path, {}, {0: Rec(5)})
    assert "only 5 of 50 frames" in err.value.message and err.value.hint
