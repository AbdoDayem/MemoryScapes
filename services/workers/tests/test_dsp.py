import numpy as np

from memoryscapes_workers.audio import dsp
from memoryscapes_workers.audio.doa import gcc_phat
from synth import SR, frac_delay, noise


def test_consistent_masks_partition_every_bin():
    rng = np.random.default_rng(1)
    mix = rng.random((65, 40))
    stems = rng.random((3, 65, 40)) * 2  # estimates that overshoot the mix
    masks, residual = dsp.consistent_masks(mix, stems)
    np.testing.assert_allclose(masks.sum(0) + residual, 1.0, atol=1e-9)
    assert (masks >= 0).all() and (residual >= 0).all()


def test_masked_stems_plus_ambient_reconstruct_multichannel_mix():
    rng = np.random.default_rng(2)
    mix = rng.standard_normal((2, SR)).astype(np.float32) * 0.1
    z = dsp.stft(dsp.to_mono(mix), 1024, 256)
    masks, residual = dsp.consistent_masks(np.abs(z), np.abs(z)[None] * rng.random((2, *z.shape)))
    stems, ambient = dsp.apply_masks(mix, masks, residual, 1024, 256)
    np.testing.assert_allclose(stems.sum(0) + ambient, mix, atol=1e-4)


def test_process_in_chunks_is_identity_for_identity_fn():
    x = noise(3.3)
    y = dsp.process_in_chunks(x, chunk=SR, overlap=SR // 10, fn=lambda s: s)
    np.testing.assert_allclose(y, x, atol=1e-5)


def test_activity_envelope_tracks_loud_section():
    x = np.concatenate([noise(1.0) * 0.01, noise(1.0), noise(1.0) * 0.01])
    t, env = dsp.activity_envelope(x, SR)
    assert env[np.argmin(abs(t - 1.5))] > 0.95
    assert env[np.argmin(abs(t - 0.25))] < 0.5


def test_gcc_phat_recovers_fractional_delay():
    x = noise(1.0, seed=3)
    for delay in (0.0002, -0.00031, 0.0):
        tau, peak = gcc_phat(x, frac_delay(x, delay), SR, max_tau=0.001)
        assert abs(tau - delay) < 1.5e-5
        assert peak > 0.5


def test_clap_cosine_scores_are_multilabel():
    """Two sounds clearly present must both score high: classes don't compete."""
    from memoryscapes_workers.audio.tagging import cosine_scores

    cos = np.array([[0.27, 0.24, 0.17, 0.03, -0.2]])
    s = cosine_scores(cos)[0]
    assert s[0] > 0.85 and s[1] > 0.75  # both present
    assert 0.3 < s[2] < 0.6  # borderline
    assert s[3] == 0 and s[4] == 0  # absent
