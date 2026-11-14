import numpy as np

from memoryscapes_workers.audio.tagging import TagResult
from memoryscapes_workers.models import StemsIndex
from memoryscapes_workers.stages import separate
from memoryscapes_workers.storage import read_json, read_wav, write_json, write_wav
from synth import SR


def tone(freq, dur=4.0, amp=0.2):
    t = np.arange(int(dur * SR)) / SR
    return (amp * np.sin(2 * np.pi * freq * t)).astype(np.float32)


class FakeTagger:
    def __init__(self, scores):
        self.scores = scores

    def tag(self, mono, sr, classes):
        times = np.array([1.0, 3.0])
        return TagResult(times, {c.label: np.full(2, self.scores.get(c.label, 0.01)) for c in classes})


class FakeSeparator:
    """Returns the true source for each label, as a perfect separator would."""

    def __init__(self, truth):
        self.truth = truth
        self.requested = []

    def separate(self, mono, sr, classes):
        self.requested = [c.label for c in classes]
        return {c.label: self.truth[c.label] for c in classes if c.label in self.truth}


def test_separate_stage_writes_consistent_multichannel_stems(ctx):
    bird, dog = tone(440), tone(3000, amp=0.1)
    rumble = tone(90, amp=0.05)
    # stereo with different gains per channel, so we can check spatial cues survive
    mix = np.stack([bird * 1.0 + dog * 0.3 + rumble, bird * 0.4 + dog * 1.0 + rumble])
    write_wav(ctx.storage, ctx.key("ingest", "audio.wav"), mix, SR)
    write_json(ctx.storage, ctx.key("ingest", "meta.json"), {"mic": {"channels": 2, "layout": "stereo"}})

    tagger = FakeTagger({"birdsong": 0.6, "dog": 0.45, "wind": 0.5, "speech": 0.05})
    separator = FakeSeparator({"birdsong": bird, "dog": dog})
    index = separate.run(ctx, tagger=tagger, separator=separator)

    # wind is diffuse (left in the bed), speech is below threshold
    assert separator.requested == ["birdsong", "dog"]
    assert [s.label for s in index.stems] == ["birdsong", "dog"]

    saved = StemsIndex.from_json(read_json(ctx.storage, ctx.key("separate", "stems.json")))
    assert saved.layout == "stereo" and saved.channels == 2
    stems = [read_wav(ctx.storage, s.key)[0] for s in saved.stems]
    ambient, _ = read_wav(ctx.storage, saved.ambient_key)
    mix_16 = read_wav(ctx.storage, ctx.key("ingest", "audio.wav"))[0]
    np.testing.assert_allclose(sum(stems) + ambient, mix_16, atol=2e-3)  # 16-bit rounding

    bird_stem = stems[0]
    # mostly the 440 Hz tone, and louder on the left channel like the recording
    corr = np.corrcoef(bird_stem[0, SR:-SR], bird[SR:-SR])[0, 1]
    assert corr > 0.95
    assert np.std(bird_stem[0]) > 2 * np.std(bird_stem[1])
    assert len(saved.stems[0].activity) == len(saved.window_times)


def test_separate_stage_with_nothing_detected_puts_everything_in_ambient(ctx):
    mix = np.stack([tone(200), tone(200)])
    write_wav(ctx.storage, ctx.key("ingest", "audio.wav"), mix, SR)
    write_json(ctx.storage, ctx.key("ingest", "meta.json"), {"mic": {"channels": 2, "layout": "stereo"}})
    index = separate.run(ctx, tagger=FakeTagger({}), separator=FakeSeparator({}))
    assert index.stems == []
    ambient, _ = read_wav(ctx.storage, index.ambient_key)
    np.testing.assert_allclose(ambient, read_wav(ctx.storage, ctx.key("ingest", "audio.wav"))[0], atol=2e-3)
