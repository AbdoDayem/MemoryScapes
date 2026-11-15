"""End-to-end runs of the example memories (examples/scenes.py) with oracle backends.

These are regression tests for placement quality. The bar everywhere: never place a
sound confidently in the wrong spot. Leaving a sound diffuse when the recording can't
pin it down is acceptable and asserted explicitly where it's expected.
"""

import dataclasses

import pytest

from evaluate import BAD_VERDICTS, evaluate
from memoryscapes_workers.context import StageContext
from memoryscapes_workers.providers.soundscape import StubProvider
from memoryscapes_workers.stagekit import STAGE_MARKER
from memoryscapes_workers.stages import localize, publish, separate, soundscape
from memoryscapes_workers.storage import LocalStorage, read_json, write_json
from scenes import SCENARIOS, OracleDetector, OracleSeparator, OracleTagger, render, write_inputs


def run(tmp_path, scenario, detector=True):
    ctx = StageContext(memory_id=scenario.name, storage=LocalStorage(tmp_path))
    r = render(scenario)
    write_inputs(ctx, r)
    separate.run(ctx, tagger=OracleTagger(r), separator=OracleSeparator(r))
    localized = localize.run(ctx, detector=OracleDetector(r) if detector else None)
    scene = soundscape.run(ctx, provider=StubProvider())
    return ctx, {row.label: row for row in evaluate(r, localized)}, scene


def publish_audio(ctx) -> dict:
    """Publish the memory with stand-in visual outputs, so the manifest is validated against
    what the audio stages really wrote (a source method the schema didn't know once made
    publish reject every memory whose voice became narration)."""
    s = ctx.storage
    for stage in ("ingest", "poses", "splat"):
        write_json(s, ctx.key(stage, STAGE_MARKER), {"fingerprint": f"fp-{stage}"})
    write_json(s, ctx.key("poses", "navigation.json"), {"spawn": {"position": [0, 1.6, 0]}})
    write_json(s, ctx.key("splat", "lods.json"), {"lods": [{"file": "lods/lod0.spz", "count": 1}]})
    publish.run(ctx)  # raises when the manifest doesn't match memory.schema.json
    manifest = read_json(s, ctx.key("publish", "manifest.json"))
    scene = read_json(s, ctx.key("soundscape", "scene.json"))
    assert [(m["id"], m.get("method"), m["position"]) for m in manifest["audio"]["sources"]] == \
        [(x["id"], x.get("method"), x["position"]) for x in scene["sources"]]
    return manifest


# scenario -> sound -> verdicts that count as a pass. Speech is placed only on strong
# evidence (localize: speech_place_min_confidence) and is otherwise narration, heard
# everywhere: a voice that can't be pinned down must never fade out (on a real phone clip
# the filmer's voice was pinned to a bystander and was lost).
SPEECH = {"placed", "narration (heard everywhere)"}
EXPECTED = {
    "park_walk": {"birdsong": {"placed"}, "water": {"placed"}, "traffic": {"placed"}},
    "street_cafe": {"speech": SPEECH, "music": {"placed"}, "traffic": {"placed"}, "dog": {"placed"}},
    # bells are above the frame and the camera never moves, so range is unobservable
    "courtyard_static": {"water": {"placed"}, "bells": {"missed (left diffuse)", "loosely placed"}},
    "beach_mono": {"speech": {"placed"}, "waves": {"placed", "loosely placed"}},  # seen clearly: placed
    "hard_mode": {"speech": SPEECH, "music": {"placed"}, "traffic": {"placed"}},
}


@pytest.mark.parametrize("name", list(SCENARIOS))
def test_example_scenario(tmp_path, name):
    ctx, rows, scene = run(tmp_path, SCENARIOS[name])
    bad = {label: row.verdict for label, row in rows.items() if row.verdict in BAD_VERDICTS}
    assert not bad, bad
    for label, ok in EXPECTED[name].items():
        assert rows[label].verdict in ok, (label, rows[label])
    # every stem made it into the soundscape, and the memory publishes
    assert {s["label"] for s in scene["sources"]} == {r.label for r in rows.values() if r.verdict != "not separated"}
    publish_audio(ctx)


def test_an_unseen_voice_is_narration_all_the_way_to_the_manifest(tmp_path):
    """No frame shows the speaker (the filmer): the voice is narration, heard everywhere,
    through the soundscape and into the published manifest."""
    ctx, rows, _ = run(tmp_path, SCENARIOS["street_cafe"], detector=False)
    assert rows["speech"].verdict == "narration (heard everywhere)"
    voice = [s for s in publish_audio(ctx)["audio"]["sources"] if s["label"] == "speech"]
    assert [(s["method"], s["position"]) for s in voice] == [("narration", None)]


def test_hard_mode_drops_the_false_tag(tmp_path):
    ctx, rows, _ = run(tmp_path, SCENARIOS["hard_mode"])
    index = read_json(ctx.storage, ctx.key("separate", "stems.json"))
    assert "bells" not in {s["label"] for s in index["stems"]}
    assert [d["label"] for d in index["dropped"]] == ["bells"]


def test_doa_alone_on_a_straight_walk_is_never_confidently_wrong(tmp_path):
    """Walking in a straight line without turning leaves stereo DOA with a cone
    ambiguity around the walking direction. It may place loosely, but not confidently
    wrong."""
    base = SCENARIOS["park_walk"]
    straight = dataclasses.replace(base, name="park_straight", conditions=dataclasses.replace(base.conditions, camera="walk_straight"))
    _, rows, _ = run(tmp_path, straight, detector=False)
    assert all(row.verdict not in BAD_VERDICTS for row in rows.values()), rows


def test_wrong_mic_spacing_without_vision_is_not_confidently_wrong(tmp_path):
    base = SCENARIOS["park_walk"]
    wrong = dataclasses.replace(
        base, name="park_wrong_spacing", conditions=dataclasses.replace(base.conditions, spacing_m=0.06, reported_spacing_m=0.16)
    )
    _, rows, _ = run(tmp_path, wrong, detector=False)
    assert all(row.verdict not in BAD_VERDICTS for row in rows.values()), rows
