import pytest

from memoryscapes_workers.models import LocalizedSource, StemsIndex
from memoryscapes_workers.providers.mcp_transport import McpError, parse_tool_result
from memoryscapes_workers.providers.soundscape import (
    SatieConfig,
    SatieMcpProvider,
    StubProvider,
    build_brief,
    build_scene_manifest,
)
from memoryscapes_workers.stages import soundscape
from memoryscapes_workers.storage import read_json, write_json

PLACED = LocalizedSource(
    "s1_birdsong", "birdsong", "event", "mem_test/separate/stems/s1_birdsong.wav",
    [2.5, 1.6, 5.0], 0.3, 0.8, "av+doa", -26.0, -12.0,
)
DIFFUSE = LocalizedSource(
    "s2_traffic", "traffic", "event", "mem_test/separate/stems/s2_traffic.wav",
    None, None, 0.0, "diffuse", -30.0, None,
)


def _write_inputs(ctx):
    write_json(
        ctx.storage,
        ctx.key("localize", "sources.json"),
        {"bounds": {"min": [-6, -1, -1], "max": [6, 5, 6]}, "sources": [PLACED.to_json(), DIFFUSE.to_json()]},
    )
    index = StemsIndex(16000, 2, "stereo", 8.0, [0.25], [], ctx.key("separate", "ambient.wav"))
    write_json(ctx.storage, ctx.key("separate", "stems.json"), index.to_json())


ORIGINAL_SCRIPT = (
    "# birds, from the recording\n"
    "loop memory/s1_birdsong\n    at s1_birdsong\n"
    "loop memory/ambient\n    bed\n"
    "loop memory/s2_traffic\n    bed"
)
CLEAN = {"clipsToSupply": ["memory/s1_birdsong", "memory/ambient", "memory/s2_traffic"],
         "libraryClips": [], "takes": {"total": 0, "maxCostCents": 0}}


def report(clips=("memory/s1_birdsong", "memory/ambient", "memory/s2_traffic"), library=(), takes=0):
    return {"clipsToSupply": list(clips), "libraryClips": list(library), "takes": {"total": takes, "maxCostCents": 3 * takes}}


class FakeSatie:
    """Scripted SATIE MCP server. Records every call."""

    def __init__(self, compose_scripts=None, realize=None, checks=None):
        self.calls = []
        self.compose_scripts = list(compose_scripts or [ORIGINAL_SCRIPT])
        self.checks = list(checks or [CLEAN])
        self.realize = list(realize or [{"status": "in_progress"}, {"samplesUrl": "https://satie.example/samples.json", "realizeId": "r_1"}])

    def call_tool(self, name, args):
        self.calls.append((name, args))
        if name == "satie_inspect_scene":
            return {"contractId": "c_1"}
        if name == "satie_brief":
            return {"ready": True, "questions": []}
        if name == "satie_compose":
            return {"text": f"Here you go:\n```satie\n{self.compose_scripts.pop(0)}\n```"}
        if name == "satie_check":
            return self.checks.pop(0) if len(self.checks) > 1 else self.checks[0]
        if name == "satie_audit_scene":
            return {"quoteId": "q_1"}
        if name == "satie_realize":
            return self.realize.pop(0)
        raise AssertionError(name)


# What satie_realize's samples.json looks like (live run, 2026-11-14): our files are
# "provided", keyed by the runtime clip path.
PROVIDED = {"samples": {"Audio/memory/s1_birdsong": {"url": "u", "source": "provided", "licence": "provided"}}}


def _satie(fake, samples=PROVIDED, **cfg):
    return SatieMcpProvider(fake, SatieConfig(**cfg), http_get_json=lambda url: samples, sleep=lambda s: None)


def test_stub_provider_writes_object_based_scene(ctx):
    _write_inputs(ctx)
    scene = soundscape.run(ctx, provider=StubProvider())
    assert scene["provider"]["name"] == "stub" and scene["satie"] is None
    bird = next(s for s in scene["sources"] if s["id"] == "s1_birdsong")
    assert bird["ref_distance"] == pytest.approx(10 ** (14 / 20), rel=1e-3)  # recorded ~5 m away
    assert scene["ambient_bed"]["format"] == "stereo"
    assert read_json(ctx.storage, ctx.key("soundscape", "scene.json")) == scene


def test_satie_provider_follows_documented_workflow(ctx):
    _write_inputs(ctx)
    fake = FakeSatie()
    scene = soundscape.run(ctx, provider=_satie(fake))

    names = [n for n, _ in fake.calls]
    assert names == ["satie_inspect_scene", "satie_brief", "satie_compose", "satie_check",
                     "satie_audit_scene", "satie_realize", "satie_realize"]

    scene_manifest = fake.calls[0][1]["scene"]  # satie.scene/v1 (satie://schema/scene)
    objects = scene_manifest["objects"]
    assert [o["id"] for o in objects] == ["s1_birdsong"]  # diffuse sources are not hosts
    assert objects[0]["position"] == [2.5, 1.6, 5.0] and objects[0]["motion"] == "static"
    assert {x["name"] for x in scene_manifest["samples"]} == {"memory/s1_birdsong", "memory/s2_traffic", "memory/ambient"}
    assert "rooms" not in scene_manifest  # an outdoor recording isn't an enclosure

    realize_args = fake.calls[5][1]
    assert realize_args["confirm"] == {"quoteId": "q_1"}
    assert realize_args["maxCents"] == 0
    assert realize_args["preferLibrary"] is False  # no library stand-ins for our recordings
    assert realize_args["saveSoundscapes"] == "off"  # memories are private
    assert set(realize_args["samples"]) == {"memory/s1_birdsong", "memory/s2_traffic", "memory/ambient"}
    assert fake.calls[5][1]["requestKey"] == fake.calls[6][1]["requestKey"]  # resume, not re-buy

    assert scene["provider"]["name"] == "satie"
    assert scene["satie"]["host_map"] == {"s1_birdsong": "s1_birdsong"}
    script = ctx.storage.read_bytes(scene["satie"]["script_key"]).decode()
    assert "loop memory/s1_birdsong\n    at s1_birdsong" in script
    assert read_json(ctx.storage, scene["satie"]["samples_key"]) == PROVIDED


def test_satie_revises_script_that_drops_a_recording(ctx):
    _write_inputs(ctx)
    fake = FakeSatie(compose_scripts=[
        "loop memory/s1_birdsong\n    at s1_birdsong\nloop memory/ambient\n    bed",
        ORIGINAL_SCRIPT,
    ], checks=[report(clips=("memory/s1_birdsong", "memory/ambient")), CLEAN])
    scene = soundscape.run(ctx, provider=_satie(fake))
    composes = [a for n, a in fake.calls if n == "satie_compose"]
    assert len(composes) == 2 and "memory/s2_traffic" in composes[1]["brief"] and "script" in composes[1]
    assert scene["provider"]["notes"] == []


def test_satie_over_budget_falls_back_to_stub(ctx):
    _write_inputs(ctx)
    fake = FakeSatie(realize=[{"status": "input_required"}])
    scene = soundscape.run(ctx, provider=_satie(fake))
    assert scene["provider"]["name"] == "stub"
    assert "SATIE_MAX_CENTS" in scene["provider"]["fallback_reason"]
    assert scene["satie"] is None


def test_satie_removes_sounds_that_were_not_recorded(ctx):
    """A generated voice and a library clip: one revision asks for them to go."""
    _write_inputs(ctx)
    fake = FakeSatie(
        compose_scripts=[ORIGINAL_SCRIPT + "\nloop gen gentle birdsong chorus\nloop community/abc123", ORIGINAL_SCRIPT],
        checks=[report(library=["abc123"], takes=1), CLEAN])
    scene = soundscape.run(ctx, provider=_satie(fake))
    composes = [a for n, a in fake.calls if n == "satie_compose"]
    assert len(composes) == 2  # one revision, not one per problem
    ask = composes[1]["brief"]
    assert "Use only the supplied recordings" in ask and "abc123" in ask and "take(s)" in ask
    assert scene["provider"]["name"] == "satie"


def test_satie_that_keeps_unrecorded_sounds_is_refused_before_realize(ctx):
    _write_inputs(ctx)
    gen = ORIGINAL_SCRIPT + "\nloop gen gentle birdsong chorus"
    fake = FakeSatie(compose_scripts=[gen, gen], checks=[report(takes=1)])
    scene = soundscape.run(ctx, provider=_satie(fake))
    assert scene["provider"]["name"] == "stub"
    assert "weren't recorded" in scene["provider"]["fallback_reason"]
    assert not [n for n, _ in fake.calls if n in ("satie_audit_scene", "satie_realize")]  # nothing bought


def test_a_budget_never_buys_sounds_when_generation_is_off(ctx):
    """Even with money allowed, an original-only memory realizes with maxCents 0."""
    _write_inputs(ctx)
    fake = FakeSatie()
    soundscape.run(ctx, provider=_satie(fake, max_cents=100))
    realize_args = next(a for n, a in fake.calls if n == "satie_realize")
    assert realize_args["maxCents"] == 0 and realize_args["preferLibrary"] is False


@pytest.mark.parametrize("realized, samples", [
    ({"samplesUrl": "s", "generated": 1}, PROVIDED),
    ({"samplesUrl": "s", "fromLibrary": 2}, PROVIDED),
    ({"samplesUrl": "s"}, {"samples": {"Audio/memory/x": {"source": "library"}}}),
])
def test_unrecorded_sounds_in_the_realized_result_are_not_used(ctx, realized, samples):
    _write_inputs(ctx)
    fake = FakeSatie(realize=[realized])
    scene = soundscape.run(ctx, provider=_satie(fake, samples=samples))
    assert scene["provider"]["name"] == "stub" and "weren't recorded" in scene["provider"]["fallback_reason"]


def test_voice_report_catches_a_generated_voice():
    from memoryscapes_workers.providers.soundscape import original_only_violations

    rep = {"voices": [{"clip": "memory/a", "generated": False}, {"clip": None, "label": "rain", "generated": True}]}
    assert original_only_violations(rep, "loop memory/a", {"memory/a"}) == ["voices that don't play our recordings: rain"]


def test_script_scan_catches_generated_and_synth_voices():
    from memoryscapes_workers.providers.soundscape import original_only_violations

    allowed = {"memory/a"}
    assert original_only_violations(CLEAN | {"clipsToSupply": ["memory/a"]}, "loop memory/a\n    bed", allowed) == []
    for line in ("loop gen soft wind", "3 * oneshot gen tap every 1to2", "on step gen footstep on {text}",
                 "loop synth an ambient piece", "gen pad_name"):
        assert original_only_violations({"clipsToSupply": ["memory/a"]}, f"loop memory/a\n{line}", allowed), line
    assert original_only_violations({"clipsToSupply": ["memory/a", "other/b"]}, "loop memory/a", allowed)


def test_brief_respects_generation_policy():
    from memoryscapes_workers.providers.soundscape import SoundscapeRequest, SoundscapeSource

    req = SoundscapeRequest("m", [SoundscapeSource(PLACED, "u1"), SoundscapeSource(DIFFUSE, "u2")], "a", "ua", 2, [0, 0, 0], [1, 1, 1])
    strict = build_brief(req, SatieConfig())
    assert "Use only the recordings above" in strict and "No `gen`, `synth`" in strict
    assert "no crowd bed" in strict and "add no rain, wind, insect or bird beds" in strict
    assert "no crowd bed" not in build_brief(req, SatieConfig(allow_generation=True))
    assert "host s1_birdsong: clip memory/s1_birdsong" in strict
    assert "clip memory/s2_traffic (traffic, no single location)" in strict
    assert "quiet voices" in build_brief(req, SatieConfig(allow_generation=True))
    manifest = build_scene_manifest(req)
    assert manifest["listener"] == "first_person" and [o["id"] for o in manifest["objects"]] == ["s1_birdsong"]


def test_parse_tool_result_variants():
    assert parse_tool_result({"contractId": "c"}, [], False)["contractId"] == "c"
    assert parse_tool_result(None, ['{"quoteId": "q"}'], False)["quoteId"] == "q"
    assert parse_tool_result(None, ["plain words"], False) == {"text": "plain words"}
    with pytest.raises(McpError):
        parse_tool_result(None, ["boom"], True)
