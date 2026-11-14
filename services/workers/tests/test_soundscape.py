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


class FakeSatie:
    """Scripted SATIE MCP server. Records every call."""

    def __init__(self, compose_scripts=None, realize=None):
        self.calls = []
        self.compose_scripts = list(compose_scripts or [
            "# birds\nloop memory/s1_birdsong at s1_birdsong\nbed memory/ambient in memory\nbed memory/s2_traffic in memory"
        ])
        self.realize = list(realize or [{"status": "in_progress"}, {"samplesUrl": "https://satie.example/samples.json", "realizeId": "r_1"}])

    def call_tool(self, name, args):
        self.calls.append((name, args))
        if name == "satie_inspect_scene":
            return {"contractId": "c_1"}
        if name == "satie_brief":
            return {"ready": True, "questions": []}
        if name == "satie_compose":
            return {"text": f"Here you go:\n```satie\n{self.compose_scripts.pop(0)}\n```"}
        if name == "satie_audit_scene":
            return {"quoteId": "q_1"}
        if name == "satie_realize":
            return self.realize.pop(0)
        raise AssertionError(name)


def _satie(fake, **cfg):
    return SatieMcpProvider(fake, SatieConfig(**cfg), http_get_json=lambda url: {"clips": {"memory/s1_birdsong": "x"}}, sleep=lambda s: None)


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
    assert names == ["satie_inspect_scene", "satie_brief", "satie_compose", "satie_audit_scene", "satie_realize", "satie_realize"]

    inspect_args = fake.calls[0][1]
    hosts = inspect_args["scene"]["hosts"]
    assert [h["id"] for h in hosts] == ["s1_birdsong"]  # diffuse sources are not hosts
    assert hosts[0]["position"] == [2.5, 1.6, 5.0]

    realize_args = fake.calls[4][1]
    assert realize_args["confirm"] == {"quoteId": "q_1"}
    assert realize_args["maxCents"] == 0
    assert set(realize_args["samples"]) == {"memory/s1_birdsong", "memory/s2_traffic", "memory/ambient"}
    assert fake.calls[4][1]["requestKey"] == fake.calls[5][1]["requestKey"]  # resume, not re-buy

    assert scene["provider"]["name"] == "satie"
    assert scene["satie"]["host_map"] == {"s1_birdsong": "s1_birdsong"}
    script = ctx.storage.read_bytes(scene["satie"]["script_key"]).decode()
    assert "loop memory/s1_birdsong at s1_birdsong" in script
    assert read_json(ctx.storage, scene["satie"]["samples_key"]) == {"clips": {"memory/s1_birdsong": "x"}}


def test_satie_revises_script_that_drops_a_recording(ctx):
    _write_inputs(ctx)
    fake = FakeSatie(compose_scripts=[
        "loop memory/s1_birdsong at s1_birdsong\nbed memory/ambient",
        "loop memory/s1_birdsong at s1_birdsong\nbed memory/ambient\nbed memory/s2_traffic",
    ])
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


def test_brief_respects_generation_policy():
    from memoryscapes_workers.providers.soundscape import SoundscapeRequest, SoundscapeSource

    req = SoundscapeRequest("m", [SoundscapeSource(PLACED, "u1"), SoundscapeSource(DIFFUSE, "u2")], "a", "ua", 2, [0, 0, 0], [1, 1, 1])
    strict = build_brief(req, SatieConfig())
    assert "Do not add library or generated sounds" in strict
    assert "host s1_birdsong: clip memory/s1_birdsong" in strict
    assert "clip memory/s2_traffic (traffic, no single location)" in strict
    assert "quiet voices" in build_brief(req, SatieConfig(allow_generation=True))
    assert build_scene_manifest(req)["up"] == "+y"


def test_parse_tool_result_variants():
    assert parse_tool_result({"contractId": "c"}, [], False)["contractId"] == "c"
    assert parse_tool_result(None, ['{"quoteId": "q"}'], False)["quoteId"] == "q"
    assert parse_tool_result(None, ["plain words"], False) == {"text": "plain words"}
    with pytest.raises(McpError):
        parse_tool_result(None, ["boom"], True)
