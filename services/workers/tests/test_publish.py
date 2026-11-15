import dataclasses
import json

import pytest

from memoryscapes_workers.stagekit import STAGE_MARKER, StageError
from memoryscapes_workers.stages import publish
from memoryscapes_workers.storage import read_json, write_json


def fake_upstream(ctx, with_scene=True, geo=True):
    """Minimal outputs of ingest, poses, splat (and soundscape), with stage markers."""
    s = ctx.storage
    for stage in ("ingest", "poses", "splat"):
        write_json(s, ctx.key(stage, STAGE_MARKER), {"fingerprint": f"fp-{stage}"})
    write_json(s, ctx.key("ingest", "meta.json"), {
        "geo": {"lat": 37.4275, "lon": -122.1697, "alt": 30.0} if geo else None,
        "captured_at": "2026-06-01T18:30:00Z",
        "mic": {"channels": 2, "layout": "stereo", "spacing_m": None},
    })
    s.write_bytes(ctx.key("ingest", "audio.wav"), b"RIFF")
    write_json(s, ctx.key("poses", "navigation.json"), {
        "spawn": {"position": [0, 1.7, 0], "quaternion_xyzw": [0, 0, 0, 1], "look_at": [0, 1.7, -1]},
        "floor_y": 0.0, "eye_height_m": 1.5,
        "walkable_bounds": {"type": "aabb+corridor", "min": [-2, 0, -10], "max": [2, 2.7, 1]},
    })
    s.write_bytes(ctx.key("poses", "path.json"), b"{}")
    write_json(s, ctx.key("splat", "lods.json"), {"lods": [
        {"file": "lods/lod0.spz", "count": 5000, "sh_degree": 3, "bytes": 1},
        {"file": "lods/lod1.spz", "count": 2000, "sh_degree": 1, "bytes": 1},
    ]})
    for i in range(2):
        s.write_bytes(ctx.key("splat", f"lods/lod{i}.spz"), b"spz")
    if with_scene:
        write_scene(ctx)


def write_scene(ctx, gain=1.0):
    m = ctx.memory_id
    for k in ("separate/ambient.wav", "separate/stems/s1_birdsong.wav", "separate/stems/s2_wind.wav"):
        ctx.storage.write_bytes(f"{m}/{k}", b"RIFF")
    write_json(ctx.storage, ctx.key("soundscape", "scene.json"), {
        "provider": {"name": "stub", "notes": []},
        "ambient_bed": {"key": f"{m}/separate/ambient.wav", "format": "stereo"},
        "sources": [
            {"id": "s1_birdsong", "label": "birdsong", "key": f"{m}/separate/stems/s1_birdsong.wav",
             "position": [3.2, 3.5, 6.3], "trajectory": None, "ref_distance": 7.3, "rolloff": 1.0,
             "gain": gain, "loop": True, "confidence": 0.6, "method": "av+doa"},
            {"id": "s2_wind", "label": "wind", "key": f"{m}/separate/stems/s2_wind.wav",
             "position": None, "method": "diffuse", "confidence": 0.0},
        ],
        "satie": None,
    })


def manifest(ctx):
    return read_json(ctx.storage, ctx.key("publish", "manifest.json"))


def test_publish_with_soundscape(ctx):
    fake_upstream(ctx)
    summary = publish.run(ctx)
    m = manifest(ctx)
    assert summary["version"] == 1 and summary["placed_sources"] == 1
    assert m["coord_frame"] == {"units": "meters", "up": "+y", "origin_geo": [37.4275, -122.1697, 30.0]}
    assert [lod["url"] for lod in m["splat"]["lods"]] == ["splat/lods/lod0.spz", "splat/lods/lod1.spz"]
    assert m["navigation"]["original_path"] == "poses/path.json"
    assert m["navigation"]["floor_y"] == 0.0 and m["navigation"]["spawn"]["position"] == [0, 1.7, 0]
    bird, wind = m["audio"]["sources"]
    assert bird["url"] == "separate/stems/s1_birdsong.wav" and bird["method"] == "av+doa"
    assert wind["position"] is None and "key" not in wind
    assert m["audio"]["ambient_bed"] == {"url": "separate/ambient.wav", "format": "stereo"}
    assert m["audio"]["soundscape_provider"] == {"name": "stub"} and m["audio"]["satie"] is None
    assert m["meta"] == {"visibility": "private", "captured_at": "2026-06-01T18:30:00Z"}
    # Every relative URL resolves to a stored object.
    urls = [lod["url"] for lod in m["splat"]["lods"]] + [s["url"] for s in m["audio"]["sources"]]
    urls += [m["audio"]["ambient_bed"]["url"], m["navigation"]["original_path"]]
    assert all(ctx.storage.exists(f"{ctx.memory_id}/{u}") for u in urls)


def test_publish_without_audio_stages_plays_the_recording(ctx):
    fake_upstream(ctx, with_scene=False, geo=False)
    publish.run(ctx)
    m = manifest(ctx)
    assert m["audio"]["sources"] == []
    assert m["audio"]["ambient_bed"] == {"url": "ingest/audio.wav", "format": "stereo"}
    assert m["audio"]["soundscape_provider"]["name"] == "none"
    assert "origin_geo" not in m["coord_frame"]


def test_republishes_when_only_the_soundscape_changes(ctx):
    fake_upstream(ctx)
    publish.run(ctx)
    assert publish.run(ctx)["version"] == 1  # unchanged: skipped
    write_scene(ctx, gain=0.5)  # audio stages re-ran; they write no stage marker
    assert publish.run(ctx)["version"] == 2
    assert manifest(ctx)["audio"]["sources"][0]["gain"] == 0.5


@pytest.mark.parametrize("visibility", ["unlisted", "public"])
def test_refuses_non_private_until_the_privacy_pass_exists(ctx, visibility):
    fake_upstream(ctx)
    ctx = dataclasses.replace(ctx, config={"publish": {"visibility": visibility}})
    with pytest.raises(StageError) as err:
        publish.run(ctx)
    assert "privacy" in err.value.message


def test_invalid_manifest_is_a_stage_error(ctx):
    fake_upstream(ctx)
    write_json(ctx.storage, ctx.key("splat", "lods.json"), {"lods": []})  # schema: minItems 1
    with pytest.raises(StageError) as err:
        publish.run(ctx)
    assert "splat/lods" in err.value.message
    assert not ctx.storage.exists(ctx.key("publish", "manifest.json"))
    json.dumps(err.value.message)


def _spz(antialiased):
    import numpy as np

    from memoryscapes_workers.visual import splat_io

    n = 4
    return splat_io.encode_spz(splat_io.GaussianCloud(
        means=np.zeros((n, 3), np.float32), scales=np.full((n, 3), -3, np.float32),
        quats=np.tile(np.array([1, 0, 0, 0], np.float32), (n, 1)), opacities=np.zeros(n, np.float32),
        sh0=np.zeros((n, 3), np.float32), shN=np.zeros((n, 0, 3), np.float32), antialiased=antialiased))


@pytest.mark.parametrize("antialiased", [True, False])
def test_spz_header_flag(antialiased):
    from memoryscapes_workers.visual import splat_io

    assert splat_io.spz_antialiased(_spz(antialiased)) is antialiased
    with pytest.raises(ValueError):
        splat_io.spz_antialiased(b"not an spz")


@pytest.mark.parametrize("antialiased", [True, False])
def test_manifest_records_the_training_mode(ctx, antialiased):
    """The viewer's blur must match training: classic gets a fixed dilation, anti-aliased
    training Spark's opacity-compensated blur."""
    fake_upstream(ctx)
    lods = read_json(ctx.storage, ctx.key("splat", "lods.json"))
    write_json(ctx.storage, ctx.key("splat", "lods.json"), {**lods, "antialiased": antialiased})
    publish.run(ctx)
    assert manifest(ctx)["splat"]["antialiased"] is antialiased


def test_training_mode_from_the_spz_for_older_splat_outputs(ctx):
    fake_upstream(ctx)  # lods.json without "antialiased"
    ctx.storage.write_bytes(ctx.key("splat", "lods/lod0.spz"), _spz(True))
    publish.run(ctx)
    assert manifest(ctx)["splat"]["antialiased"] is True


def test_unreadable_spz_leaves_the_training_mode_out(ctx):
    fake_upstream(ctx)  # lods.json without "antialiased", lod0.spz isn't a real spz
    publish.run(ctx)
    assert "antialiased" not in manifest(ctx)["splat"]
