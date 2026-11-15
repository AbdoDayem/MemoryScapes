"""The streaming LoD tree (visual/lod_tree.py) and its place in the publish stage, with a
fake build-lod that writes files the way the real one does (no Rust needed)."""

import dataclasses
import sys
import textwrap

import pytest
from test_publish import fake_upstream, manifest

from memoryscapes_workers.stagekit import StageError
from memoryscapes_workers.stages import publish
from memoryscapes_workers.visual import lod_tree

# Writes <input stem>-lod.rad and three chunks next to the input, prints its JSON report.
FAKE = textwrap.dedent('''
    import sys
    from pathlib import Path
    src = Path(sys.argv[-1])
    assert "--rad-chunked" in sys.argv and ("--quality" in sys.argv or "--quick" in sys.argv)
    if src.read_bytes() == b"broken":
        print("Decoding failed: bad file", file=sys.stderr)  # the real tool still exits 0
        sys.exit(0)
    stem = src.with_suffix("")
    (stem.parent / (stem.name + "-lod.rad")).write_bytes(b"RAD header")
    for i in (0, 1, 10, 2):
        (stem.parent / f"{stem.name}-lod-{i}.radc").write_bytes(b"chunk" * (i + 1))
    print('Comment: {"final_splat_count": 7700, "lod_duration": 0.1}')
''')


@pytest.fixture
def fake_tool(tmp_path, monkeypatch):
    script = tmp_path / "fake_build_lod.py"
    script.write_text(FAKE)
    cmd = [sys.executable, str(script)]
    monkeypatch.setattr(lod_tree, "find_tool", lambda: cmd)
    return cmd


def test_build_moves_header_and_chunks_in_order(fake_tool, tmp_path):
    src = tmp_path / "lod0.spz"
    src.write_bytes(b"spz")
    tree = lod_tree.build(fake_tool, src, tmp_path / "out")
    assert tree.header == "splat-lod.rad"
    assert tree.chunks == [f"splat-lod-{i}.radc" for i in (0, 1, 2, 10)]
    assert tree.count == 7700 and tree.bytes == len(b"RAD header") + 5 * (1 + 2 + 3 + 11)
    assert sorted(p.name for p in (tmp_path / "out").iterdir()) == sorted([tree.header, *tree.chunks])
    assert src.exists()  # the input is copied, not consumed


def test_a_failure_that_exits_zero_is_still_an_error(fake_tool, tmp_path):
    src = tmp_path / "lod0.spz"
    src.write_bytes(b"broken")
    with pytest.raises(lod_tree.LodTreeError, match="Decoding failed"):
        lod_tree.build(fake_tool, src, tmp_path / "out")


def test_tool_lookup(monkeypatch, tmp_path):
    monkeypatch.setenv(lod_tree.TOOL_ENV, "off")
    assert lod_tree.find_tool() is None
    exe = tmp_path / "build-lod"
    exe.write_text("")
    monkeypatch.setenv(lod_tree.TOOL_ENV, str(exe))
    assert lod_tree.find_tool() == [str(exe)]
    monkeypatch.setenv(lod_tree.TOOL_ENV, str(tmp_path / "missing"))
    assert lod_tree.find_tool() is None


def trees(ctx):
    return [lod.get("rad") for lod in manifest(ctx)["splat"]["lods"]]


def test_publish_builds_trees_and_lists_every_chunk(ctx, fake_tool):
    """lod0's tree for desktops, lod1's for phones, each on its level."""
    fake_upstream(ctx)
    summary = publish.run(ctx)
    lods = manifest(ctx)["splat"]["lods"]
    assert "rad" not in manifest(ctx)["splat"]  # the older single-tree field
    for n, lod in enumerate(lods):
        rad = lod["rad"]
        assert rad["url"] == f"publish/lod/lod{n}-lod.rad" and rad["count"] == 7700
        assert rad["chunks"] == [f"publish/lod/lod{n}-lod-{i}.radc" for i in (0, 1, 2, 10)]
        # The API stores exactly the files the manifest names, so every chunk must be there.
        assert all(ctx.storage.exists(f"{ctx.memory_id}/{u}") for u in [rad["url"], *rad["chunks"]])
        assert lod["url"] == f"splat/lods/lod{n}.spz"  # the .spz levels stay, for the fallback and the debugger
    assert [t["lod"] for t in summary["lod_trees"]] == [0, 1] and summary["lod_trees"][0]["chunks"] == 4


def test_which_levels_get_a_tree(ctx, fake_tool):
    fake_upstream(ctx)
    publish.run(dataclasses.replace(ctx, config={"publish": {"lod_tree_levels": [0, 5]}}))  # no lod5: skipped
    assert [t is not None for t in trees(ctx)] == [True, False]


def test_without_the_tool_publish_skips_the_tree(ctx):
    fake_upstream(ctx)
    publish.run(ctx)
    assert trees(ctx) == [None, None]


def test_installing_the_tool_republishes(ctx, monkeypatch, tmp_path):
    fake_upstream(ctx)
    publish.run(ctx)
    assert trees(ctx) == [None, None]
    script = tmp_path / "fake_build_lod.py"
    script.write_text(FAKE)
    monkeypatch.setattr(lod_tree, "find_tool", lambda: [sys.executable, str(script)])
    assert publish.run(ctx)["version"] == 2  # "auto" resolved differently: not skipped
    assert all(trees(ctx))


def test_a_failing_tool_doesnt_block_publishing_unless_asked_for(ctx, fake_tool):
    fake_upstream(ctx)
    ctx.storage.write_bytes(ctx.key("splat", "lods/lod0.spz"), b"broken")
    summary = publish.run(ctx)
    assert trees(ctx)[0] is None and trees(ctx)[1] is not None  # lod1's tree still built
    assert summary["lod_tree_error"].startswith("lod0: ") and "Decoding failed" in summary["lod_tree_error"]
    strict = dataclasses.replace(ctx, config={"publish": {"lod_tree": "quick"}})
    with pytest.raises(StageError, match="LoD tree"):
        publish.run(strict)


def test_asking_for_a_tree_without_the_tool_is_an_error(ctx):
    fake_upstream(ctx)
    with pytest.raises(StageError) as err:
        publish.run(dataclasses.replace(ctx, config={"publish": {"lod_tree": "quality"}}))
    assert "build-lod" in err.value.message and "setup.py" in err.value.hint
