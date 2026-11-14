from memoryscapes_workers import envfile


def test_parse_handles_comments_quotes_and_export():
    text = """
# SATIE
export SATIE_MAX_CENTS=0
SATIE_MCP_TOKEN="tok en"
EMPTY=
SINGLE='x=y'
TRAILING=abc # note
not a line
"""
    assert envfile.parse(text) == {
        "SATIE_MAX_CENTS": "0", "SATIE_MCP_TOKEN": "tok en", "EMPTY": "",
        "SINGLE": "x=y", "TRAILING": "abc",
    }


def test_load_never_overrides_the_real_environment(tmp_path, monkeypatch):
    f = tmp_path / ".env"
    f.write_text("MS_TEST_A=from_file\nMS_TEST_B=from_file\n", encoding="utf-8")
    monkeypatch.setenv("MS_TEST_A", "from_env")
    monkeypatch.delenv("MS_TEST_B", raising=False)
    assert envfile.load([f, tmp_path / "missing.env"]) == ["MS_TEST_B"]
    import os

    assert os.environ["MS_TEST_A"] == "from_env" and os.environ["MS_TEST_B"] == "from_file"
    monkeypatch.delenv("MS_TEST_B")
