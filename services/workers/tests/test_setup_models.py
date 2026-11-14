"""setup-models' download logic (no network): resume, checksum, safe unpacking."""

import hashlib
import io
import zipfile

import pytest

from memoryscapes_workers.devtools import models

DATA = bytes(range(256)) * 400  # 100 KB


class FakeResponse(io.BytesIO):
    def __init__(self, data, status):
        super().__init__(data)
        self.status = status

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False


def fake_urlopen(data, calls):
    def urlopen(req, timeout=None):
        rng = req.headers.get("Range")
        calls.append(rng)
        if rng:
            start = int(rng.split("=")[1].rstrip("-"))
            return FakeResponse(data[start:], 206)
        return FakeResponse(data, 200)
    return urlopen


def test_download_resumes_and_verifies(tmp_path, monkeypatch):
    sha = hashlib.sha256(DATA).hexdigest()
    dest = tmp_path / "checkpoint" / "model.ckpt"  # parent doesn't exist yet
    dest.parent.mkdir()
    (dest.parent / "model.ckpt.part").write_bytes(DATA[:30000])  # an interrupted download
    calls = []
    monkeypatch.setattr(models.urllib.request, "urlopen", fake_urlopen(DATA, calls))
    models.download_checked("https://x/model.ckpt", dest, len(DATA), sha)
    assert dest.read_bytes() == DATA and calls == ["bytes=30000-"]
    models.download_checked("https://x/model.ckpt", dest, len(DATA), sha)  # already there
    assert len(calls) == 1


def test_checksum_mismatch_deletes_the_partial(tmp_path, monkeypatch):
    monkeypatch.setattr(models.urllib.request, "urlopen", fake_urlopen(DATA, []))
    dest = tmp_path / "sub" / "model.ckpt"
    with pytest.raises(RuntimeError, match="checksum"):
        models.download_checked("https://x/m", dest, len(DATA), "0" * 64)
    assert not dest.exists() and not (tmp_path / "sub" / "model.ckpt.part").exists()


def zip_bytes(files):
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as z:
        for name, data in files.items():
            z.writestr(name, data)
    return buf.getvalue()


def test_audiosep_code_unpacks_the_pinned_commit(tmp_path, monkeypatch):
    archive = zip_bytes({"AudioSep-abc/pipeline.py": "x = 1\n", "AudioSep-abc/config/a.yaml": "k: v\n"})
    monkeypatch.setattr(models.urllib.request, "urlopen", lambda req, timeout=None: FakeResponse(archive, 200))
    models.install_audiosep_code(tmp_path / "AudioSep")
    assert (tmp_path / "AudioSep" / "pipeline.py").read_text() == "x = 1\n"
    assert (tmp_path / "AudioSep" / ".memoryscapes-commit").read_text().strip() == models.AUDIOSEP_COMMIT


def test_audiosep_code_refuses_paths_outside_the_folder(tmp_path, monkeypatch):
    archive = zip_bytes({"AudioSep-abc/../../evil.py": "boom"})
    monkeypatch.setattr(models.urllib.request, "urlopen", lambda req, timeout=None: FakeResponse(archive, 200))
    with pytest.raises(ValueError, match="unsafe"):
        models.install_audiosep_code(tmp_path / "AudioSep")
    assert not (tmp_path / "evil.py").exists()
