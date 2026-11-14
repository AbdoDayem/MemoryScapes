"""The debugger's upload runner (memoryscapes_workers.run)."""

import json

from memoryscapes_workers import run as runner


def statuses(tmp_path, memory="m"):
    status = json.loads((tmp_path / memory / "job" / "status.json").read_text())
    return status, {k: v["status"] for k, v in status["stages"].items()}


def fake_runner(results: dict):
    calls = []

    def run_stage(stage, memory_id, root, config, env, on_line):
        calls.append(stage)
        on_line(f"[{stage}] working")
        return results.get(stage, ("done", None))

    return run_stage, calls


def test_failure_blocks_dependents_and_others_keep_going(tmp_path):
    run_stage, calls = fake_runner({"poses": ("failed", "too little texture. Hint: move slowly"), "publish": ("skipped", "not implemented yet")})
    code = runner.main(["--root", str(tmp_path), "--memory", "m"], run_stage=run_stage)
    status, stages = statuses(tmp_path)
    assert code == 1 and status["state"] == "failed"
    assert stages == {"ingest": "done", "poses": "failed", "splat": "blocked", "separate": "done", "localize": "blocked", "soundscape": "blocked", "publish": "blocked"}
    assert calls == ["ingest", "poses", "separate"]
    assert "move slowly" in status["stages"]["poses"]["error"]
    assert "[separate] working" in (tmp_path / "m" / "job" / "log.txt").read_text()


def test_unimplemented_publish_is_not_a_failed_run(tmp_path):
    run_stage, _ = fake_runner({"publish": ("skipped", "not implemented yet")})
    assert runner.main(["--root", str(tmp_path), "--memory", "m"], run_stage=run_stage) == 0
    status, stages = statuses(tmp_path)
    assert status["state"] == "done" and stages["publish"] == "skipped" and stages["soundscape"] == "done"


def test_options_reach_the_stages(tmp_path):
    seen = {}

    def run_stage(stage, memory_id, root, config, env, on_line):
        seen[stage] = (config, env)
        return "done", None

    runner.main(["--root", str(tmp_path), "--memory", "m", "--max-seconds", "12", "--max-keyframes", "50", "--no-detector", "--only", "ingest,localize"], run_stage=run_stage)
    _, stages = statuses(tmp_path)
    assert set(seen) == {"ingest", "localize"} and stages["poses"] == "not requested"
    config, env = seen["ingest"]
    assert config["ingest"]["max_duration_s"] == 12 and config["ingest"]["max_keyframes"] == 50
    assert env == {"DETECTOR": "none"}


def test_real_child_process_reports_stage_errors(tmp_path):
    """The real per-stage child (worker CLI) with nothing uploaded: ingest's own error
    comes back as a failed stage with its hint, and everything downstream is blocked."""
    code = runner.main(["--root", str(tmp_path), "--memory", "empty", "--only", "ingest"])
    status, stages = statuses(tmp_path, "empty")
    assert code == 1 and stages["ingest"] == "failed"
    assert "No uploaded files found" in status["stages"]["ingest"]["error"]
    assert "Upload a video first" in status["stages"]["ingest"]["error"]
