import hashlib
import json
import sys
from contextlib import nullcontext

import pytest

from benchmarks import resume_direct_skills_exploration_v1 as guard
from benchmarks.resume_direct_skills_exploration_v1 import AMENDMENT_SHA, validate_resume


@pytest.fixture
def recorded(tmp_path):
    checkpoint = tmp_path / "latest.pt"
    checkpoint.write_bytes(b"opaque example; frozen runner performs payload validation")
    digest = hashlib.sha256(checkpoint.read_bytes()).hexdigest().upper()
    latest = dict(path=str(checkpoint), sha256=digest, accepted_updates=570)
    state = dict(status="failed", accepted_updates=570, exploration_amendment_sha256=AMENDMENT_SHA)
    (tmp_path / "latest.json").write_text(json.dumps(latest))
    (tmp_path / "campaign_state.json").write_text(json.dumps(state))
    return checkpoint, digest, latest, state


def test_exact_latest_can_be_recovered_without_changing_it(recorded, tmp_path):
    path, digest, latest, _ = recorded
    before = path.read_bytes()
    result = validate_resume(path, digest.lower(), external=tmp_path)
    assert result["accepted_updates"] == latest["accepted_updates"]
    assert path.read_bytes() == before


def test_old_parent_sha_cannot_restart_the_segment(recorded, tmp_path):
    path, _, _, _ = recorded
    with pytest.raises(RuntimeError, match="latest accepted"):
        validate_resume(
            path,
            "CF5C9D022FFB4EF18DBD728206D78A3601AB8C9128BFA0F0E9ADE31BCF961BAD",
            external=tmp_path,
        )


def test_corrupt_latest_never_falls_back(recorded, tmp_path):
    path, digest, _, _ = recorded
    path.write_bytes(b"changed")
    with pytest.raises(RuntimeError, match="hash mismatch"):
        validate_resume(path, digest, external=tmp_path)


def test_any_stop_is_respected(recorded, tmp_path):
    path, digest, _, _ = recorded
    stop = tmp_path / "STOP"
    stop.write_text("user or unknown stop")
    with pytest.raises(RuntimeError, match="STOP present"):
        validate_resume(path, digest, external=tmp_path)
    assert stop.read_text() == "user or unknown stop"


def test_review_boundary_cannot_be_mistaken_for_a_crash(recorded, tmp_path):
    path, digest, latest, state = recorded
    latest["accepted_updates"] = state["accepted_updates"] = 600
    state["status"] = "stopped_for_exploration_review"
    (tmp_path / "latest.json").write_text(json.dumps(latest))
    (tmp_path / "campaign_state.json").write_text(json.dumps(state))
    with pytest.raises(RuntimeError, match="review boundary"):
        validate_resume(path, digest, external=tmp_path)


def test_atomic_latest_can_be_one_boundary_ahead_of_status(recorded, tmp_path):
    path, digest, _, state = recorded
    state["accepted_updates"] = 569
    (tmp_path / "campaign_state.json").write_text(json.dumps(state))
    assert validate_resume(path, digest, external=tmp_path)["accepted_updates"] == 570
    state["accepted_updates"] = 568
    (tmp_path / "campaign_state.json").write_text(json.dumps(state))
    with pytest.raises(RuntimeError, match="discrepancy"):
        validate_resume(path, digest, external=tmp_path)


def test_wrong_amendment_rejected(recorded, tmp_path):
    path, digest, _, state = recorded
    state["exploration_amendment_sha256"] = "old run"
    (tmp_path / "campaign_state.json").write_text(json.dumps(state))
    with pytest.raises(RuntimeError, match="amendment"):
        validate_resume(path, digest, external=tmp_path)


def test_cli_requires_explicit_resume_identity(monkeypatch):
    monkeypatch.setattr(sys, "argv", ["resume_guard"])
    with pytest.raises(SystemExit) as error:
        guard.main()
    assert error.value.code == 2


@pytest.mark.parametrize("check_only", [True, False])
def test_dispatch_checks_identity_before_frozen_run_and_check_only_never_runs(
    monkeypatch, check_only
):
    from benchmarks import run_direct_skills_exploration_v1 as frozen

    events = []
    monkeypatch.setattr(frozen.base, "gpu_lease", nullcontext)

    def validate(*args):
        events.append("validated")
        return {"accepted_updates": 570}

    monkeypatch.setattr(guard, "validate_resume", validate)
    monkeypatch.setattr(frozen, "run", lambda args: events.append("run"))
    argv = ["resume_guard", "--resume", "test.pt", "--resume-sha256", "test-sha"]
    if check_only:
        argv.append("--check-only")
    monkeypatch.setattr(sys, "argv", argv)
    guard.main()
    assert events == (["validated"] if check_only else ["validated", "run"])
