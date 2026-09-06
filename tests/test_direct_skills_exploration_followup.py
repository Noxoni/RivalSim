import ast
import hashlib
import json
from pathlib import Path

import pytest

from benchmarks import run_direct_skills_exploration_followup_v1 as follow
from benchmarks import run_direct_skills_exploration_v1 as prior


@pytest.fixture
def state(tmp_path):
    checkpoint = tmp_path / "example.pt"
    checkpoint.write_bytes(b"opaque fixture; strict loader checks payload separately")
    digest = hashlib.sha256(checkpoint.read_bytes()).hexdigest().upper()
    latest = dict(accepted_updates=600, sha256=digest)
    status = dict(accepted_updates=600, status="stopped_for_exploration_review",
                  exploration_amendment_sha256=follow.PRIOR_AMENDMENT_SHA)
    def save():
        (tmp_path / "latest.json").write_text(json.dumps(latest))
        (tmp_path / "campaign_state.json").write_text(json.dumps(status))
    save()
    return checkpoint, digest, latest, status, save


def test_reviewed_entry_requires_exact_latest(state, tmp_path):
    p, sha, latest, _, _ = state
    assert follow.validate_latest(p, sha, external=tmp_path) == latest
    with pytest.raises(RuntimeError, match="latest accepted"):
        follow.validate_latest(p, "older checkpoint", external=tmp_path)


def test_stop_never_cleared(state, tmp_path):
    p, sha, _, _, _ = state
    stop = tmp_path / "STOP"
    stop.write_text("user stop")
    with pytest.raises(RuntimeError, match="STOP"):
        follow.validate_latest(p, sha, external=tmp_path)
    assert stop.read_text() == "user stop"


@pytest.mark.parametrize("offset,state_offset,amendment", [
    (600, 600, "new"), (601, 600, "new"), (601, 600, "prior"),
    (625, 624, "new"), (650, 649, "new")])
def test_publication_and_recovery_windows(state, tmp_path, offset, state_offset, amendment):
    p, sha, latest, status, save = state
    latest["accepted_updates"] = offset
    status.update(accepted_updates=state_offset, status="failed",
                  exploration_amendment_sha256=follow.PRIOR_AMENDMENT_SHA
                  if amendment == "prior" else follow.content_hash(follow.amendment()))
    save()
    assert follow.validate_latest(p, sha, external=tmp_path) == latest


@pytest.mark.parametrize("failure", ["old_authority", "stale_state", "completed", "out_of_range", "corrupt"])
def test_refuses_invalid_resume(state, tmp_path, failure):
    p, sha, latest, status, save = state
    latest["accepted_updates"] = status["accepted_updates"] = 625
    status.update(status="failed", exploration_amendment_sha256=follow.content_hash(follow.amendment()))
    if failure == "old_authority":
        status["exploration_amendment_sha256"] = follow.PRIOR_AMENDMENT_SHA
    elif failure == "stale_state":
        status["accepted_updates"] = 622
    elif failure == "completed":
        latest["accepted_updates"] = status["accepted_updates"] = 650
        status["status"] = "stopped_for_exploration_review"
    elif failure == "out_of_range":
        latest["accepted_updates"] = 651
    else:
        p.write_bytes(b"corrupt")
    save()
    with pytest.raises(RuntimeError):
        follow.validate_latest(p, sha, external=tmp_path)


def test_frozen_learning_implementations_and_ppo_identical():
    assert follow.TrainingExplorationPolicy is prior.TrainingExplorationPolicy
    assert follow.base.DirectSkillsEnv is prior.base.DirectSkillsEnv
    assert follow.base.DirectSkillCollector is prior.base.DirectSkillCollector
    assert follow.base.mixed_joint_ppo_update is prior.base.mixed_joint_ppo_update
    assert follow.ppo_config() == prior.ppo_config()
    assert follow.TEMPERATURE == prior.TEMPERATURE == 2
    assert follow.START == 600 and follow.REVIEW == 650
    assert follow.amendment()["prior_amendment_sha256"] == follow.content_hash(prior.amendment())


def test_loader_and_preflight_are_same_except_new_parent_counter():
    def nodes(module):
        tree = ast.parse(Path(module.__file__).read_text())
        return {n.name: ast.unparse(n) for n in tree.body if isinstance(n, ast.FunctionDef)}
    a, b = nodes(prior), nodes(follow)
    assert a["load"] == b["load"]
    assert a["preflight"].replace("131038", "137558") == b["preflight"]


def test_actual_collect_and_ppo_call_sites_identical():
    def calls(module):
        tree = ast.parse(Path(module.__file__).read_text())
        run = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "run")
        return [ast.unparse(n) for n in ast.walk(run) if isinstance(n, ast.Call)
                and ast.unparse(n.func) in {"collector.collect", "base.mixed_joint_ppo_update"}]
    assert calls(prior) == calls(follow)
    assert len(calls(follow)) == 2


def test_full_evaluation_body_remains_exactly_the_same():
    def evaluation(module):
        tree = ast.parse(Path(module.__file__).read_text())
        run = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "run")
        return ast.unparse(next(n for n in run.body if isinstance(n, ast.FunctionDef)
                               and n.name == "evaluate"))
    assert evaluation(prior) == evaluation(follow)
