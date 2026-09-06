import ast
import json
from pathlib import Path

import pytest

from benchmarks import run_direct_skills_native_nexto_v1 as run
from rivalsim.direct_skills_finishing_goal_v2 import FinishingGoalEnv
from rivalsim.direct_skills_native_nexto import DirectSkillNativeNextoCollector
from rivalsim.fresh_ground_30hz import content_hash


def spec():
    return dict(parent=dict(path="frozen.pt", sha256="frozen", accepted_updates=650), version="unit")


def test_no_legacy_opponent_or_new_reward_implementation():
    assert run.FinishingGoalEnv is FinishingGoalEnv
    assert run.DirectSkillNativeNextoCollector is DirectSkillNativeNextoCollector
    assert run.LIMIT == 25 and run.EVALUATIONS == (10, 25)
    assert run.TEMPERATURE == 2
    assert run.ppo_config().learning_rate == 1e-4 and run.ppo_config().epochs == 2
    assert run.ppo_config().rollout_horizon == 90
    source = Path(run.__file__).read_text()
    tree = ast.parse(source)
    run_function = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "run")
    rendered = ast.unparse(run_function)
    assert "base.mixed_joint_ppo_update(model, optimizer, rollout, ppo_config(), shuffle)" in rendered
    assert "collector.restore_opponent_for_fresh_episodes(payload['opponent_state'])" in rendered
    assert "evaluation.evaluate(" in rendered
    assert "while child < LIMIT" in rendered
    assert "NativeNexto" not in [n.func.id for n in ast.walk(tree) if isinstance(n, ast.Call) and isinstance(n.func, ast.Name)]
    assert not any(isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute) and n.func.attr == "unlink" for n in ast.walk(tree))


def test_make_collector_binds_sampling_seed(monkeypatch):
    calls = []
    monkeypatch.setattr(run, "DirectSkillNativeNextoCollector", lambda *args, **kwargs: calls.append(kwargs))
    run.make_collector(None, None)
    assert calls == [dict(seed=run.base.SEED, nexto_sampling_mode="native_v5", nexto_seed=2026090691)]


def test_fresh_only_selected_parent(tmp_path):
    assert run.validate_resume(None, None, spec(), tmp_path) == (run.ROOT / "frozen.pt", "frozen", 0)
    with pytest.raises(RuntimeError): run.validate_resume(tmp_path / "other.pt", "other", spec(), tmp_path)
    (tmp_path / "campaign_state.json").write_text("{}")
    with pytest.raises(RuntimeError): run.validate_resume(None, None, spec(), tmp_path)


@pytest.fixture
def checkpoint(tmp_path):
    p = tmp_path / "rolling_1.pt"; p.write_bytes(b"metadata-only test checkpoint")
    digest = run.sha(p)
    latest = dict(branch_updates=3, path=str(p), sha256=digest)
    state = dict(branch_updates=3, status="rollout", native_nexto_authority_sha256=content_hash(spec()))
    def write():
        (tmp_path / "latest.json").write_text(json.dumps(latest))
        (tmp_path / "campaign_state.json").write_text(json.dumps(state))
    write()
    return p, digest, latest, state, write


def test_exact_latest_only(checkpoint, tmp_path):
    p, digest, _, state, write = checkpoint
    assert run.validate_resume(p, digest, spec(), tmp_path) == (p, digest, 3)
    state["branch_updates"] = 2; write()
    assert run.validate_resume(p, digest, spec(), tmp_path)[2] == 3


@pytest.mark.parametrize("condition", ["stop", "wrong_hash", "other_file", "corrupt", "wrong_authority",
    "wrong_counter", "complete_review", "nonfinite_or_runtime_failure", "stopped_at_accepted_boundary", "beyond_limit"])
def test_refuse_unsafe_resume(checkpoint, tmp_path, condition):
    p, digest, latest, state, write = checkpoint
    if condition == "stop": (tmp_path / "STOP").write_text("user stop")
    elif condition == "wrong_hash": digest = "wrong"
    elif condition == "other_file": p = tmp_path / "other.pt"; p.write_bytes(b"metadata-only test checkpoint")
    elif condition == "corrupt": p.write_bytes(b"changed")
    elif condition == "wrong_authority": state["native_nexto_authority_sha256"] = "old"
    elif condition == "wrong_counter": state["branch_updates"] = 1
    elif condition == "beyond_limit": latest["branch_updates"] = 26
    else: state["status"] = condition
    write()
    with pytest.raises(RuntimeError): run.validate_resume(p, digest, spec(), tmp_path)
