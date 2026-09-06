"""CPU terminal-state tests; synthetic fixtures are not gameplay evidence."""
import copy

import pytest

from benchmarks.finalize_direct_skills_native_nexto_v1 import validate_terminal
from benchmarks import report_direct_skills_native_nexto_v1 as report


def fixture():
    authority = dict(child_updates=25, parent=dict(accepted_updates=650))
    latest = dict(branch_updates=25, accepted_updates=675, path="synthetic", sha256="synthetic")
    rows = [dict(branch_updates=i, cumulative_optimizer_steps=100+i, ppo=dict(kl_rejections=0)) for i in range(1,26)]
    state = dict(status="complete_review", branch_updates=25, cumulative_optimizer_steps=125,
                 latest_checkpoint=latest)
    return state, latest, rows, authority


def test_complete_review_admitted():
    validate_terminal(*fixture())


@pytest.mark.parametrize("field,value", [("status","evaluating"), ("branch_updates",24), ("cumulative_optimizer_steps",124)])
def test_unfinished_or_wrong_counters_rejected(field,value):
    state,latest,rows,authority = fixture()
    state[field] = value
    with pytest.raises(ValueError): validate_terminal(state,latest,rows,authority)


def test_stale_latest_and_incomplete_curve_rejected():
    state,latest,rows,authority = fixture()
    stale = copy.deepcopy(latest); stale["sha256"] = "stale"
    with pytest.raises(ValueError): validate_terminal(state,stale,rows,authority)
    with pytest.raises(ValueError): validate_terminal(state,latest,rows[:-1],authority)
    rows[1]["branch_updates"] = 1
    with pytest.raises(ValueError): validate_terminal(state,latest,rows,authority)


def test_forbidden_kl_rejection_not_silently_accepted():
    state,latest,rows,authority = fixture()
    rows[0]["ppo"]["kl_rejections"] = 1
    with pytest.raises(ValueError,match="KL-telemetry"):
        validate_terminal(state,latest,rows,authority)


def test_real_closed_block_artifacts():
    out = report.OUT
    completion = report.read(out / "completion.json")
    progress = report.read(out / "progress_000025.json")
    state = report.read(out / "final_campaign_state.json")
    latest = report.read(out / "final_latest.json")
    authority = report.read(out / "training_authority.json")
    _, rows = report.training_prefix(out / "through_000025.jsonl", 25)
    validate_terminal(state, latest, rows, authority)
    for name, digest in completion["captures"].items():
        assert report.sha(out / name) == digest
    for key, name in (("training_prefix_sha256","through_000025.jsonl"),
                      ("complete_progress_sha256","progress_000025.json"),
                      ("early_progress_sha256","progress_000010.json")):
        assert report.sha(out / name) == completion[key]
    assert all(progress["checkpoint_integrity"].values())
    assert progress["training"]["learner_decisions"] == 110592000
    assert progress["training"]["optimizer_steps"] == 3536
    final = report.read(out / "child_000025.json")
    assert final["summary"]["wins"] == 0 and final["summary"]["losses"] == 10
    assert final["summary"]["goals_for"] == 2 and final["summary"]["goals_against"] == 203
    assert completion["status"] == "complete_review_not_SSL"
    assert completion["optimizer_steps_in_finalization"] == 0
    assert not (out / "stderr.log").read_bytes()
