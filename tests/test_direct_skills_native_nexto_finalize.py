"""Synthetic terminal-state admission tests; not model or gameplay evidence."""
import copy

import pytest

from benchmarks.finalize_direct_skills_native_nexto_v1 import validate_terminal


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
