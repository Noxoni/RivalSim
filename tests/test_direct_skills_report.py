import copy
import json

import pytest

from benchmarks.report_rival2_direct_skills import RESULTS, compare_documents, group_start_cases


@pytest.fixture
def baseline():
    return json.loads((RESULTS / "evaluation_000000.json").read_text())


def test_identical_evaluation_is_not_improvement(baseline):
    report = compare_documents(baseline, baseline)
    assert all(
        row["paired_goal_outcome"] == {"improved": 0, "worsened": 0, "unchanged": 64}
        for row in report.values()
    )
    assert all(row["goals_for"]["change"] == 0 for row in report.values())


@pytest.mark.parametrize(
    "fault", ["scenario", "count", "authority", "missing", "optimizer", "nonfinite", "outcome"]
)
def test_rejects_incomparable_or_corrupt_evidence(baseline, fault):
    candidate = copy.deepcopy(baseline)
    skill = candidate["skills"]["finishing"]
    if fault == "scenario":
        skill["scenario_sha256"] = "different"
    elif fault == "count":
        skill["goals_for"] += 1
    elif fault == "authority":
        candidate["authority_sha256"] = "different"
    elif fault == "missing":
        del candidate["skills"]["defense"]
    elif fault == "optimizer":
        candidate["optimizer_steps"] = 1
    elif fault == "nonfinite":
        skill["raw"]["seconds"][0] = float("nan")
    else:
        skill["raw"]["goals"][0] = skill["raw"]["concedes"][0] = 1
    with pytest.raises(AssertionError):
        compare_documents(baseline, candidate)


def test_paired_concede_to_goal_is_one_better_case(baseline):
    candidate = copy.deepcopy(baseline)
    candidate["accepted_updates"] = 10
    skill = candidate["skills"]["finishing"]
    i = skill["raw"]["concedes"].index(1)
    skill["raw"]["concedes"][i], skill["raw"]["goals"][i] = 0, 1
    skill["goals_against"] -= 1
    skill["goals_for"] += 1
    row = compare_documents(baseline, candidate)["finishing"]
    assert row["paired_goal_outcome"] == {"improved": 1, "worsened": 0, "unchanged": 63}
    assert row["goals_for"]["change"] == 1 and row["goals_against"]["change"] == -1


def test_start_groups_preserve_every_case_and_outcome(baseline):
    item = baseline["skills"]["natural"]
    grouped = group_start_cases(item, [1] * 32 + [0] * 32, [0] * 32 + [-1] * 32, [0] * 64)
    assert len(grouped) == 2
    assert grouped["ongoing_ground"]["case_indices"] == list(range(32, 64))
    assert sorted(i for g in grouped.values() for i in g["case_indices"]) == list(range(64))
    for key in ("goals_for", "goals_against", "touches"):
        assert sum(g[key] for g in grouped.values()) == item[key]


@pytest.mark.parametrize("fault", ["length", "layout", "side"])
def test_start_groups_reject_invalid_metadata(baseline, fault):
    kickoff, layouts, sides = [1] * 64, [0] * 64, [0] * 64
    if fault == "length":
        kickoff.pop()
    elif fault == "layout":
        layouts[0] = -1
    else:
        sides[0] = 2
    with pytest.raises(AssertionError):
        group_start_cases(baseline["skills"]["kickoff"], kickoff, layouts, sides)
