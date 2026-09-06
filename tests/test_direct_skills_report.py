import copy
import json

import pytest

from benchmarks import report_rival2_direct_skills as reporting
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


@pytest.mark.parametrize("grouped", [False, True])
def test_prior_review_comparison_keeps_root_evidence_and_rebuilds(
    baseline, tmp_path, monkeypatch, grouped
):
    monkeypatch.setattr(reporting, "RESULTS", tmp_path)
    for offset in (0, 50, 100):
        document = copy.deepcopy(baseline)
        document["accepted_updates"] = offset
        (tmp_path / f"evaluation_{offset:06d}.json").write_text(json.dumps(document))
    build = reporting.start_groups if grouped else reporting.main
    prefix = "start_groups" if grouped else "comparison"
    build(100)
    root_output = tmp_path / f"{prefix}_000100.json"
    original = root_output.read_bytes()
    build(100, 50)
    pair_output = tmp_path / f"{prefix}_000050_to_000100.json"
    pair = json.loads(pair_output.read_text())
    assert pair["baseline_updates"] == 50 and pair["accepted_updates"] == 100
    assert set(pair["sources"]) == {"evaluation_000050.json", "evaluation_000100.json"}
    first = pair_output.read_bytes()
    build(100, 50)
    assert pair_output.read_bytes() == first
    assert root_output.read_bytes() == original
    assert pair["optimizer_steps"] == pair["policy_evaluations_run"] == 0


@pytest.mark.parametrize("baseline_offset,offset", [(-1, 100), (100, 50)])
def test_prior_review_rejects_invalid_offset_order(baseline_offset, offset):
    with pytest.raises(AssertionError):
        reporting.load_pair(offset, baseline_offset)


def test_prior_review_rejects_mislabeled_source(baseline, tmp_path, monkeypatch):
    monkeypatch.setattr(reporting, "RESULTS", tmp_path)
    for offset in (50, 100):
        (tmp_path / f"evaluation_{offset:06d}.json").write_text(json.dumps(baseline))
    with pytest.raises(AssertionError):
        reporting.load_pair(100, 50)


def test_prior_review_reports_regression_even_when_root_improved(baseline):
    previous, candidate = copy.deepcopy(baseline), copy.deepcopy(baseline)
    previous["accepted_updates"], candidate["accepted_updates"] = 50, 100
    for document, gains in ((previous, 2), (candidate, 1)):
        skill = document["skills"]["finishing"]
        for _ in range(gains):
            i = skill["raw"]["concedes"].index(1)
            skill["raw"]["concedes"][i], skill["raw"]["goals"][i] = 0, 1
            skill["goals_against"] -= 1
            skill["goals_for"] += 1
    assert compare_documents(baseline, candidate)["finishing"]["goals_for"]["change"] == 1
    row = compare_documents(previous, candidate)["finishing"]
    assert row["goals_for"]["change"] == -1
    assert row["paired_goal_outcome"] == {"improved": 0, "worsened": 1, "unchanged": 63}
