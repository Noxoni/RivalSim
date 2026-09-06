import copy
import json

import pytest

from benchmarks.report_rival2_direct_skills_matches import ROOT, compare, validate_comparability

RESULTS = ROOT / "results/rival2/direct_skills_v1"


def inputs():
    return [json.loads((RESULTS / "corrected_reset" / f"full_match_{n:06d}.json").read_text()) for n in (0, 451)]


@pytest.mark.parametrize("field", ["match_reset_version", "runtime_package_sha256", "authority_sha256"])
def test_missing_or_different_method_rejected(field):
    a, b = inputs()
    b.pop(field)
    with pytest.raises(ValueError, match=field):
        validate_comparability(a, b)
    b[field] = "different"
    with pytest.raises(ValueError, match=field):
        validate_comparability(a, b)


@pytest.mark.parametrize("field", ["match.rival_side", "match.starting_layout"])
def test_case_changes_rejected(field):
    a, b = inputs()
    b["raw"][field][0] = 99
    with pytest.raises(ValueError, match="Match cases differ"):
        validate_comparability(a, b)


def test_old_reset_cannot_be_used_as_learning_baseline():
    a = json.loads((RESULTS / "full_match_000450.json").read_text())
    b = inputs()[1]
    with pytest.raises(ValueError, match="match_reset_version"):
        validate_comparability(a, b)


def test_same_or_backward_offsets_rejected():
    a, b = inputs()
    with pytest.raises(ValueError, match="Target must follow"):
        validate_comparability(b, a)
    with pytest.raises(ValueError, match="Target must follow"):
        validate_comparability(a, copy.deepcopy(a))


def test_committed_real_comparison_is_deterministic_and_conserved():
    a = RESULTS / "corrected_reset/full_match_000000.json"
    b = RESULTS / "corrected_reset/full_match_000451.json"
    result = compare(a, b)
    assert result == compare(a, b)
    assert result["goal_difference_cases"] == dict(improved=10, unchanged=0, worsened=0)
    assert result["target"]["summary"]["goals_for"] == 59
    assert result["target"]["summary"]["goals_against"] == 156
    assert result["target"]["summary"]["wins"] == 0
    assert result["target"]["later_goals"] == 49
    assert result["optimizer_steps"] == 0 and not result["automatic_training_change"]
    assert sum(row["target"]["contacts"] for row in result["per_match"]) == 538
