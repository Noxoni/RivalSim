"""CPU-only report tests; fixtures cannot be mistaken for gameplay evidence."""
import copy
import json

import pytest

from benchmarks import report_direct_skills_native_nexto_v1 as report


def baseline():
    return report.read(report.OUT / "finishing650.json")


def test_identical_real_baseline_has_zero_deltas():
    data = baseline()
    result = report.compare_records(data, data)
    assert all(row["delta"] == 0 for row in result["summary"].values())
    assert result["goal_difference_worlds"] == dict(improved=0, unchanged=10, worsened=0)
    assert sum(row["before"]["goals_for"] for row in result["paired_worlds"]) == 7
    assert sum(row["before"]["goals_against"] for row in result["paired_worlds"]) == 196


@pytest.mark.parametrize("key", ["nexto_seed", "nexto_sampling_mode", "evaluation_spec_sha256"])
def test_reject_mismatched_evaluation_domain(key):
    parent = baseline(); child = copy.deepcopy(parent)
    child[key] = "synthetic incompatible domain"
    with pytest.raises(ValueError, match="domains differ"):
        report.compare_records(parent, child)


def test_reject_unfinished_or_mutated_evaluation():
    parent = baseline(); child = copy.deepcopy(parent)
    child["raw"]["match.done"][0] = False
    with pytest.raises(ValueError, match="Incomplete"):
        report.compare_records(parent, child)
    child = copy.deepcopy(parent); child["optimizer_steps"] = 1
    with pytest.raises(ValueError, match="immutable"):
        report.compare_records(parent, child)


def test_log_prefix_ignores_partial_tail_and_rejects_gaps(tmp_path):
    path = tmp_path / "synthetic.jsonl"
    first = json.dumps(dict(branch_updates=1)).encode()+b"\n"
    path.write_bytes(first+b'{"branch_updates":')
    data, rows = report.training_prefix(path, 1)
    assert data == first and rows == [dict(branch_updates=1)]
    with pytest.raises(ValueError, match="not complete"):
        report.training_prefix(path, 2)
    path.write_bytes(first+json.dumps(dict(branch_updates=3)).encode()+b"\n")
    with pytest.raises(ValueError, match="gap or duplicate"):
        report.training_prefix(path, 2)


def test_immutable_output_never_overwrites(tmp_path):
    path = tmp_path / "fixture.bin"
    report.save_once(path, b"original")
    report.save_once(path, b"original")
    with pytest.raises(ValueError, match="immutable report differs"):
        report.save_once(path, b"replacement")
    assert path.read_bytes() == b"original"


def test_real_first_checkpoint_state_and_counter_audit():
    first = report.read(report.OUT / "first_update_audit.json")
    authority = report.read(report.OUT / "training_authority.json")
    checks = report.checkpoint_checks(first["parent"], first["first"], authority, [first["first_update"]])
    assert len(checks) == 17 and all(checks.values())


def test_checkpoint_audit_rejects_wrong_log_counters():
    first = report.read(report.OUT / "first_update_audit.json")
    authority = report.read(report.OUT / "training_authority.json")
    synthetic = copy.deepcopy(first["first_update"])
    synthetic["ppo"]["optimizer_steps"] += 1
    with pytest.raises(AssertionError):
        report.checkpoint_checks(first["parent"], first["first"], authority, [synthetic])
