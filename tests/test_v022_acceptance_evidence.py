from __future__ import annotations

import copy

import pytest

from benchmarks.build_v022_acceptance_evidence import _validate_full, _validate_pilot


def _full() -> dict:
    return {
        "milestone": "v0.2.2",
        "run": {
            "selection_kind": "complete",
            "oracle_execution": "cached_native_authority_only",
            "selected_case_count": 39236,
            "oracle_authority_identity_sha256": "AUTHORITY",
            "corpus_sha256": "CORPUS",
        },
        "counts": {
            "checkpoint_comparisons": 156944,
            "hard_mismatch_events": 0,
            "numeric_failure_events": 0,
            "failed_cases": 0,
        },
        "gate": {
            "selection_complete": True,
            "complete_v022_gate_pass": True,
            "classification": "PASS_GREEN",
        },
    }


def _pilot() -> dict:
    return {
        "run": {
            "oracle_execution": "cached_native_authority_only",
            "selected_case_count": 1043,
            "oracle_authority_identity_sha256": "AUTHORITY",
            "corpus_sha256": "CORPUS",
        },
        "counts": {
            "hard_mismatch_events": 0,
            "numeric_failure_events": 0,
            "failed_cases": 0,
        },
        "gate": {
            "selected_run_pass": True,
            "classification": "PILOT_PASS",
        },
    }


def test_acceptance_evidence_requires_complete_green_cached_run() -> None:
    full = _full()
    _validate_full(full)

    failed = copy.deepcopy(full)
    failed["counts"]["numeric_failure_events"] = 1
    with pytest.raises(RuntimeError, match="blocking failures"):
        _validate_full(failed)


def test_acceptance_evidence_binds_representative_to_full_authority() -> None:
    full = _full()
    pilot = _pilot()
    _validate_pilot(pilot, full)

    mismatched = copy.deepcopy(pilot)
    mismatched["run"]["corpus_sha256"] = "OTHER"
    with pytest.raises(RuntimeError, match="identity mismatch"):
        _validate_pilot(mismatched, full)
