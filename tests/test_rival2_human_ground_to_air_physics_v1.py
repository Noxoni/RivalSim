from __future__ import annotations

import json
from pathlib import Path

from benchmarks.analyze_rival2_human_ground_to_air_physics_v1 import (
    analyze_attempt,
    distribution,
    load_attempts,
    summarize,
)

ROOT = Path(__file__).resolve().parents[1]


def test_distribution_is_deterministic_and_bounded() -> None:
    assert distribution([])["count"] == 0
    measured = distribution([1, 2, 3, 4, 5])
    assert measured == {
        "count": 5,
        "minimum": 1.0,
        "p10": 1.4,
        "p50": 3.0,
        "p90": 4.6,
        "maximum": 5.0,
    }


def test_accepted_attempts_produce_source_bound_transition_evidence() -> None:
    attempts = load_attempts()
    rows = [analyze_attempt(row) for row in attempts]
    summary = summarize(rows)
    assert len(rows) == 35
    assert summary["attempts_with_one_jump_onset_before_first_air_contact"] == 13
    assert summary["attempts_with_two_jump_onsets_before_first_air_contact"] == 22
    assert summary["attempts_with_at_least_two_airborne_contacts"] == 27
    assert summary["attempts_with_one_to_six_airborne_contacts"] == 33
    assert summary["pop_to_first_air_contact_ticks"]["minimum"] == 11.0
    assert summary["pop_to_first_air_contact_ticks"]["maximum"] == 152.0
    assert summary["airborne_contact_count"]["p50"] == 3.0
    assert all(
        row["within_six_airborne_contacts"]
        for row in rows
        if row["airborne_contact_count"] <= 6
    )


def test_committed_result_remains_rebuildable_when_present() -> None:
    attempts = load_attempts()
    rebuilt = summarize([analyze_attempt(row) for row in attempts])
    result_path = (
        ROOT
        / "results/rival2/ground_to_air_human_physics_v1/human_transition.json"
    )
    if result_path.exists():
        committed = json.loads(result_path.read_text(encoding="utf-8"))
        assert committed["summary"] == rebuilt
