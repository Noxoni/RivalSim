from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from benchmarks.review_rival2_human_demos import (
    _assign_outcome_groups,
    _closest_soccar_surface,
    _event_index,
    _load_adjudication,
    _paired_stream_assessment,
    _segment_attempts,
    _select_session_dirs,
)

HUMAN_ID = "human-1"


def _frame(index: int, *, stable: bool) -> dict[str, Any]:
    return {
        "sequence": index,
        "physics_frame": 10_000 + index,
        "engine_physics_time": index / 120.0,
        "cars": [
            {
                "stable_id": HUMAN_ID,
                "flags": {
                    "is_local_human": True,
                    "on_ground": stable,
                    "demolished": False,
                },
                "num_wheel_world_contacts": 4 if stable else 0,
                "jump_component": {"active": not stable},
                "dodge_component": {"active": False},
                "flip_component": {"active": False},
            }
        ],
    }


def _jump(index: int) -> dict[str, Any]:
    return {
        "kind": "jump_onset",
        "actor_id": HUMAN_ID,
        "physics_frame": 10_000 + index,
        "monotonic_ns": index,
    }


def test_generic_segmentation_groups_multi_press_activity_and_splits_on_recovery() -> None:
    frames = [
        _frame(index, stable=not (90 <= index < 200 or 290 <= index < 400)) for index in range(500)
    ]
    spans, audit = _segment_attempts(
        frames,
        [_jump(100), _jump(115), _jump(300)],
        HUMAN_ID,
    )

    assert audit["anchor_count"] == 3
    assert audit["attempt_count"] == 2
    assert spans[0]["anchor_indices"] == [100, 115]
    assert spans[1]["anchor_indices"] == [300]
    assert spans[0]["end_index"] < spans[1]["start_index"]
    assert all(span["recovered"] for span in spans)


def _valid_validation(*, clean: bool = True) -> dict[str, Any]:
    return {
        "valid": clean,
        "container_valid": True,
        "manifest_hashes_valid": True,
        "partial_chunks": 0,
        "missing_sequence_count": 0,
        "sequence_gap_count": 0,
        "duplicate_sequence_count": 0,
        "out_of_order_sequence_count": 0,
        "invalid_action_frames": 0,
        "invalid_human_car_frames": 0,
        "capture_complete": clean,
        "clean_termination": clean,
    }


def test_tail_event_is_not_clamped_onto_final_action_frame() -> None:
    physics = [100, 101, 102]

    assert _event_index({"physics_frame": 103}, physics) is None
    assert _event_index({"physics_frame": 99}, physics) is None
    assert _event_index({"physics_frame": 101}, physics) == 1


def test_lifecycle_gap_is_boundary_not_missing_paired_action() -> None:
    frames = [_frame(0, stable=True), _frame(1, stable=True)]
    frames[1]["physics_frame"] = 10_010
    events = [
        {"kind": "local_car_rebind", "physics_frame": 10_010},
        {"kind": "jump_onset", "physics_frame": 10_011},
    ]

    result = _paired_stream_assessment(
        {"queue_dropped_frame_count": 0, "termination_reason": "user_stop"},
        _valid_validation(),
        frames,
        events,
        [],
    )

    assert result["review_usable"]
    assert result["lifecycle_gap_episode_count"] == 1
    assert result["lifecycle_explained_global_frame_id_gap_count"] == 9
    assert result["unexplained_global_frame_id_gap_count"] == 0
    assert result["unpaired_event_count"] == 1


def test_completed_match_post_match_identity_shutdown_is_review_usable() -> None:
    frames = [_frame(0, stable=True)]
    frames[0]["match"] = {"flags": {"match_ended": True}}

    result = _paired_stream_assessment(
        {
            "queue_dropped_frame_count": 0,
            "termination_reason": "local_human_identity_lost:GetLocalCar returned null",
        },
        _valid_validation(clean=False),
        frames,
        [],
        [],
    )

    assert result["review_usable"]
    assert result["match_ended_in_paired_frames"]
    assert result["post_match_identity_shutdown"]


def test_source_bound_adjudication_partitions_all_attempts() -> None:
    document = _load_adjudication()
    success_count = 0
    total_count = 0
    for session in document["sessions"].values():
        attempts = set(range(1, int(session["attempt_count"]) + 1))
        successes = set(session["success_attempts"])
        ambiguous = set(session["ambiguous_attempts"])
        assert successes.isdisjoint(ambiguous)
        assert successes | ambiguous <= attempts
        success_count += len(successes)
        total_count += len(attempts)

    assert total_count == 195
    assert success_count == 110


def test_soccar_surface_geometry_distinguishes_ground_ceiling_and_wall() -> None:
    assert _closest_soccar_surface([0.0, 0.0, 92.75])["surface"] == "ground"
    assert _closest_soccar_surface([0.0, 0.0, 2044.0 - 92.75])["surface"] == "ceiling"
    assert _closest_soccar_surface([4096.0 - 92.75, 0.0, 500.0])["surface"] == "side_wall"


def _outcome_attempt(
    *,
    speed_gain: float,
    recovered: bool = True,
    transfer: float | None = None,
    goals: int = 0,
) -> dict[str, Any]:
    contacts = []
    primary_index = None
    if transfer is not None:
        contacts = [{"ball_delta_velocity_magnitude_12_ticks": transfer}]
        primary_index = 0
    return {
        "segmentation": {"generic_recovery_observed": recovered},
        "event_counts": {"goal": goals},
        "post_attempt_events_within_3_seconds": {"goals": []},
        "ball_outcome": {
            "human_touch_episode_count": len(contacts),
            "primary_contact_index": primary_index,
            "contacts": contacts,
        },
        "car_motion": {
            "planar_speed_gain_from_start_to_peak_uu_per_s": speed_gain,
        },
        "outcome_group": None,
    }


def test_relative_grouping_preserves_every_attempt_and_uses_session_quartiles() -> None:
    attempts = [_outcome_attempt(speed_gain=float(value)) for value in (10, 20, 30, 40)]

    report = _assign_outcome_groups(attempts)

    assert sum(report["counts"].values()) == len(attempts)
    assert report["relative_quartile_comparison_available"]
    assert attempts[0]["outcome_group"]["name"] == ("limited_or_failed_physical_outcome_candidate")
    assert attempts[-1]["outcome_group"]["name"] == "stronger_physical_outcome_candidate"


def test_single_contact_attempt_is_not_artificially_top_quartile() -> None:
    attempts = [_outcome_attempt(speed_gain=0.0, transfer=900.0)]

    report = _assign_outcome_groups(attempts)

    assert not report["relative_quartile_comparison_available"]
    assert attempts[0]["outcome_group"]["name"] == "ambiguous_middle_physical_outcome"


def _session_dir(root: Path, session_id: str, capture_start_utc: str) -> Path:
    session_dir = root / session_id
    session_dir.mkdir()
    (session_dir / "manifest.json").write_text(
        json.dumps({"capture_start_utc": capture_start_utc}),
        encoding="utf-8",
    )
    return session_dir


def test_explicit_session_selection_is_exact_and_chronological(tmp_path: Path) -> None:
    later = _session_dir(tmp_path, "later", "2026-08-28T19:01:00Z")
    earlier = _session_dir(tmp_path, "earlier", "2026-08-28T19:00:00Z")
    _session_dir(tmp_path, "unselected", "2026-08-28T18:59:00Z")

    selected = _select_session_dirs(tmp_path, [later.name, earlier.name])

    assert selected == [earlier, later]


def test_explicit_session_selection_rejects_missing_or_duplicate_ids(tmp_path: Path) -> None:
    present = _session_dir(tmp_path, "present", "2026-08-28T19:00:00Z")

    with pytest.raises(ValueError, match="requested session directories not found: missing"):
        _select_session_dirs(tmp_path, [present.name, "missing"])
    with pytest.raises(ValueError, match="duplicate --session-id values: present"):
        _select_session_dirs(tmp_path, [present.name, present.name])
