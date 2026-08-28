from __future__ import annotations

from typing import Any

from benchmarks.review_rival2_human_demos import (
    _assign_outcome_groups,
    _segment_attempts,
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
