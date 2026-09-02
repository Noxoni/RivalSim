from __future__ import annotations

import numpy as np
import pytest

from benchmarks.analyze_rival2_v23_physical_telemetry import _analyze_subset


def test_inactive_overtime_padding_is_excluded_from_physical_metrics() -> None:
    ticks = 4
    scalar = lambda dtype=np.float32: np.zeros((ticks, 1), dtype=dtype)
    trace = {
        name: scalar()
        for name in (
            "horizontal_speed",
            "car_z",
            "ball_z",
            "ball_velocity_y_before",
            "ball_velocity_y_after",
            "world_contact_normal_z",
            "ball_speed_after",
            "car_y",
            "ball_y",
            "car_x",
            "opponent_x",
            "opponent_y",
        )
    }
    trace.update(
        {
            name: scalar(np.int16)
            for name in (
                "match_active",
                "on_ground",
                "rival_hit_raw",
                "nexto_hit_raw",
                "has_flipped",
                "is_flipping",
                "goal_scored",
                "scoring_team",
                "pre_tick_first_car",
                "rival_demo_count",
                "is_supersonic",
            )
        }
    )
    trace["action"] = np.zeros((ticks, 1, 8), dtype=np.float32)
    trace["match_active"][:2] = 1
    trace["on_ground"][0] = 1
    trace["car_z"][:, 0] = (17.0, 120.0, 9999.0, 9999.0)
    trace["ball_z"][:, 0] = (93.15, 200.0, 9999.0, 9999.0)
    trace["rival_hit_raw"][1:] = 1
    trace["action"][1:, 0, 5] = 1.0

    report = _analyze_subset(trace, np.asarray([0]), np.asarray([0]))

    assert report["match_minutes"] == pytest.approx(2.0 / 120.0 / 60.0)
    assert report["air"]["airborne_tick_fraction"] == pytest.approx(0.5)
    assert report["air"]["maximum_car_height_uu"] == pytest.approx(120.0)
    assert report["touches"]["total"] == 1
    assert report["jump_flip_recovery"]["jump_onsets"] == 1
    assert report["jump_flip_recovery"]["landings"] == 0

