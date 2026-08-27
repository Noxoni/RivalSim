from __future__ import annotations

import inspect
import json
import os
from pathlib import Path

import numpy as np
import pytest
import warp as wp

from benchmarks.run_rival2_mechanics_calibration import (
    CALIBRATION_FEATURES,
    TARGET_FAMILIES,
    TARGET_HELDOUT_SCENARIO_OFFSET,
    _scenario_rows,
)
from rivalsim.kernels.rival2 import (
    REWARD_MODE_GAMEPLAY,
    STRICT_DASH_LANDING_WINDOW_TICKS,
    STRICT_DASH_LOW_AIR_TICKS,
    STRICT_DOUBLE_DASH_WINDOW_TICKS,
)
from rivalsim.mechanics_calibration import (
    DASH_AIR_TICKS,
    DASH_LANDING_TICKS,
    DOUBLE_DASH_TICKS,
    FAMILY_NAMES,
    RESET_SUPPORT_BALL,
    RESET_SUPPORT_CAR,
    SURFACE_FLOOR_CEILING_NZ,
    SURFACE_WALL_NZ,
    THRESHOLD_NAMES,
    ZAP_DODGE_TICKS,
    ZAP_JUMP_TICKS,
    MechanicsShadowObserver,
    canonical_family_events,
    classify,
    midpoint_boundary,
    source_exact_reset_acquisition,
    source_exact_reset_rearmed,
)
from rivalsim.rival2_env import Rival2WorldSim


def test_midpoint_uses_narrowest_clean_physical_margin() -> None:
    positives = [{"value": 8.0}, {"value": 10.0}, {"value": 12.0}]
    negatives = [{"value": 1.0}, {"value": 5.0}, {"value": 7.0}]
    boundary = midpoint_boundary("value", "min", positives, negatives)
    assert boundary is not None
    assert boundary.threshold == 7.5
    assert boundary.margin == 1.0
    assert classify({"value": 8.0}, [boundary])
    assert not classify({"value": 7.0}, [boundary])


def test_overlap_is_not_force_fit() -> None:
    assert midpoint_boundary("value", "min", [{"value": 2.0}], [{"value": 3.0}]) is None


def test_every_continuous_detector_has_exact_72_case_split() -> None:
    for family in FAMILY_NAMES:
        rows = _scenario_rows(family)
        assert len(rows) == 72
        for class_name in ("positive", "near_miss", "ordinary_control"):
            selected = [row for row in rows if row["class"] == class_name]
            assert len(selected) == 24
            assert sum(row["split"] == "derivation" for row in selected) == 16
            assert sum(row["split"] == "heldout" for row in selected) == 8

        if family in TARGET_FAMILIES:
            derivation = [row for row in rows if row["split"] == "derivation"]
            heldout = [row for row in rows if row["split"] == "heldout"]
            assert {row["scenario_variant"] for row in derivation} == {0}
            assert {row["scenario_variant"] for row in heldout} == {
                TARGET_HELDOUT_SCENARIO_OFFSET
            }

    runtime_names = set(THRESHOLD_NAMES)
    for candidates in CALIBRATION_FEATURES.values():
        for _feature, _direction, runtime_name in candidates:
            assert runtime_name.startswith("discrete_") or runtime_name in runtime_names


def test_frozen_threshold_abi_prefix_is_not_renumbered() -> None:
    assert THRESHOLD_NAMES[:36] == (
        "speedflip_pitch_rotation_max",
        "speedflip_alignment_min",
        "speedflip_cancel_ticks_max",
        "half_flip_pitch_rotation_max",
        "half_flip_heading_dot_max",
        "half_flip_new_forward_speed_min",
        "possession_distance_max",
        "possession_relative_speed_max",
        "possession_gap_ticks_max",
        "carry_distance_max",
        "carry_relative_speed_max",
        "carry_support_ticks_min",
        "musty_rotational_normal_speed_min",
        "musty_rotational_fraction_min",
        "musty_ball_delta_v_min",
        "breezi_roll_path_min",
        "breezi_yaw_path_min",
        "breezi_setup_ticks_min",
        "redirect_incoming_speed_min",
        "redirect_outgoing_speed_min",
        "redirect_angle_min_radians",
        "pinch_overlap_ticks_max",
        "pinch_opposition_min",
        "pinch_closing_speed_min",
        "pogo_corner_region_min",
        "pogo_incoming_normal_speed_min",
        "pogo_outgoing_normal_speed_min",
        "pogo_wheel_support_max",
        "pogo_separation_ticks_max",
        "half_flip_cancel_ticks_min",
        "half_flip_cancel_ticks_max",
        "pinch_ball_delta_v_min",
        "breezi_setup_ticks_max",
        "breezi_nose_up_min",
        "breezi_inverted_depth_min",
        "breezi_nose_down_depth_min",
    )


def test_targeted_gpu_kernel_uses_signed_sweep_and_no_host_hot_path() -> None:
    kernel_source = inspect.getsource(
        __import__(
            "rivalsim.mechanics_calibration", fromlist=["collect_mechanics_shadow_tick"]
        ).collect_mechanics_shadow_tick
    )
    launch_source = inspect.getsource(MechanicsShadowObserver._launch_tick)
    assert "flip_rel_torque[car][1] < -0.25" in kernel_source
    assert "rotational_closing = wp.dot(rotational_velocity, car_to_ball)" in kernel_source
    assert "sweep_closure" in kernel_source
    assert "setup_terminal_pending" in kernel_source
    assert "approach_cross_fraction" in kernel_source
    assert ".numpy(" not in launch_source


def test_source_exact_reset_resource_and_body_identity() -> None:
    common = {
        "pre_untimed_resource": False,
        "post_untimed_resource": True,
        "supporting_wheels": 3,
        "separated": True,
        "airborne": True,
        "new_ground_jump": False,
    }
    assert source_exact_reset_acquisition(
        **common,
        support_body=RESET_SUPPORT_BALL,
        expected_support_body=RESET_SUPPORT_BALL,
    )
    assert source_exact_reset_acquisition(
        **common,
        support_body=RESET_SUPPORT_CAR,
        expected_support_body=RESET_SUPPORT_CAR,
    )
    assert not source_exact_reset_acquisition(
        **common,
        support_body=RESET_SUPPORT_BALL,
        expected_support_body=RESET_SUPPORT_CAR,
    )
    assert not source_exact_reset_acquisition(
        **{**common, "pre_untimed_resource": True},
        support_body=RESET_SUPPORT_BALL,
        expected_support_body=RESET_SUPPORT_BALL,
    )


def test_chain_and_preflip_rearm_require_real_resource_transition() -> None:
    assert not source_exact_reset_rearmed(
        acquired_token_active=True,
        resource_consumed_or_lost=False,
        lockout_ended=False,
    )
    assert source_exact_reset_rearmed(
        acquired_token_active=True,
        resource_consumed_or_lost=True,
        lockout_ended=False,
    )
    assert source_exact_reset_rearmed(
        acquired_token_active=True,
        resource_consumed_or_lost=False,
        lockout_ended=True,
    )


def test_source_exact_dash_windows_and_surface_classes_are_frozen() -> None:
    assert (DASH_AIR_TICKS, DASH_LANDING_TICKS) == (42, 24)
    assert (ZAP_JUMP_TICKS, ZAP_DODGE_TICKS) == (12, 30)
    assert DOUBLE_DASH_TICKS == 90
    assert DASH_AIR_TICKS == STRICT_DASH_LOW_AIR_TICKS
    assert DASH_LANDING_TICKS == STRICT_DASH_LANDING_WINDOW_TICKS
    assert DOUBLE_DASH_TICKS == STRICT_DOUBLE_DASH_WINDOW_TICKS
    assert SURFACE_FLOOR_CEILING_NZ == 0.85
    assert SURFACE_WALL_NZ == 0.25


def test_same_family_subtype_dedup_and_compound_observability() -> None:
    events = canonical_family_events(
        [
            ("movement", "wavedash"),
            ("movement", "wall_wavedash"),
            ("reset", "ball_reset"),
            ("flick", "musty"),
            ("flick", "breezi"),
        ]
    )
    assert events == [
        ("movement", "wavedash"),
        ("reset", "ball_reset"),
        ("flick", "breezi"),
    ]


def test_shadow_observer_is_gpu_resident_and_reward_free(tmp_path: Path) -> None:
    collision_root = os.environ.get("RIVALSIM_COLLISION_DIR")
    if not collision_root or not wp.is_cuda_available():
        pytest.skip("exact local CMFs and CUDA are required")
    threshold_path = tmp_path / "thresholds.json"
    threshold_path.write_text(
        json.dumps(
            {
                "detectors": {
                    family: {"status": "NOT_READY_FOR_REWARD", "boundaries": []}
                    for family in FAMILY_NAMES
                }
            }
        ),
        encoding="utf-8",
    )
    world = Rival2WorldSim(
        2,
        collision_root,
        reward_mode=REWARD_MODE_GAMEPLAY,
        seed=2026082703,
    )
    observer = MechanicsShadowObserver(world, threshold_path)
    observer.attach()
    world.begin_decision()
    world.step(1, synchronize=True)
    raw = observer.numpy()
    assert raw["family_event_count"].shape == (2, 2, len(FAMILY_NAMES))
    np.testing.assert_array_equal(raw["family_event_count"], 0)
    np.testing.assert_array_equal(raw["reward_contribution"], 0.0)
