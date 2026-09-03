from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path

import numpy as np

from rivalsim.rival2_dash_scripted_probe import (
    action_for_case,
    analyze_dash_trace,
    build_dash_probe_cases,
    summarize_dash_results,
)

ROOT = Path(__file__).resolve().parents[1]
TIMING = ROOT / "results/rival2/dash_physical_calibration_v1/human_timing.json"


def cases():
    return build_dash_probe_cases(json.loads(TIMING.read_text(encoding="utf-8")))


def test_case_sweep_is_source_bounded_and_has_controls() -> None:
    selected = cases()
    floor = [case for case in selected if case.family == "floor_wavedash"]
    wall = [case for case in selected if case.family == "wall_dash"]
    assert len(selected) == 263
    assert {case.first_jump_hold_ticks for case in floor} == {5, 6, 7}
    assert min(case.second_jump_tick for case in floor) == 116
    assert max(case.second_jump_tick for case in floor) == 131
    assert {case.first_jump_hold_ticks for case in wall} == {4, 5}
    assert {case.second_jump_tick for case in wall} == {11, 12, 13}
    assert {case.wall_sign for case in wall} == {-1, 1}
    assert {case.boost for case in wall} == {False, True}
    assert sum(not case.fire_second_jump for case in floor) == 15
    assert sum(not case.fire_second_jump for case in wall) == 32


def test_actions_encode_human_timing_without_input_injection() -> None:
    floor = next(
        case
        for case in cases()
        if case.family == "floor_wavedash"
        and case.fire_second_jump
        and case.second_jump_tick == 116
    )
    assert action_for_case(floor, 0)[5] == 1.0
    assert action_for_case(floor, floor.first_jump_hold_ticks)[5] == 0.0
    assert action_for_case(floor, 110)[2] == -1.0
    assert action_for_case(floor, 116)[5] == 1.0

    wall = next(
        case
        for case in cases()
        if case.family == "wall_dash"
        and case.fire_second_jump
        and case.second_jump_tick == 12
        and case.wall_sign == -1
        and case.boost
    )
    start = action_for_case(wall, 0)
    assert start[0] == 1.0
    assert start[1] == -1.0
    assert start[2] == 1.0
    assert start[3] == -1.0
    assert start[5] == 1.0
    assert start[6] == 1.0
    dodge = action_for_case(wall, 12)
    assert dodge[1] == 1.0
    assert dodge[2] == -1.0
    assert dodge[3] == 1.0
    assert dodge[5] == 1.0


def synthetic_trace(worlds: int, state_ticks: int = 140) -> dict[str, np.ndarray]:
    shape = (state_ticks, worlds)
    trace = {
        "car_position": np.zeros((state_ticks, worlds, 3), dtype=np.float32),
        "car_velocity": np.zeros((state_ticks, worlds, 3), dtype=np.float32),
        "on_ground": np.ones(shape, dtype=np.int32),
        "has_flipped": np.zeros(shape, dtype=np.int32),
        "is_flipping": np.zeros(shape, dtype=np.int32),
        "wheel_contact_count": np.full(shape, 4, dtype=np.int32),
        "world_contact_normal": np.zeros(
            (state_ticks, worlds, 3), dtype=np.float32
        ),
        "action": np.zeros((state_ticks - 1, worlds, 8), dtype=np.float32),
    }
    trace["world_contact_normal"][..., 2] = 1.0
    return trace


def test_trace_analysis_requires_a_source_timed_productive_landing() -> None:
    source = next(
        case
        for case in cases()
        if case.family == "floor_wavedash"
        and case.fire_second_jump
        and case.second_jump_tick == 116
    )
    control = replace(source, case_id="control", fire_second_jump=False)
    trace = synthetic_trace(2)
    trace["on_ground"][1:118] = 0
    trace["wheel_contact_count"][1:118] = 0
    trace["has_flipped"][117:] = 1
    trace["car_velocity"][:, :, 1] = 100.0
    trace["car_velocity"][118:, :, 1] = 300.0
    rows = analyze_dash_trace([source, control], trace)
    assert rows[0]["flip_action_tick"] == 116
    assert rows[0]["landing_action_tick"] == 117
    assert rows[0]["requested_to_landing_ticks"] == 1
    assert rows[0]["surface_tangent_speed_gain_uu_per_second"] == 200.0
    assert rows[0]["productive_source_timed_landing"] is True
    assert rows[1]["productive_source_timed_landing"] is False
    summary = summarize_dash_results(rows)
    assert summary["floor_wavedash"]["positive_success_fraction"] == 1.0
    assert summary["floor_wavedash"]["control_success_fraction"] == 0.0
