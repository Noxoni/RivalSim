from __future__ import annotations

import json
from pathlib import Path

RESULTS = Path(__file__).resolve().parents[1] / "results" / "v0.3"


def _load(name: str) -> dict:
    return json.loads((RESULTS / name).read_text(encoding="utf-8"))


def test_v03_phase_evidence_is_complete_and_green() -> None:
    ball_world = _load("ball_world.json")
    car_ball = _load("car_ball.json")
    car_car = _load("car_car.json")
    integrated = _load("integrated.json")
    assert ball_world["selection"]["selected_case_count"] == 31_216
    assert ball_world["gate"]["phase_a_complete_gate_pass"] is True
    assert car_ball["selection"]["selected_case_count"] == 8_192
    assert car_ball["gate"]["phase_b_complete_gate_pass"] is True
    assert car_car["full_gate"]["selected_case_count"] == 8_192
    assert car_car["full_gate"]["failed_case_count"] == 0
    assert integrated["case_count"] == 512
    assert integrated["gate"]["phase_d_complete_gate_pass"] is True


def test_v03_relational_authority_keeps_complete_native_branches() -> None:
    car_car = _load("car_car.json")
    integrated = _load("integrated.json")
    relation = car_car["native_multi_outcome_relation"]
    assert relation["branches"] == ["a_then_b", "b_then_a"]
    assert relation["metric_branch_mixing"] is False
    assert relation["runtime_best_match_selection"] is False
    assert relation["native_pointer_or_allocator_emulation"] is False
    assert integrated["native_branches"] == ["a_then_b", "b_then_a"]
    assert integrated["relation"]["metric_mixing"] is False
    assert integrated["relation"]["best_match_runtime_selection"] is False


def test_v03_oracle_custody_binds_every_phase() -> None:
    oracle = _load("oracle_data.json")
    assert oracle["status"] == "COMPLETE_NATIVE_AUTHORITY"
    assert oracle["policy"]["isolated_native_world_per_case"] is True
    assert oracle["policy"]["all_ticks_1_through_12_cached"] is True
    assert oracle["policy"]["live_fallback_after_freeze"] is False
    assert {name: phase["case_count"] for name, phase in oracle["phases"].items()} == {
        "ball_world": 31_216,
        "car_ball": 8_192,
        "car_car": 8_192,
        "integrated": 512,
    }
    assert all(
        phase["status"] == "COMPLETE_NATIVE_AUTHORITY"
        for phase in oracle["phases"].values()
    )


def test_v03_regression_determinism_and_performance_are_green() -> None:
    regression = _load("regression.json")
    benchmark = _load("benchmark.json")
    assert regression["status"] == "PASS_GREEN"
    assert regression["v0_2_2_static_acceptance"]["counts"]["failed_cases"] == 0
    assert regression["v0_1_live_rocketsim"]["scenario_count"] == 27
    assert all(
        backend["exact_hit_distance_normal_pass"]
        for backend in regression["arena_query_backends"]["backends"].values()
    )
    assert regression["deterministic_stress"]["deterministic_equal"] is True
    assert benchmark["summary"]["verdict"] == "PASS_GREEN"
    assert benchmark["summary"]["best_aggregate_simulated_game_seconds_per_s"] >= 100_000
    assert benchmark["summary"]["hot_loop_gpu_resident"] is True
