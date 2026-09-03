from __future__ import annotations

import numpy as np
import torch

from benchmarks import run_rival2_ground_to_air_hybrid_v1 as hybrid
from rivalsim.rival2_aerial_option import FIELD
from rivalsim.rival2_contracts import BALL_LINEAR_SPEED_SCALE, POSITION_SCALE
from rivalsim.rival2_ground_to_air_hybrid import (
    NaturalGroundToAirGateConfig,
    natural_ground_to_air_eligibility,
)
from rivalsim.rival2_ground_to_air_option import GroundToAirConfig


def _trace(ticks: int = 12, worlds: int = 2) -> dict[str, np.ndarray]:
    integer = np.zeros((ticks, worlds), dtype=np.int16)
    floating = np.zeros((ticks, worlds), dtype=np.float32)
    trace = {
        "match_active": np.ones((ticks, worlds), dtype=np.int16),
        "option_active": integer.copy(),
        "option_activated": integer.copy(),
        "option_eligible": integer.copy(),
        "option_pop_started": integer.copy(),
        "option_pop_primitive": integer.copy(),
        "option_learned_control": integer.copy(),
        "rival_hit_raw": integer.copy(),
        "on_ground": np.ones((ticks, worlds), dtype=np.int16),
        "ball_z": floating.copy(),
        "car_z": floating.copy(),
        "goal_scored": integer.copy(),
        "scoring_team": integer.copy(),
    }
    return trace


def test_trace_summary_counts_contact_onsets_and_option_goal() -> None:
    trace = _trace()
    trace["option_active"][1:9, 0] = 1
    trace["option_activated"][1, 0] = 1
    trace["option_eligible"][1, 0] = 1
    trace["option_pop_started"][2, 0] = 1
    trace["option_pop_primitive"][2:4, 0] = 1
    trace["option_learned_control"][4:9, 0] = 1
    trace["rival_hit_raw"][[2, 5, 7], 0] = 1
    trace["on_ground"][5:9, 0] = 0
    trace["ball_z"][5:9, 0] = 420.0
    trace["car_z"][5:9, 0] = 350.0
    trace["goal_scored"][8, 0] = 1
    trace["scoring_team"][8, 0] = 0
    result = hybrid.option_trace_summary(trace, np.asarray([0, 1]))
    assert result["activations"] == 1
    assert result["pop_starts"] == 1
    assert result["option_touches"] == 3
    assert result["elevated_option_touches"] == 2
    assert result["high_option_touches"] == 2
    assert result["option_goals"] == 1
    assert result["activation_contact_counts"] == [3]
    assert result["activations_over_six_contacts"] == 0


def test_trace_summary_flags_more_than_six_distinct_contacts() -> None:
    trace = _trace(ticks=16, worlds=1)
    trace["option_active"][1:15, 0] = 1
    trace["option_activated"][1, 0] = 1
    trace["rival_hit_raw"][[1, 3, 5, 7, 9, 11, 13], 0] = 1
    result = hybrid.option_trace_summary(trace, np.asarray([0]))
    assert result["maximum_contacts_per_activation"] == 7
    assert result["activations_over_six_contacts"] == 1


def test_promotion_gate_does_not_average_away_option_failures() -> None:
    authority = {
        "validation": {
            "baseline": {"touches": 100, "touches_per_minute": 10.0},
            "promotion_gate": {
                "minimum_wins": 6,
                "maximum_losses": 4,
                "minimum_goal_differential": 0,
                "minimum_touch_fraction": 0.8,
                "minimum_touch_rate_fraction": 0.8,
                "maximum_no_touch_worlds": 0,
                "minimum_activations": 2,
                "minimum_activations_per_side": 1,
                "minimum_pop_starts": 1,
                "minimum_elevated_option_touches": 1,
                "minimum_high_option_touches": 1,
                "minimum_option_goals": 1,
                "maximum_activations_over_six_contacts": 0,
            },
        }
    }
    report = {
        "evaluation": {
            "score": {"wins": 6, "losses": 4, "rival_goals": 12, "nexto_goals": 10},
            "no_touch_worlds": 0,
            "finite_actions_and_observations": True,
        },
        "overall": {"touches": {"total": 90, "per_minute": 9.0}},
        "option": {
            "activations": 2,
            "pop_starts": 2,
            "elevated_option_touches": 2,
            "high_option_touches": 1,
            "option_goals": 0,
            "activations_over_six_contacts": 0,
            "per_side": {
                "blue": {"activations": 1},
                "orange": {"activations": 1},
            },
        },
    }
    passed, checks = hybrid.promotion_verdict(report, authority)
    assert not passed
    assert not checks["option_goals"]
    report["option"]["option_goals"] = 1
    passed, checks = hybrid.promotion_verdict(report, authority)
    assert passed
    assert all(checks.values())


def test_natural_gate_requires_validated_attacking_goal_envelope() -> None:
    observation = torch.zeros((3, 182), dtype=torch.float32)
    observation[:, FIELD["self.on_ground"]] = 1.0
    observation[:, FIELD["self.boost"]] = 1.0
    observation[:, FIELD["self.forward.y"]] = 1.0
    observation[:, FIELD["self.up.z"]] = 1.0
    observation[:, FIELD["relative.ball_position.y"]] = 40.0 / POSITION_SCALE[1]
    observation[:, FIELD["ball.position.z"]] = 150.0 / POSITION_SCALE[2]
    observation[:, FIELD["ball.position.y"]] = (
        torch.tensor([3_500.0, 2_500.0, 3_500.0]) / POSITION_SCALE[1]
    )
    observation[:, FIELD["ball.linear_velocity.y"]] = (
        torch.tensor([300.0, 300.0, 1_200.0]) / BALL_LINEAR_SPEED_SCALE
    )
    result = natural_ground_to_air_eligibility(
        observation,
        option_config=GroundToAirConfig(),
        gate_config=NaturalGroundToAirGateConfig(),
    )
    assert result.physical_eligible.tolist() == [True, True, True]
    assert result.eligible.tolist() == [True, False, False]
