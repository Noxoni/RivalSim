from __future__ import annotations

import json
from pathlib import Path

import torch

from benchmarks import run_rival2_ground_to_air_natural_v10 as v10
from rivalsim.rival2_contracts import POSITION_SCALE
from rivalsim.rival2_ground_to_air_option import FIELD
from rivalsim.rival2_ground_to_air_prompt_follow_v10 import (
    PromptAerialFollowTrainingTracker,
)

ROOT = Path(__file__).resolve().parents[1]


def _observation(*, car_z: float, ball_z: float, touch: bool, grounded: bool) -> torch.Tensor:
    observation = torch.zeros((1, 2, 182), dtype=torch.float32)
    observation[:, 0, FIELD["self.position.z"]] = car_z / POSITION_SCALE[2]
    observation[:, 0, FIELD["ball.position.z"]] = ball_z / POSITION_SCALE[2]
    observation[:, 0, FIELD["relative.ball_position.z"]] = (
        ball_z - car_z
    ) / POSITION_SCALE[2]
    observation[:, 0, FIELD["lifecycle.self_touch_event"]] = float(touch)
    observation[:, 0, FIELD["self.on_ground"]] = float(grounded)
    return observation


def _authority() -> dict[str, object]:
    authority = json.loads(
        (
            ROOT / "results/rival2/ground_to_air_natural_v9/authority.json"
        ).read_text(encoding="utf-8")
    )
    authority["reward"]["prompt_airborne_follow_event"] = 1.5
    return authority


def test_prompt_follow_reward_is_once_per_attempt_and_auditable() -> None:
    tracker = PromptAerialFollowTrainingTracker(
        1,
        attacker_side=0,
        horizon=600,
        authority=_authority(),
    )
    active = torch.ones(1, dtype=torch.bool)
    no_goal = torch.zeros(1, dtype=torch.bool)
    before = _observation(car_z=17.0, ball_z=120.0, touch=False, grounded=True)
    setup = _observation(car_z=20.0, ball_z=130.0, touch=True, grounded=True)
    tracker.step(
        before,
        setup,
        tick=2,
        goal_for_attacker=no_goal,
        any_goal=no_goal,
        active=active,
    )
    before_follow = _observation(
        car_z=70.0, ball_z=180.0, touch=False, grounded=False
    )
    follow = _observation(
        car_z=75.0, ball_z=185.0, touch=True, grounded=False
    )
    tracker.step(
        before_follow,
        follow,
        tick=24,
        goal_for_attacker=no_goal,
        any_goal=no_goal,
        active=active,
    )
    tracker.step(
        before_follow,
        follow,
        tick=30,
        goal_for_attacker=no_goal,
        any_goal=no_goal,
        active=active,
    )
    telemetry = tracker.telemetry()
    assert telemetry["prompt_airborne_follow_touches"] == 1
    assert telemetry["prompt_airborne_follow_reward_sum"] == 1.5
    assert (
        telemetry["touch_geometry_probe"]["categories"][
            "first_prompt_airborne_follow"
        ]["attempt_fraction"]
        == 1.0
    )


def test_selection_caps_completed_prompt_gate_before_downstream_score() -> None:
    authority = {
        "acceptance": {
            "per_setup_and_defender": {
                "low_bounce": {
                    "parked": {"prompt_airborne_follow_fraction_min": 0.4}
                }
            }
        }
    }
    row = {
        "setup": "low_bounce",
        "defender_mode": "parked",
        "fractions": {
            "prompt_airborne_follow": 0.8,
            "goal_within_contact_budget": 0.0,
            "productive_continuation": 0.0,
            "sustained_control": 0.0,
            "second_airborne_touch": 0.0,
            "high_follow_touch": 0.0,
            "elevated_follow_touch": 0.0,
            "goalward_velocity_contact": 0.0,
            "contact_budget_exceeded": 0.0,
        },
    }
    primary, secondary = v10.selection_key([row], authority)
    assert primary == 1.0
    assert secondary == 0.4
