from __future__ import annotations

import numpy as np
import pytest
import torch

from rivalsim.rival2_aerial_option import FIELD
from rivalsim.rival2_contracts import BALL_LINEAR_SPEED_SCALE, POSITION_SCALE
from rivalsim.rival2_ground_to_air_goal_v3 import (
    PHASE_ATTACKING_HALF,
    PHASE_EASY_FINISH,
    GoalDirectedTracker,
    build_goal_directed_pop_scenarios,
)


@pytest.mark.parametrize("side", [0, 1])
@pytest.mark.parametrize("phase", [PHASE_EASY_FINISH, PHASE_ATTACKING_HALF])
def test_goal_directed_scenarios_are_low_ball_possession_starts(
    side: int, phase: int
) -> None:
    state = build_goal_directed_pop_scenarios(
        128, seed=19, attacker_side=side, phase=phase
    )
    sign = 1.0 if side == 0 else -1.0
    assert np.all((state.ball_pos[:, 2] >= 142.0) & (state.ball_pos[:, 2] <= 168.0))
    assert np.all(state.on_ground[:, side] == 1)
    assert np.all(state.boost[:, side] == 100.0)
    assert np.all(sign * state.ball_pos[:, 1] > 2_100.0)
    assert np.all(
        sign * (state.ball_pos[:, 1] - state.car_pos[:, side, 1]) >= 12.0
    )
    assert np.all(
        sign * (state.ball_pos[:, 1] - state.car_pos[:, side, 1]) <= 45.0
    )


def test_goal_directed_scenario_rejects_invalid_phase() -> None:
    with pytest.raises(ValueError):
        build_goal_directed_pop_scenarios(1, seed=1, attacker_side=0, phase=2)


def _observation(*, car_z: float, ball_z: float, touch: bool, ball_vy: float) -> torch.Tensor:
    observation = torch.zeros((1, 2, 182), dtype=torch.float32)
    for side in (0, 1):
        observation[0, side, FIELD["self.position.z"]] = car_z / POSITION_SCALE[2]
        observation[0, side, FIELD["ball.position.z"]] = ball_z / POSITION_SCALE[2]
        observation[0, side, FIELD["ball.linear_velocity.y"]] = (
            ball_vy / BALL_LINEAR_SPEED_SCALE
        )
        observation[0, side, FIELD["lifecycle.self_touch_event"]] = float(touch)
    return observation


@pytest.mark.parametrize("side", [0, 1])
def test_goal_tracker_requires_pop_and_elevated_follow_before_goal_credit(side: int) -> None:
    tracker = GoalDirectedTracker(1, attacker_side=side, horizon=40)
    active = torch.ones(1, dtype=torch.bool)
    false = torch.zeros(1, dtype=torch.bool)
    before = _observation(car_z=17.0, ball_z=160.0, touch=False, ball_vy=500.0)
    after = _observation(car_z=80.0, ball_z=190.0, touch=True, ball_vy=700.0)
    tracker.step(
        before,
        after,
        tick=0,
        goal_for_attacker=false,
        any_goal=false,
        active=active,
    )
    before = _observation(car_z=180.0, ball_z=280.0, touch=False, ball_vy=700.0)
    after = _observation(car_z=200.0, ball_z=310.0, touch=True, ball_vy=900.0)
    first = tracker.step(
        before,
        after,
        tick=10,
        goal_for_attacker=false,
        any_goal=false,
        active=active,
    )
    before = _observation(car_z=320.0, ball_z=360.0, touch=False, ball_vy=900.0)
    after = _observation(car_z=330.0, ball_z=370.0, touch=True, ball_vy=1100.0)
    second = tracker.step(
        before,
        after,
        tick=20,
        goal_for_attacker=false,
        any_goal=false,
        active=active,
    )
    goal = tracker.step(
        after,
        after,
        tick=30,
        goal_for_attacker=active,
        any_goal=active,
        active=active,
    )
    assert bool(first["first_elevated"].item())
    assert bool(second["second"].item())
    assert bool(second["high"].item())
    assert bool(goal["goal_after_follow"].item())
    assert tracker.telemetry.low_pop_touches == 1
    assert tracker.telemetry.elevated_follow_touches == 1
    assert tracker.telemetry.second_airborne_touches == 1
    assert tracker.telemetry.high_follow_touches == 1
    assert tracker.telemetry.goals_after_elevated_follow == 1
