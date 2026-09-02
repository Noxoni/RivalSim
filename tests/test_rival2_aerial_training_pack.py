from __future__ import annotations

import numpy as np
import torch

from rivalsim.rival2_aerial_option import FIELD
from rivalsim.rival2_aerial_training_pack import (
    PACK_AIRBORNE_POSSESSION,
    PACK_CENTER_POP,
    PACK_LATERAL_POP,
    AerialTrainingPackTracker,
    build_training_pack_scenarios,
)
from rivalsim.rival2_contracts import POSITION_SCALE


def test_training_pack_scenarios_are_deterministic_post_pop_launches() -> None:
    for pack in (PACK_CENTER_POP, PACK_LATERAL_POP, PACK_AIRBORNE_POSSESSION):
        first = build_training_pack_scenarios(64, seed=811, attacker_side=1, pack=pack)
        second = build_training_pack_scenarios(64, seed=811, attacker_side=1, pack=pack)
        assert np.array_equal(first.state.ball_pos, second.state.ball_pos)
        assert np.array_equal(first.state.ball_vel, second.state.ball_vel)
        assert np.all(first.state.on_ground[:, 1] == 1)
        assert np.all(first.state.ball_pos[:, 2] >= 280.0)
        assert np.all(first.state.ball_vel[:, 2] >= 80.0)
        assert np.all(first.state.ball_pos[:, 2] > 2.0 * 92.75)


def _state(
    *, car_height: float, ball_height: float, touch: bool = False
) -> torch.Tensor:
    observation = torch.zeros((1, 2, 182))
    observation[0, 0, FIELD["self.position.z"]] = car_height / POSITION_SCALE[2]
    observation[0, 0, FIELD["ball.position.z"]] = ball_height / POSITION_SCALE[2]
    observation[0, 0, FIELD["self.on_ground"]] = float(car_height <= 20.0)
    observation[0, 0, FIELD["lifecycle.self_touch_event"]] = float(touch)
    return observation


def _prime_launch(tracker: AerialTrainingPackTracker) -> None:
    ground = _state(car_height=17.0, ball_height=180.0)
    airborne = _state(car_height=40.0, ball_height=190.0)
    tracker.step(
        ground,
        airborne,
        tick=1,
        goal_for_attacker=torch.zeros(1, dtype=torch.bool),
        any_goal=torch.zeros(1, dtype=torch.bool),
        active=torch.ones(1, dtype=torch.bool),
    )


def test_pack_fails_immediately_when_launched_ball_returns_to_ground() -> None:
    tracker = AerialTrainingPackTracker(
        1,
        attacker_side=0,
        pack=PACK_CENTER_POP,
        first_touch_deadline=180,
        horizon=360,
    )
    _prime_launch(tracker)
    airborne = _state(car_height=200.0, ball_height=300.0)
    tracker.step(
        airborne,
        airborne,
        tick=50,
        goal_for_attacker=torch.zeros(1, dtype=torch.bool),
        any_goal=torch.zeros(1, dtype=torch.bool),
        active=torch.ones(1, dtype=torch.bool),
    )
    grounded = _state(car_height=50.0, ball_height=95.0)
    reward, done = tracker.step(
        airborne,
        grounded,
        tick=80,
        goal_for_attacker=torch.zeros(1, dtype=torch.bool),
        any_goal=torch.zeros(1, dtype=torch.bool),
        active=torch.ones(1, dtype=torch.bool),
    )
    assert bool(done[0])
    assert float(reward[0]) < 0.0
    assert tracker.telemetry.ball_ground_failures == 1


def test_pack_requires_high_touch_before_goal_credit() -> None:
    tracker = AerialTrainingPackTracker(
        1,
        attacker_side=0,
        pack=PACK_CENTER_POP,
        first_touch_deadline=180,
        horizon=360,
    )
    _prime_launch(tracker)
    low = _state(car_height=180.0, ball_height=260.0, touch=True)
    _reward, low_done = tracker.step(
        low,
        low,
        tick=70,
        goal_for_attacker=torch.zeros(1, dtype=torch.bool),
        any_goal=torch.zeros(1, dtype=torch.bool),
        active=torch.ones(1, dtype=torch.bool),
    )
    assert not bool(low_done[0])
    high = _state(car_height=330.0, ball_height=400.0, touch=True)
    tracker.step(
        high,
        high,
        tick=90,
        goal_for_attacker=torch.zeros(1, dtype=torch.bool),
        any_goal=torch.zeros(1, dtype=torch.bool),
        active=torch.ones(1, dtype=torch.bool),
    )
    goal_state = _state(car_height=250.0, ball_height=350.0)
    reward, goal_done = tracker.step(
        goal_state,
        goal_state,
        tick=120,
        goal_for_attacker=torch.ones(1, dtype=torch.bool),
        any_goal=torch.ones(1, dtype=torch.bool),
        active=torch.ones(1, dtype=torch.bool),
    )
    assert bool(goal_done[0])
    assert float(reward[0]) >= 8.0
    assert tracker.telemetry.first_high_touches == 1
    assert tracker.telemetry.goals == 1


def test_pack_fails_at_intercept_deadline_without_high_touch() -> None:
    tracker = AerialTrainingPackTracker(
        1,
        attacker_side=0,
        pack=PACK_LATERAL_POP,
        first_touch_deadline=240,
        horizon=420,
    )
    _prime_launch(tracker)
    state = _state(car_height=250.0, ball_height=400.0)
    _reward, done = tracker.step(
        state,
        state,
        tick=240,
        goal_for_attacker=torch.zeros(1, dtype=torch.bool),
        any_goal=torch.zeros(1, dtype=torch.bool),
        active=torch.ones(1, dtype=torch.bool),
    )
    assert bool(done[0])
    assert tracker.telemetry.missed_intercept_failures == 1
