from __future__ import annotations

import torch

from rivalsim.rival2_aerial_option import FIELD
from rivalsim.rival2_aerial_option_v2 import (
    PHASE_GOAL_DIRECTED,
    PHASE_MOVING_INTERCEPT,
    AerialRewardTrackerV2,
    apply_fast_aerial_initiation,
)
from rivalsim.rival2_contracts import POSITION_SCALE


def test_fast_aerial_initiation_has_release_and_neutral_second_jump() -> None:
    learned = torch.full((4, 8), 0.25)
    age = torch.tensor((0, 24, 28, 29))
    action, overridden = apply_fast_aerial_initiation(
        learned, age, torch.ones(4, dtype=torch.bool)
    )
    assert overridden.tolist() == [True, True, True, False]
    assert action[0, 2].item() == -1.0
    assert action[0, 5].item() == 1.0
    assert action[1, 5].item() == 0.0
    assert action[2].tolist() == [1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 0.0]
    assert torch.equal(action[3], learned[3])


def _airborne_contact(car_height: float) -> tuple[torch.Tensor, torch.Tensor]:
    before = torch.zeros((1, 2, 182))
    after = before.clone()
    before[0, 0, FIELD["self.on_ground"]] = 0.0
    after[0, 0, FIELD["self.on_ground"]] = 0.0
    after[0, 0, FIELD["self.position.z"]] = car_height / POSITION_SCALE[2]
    after[0, 0, FIELD["ball.position.z"]] = 400.0 / POSITION_SCALE[2]
    after[0, 0, FIELD["lifecycle.self_touch_event"]] = 1.0
    return before, after


def test_moving_intercept_does_not_terminate_on_low_elevated_touch() -> None:
    tracker = AerialRewardTrackerV2(1, attacker_side=0, phase=PHASE_MOVING_INTERCEPT)
    before, after = _airborne_contact(200.0)
    reward, done = tracker.step(
        before,
        after,
        tick=60,
        goal_for_attacker=torch.zeros(1, dtype=torch.bool),
        active=torch.ones(1, dtype=torch.bool),
    )
    assert float(reward[0]) >= 1.0
    assert not bool(done[0])
    assert tracker.telemetry.elevated_contacts == 1
    assert tracker.telemetry.high_contacts == 0


def test_moving_intercept_terminates_on_genuine_high_touch() -> None:
    tracker = AerialRewardTrackerV2(1, attacker_side=0, phase=PHASE_MOVING_INTERCEPT)
    before, after = _airborne_contact(330.0)
    reward, done = tracker.step(
        before,
        after,
        tick=60,
        goal_for_attacker=torch.zeros(1, dtype=torch.bool),
        active=torch.ones(1, dtype=torch.bool),
    )
    assert float(reward[0]) >= 3.0
    assert bool(done[0])
    assert tracker.telemetry.high_contacts == 1


def test_goal_directed_requires_goal_after_high_touch() -> None:
    tracker = AerialRewardTrackerV2(1, attacker_side=0, phase=PHASE_GOAL_DIRECTED)
    before, after = _airborne_contact(330.0)
    _reward, touch_done = tracker.step(
        before,
        after,
        tick=60,
        goal_for_attacker=torch.zeros(1, dtype=torch.bool),
        active=torch.ones(1, dtype=torch.bool),
    )
    assert not bool(touch_done[0])
    goal_before = after.clone()
    goal_after = after.clone()
    goal_after[0, 0, FIELD["lifecycle.self_touch_event"]] = 0.0
    reward, goal_done = tracker.step(
        goal_before,
        goal_after,
        tick=61,
        goal_for_attacker=torch.ones(1, dtype=torch.bool),
        active=torch.ones(1, dtype=torch.bool),
    )
    assert bool(goal_done[0])
    assert float(reward[0]) >= 5.0
    assert tracker.telemetry.aerial_origin_goals == 1
