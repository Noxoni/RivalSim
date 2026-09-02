from __future__ import annotations

import numpy as np
import torch

from rivalsim.rival2_aerial_option import (
    FIELD,
    PHASE_EASY_LAUNCH,
    PHASE_GOAL_DIRECTED,
    PHASE_MOVING_INTERCEPT,
    AerialRewardTracker,
    build_aerial_scenarios,
)
from rivalsim.rival2_contracts import BALL_LINEAR_SPEED_SCALE, POSITION_SCALE


def test_aerial_scenarios_are_deterministic_grounded_and_unreachable_by_driving() -> None:
    for phase in (
        PHASE_EASY_LAUNCH,
        PHASE_MOVING_INTERCEPT,
        PHASE_GOAL_DIRECTED,
    ):
        first = build_aerial_scenarios(64, seed=7123, attacker_side=1, phase=phase)
        second = build_aerial_scenarios(64, seed=7123, attacker_side=1, phase=phase)
        first.state.validate()
        assert np.array_equal(first.state.car_pos, second.state.car_pos)
        assert np.array_equal(first.state.ball_pos, second.state.ball_pos)
        assert np.all(first.state.on_ground[:, 1] == 1)
        assert np.all(first.state.car_pos[:, 1, 2] == 17.0)
        assert np.all(first.state.ball_pos[:, 2] >= 260.0)
        assert np.all(first.state.ball_pos[:, 2] > first.state.car_pos[:, 1, 2] + 200.0)


def test_aerial_tracker_requires_physical_launch_and_elevated_contact() -> None:
    tracker = AerialRewardTracker(2, attacker_side=0, phase=PHASE_EASY_LAUNCH)
    before = torch.zeros((2, 2, 182), dtype=torch.float32)
    after = before.clone()
    before[:, 0, FIELD["self.on_ground"]] = 1.0
    after[:, 0, FIELD["self.on_ground"]] = 0.0
    before[:, 0, FIELD["self.position.z"]] = 17.0 / POSITION_SCALE[2]
    after[:, 0, FIELD["self.position.z"]] = 170.0 / POSITION_SCALE[2]
    before[:, 0, FIELD["ball.position.z"]] = 350.0 / POSITION_SCALE[2]
    after[:, 0, FIELD["ball.position.z"]] = 350.0 / POSITION_SCALE[2]
    before[:, 0, FIELD["relative.ball_position.z"]] = 333.0 / POSITION_SCALE[2]
    after[:, 0, FIELD["relative.ball_position.z"]] = 180.0 / POSITION_SCALE[2]
    before[:, 0, FIELD["self.forward.y"]] = 1.0
    after[:, 0, FIELD["self.forward.z"]] = 1.0
    active = torch.ones(2, dtype=torch.bool)
    no_goal = torch.zeros(2, dtype=torch.bool)

    launch_reward, launch_done = tracker.step(
        before, after, tick=1, goal_for_attacker=no_goal, active=active
    )
    assert torch.all(launch_reward > 0.0)
    assert not bool(launch_done.any())

    contact_before = after.clone()
    contact_after = contact_before.clone()
    contact_after[0, 0, FIELD["lifecycle.self_touch_event"]] = 1.0
    contact_after[0, 0, FIELD["ball.linear_velocity.y"]] = (
        800.0 / BALL_LINEAR_SPEED_SCALE
    )
    contact_reward, contact_done = tracker.step(
        contact_before,
        contact_after,
        tick=2,
        goal_for_attacker=no_goal,
        active=active,
    )
    assert float(contact_reward[0]) >= 2.0
    assert bool(contact_done[0])
    assert not bool(contact_done[1])
    assert tracker.telemetry.launches == 2
    assert tracker.telemetry.elevated_contacts == 1
    assert tracker.telemetry.high_contacts == 0
    assert tracker.telemetry.aerial_origin_goals == 0


def test_ground_contact_gets_no_aerial_event_reward() -> None:
    tracker = AerialRewardTracker(1, attacker_side=0, phase=PHASE_MOVING_INTERCEPT)
    before = torch.zeros((1, 2, 182), dtype=torch.float32)
    after = before.clone()
    before[0, 0, FIELD["self.on_ground"]] = 1.0
    after[0, 0, FIELD["self.on_ground"]] = 1.0
    after[0, 0, FIELD["lifecycle.self_touch_event"]] = 1.0
    after[0, 0, FIELD["ball.position.z"]] = 500.0 / POSITION_SCALE[2]
    reward, done = tracker.step(
        before,
        after,
        tick=1,
        goal_for_attacker=torch.zeros(1, dtype=torch.bool),
        active=torch.ones(1, dtype=torch.bool),
    )
    assert float(reward[0]) == 0.0
    assert not bool(done[0])
    assert tracker.telemetry.elevated_contacts == 0


def test_aerial_origin_goal_is_paid_once_after_a_high_contact() -> None:
    tracker = AerialRewardTracker(1, attacker_side=0, phase=PHASE_GOAL_DIRECTED)
    before = torch.zeros((1, 2, 182), dtype=torch.float32)
    after = before.clone()
    before[0, 0, FIELD["self.on_ground"]] = 1.0
    after[0, 0, FIELD["self.on_ground"]] = 0.0
    after[0, 0, FIELD["self.position.z"]] = 320.0 / POSITION_SCALE[2]
    after[0, 0, FIELD["ball.position.z"]] = 420.0 / POSITION_SCALE[2]
    after[0, 0, FIELD["lifecycle.self_touch_event"]] = 1.0
    active = torch.ones(1, dtype=torch.bool)
    tracker.step(
        before,
        after,
        tick=1,
        goal_for_attacker=torch.zeros(1, dtype=torch.bool),
        active=active,
    )
    goal_before = after.clone()
    goal_after = after.clone()
    first_reward, first_done = tracker.step(
        goal_before,
        goal_after,
        tick=2,
        goal_for_attacker=torch.ones(1, dtype=torch.bool),
        active=active,
    )
    second_reward, _ = tracker.step(
        goal_before,
        goal_after,
        tick=3,
        goal_for_attacker=torch.ones(1, dtype=torch.bool),
        active=active,
    )
    assert bool(first_done[0])
    assert float(first_reward[0]) >= 5.0
    assert float(second_reward[0]) < 5.0
    assert tracker.telemetry.high_contacts == 1
    assert tracker.telemetry.aerial_origin_goals == 1
