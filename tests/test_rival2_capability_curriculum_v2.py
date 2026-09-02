from __future__ import annotations

import numpy as np
import torch

from rivalsim.rival2_capability_curriculum_v2 import (
    FIELD,
    SCENARIO_FLOOR_LANDING,
    SCENARIO_HIGH_BALL,
    SCENARIO_OFFENSIVE_DEMO,
    CapabilityRewardTrackerV2,
    build_capability_scenarios_v2,
)
from rivalsim.rival2_contracts import CAR_LINEAR_SPEED_SCALE, POSITION_SCALE


def test_v2_scenarios_are_deterministic_side_fixed_and_ground_origin() -> None:
    first = build_capability_scenarios_v2(100, seed=991, attacker_side=1)
    second = build_capability_scenarios_v2(100, seed=991, attacker_side=1)
    first.state.validate()
    assert np.array_equal(first.scenario, second.scenario)
    assert np.array_equal(first.state.car_pos, second.state.car_pos)
    assert np.bincount(first.scenario, minlength=4).tolist() == [45, 20, 10, 25]
    assert np.unique(first.attacker_side).tolist() == [1]
    high = first.scenario == SCENARIO_HIGH_BALL
    assert np.all(first.state.on_ground[high, 1] == 1)
    assert np.all(first.state.car_pos[high, 1, 2] == 17.0)
    assert np.all(first.state.ball_pos[high, 2] >= 430.0)


def test_v2_tracker_pays_only_literal_high_contact_landing_and_demo_events() -> None:
    scenario = torch.tensor(
        [SCENARIO_HIGH_BALL, SCENARIO_FLOOR_LANDING, SCENARIO_OFFENSIVE_DEMO]
    )
    tracker = CapabilityRewardTrackerV2(scenario, attacker_side=0)
    before = torch.zeros((3, 2, 182), dtype=torch.float32)
    after = before.clone()
    normal = torch.zeros((3, 3), dtype=torch.float32)
    normal[:, 2] = 1.0
    no_goal = torch.zeros(3, dtype=torch.bool)

    after[0, 0, FIELD["self.position.z"]] = 350.0 / POSITION_SCALE[2]
    after[0, 0, FIELD["ball.position.z"]] = 500.0 / POSITION_SCALE[2]
    after[0, 0, FIELD["ball.linear_velocity.y"]] = 1000.0 / 6000.0
    after[0, 0, FIELD["lifecycle.self_touch_event"]] = 1.0
    after[1, 0, FIELD["self.linear_velocity.y"]] = 1000.0 / CAR_LINEAR_SPEED_SCALE
    after[1, 0, FIELD["self.is_flipping"]] = 1.0
    onset = tracker.step(
        before,
        after,
        tick=4,
        world_contact_normal=normal,
        goal_for_attacker=no_goal,
    )
    assert float(onset[0]) >= 1.0
    assert float(onset[1]) == 0.0

    landing_before = after.clone()
    landing_before[0, 0, FIELD["lifecycle.self_touch_event"]] = 0.0
    landing_after = landing_before.clone()
    landing_after[1, 0, FIELD["self.is_flipping"]] = 0.0
    landing_after[1, 0, FIELD["self.on_ground"]] = 1.0
    landing_after[1, 0, FIELD["self.wheel_contact.front_left"]] = 1.0
    landing_after[1, 0, FIELD["self.linear_velocity.y"]] = 1250.0 / CAR_LINEAR_SPEED_SCALE
    landing = tracker.step(
        landing_before,
        landing_after,
        tick=12,
        world_contact_normal=normal,
        goal_for_attacker=no_goal,
    )
    assert float(landing[1]) > 0.0

    demo_before = landing_after.clone()
    demo_after = demo_before.clone()
    demo_after[2, 0, FIELD["ball.position.y"]] = 0.5
    no_demo = tracker.step(
        demo_before,
        demo_after,
        tick=13,
        world_contact_normal=normal,
        goal_for_attacker=no_goal,
    )
    assert float(no_demo[2]) == 0.0
    demo_after[2, 0, FIELD["lifecycle.opponent_demoed_event"]] = 1.0
    demo = tracker.step(
        demo_before,
        demo_after,
        tick=14,
        world_contact_normal=normal,
        goal_for_attacker=no_goal,
    )
    assert float(demo[2]) == 1.5

    follow_before = demo_after.clone()
    follow_before[2, 0, FIELD["lifecycle.opponent_demoed_event"]] = 0.0
    follow_after = follow_before.clone()
    follow_after[2, 0, FIELD["lifecycle.self_touch_event"]] = 1.0
    goal = torch.tensor([True, False, False])
    follow = tracker.step(
        follow_before,
        follow_after,
        tick=15,
        world_contact_normal=normal,
        goal_for_attacker=goal,
    )
    assert float(follow[0]) == 2.0
    assert float(follow[2]) == 0.5
    assert tracker.telemetry.ground_origin_high_contacts == 1
    assert tracker.telemetry.high_contact_goals == 1
    assert tracker.telemetry.productive_floor_landings == 1
    assert tracker.telemetry.actual_demos == 1
    assert tracker.telemetry.demo_followup_touches == 1
