from __future__ import annotations

import numpy as np
import torch

from rivalsim.rival2_capability_curriculum import (
    FIELD,
    SCENARIO_AIRBORNE_INTERCEPT,
    SCENARIO_FLOOR_RECOVERY,
    SCENARIO_OFFENSIVE_DEMO,
    CapabilityRewardTracker,
    build_capability_scenarios,
)
from rivalsim.rival2_contracts import CAR_LINEAR_SPEED_SCALE, POSITION_SCALE


def test_scenario_bank_is_deterministic_balanced_and_valid() -> None:
    first = build_capability_scenarios(100, seed=123)
    second = build_capability_scenarios(100, seed=123)
    first.state.validate()
    assert np.array_equal(first.scenario, second.scenario)
    assert np.array_equal(first.attacker_side, second.attacker_side)
    assert np.array_equal(first.state.car_pos, second.state.car_pos)
    assert np.bincount(first.scenario, minlength=5).tolist() == [25, 25, 15, 10, 25]
    assert set(np.unique(first.attacker_side)).issubset({0, 1})


def test_physical_overlay_uses_elevated_contact_dash_gain_and_actual_demo() -> None:
    scenario = torch.tensor(
        [SCENARIO_AIRBORNE_INTERCEPT, SCENARIO_FLOOR_RECOVERY, SCENARIO_OFFENSIVE_DEMO]
    )
    side = torch.tensor([0, 1, 0])
    tracker = CapabilityRewardTracker(scenario, side)
    before = torch.zeros((3, 2, 182), dtype=torch.float32)
    after = before.clone()
    # Elevated authoritative touch with canonical forward ball velocity.
    after[0, 0, FIELD["self.position.z"]] = 350.0 / POSITION_SCALE[2]
    after[0, 0, FIELD["ball.position.z"]] = 500.0 / POSITION_SCALE[2]
    after[0, 0, FIELD["ball.linear_velocity.y"]] = 1000.0 / 6000.0
    after[0, 0, FIELD["lifecycle.self_touch_event"]] = 1.0
    # Orange-perspective flip onset at 1000 uu/s.
    before[1, 1, FIELD["self.linear_velocity.y"]] = 1000.0 / CAR_LINEAR_SPEED_SCALE
    after[1, 1, FIELD["self.linear_velocity.y"]] = 1000.0 / CAR_LINEAR_SPEED_SCALE
    after[1, 1, FIELD["self.is_flipping"]] = 1.0
    onset = tracker.step(before, after, tick=4)
    assert float(onset[0]) >= 0.75
    assert float(onset[1]) == 0.0
    # Landing has an authoritative wheel state and gains 200 uu/s.
    landing_before = after.clone()
    landing_after = after.clone()
    landing_before[0, 0, FIELD["lifecycle.self_touch_event"]] = 0.0
    landing_after[0, 0, FIELD["lifecycle.self_touch_event"]] = 0.0
    landing_after[1, 1, FIELD["self.is_flipping"]] = 0.0
    landing_after[1, 1, FIELD["self.on_ground"]] = 1.0
    landing_after[1, 1, FIELD["self.wheel_contact.front_left"]] = 1.0
    landing_after[1, 1, FIELD["self.linear_velocity.y"]] = 1200.0 / CAR_LINEAR_SPEED_SCALE
    landing = tracker.step(landing_before, landing_after, tick=12)
    assert float(landing[1]) > 0.0
    # A non-demo bump does not pay; the actual lifecycle event does.
    demo_before = landing_after.clone()
    demo_after = landing_after.clone()
    demo_after[2, 0, FIELD["ball.position.y"]] = 0.5
    no_demo = tracker.step(demo_before, demo_after, tick=13)
    assert float(no_demo[2]) == 0.0
    demo_after[2, 0, FIELD["lifecycle.opponent_demoed_event"]] = 1.0
    actual_demo = tracker.step(demo_before, demo_after, tick=14)
    assert float(actual_demo[2]) == 1.5
    assert tracker.telemetry.elevated_contacts == 1
    assert tracker.telemetry.productive_floor_landings == 1
    assert tracker.telemetry.actual_demos == 1
