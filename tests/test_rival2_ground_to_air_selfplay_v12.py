from __future__ import annotations

import torch

from rivalsim.rival2_aerial_option import FIELD
from rivalsim.rival2_contracts import (
    BALL_LINEAR_SPEED_SCALE,
    POSITION_SCALE,
)
from rivalsim.rival2_ground_to_air_selfplay_v12 import (
    ROUTE_ASSISTED_LOW_BOUNCE,
    ROUTE_RISING_DOUBLE_JUMP,
    ROUTE_ROOF_CARRY,
    ROUTE_SOFT_INCOMING_CHIP,
    AerialOptionRouterConfig,
    AerialOptionSelfPlayRouter,
    AerialSelfPlayRewardConfig,
    aerial_route_eligibility,
)


def _observation(lanes: int = 1) -> torch.Tensor:
    result = torch.zeros((lanes, 182), dtype=torch.float32)
    result[:, FIELD["self.position.z"]] = 17.0 / POSITION_SCALE[2]
    result[:, FIELD["self.forward.y"]] = 1.0
    result[:, FIELD["self.up.z"]] = 1.0
    result[:, FIELD["self.boost"]] = 0.75
    result[:, FIELD["self.on_ground"]] = 1.0
    result[:, FIELD["opponent.position.y"]] = 2_000.0 / POSITION_SCALE[1]
    return result


def _entry(
    *,
    ball_height: float,
    distance: float,
    vertical_speed: float,
) -> torch.Tensor:
    result = _observation()
    result[:, FIELD["ball.position.y"]] = distance / POSITION_SCALE[1]
    result[:, FIELD["ball.position.z"]] = ball_height / POSITION_SCALE[2]
    result[:, FIELD["relative.ball_position.y"]] = distance / POSITION_SCALE[1]
    result[:, FIELD["relative.ball_position.z"]] = (
        ball_height - 17.0
    ) / POSITION_SCALE[2]
    result[:, FIELD["ball.linear_velocity.z"]] = (
        vertical_speed / BALL_LINEAR_SPEED_SCALE
    )
    return result


def test_route_classifier_covers_all_four_direct_entry_families() -> None:
    config = AerialOptionRouterConfig()
    rows = (
        (_entry(ball_height=180.0, distance=250.0, vertical_speed=30.0), ROUTE_ASSISTED_LOW_BOUNCE),
        (_entry(ball_height=96.0, distance=250.0, vertical_speed=0.0), ROUTE_SOFT_INCOMING_CHIP),
        (_entry(ball_height=220.0, distance=250.0, vertical_speed=250.0), ROUTE_RISING_DOUBLE_JUMP),
        (_entry(ball_height=132.0, distance=45.0, vertical_speed=0.0), ROUTE_ROOF_CARRY),
    )
    for observation, expected in rows:
        eligibility = aerial_route_eligibility(observation, config)
        assert bool(eligibility.eligible.item())
        assert int(eligibility.route.item()) == expected


def test_route_classifier_requires_possession_alignment_and_visible_state() -> None:
    config = AerialOptionRouterConfig()
    observation = _entry(ball_height=180.0, distance=250.0, vertical_speed=30.0)
    observation[:, FIELD["opponent.position.y"]] = 220.0 / POSITION_SCALE[1]
    assert not bool(aerial_route_eligibility(observation, config).eligible.item())
    observation[:, FIELD["opponent.position.y"]] = 2_000.0 / POSITION_SCALE[1]
    observation[:, FIELD["self.forward.y"]] = -1.0
    assert not bool(aerial_route_eligibility(observation, config).eligible.item())
    observation[:, FIELD["self.forward.y"]] = 1.0
    observation[:, FIELD["self.boost"]] = 0.0
    assert not bool(aerial_route_eligibility(observation, config).eligible.item())


def test_router_latches_direct_control_and_releases_on_reset() -> None:
    observation = _entry(ball_height=180.0, distance=250.0, vertical_speed=30.0)
    router = AerialOptionSelfPlayRouter(1, device="cpu")
    false = torch.zeros(1, dtype=torch.bool)
    selected = router.select(
        observation, kickoff_active=false, match_done=false
    )
    assert bool(selected.activated.item())
    assert bool(selected.active.item())
    blocked = router.select(
        observation,
        kickoff_active=torch.ones(1, dtype=torch.bool),
        match_done=false,
    )
    assert bool(blocked.released.item())
    assert not bool(blocked.active.item())


def test_physical_reward_requires_contact_not_airtime_and_pays_second_touch() -> None:
    before = _entry(ball_height=180.0, distance=250.0, vertical_speed=30.0)
    router = AerialOptionSelfPlayRouter(1, device="cpu")
    false = torch.zeros(1, dtype=torch.bool)
    active = router.select(
        before, kickoff_active=false, match_done=false
    ).active

    airborne = before.clone()
    airborne[:, FIELD["self.on_ground"]] = 0.0
    airborne[:, FIELD["self.position.z"]] = 180.0 / POSITION_SCALE[2]
    airborne[:, FIELD["ball.position.z"]] = 320.0 / POSITION_SCALE[2]
    no_contact = router.observe(
        before, airborne, active_before=active, goal_for_lane=false
    )
    assert float(no_contact.supplemental_reward.item()) == 0.0
    assert not bool(no_contact.entry_airborne_contact.item())

    first = airborne.clone()
    first[:, FIELD["lifecycle.self_touch_event"]] = 1.0
    first[:, FIELD["ball.linear_velocity.y"]] = 800.0 / BALL_LINEAR_SPEED_SCALE
    entry = router.observe(
        airborne, first, active_before=active, goal_for_lane=false
    )
    assert bool(entry.entry_airborne_contact.item())
    assert float(entry.supplemental_reward.item()) > 0.0

    quiet = first.clone()
    quiet[:, FIELD["lifecycle.self_touch_event"]] = 0.0
    for _ in range(4):
        router.observe(quiet, quiet, active_before=active, goal_for_lane=false)
    second_state = quiet.clone()
    second_state[:, FIELD["lifecycle.self_touch_event"]] = 1.0
    second_state[:, FIELD["ball.linear_velocity.y"]] = 900.0 / BALL_LINEAR_SPEED_SCALE
    second = router.observe(
        quiet, second_state, active_before=active, goal_for_lane=false
    )
    assert bool(second.second_airborne_contact.item())
    assert bool(second.productive_goalward_contact.item())
    assert float(second.supplemental_reward.item()) >= 5.0


def test_goal_reward_requires_prior_airborne_entry_and_contact_budget() -> None:
    start = _entry(ball_height=180.0, distance=250.0, vertical_speed=30.0)
    false = torch.zeros(1, dtype=torch.bool)
    true = torch.ones(1, dtype=torch.bool)
    router = AerialOptionSelfPlayRouter(1, device="cpu")
    active = router.select(start, kickoff_active=false, match_done=false).active
    without_entry = router.observe(
        start, start, active_before=active, goal_for_lane=true
    )
    assert not bool(without_entry.goal_within_contact_budget.item())

    airborne = start.clone()
    airborne[:, FIELD["self.on_ground"]] = 0.0
    airborne[:, FIELD["self.position.z"]] = 200.0 / POSITION_SCALE[2]
    airborne[:, FIELD["ball.position.z"]] = 350.0 / POSITION_SCALE[2]
    airborne[:, FIELD["lifecycle.self_touch_event"]] = 1.0
    router.observe(start, airborne, active_before=active, goal_for_lane=false)
    goal = airborne.clone()
    goal[:, FIELD["lifecycle.self_touch_event"]] = 0.0
    outcome = router.observe(
        airborne, goal, active_before=active, goal_for_lane=true
    )
    assert bool(outcome.goal_within_contact_budget.item())
    assert float(outcome.supplemental_reward.item()) > 0.0


def test_positive_reward_budget_is_bounded_and_raw_airtime_is_zero() -> None:
    reward = AerialSelfPlayRewardConfig(maximum_supplemental_reward_per_attempt=4.5)
    start = _entry(ball_height=180.0, distance=250.0, vertical_speed=30.0)
    router = AerialOptionSelfPlayRouter(1, device="cpu", reward_config=reward)
    false = torch.zeros(1, dtype=torch.bool)
    active = router.select(start, kickoff_active=false, match_done=false).active
    airborne = start.clone()
    airborne[:, FIELD["self.on_ground"]] = 0.0
    airborne[:, FIELD["self.position.z"]] = 350.0 / POSITION_SCALE[2]
    airborne[:, FIELD["ball.position.z"]] = 400.0 / POSITION_SCALE[2]
    airborne[:, FIELD["ball.linear_velocity.y"]] = 1_200.0 / BALL_LINEAR_SPEED_SCALE
    airborne[:, FIELD["lifecycle.self_touch_event"]] = 1.0
    outcome = router.observe(
        start, airborne, active_before=active, goal_for_lane=false
    )
    assert float(outcome.supplemental_reward.item()) <= 4.5
    assert float(router.supplemental_paid.item()) <= 4.5

