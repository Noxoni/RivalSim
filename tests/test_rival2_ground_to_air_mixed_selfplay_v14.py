from __future__ import annotations

import numpy as np
import torch

from rivalsim.rival2_aerial_option import FIELD
from rivalsim.rival2_contracts import POSITION_SCALE
from rivalsim.rival2_ground_to_air_entry_v11 import SETUP_NAMES
from rivalsim.rival2_ground_to_air_mixed_selfplay_v14 import (
    AerialContinuationFailureConfig,
    AerialOptionSelfPlayRouterV14,
    build_mixed_selfplay_initial_state,
    mixed_state_summary,
)
from rivalsim.rival2_ground_to_air_selfplay_v12 import (
    AerialOptionRouterConfig,
    AerialSelfPlayRewardConfig,
)


def _router() -> AerialOptionSelfPlayRouterV14:
    return AerialOptionSelfPlayRouterV14(
        1,
        device="cpu",
        router_config=AerialOptionRouterConfig(),
        reward_config=AerialSelfPlayRewardConfig(),
        failure_config=AerialContinuationFailureConfig(),
    )


def _observation(*, car_on_ground: bool, ball_height: float) -> torch.Tensor:
    result = torch.zeros((1, 182), dtype=torch.float32)
    result[:, FIELD["self.on_ground"]] = float(car_on_ground)
    result[:, FIELD["ball.position.z"]] = ball_height / POSITION_SCALE[2]
    return result


def _started_chain(router: AerialOptionSelfPlayRouterV14) -> None:
    router.active.fill_(True)
    router.entry_seen.fill_(True)
    router.air_contact_count.fill_(1)
    router.ever_airborne_car.fill_(True)
    router.ever_airborne_ball.fill_(True)
    router.age.fill_(router.config.minimum_landing_release_tick)


def test_mixed_state_is_exactly_bounded_side_balanced_and_deterministic() -> None:
    first = build_mixed_selfplay_initial_state(
        400,
        seed=31,
        controlled_fraction=0.25,
        setup_weights=(0.35, 0.15, 0.20, 0.30),
        difficulty=0.0,
    )
    second = build_mixed_selfplay_initial_state(
        400,
        seed=31,
        controlled_fraction=0.25,
        setup_weights=(0.35, 0.15, 0.20, 0.30),
        difficulty=0.0,
    )
    summary = mixed_state_summary(first)
    assert summary["controlled_worlds"] == 100
    assert summary["ordinary_kickoff_worlds"] == 300
    assert summary["controlled_by_attacker_side"] == {"0": 50, "1": 50}
    assert set(summary["controlled_by_setup"]) == set(SETUP_NAMES)
    assert all(value > 0 for value in summary["controlled_by_setup"].values())
    for name in first.state.__dataclass_fields__:
        assert np.array_equal(getattr(first.state, name), getattr(second.state, name))
    assert np.array_equal(first.controlled_world, second.controlled_world)
    assert np.array_equal(first.attacker_side, second.attacker_side)
    assert np.array_equal(first.setup, second.setup)


def test_landing_after_entry_cancels_the_entry_reward_once() -> None:
    router = _router()
    _started_chain(router)
    before = _observation(car_on_ground=False, ball_height=220.0)
    after = _observation(car_on_ground=True, ball_height=210.0)
    first = router.observe(
        before,
        after,
        active_before=torch.ones(1, dtype=torch.bool),
        goal_for_lane=torch.zeros(1, dtype=torch.bool),
    )
    second = router.observe(
        after,
        after,
        active_before=torch.ones(1, dtype=torch.bool),
        goal_for_lane=torch.zeros(1, dtype=torch.bool),
    )
    assert first.supplemental_reward.item() == -4.0
    assert second.supplemental_reward.item() == 0.0
    assert router.counters["landing_before_second_contact"].item() == 1


def test_ball_ground_after_entry_cancels_the_entry_reward_once() -> None:
    router = _router()
    _started_chain(router)
    router.age.zero_()
    before = _observation(car_on_ground=False, ball_height=150.0)
    after = _observation(car_on_ground=False, ball_height=90.0)
    outcome = router.observe(
        before,
        after,
        active_before=torch.ones(1, dtype=torch.bool),
        goal_for_lane=torch.zeros(1, dtype=torch.bool),
    )
    assert outcome.supplemental_reward.item() == -4.0
    assert router.counters["ball_ground_before_second_contact"].item() == 1


def test_second_contact_is_not_treated_as_failed_continuation() -> None:
    router = _router()
    _started_chain(router)
    before = _observation(car_on_ground=False, ball_height=220.0)
    after = _observation(car_on_ground=False, ball_height=220.0)
    after[:, FIELD["lifecycle.self_touch_event"]] = 1.0
    router.last_air_contact_tick.fill_(-100)
    outcome = router.observe(
        before,
        after,
        active_before=torch.ones(1, dtype=torch.bool),
        goal_for_lane=torch.zeros(1, dtype=torch.bool),
    )
    assert outcome.second_airborne_contact.item()
    # The inherited physical goalward-speed term may be negative for this
    # deliberately stationary fixture, but the V14 -4 continuation-failure
    # outcome must not be added after the separated second contact.
    assert outcome.supplemental_reward.item() == -1.5
    assert router.counters["landing_before_second_contact"].item() == 0
    assert router.counters["ball_ground_before_second_contact"].item() == 0
