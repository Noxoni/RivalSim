from __future__ import annotations

import numpy as np
import pytest
import torch

from rivalsim.rival2_contracts import OBS_DIM, POSITION_SCALE
from rivalsim.rival2_offensive_demo_v1 import (
    FIELD,
    ROUTE_OPEN_GOAL,
    ROUTE_RECOVER_POSSESSION,
    OffensiveDemoOutcomeTracker,
    build_offensive_demo_scenarios,
)


@pytest.mark.parametrize("route", [ROUTE_RECOVER_POSSESSION, ROUTE_OPEN_GOAL])
def test_offensive_demo_scenarios_are_deterministic_and_side_symmetric(
    route: int,
) -> None:
    blue = build_offensive_demo_scenarios(
        256, seed=20260903, attacker_side=0, route=route
    )
    blue_again = build_offensive_demo_scenarios(
        256, seed=20260903, attacker_side=0, route=route
    )
    orange = build_offensive_demo_scenarios(
        256, seed=20260903, attacker_side=1, route=route
    )
    assert np.array_equal(blue.state.car_pos, blue_again.state.car_pos)
    assert np.array_equal(blue.state.car_vel, blue_again.state.car_vel)
    assert np.array_equal(blue.state.ball_pos, blue_again.state.ball_pos)
    assert np.array_equal(blue.route, blue_again.route)
    assert np.array_equal(blue.state.ball_pos[:, 0], orange.state.ball_pos[:, 0])
    assert np.array_equal(blue.state.ball_pos[:, 1], -orange.state.ball_pos[:, 1])
    assert np.array_equal(blue.state.car_pos[:, 0, 0], orange.state.car_pos[:, 1, 0])
    assert np.array_equal(blue.state.car_pos[:, 0, 1], -orange.state.car_pos[:, 1, 1])
    assert np.array_equal(
        blue.initial_defender_lateral_offset_uu,
        orange.initial_defender_lateral_offset_uu,
    )
    assert np.all(np.abs(blue.initial_defender_lateral_offset_uu) >= 250.0)
    assert np.all(np.abs(blue.initial_defender_lateral_offset_uu) <= 550.0)
    assert np.all(blue.state.car_vel[:, 0, 1] >= 2_200.0)
    assert np.all(blue.state.is_supersonic[:, 0] == 1)


@pytest.mark.parametrize("side", [0, 1])
def test_route_geometry_places_the_relevant_defender_goalward(side: int) -> None:
    sign = 1.0 if side == 0 else -1.0
    other = 1 - side
    recover = build_offensive_demo_scenarios(
        1_024,
        seed=18,
        attacker_side=side,
        route=ROUTE_RECOVER_POSSESSION,
    )
    recover_self = sign * recover.state.car_pos[:, side, 1]
    recover_defender = sign * recover.state.car_pos[:, other, 1]
    recover_ball = sign * recover.state.ball_pos[:, 1]
    assert np.all(recover_self < recover_defender)
    assert np.all(recover_defender < recover_ball)

    opening = build_offensive_demo_scenarios(
        1_024,
        seed=19,
        attacker_side=side,
        route=ROUTE_OPEN_GOAL,
    )
    opening_self = sign * opening.state.car_pos[:, side, 1]
    opening_ball = sign * opening.state.ball_pos[:, 1]
    opening_defender = sign * opening.state.car_pos[:, other, 1]
    assert np.all(opening_self < opening_ball)
    assert np.all(opening_ball < opening_defender)


def _observation(
    *,
    self_y: float,
    opponent_y: float,
    ball_y: float,
    demo: bool = False,
    touch: bool = False,
) -> torch.Tensor:
    result = torch.zeros((1, 2, OBS_DIM), dtype=torch.float32)
    for side in (0, 1):
        result[0, side, FIELD["self.position.y"]] = self_y / POSITION_SCALE[1]
        result[0, side, FIELD["opponent.position.y"]] = (
            opponent_y / POSITION_SCALE[1]
        )
        result[0, side, FIELD["ball.position.y"]] = ball_y / POSITION_SCALE[1]
        result[0, side, FIELD["lifecycle.opponent_demoed_event"]] = float(demo)
        result[0, side, FIELD["lifecycle.self_touch_event"]] = float(touch)
    return result


@pytest.mark.parametrize("attacker_side", [0, 1])
def test_team_normalized_offensive_context_is_identical_for_both_sides(
    attacker_side: int,
) -> None:
    tracker = OffensiveDemoOutcomeTracker(
        torch.tensor([ROUTE_RECOVER_POSSESSION]),
        attacker_side=attacker_side,
    )
    before = _observation(self_y=0.0, opponent_y=700.0, ball_y=1_300.0)
    after = _observation(
        self_y=100.0, opponent_y=650.0, ball_y=1_320.0, demo=True
    )
    events = tracker.step(
        before,
        after,
        tick=10,
        goal_for_attacker=torch.tensor([False]),
    )
    assert bool(events.actual_demo.item())
    assert bool(events.offensive_context_demo.item())
    assert tracker.telemetry.offensive_context_demos == 1


def test_demo_is_not_recounted_and_followup_conversion_is_causal() -> None:
    tracker = OffensiveDemoOutcomeTracker(
        torch.tensor([ROUTE_RECOVER_POSSESSION]),
        attacker_side=0,
        followup_window_ticks=120,
        minimum_goalward_progress_uu=300.0,
    )
    before = _observation(self_y=0.0, opponent_y=700.0, ball_y=1_300.0)
    demo = _observation(
        self_y=100.0, opponent_y=650.0, ball_y=1_320.0, demo=True
    )
    first = tracker.step(
        before,
        demo,
        tick=20,
        goal_for_attacker=torch.tensor([False]),
    )
    held = tracker.step(
        demo,
        demo,
        tick=21,
        goal_for_attacker=torch.tensor([False]),
    )
    assert bool(first.actual_demo.item())
    assert not bool(held.actual_demo.item())

    no_touch_progress = _observation(
        self_y=200.0, opponent_y=650.0, ball_y=1_700.0
    )
    no_conversion = tracker.step(
        demo,
        no_touch_progress,
        tick=22,
        goal_for_attacker=torch.tensor([False]),
    )
    assert not bool(no_conversion.post_demo_goalward_progress.item())

    touch = _observation(
        self_y=300.0, opponent_y=650.0, ball_y=1_500.0, touch=True
    )
    touched = tracker.step(
        no_touch_progress,
        touch,
        tick=23,
        goal_for_attacker=torch.tensor([False]),
    )
    assert bool(touched.post_demo_touch.item())

    progressed = _observation(
        self_y=400.0, opponent_y=650.0, ball_y=1_700.0
    )
    converted = tracker.step(
        touch,
        progressed,
        tick=24,
        goal_for_attacker=torch.tensor([False]),
    )
    assert bool(converted.post_demo_goalward_progress.item())
    goal = tracker.step(
        progressed,
        progressed,
        tick=25,
        goal_for_attacker=torch.tensor([True]),
    )
    assert bool(goal.post_demo_goal.item())
    assert tracker.telemetry.actual_demos == 1
    assert tracker.telemetry.post_demo_touches == 1
    assert tracker.telemetry.post_demo_goalward_progress == 1
    assert tracker.telemetry.post_demo_goals == 1


def test_open_goal_route_can_convert_ball_progress_without_a_followup_touch() -> None:
    tracker = OffensiveDemoOutcomeTracker(
        torch.tensor([ROUTE_OPEN_GOAL]),
        attacker_side=1,
        minimum_goalward_progress_uu=300.0,
    )
    before = _observation(self_y=1_000.0, opponent_y=1_800.0, ball_y=1_500.0)
    demo = _observation(
        self_y=1_100.0,
        opponent_y=1_750.0,
        ball_y=1_550.0,
        demo=True,
    )
    tracker.step(
        before,
        demo,
        tick=30,
        goal_for_attacker=torch.tensor([False]),
    )
    progress = _observation(
        self_y=1_200.0, opponent_y=1_750.0, ball_y=1_900.0
    )
    events = tracker.step(
        demo,
        progress,
        tick=31,
        goal_for_attacker=torch.tensor([False]),
    )
    assert bool(events.post_demo_goalward_progress.item())


def test_out_of_context_demo_does_not_open_conversion_window() -> None:
    tracker = OffensiveDemoOutcomeTracker(
        torch.tensor([ROUTE_RECOVER_POSSESSION]), attacker_side=0
    )
    before = _observation(self_y=1_000.0, opponent_y=500.0, ball_y=1_300.0)
    after = _observation(
        self_y=1_100.0, opponent_y=450.0, ball_y=1_320.0, demo=True
    )
    events = tracker.step(
        before,
        after,
        tick=40,
        goal_for_attacker=torch.tensor([False]),
    )
    assert bool(events.actual_demo.item())
    assert not bool(events.offensive_context_demo.item())
    follow = _observation(
        self_y=1_200.0, opponent_y=450.0, ball_y=1_700.0, touch=True
    )
    later = tracker.step(
        after,
        follow,
        tick=41,
        goal_for_attacker=torch.tensor([False]),
    )
    assert not bool(later.post_demo_touch.item())


def test_unconverted_offensive_demo_expires_exactly_once() -> None:
    tracker = OffensiveDemoOutcomeTracker(
        torch.tensor([ROUTE_RECOVER_POSSESSION]),
        attacker_side=0,
        followup_window_ticks=2,
    )
    before = _observation(self_y=0.0, opponent_y=700.0, ball_y=1_300.0)
    demo = _observation(
        self_y=100.0, opponent_y=650.0, ball_y=1_320.0, demo=True
    )
    tracker.step(
        before,
        demo,
        tick=50,
        goal_for_attacker=torch.tensor([False]),
    )
    quiet = _observation(self_y=200.0, opponent_y=650.0, ball_y=1_330.0)
    tracker.step(
        demo,
        quiet,
        tick=51,
        goal_for_attacker=torch.tensor([False]),
    )
    tracker.step(
        quiet,
        quiet,
        tick=52,
        goal_for_attacker=torch.tensor([False]),
    )
    expired = tracker.step(
        quiet,
        quiet,
        tick=53,
        goal_for_attacker=torch.tensor([False]),
    )
    repeated = tracker.step(
        quiet,
        quiet,
        tick=54,
        goal_for_attacker=torch.tensor([False]),
    )
    assert bool(expired.expired_without_conversion.item())
    assert not bool(repeated.expired_without_conversion.item())
    assert tracker.telemetry.expired_without_conversion == 1


@pytest.mark.parametrize(
    "kwargs",
    [
        {"worlds": 0},
        {"attacker_side": 2},
        {"route": 4},
        {"route_mix": (1.0,)},
        {"route_mix": (0.0, 0.0)},
    ],
)
def test_invalid_scenario_request_fails_closed(kwargs: dict[str, object]) -> None:
    request: dict[str, object] = {
        "worlds": 1,
        "seed": 20,
        "attacker_side": 0,
    }
    request.update(kwargs)
    with pytest.raises(ValueError):
        build_offensive_demo_scenarios(**request)  # type: ignore[arg-type]


def test_invalid_tracker_request_fails_closed() -> None:
    with pytest.raises(ValueError):
        OffensiveDemoOutcomeTracker(torch.tensor([[0]]), attacker_side=0)
    with pytest.raises(ValueError):
        OffensiveDemoOutcomeTracker(torch.tensor([5]), attacker_side=0)
    with pytest.raises(ValueError):
        OffensiveDemoOutcomeTracker(
            torch.tensor([0]), attacker_side=0, followup_window_ticks=0
        )
