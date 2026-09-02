from __future__ import annotations

import torch

from rivalsim.rival2_aerial_option import FIELD
from rivalsim.rival2_contracts import POSITION_SCALE
from rivalsim.rival2_ground_to_air_option import (
    GroundToAirConfig,
    GroundToAirController,
    GroundToAirTracker,
    build_ground_to_air_scenarios,
    ground_to_air_eligibility,
)


def _observation(rows: int = 1) -> torch.Tensor:
    observation = torch.zeros((rows, 182), dtype=torch.float32)
    observation[:, FIELD["self.forward.y"]] = 1.0
    observation[:, FIELD["self.up.z"]] = 1.0
    observation[:, FIELD["self.on_ground"]] = 1.0
    observation[:, FIELD["self.boost"]] = 1.0
    observation[:, FIELD["ball.position.z"]] = 150.0 / POSITION_SCALE[2]
    observation[:, FIELD["relative.ball_position.y"]] = 30.0 / POSITION_SCALE[1]
    observation[:, FIELD["relative.ball_position.z"]] = 133.0 / POSITION_SCALE[2]
    return observation


def test_scenarios_are_deterministic_low_ball_possession_starts() -> None:
    first = build_ground_to_air_scenarios(8, seed=73, attacker_side=0)
    second = build_ground_to_air_scenarios(8, seed=73, attacker_side=0)
    assert torch.equal(
        torch.from_numpy(first.state.ball_pos), torch.from_numpy(second.state.ball_pos)
    )
    assert (first.state.ball_pos[:, 2] >= 142.0).all()
    assert (first.state.ball_pos[:, 2] <= 168.0).all()
    assert (first.state.on_ground == 1).all()


def test_gate_requires_low_aligned_ground_possession() -> None:
    observation = _observation(2)
    result = ground_to_air_eligibility(observation, GroundToAirConfig())
    assert result.eligible.tolist() == [True, True]
    observation[0, FIELD["ball.position.z"]] = 300.0 / POSITION_SCALE[2]
    observation[1, FIELD["self.forward.y"]] = -1.0
    result = ground_to_air_eligibility(observation, GroundToAirConfig())
    assert result.eligible.tolist() == [False, False]


def test_controller_approaches_then_runs_pop_and_pursuit() -> None:
    observation = _observation()
    controller = GroundToAirController(1, device="cpu")
    lifecycle = torch.zeros(1, dtype=torch.bool)
    first = controller.step(
        torch.zeros((1, 8)),
        observation,
        kickoff_active=lifecycle,
        match_done=lifecycle,
    )
    assert first.activated.item()
    assert first.approach.item()
    assert first.action[0, 0].item() == 1.0
    observation[:, FIELD["lifecycle.self_touch_event"]] = 1.0
    pop = controller.step(
        torch.zeros((1, 8)),
        observation,
        kickoff_active=lifecycle,
        match_done=lifecycle,
    )
    assert pop.pop_started.item()
    assert not pop.waiting_to_launch.item()
    assert pop.pop_primitive.item()
    assert pop.action[0, 5].item() == 1.0
    observation[:, FIELD["lifecycle.self_touch_event"]] = 0.0
    observation[:, FIELD["ball.position.z"]] = 300.0 / POSITION_SCALE[2]
    for _ in range(controller.config.pursuit_tick):
        observation[:, FIELD["self.on_ground"]] = 0.0
        pursuit = controller.step(
            torch.zeros((1, 8)),
            observation,
            kickoff_active=lifecycle,
            match_done=lifecycle,
        )
    assert pursuit.pursuit.item()


def test_tracker_requires_causal_pop_then_elevated_follow_touch() -> None:
    tracker = GroundToAirTracker(1, attacker_side=0, horizon=120)
    before = torch.zeros((1, 2, 182), dtype=torch.float32)
    after = before.clone()
    before[0, 0, FIELD["self.on_ground"]] = 1.0
    after[0, 0, FIELD["self.on_ground"]] = 0.0
    before[0, 0, FIELD["ball.position.z"]] = 92.75 / POSITION_SCALE[2]
    after[0, 0, FIELD["ball.position.z"]] = 100.0 / POSITION_SCALE[2]
    after[0, 0, FIELD["ball.linear_velocity.z"]] = 0.2
    after[0, 0, FIELD["lifecycle.self_touch_event"]] = 1.0
    done = tracker.step(
        before,
        after,
        tick=10,
        goal_for_attacker=torch.zeros(1, dtype=torch.bool),
        active=torch.ones(1, dtype=torch.bool),
    )
    assert not done.item()
    assert tracker.telemetry.qualified_pops == 1
    follow_before = after.clone()
    follow_before[0, 0, FIELD["lifecycle.self_touch_event"]] = 0.0
    follow_after = follow_before.clone()
    follow_after[0, 0, FIELD["self.position.z"]] = 320.0 / POSITION_SCALE[2]
    follow_after[0, 0, FIELD["ball.position.z"]] = 390.0 / POSITION_SCALE[2]
    follow_after[0, 0, FIELD["lifecycle.self_touch_event"]] = 1.0
    tracker.step(
        follow_before,
        follow_after,
        tick=40,
        goal_for_attacker=torch.zeros(1, dtype=torch.bool),
        active=torch.ones(1, dtype=torch.bool),
    )
    assert tracker.telemetry.elevated_follow_touches == 1
    assert tracker.telemetry.high_follow_touches == 1


def test_learned_controller_takes_over_only_after_neutral_second_jump() -> None:
    observation = _observation()
    config = GroundToAirConfig(
        first_jump_hold_ticks=2,
        jump_release_ticks=1,
        carry_ticks_after_second_jump=3,
        learned_after_second_jump=True,
    )
    controller = GroundToAirController(1, device="cpu", config=config)
    lifecycle = torch.zeros(1, dtype=torch.bool)
    learned = torch.tensor([[0.25, -0.5, 0.75, 0.1, -0.2, 0.0, 1.0, 0.0]])
    controller.step(
        learned,
        observation,
        kickoff_active=lifecycle,
        match_done=lifecycle,
    )
    observation[:, FIELD["lifecycle.self_touch_event"]] = 1.0
    for _ in range(config.carry_tick):
        primitive = controller.step(
            learned,
            observation,
            kickoff_active=lifecycle,
            match_done=lifecycle,
        )
        observation[:, FIELD["lifecycle.self_touch_event"]] = 0.0
        assert not primitive.learned_control.item()
    takeover = controller.step(
        learned,
        observation,
        kickoff_active=lifecycle,
        match_done=lifecycle,
    )
    assert takeover.carry.item()
    assert takeover.learned_control.item()
    assert torch.equal(takeover.action, learned)


def test_nonfinite_observation_fails_closed() -> None:
    observation = _observation()
    observation[0, 0] = float("nan")
    try:
        ground_to_air_eligibility(observation, GroundToAirConfig())
    except ValueError as exc:
        assert "nonfinite" in str(exc)
    else:
        raise AssertionError("nonfinite observation did not fail closed")
