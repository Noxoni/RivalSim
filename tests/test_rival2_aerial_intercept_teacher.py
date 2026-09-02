from __future__ import annotations

import torch

from rivalsim.rival2_aerial_intercept_teacher import plan_aerial_intercept
from rivalsim.rival2_aerial_option import FIELD
from rivalsim.rival2_contracts import BALL_LINEAR_SPEED_SCALE, POSITION_SCALE


def _observation() -> torch.Tensor:
    observation = torch.zeros((3, 182), dtype=torch.float32)
    observation[:, FIELD["self.forward.y"]] = 1.0
    observation[:, FIELD["self.up.z"]] = 1.0
    observation[:, FIELD["relative.ball_position.y"]] = 700.0 / POSITION_SCALE[1]
    observation[:, FIELD["relative.ball_position.z"]] = 450.0 / POSITION_SCALE[2]
    observation[:, FIELD["relative.ball_velocity.y"]] = 150.0 / BALL_LINEAR_SPEED_SCALE
    observation[:, FIELD["relative.ball_velocity.z"]] = 100.0 / BALL_LINEAR_SPEED_SCALE
    return observation


def test_intercept_teacher_is_deterministic_finite_and_bounded() -> None:
    observation = _observation()
    first = plan_aerial_intercept(observation)
    second = plan_aerial_intercept(observation)
    assert torch.equal(first.action, second.action)
    assert torch.isfinite(first.action).all()
    assert bool((first.action >= -1.0).all() and (first.action <= 1.0).all())
    assert bool((first.intercept_time >= 0.12).all())
    assert bool((first.intercept_time <= 1.20).all())
    assert torch.equal(first.action[:, 5], torch.zeros(3))
    assert torch.equal(first.action[:, 7], torch.zeros(3))


def test_intercept_teacher_uses_yaw_to_correct_lateral_error() -> None:
    observation = _observation()
    observation[0, FIELD["relative.ball_position.x"]] = 350.0 / POSITION_SCALE[0]
    observation[1, FIELD["relative.ball_position.x"]] = -350.0 / POSITION_SCALE[0]
    plan = plan_aerial_intercept(observation)
    assert float(plan.action[0, 3]) < 0.0
    assert float(plan.action[1, 3]) > 0.0


def test_intercept_teacher_only_boosts_when_nose_is_aligned() -> None:
    observation = _observation()
    aligned = plan_aerial_intercept(observation)
    observation[:, FIELD["self.forward.y"]] = -1.0
    misaligned = plan_aerial_intercept(observation)
    assert bool((aligned.action[:, 6] == 1.0).all())
    assert bool((misaligned.action[:, 6] == 0.0).all())
