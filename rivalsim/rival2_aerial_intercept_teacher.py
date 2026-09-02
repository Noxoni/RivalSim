"""Observation-only ballistic intercept teacher for aerial-option distillation.

The teacher consumes the same normalized 182-field observation as Rival.  It
does not read simulator internals or mutate state.  Its output is an ordinary
eight-channel action which can be evaluated physically and, only after it is
validated, used as a supervised target for an aerial option.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch

from rivalsim.rival2_aerial_option import FIELD
from rivalsim.rival2_contracts import (
    ANGULAR_SPEED_SCALE,
    BALL_LINEAR_SPEED_SCALE,
    POSITION_SCALE,
)

GRAVITY_Z = -650.0
BOOST_ACCELERATION = 991.0


@dataclass(frozen=True, slots=True)
class AerialInterceptPlan:
    action: torch.Tensor
    target_direction: torch.Tensor
    intercept_time: torch.Tensor
    predicted_distance: torch.Tensor
    nose_alignment: torch.Tensor


def _vector(observation: torch.Tensor, prefix: str) -> torch.Tensor:
    return torch.stack(
        [observation[:, FIELD[f"{prefix}.{axis}"]] for axis in "xyz"], dim=-1
    )


def _normalize(value: torch.Tensor) -> torch.Tensor:
    return value / torch.linalg.vector_norm(value, dim=-1, keepdim=True).clamp_min(
        1.0e-6
    )


def plan_aerial_intercept(observation: torch.Tensor) -> AerialInterceptPlan:
    """Return a deterministic pursuit plan using only policy-visible fields."""

    if observation.ndim != 2 or observation.shape[1] != 182:
        raise ValueError("aerial intercept teacher expects [N,182] observations")
    device = observation.device
    dtype = observation.dtype
    position_scale = torch.as_tensor(POSITION_SCALE, device=device, dtype=dtype)
    relative = _vector(observation, "relative.ball_position") * position_scale
    relative_velocity = (
        _vector(observation, "relative.ball_velocity") * BALL_LINEAR_SPEED_SCALE
    )
    forward = _normalize(_vector(observation, "self.forward"))
    up = _normalize(_vector(observation, "self.up"))
    right = _normalize(torch.linalg.cross(forward, up, dim=-1))
    angular = _vector(observation, "self.angular_velocity") * ANGULAR_SPEED_SCALE

    times = torch.linspace(0.12, 1.20, 37, device=device, dtype=dtype)
    future = relative[:, None, :] + relative_velocity[:, None, :] * times[None, :, None]
    future[..., 2] += 0.5 * GRAVITY_Z * times.square()[None, :]
    direction = _normalize(future)
    distance = torch.linalg.vector_norm(future, dim=-1)
    # `relative.ball_velocity` already subtracts the car's current velocity,
    # so `future` is the residual separation if the car simply coasts.  The
    # reachability envelope must therefore contain only *additional* boost
    # displacement; adding current car speed again double-counts it.
    reachable = 0.5 * BOOST_ACCELERATION * times.square()[None, :]
    # Prefer the earliest physically plausible solution while allowing a soft
    # match when the car is not yet exactly on the reachable envelope.
    ball_height = (
        observation[:, FIELD["ball.position.z"]] * POSITION_SCALE[2]
    )[:, None]
    ball_vertical_velocity = (
        observation[:, FIELD["ball.linear_velocity.z"]] * BALL_LINEAR_SPEED_SCALE
    )[:, None]
    future_ball_height = (
        ball_height
        + ball_vertical_velocity * times[None, :]
        + 0.5 * GRAVITY_Z * times.square()[None, :]
    )
    score = (distance - reachable).abs() + 100.0 * times[None, :]
    score = score + (future_ball_height < 300.0).to(dtype) * 1.0e6
    index = score.argmin(dim=1)
    rows = torch.arange(observation.shape[0], device=device)
    target = direction[rows, index]
    target_distance = distance[rows, index]
    intercept_time = times[index]

    local_forward = (target * forward).sum(dim=-1)
    local_right = (target * right).sum(dim=-1)
    local_up = (target * up).sum(dim=-1)
    pitch_angle = torch.atan2(local_up, local_forward)
    yaw_angle = torch.atan2(local_right, local_forward)
    omega_right = (angular * right).sum(dim=-1)
    omega_up = (angular * up).sum(dim=-1)
    omega_forward = (angular * forward).sum(dim=-1)

    world_up = torch.zeros_like(target)
    world_up[:, 2] = 1.0
    desired_up = world_up - target * (world_up * target).sum(dim=-1, keepdim=True)
    weak_up = torch.linalg.vector_norm(desired_up, dim=-1) < 0.05
    fallback_up = _normalize(torch.linalg.cross(target, right, dim=-1))
    desired_up = _normalize(torch.where(weak_up[:, None], fallback_up, desired_up))
    roll_angle = torch.atan2(
        (torch.linalg.cross(up, desired_up, dim=-1) * forward).sum(dim=-1),
        (up * desired_up).sum(dim=-1),
    )

    pitch = (-3.2 * pitch_angle + 0.38 * omega_right).clamp(-1.0, 1.0)
    yaw = (-3.2 * yaw_angle - 0.42 * omega_up).clamp(-1.0, 1.0)
    roll = (-2.4 * roll_angle + 0.32 * omega_forward).clamp(-1.0, 1.0)
    alignment = (forward * target).sum(dim=-1)
    boost = ((alignment >= 0.78) & (target_distance >= 135.0)).to(dtype)

    action = torch.zeros((observation.shape[0], 8), device=device, dtype=dtype)
    action[:, 0] = 1.0
    action[:, 2] = pitch
    action[:, 3] = yaw
    action[:, 4] = roll
    action[:, 6] = boost
    return AerialInterceptPlan(
        action=action,
        target_direction=target,
        intercept_time=intercept_time,
        predicted_distance=target_distance,
        nose_alignment=alignment,
    )
