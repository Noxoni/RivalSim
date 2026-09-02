"""Ground-launch aerial-option curriculum with literal physical outcomes.

This module is training-only.  It does not define or modify the production
Gameplay 120 V2 reward and it does not classify named mechanics.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import torch

from rivalsim.rival2_contracts import (
    BALL_LINEAR_SPEED_SCALE,
    OBS_FIELD_NAMES,
    POSITION_SCALE,
)
from rivalsim.state import StateSnapshot

PHASE_EASY_LAUNCH = 0
PHASE_MOVING_INTERCEPT = 1
PHASE_GOAL_DIRECTED = 2
PHASE_NAMES = ("easy_launch", "moving_intercept", "goal_directed")
FIELD = {name: index for index, name in enumerate(OBS_FIELD_NAMES)}


def _yaw_quat(yaw: float) -> np.ndarray:
    return np.asarray(
        (0.0, 0.0, math.sin(yaw * 0.5), math.cos(yaw * 0.5)),
        dtype=np.float32,
    )


@dataclass(frozen=True, slots=True)
class AerialScenarioBatch:
    state: StateSnapshot
    attacker_side: np.ndarray
    phase: int


def build_aerial_scenarios(
    worlds: int,
    *,
    seed: int,
    attacker_side: int,
    phase: int,
) -> AerialScenarioBatch:
    """Create deterministic grounded launches whose objective is unreachable by driving."""

    if worlds <= 0 or attacker_side not in (0, 1):
        raise ValueError("invalid aerial scenario request")
    if phase not in range(len(PHASE_NAMES)):
        raise ValueError("invalid aerial phase")
    rng = np.random.default_rng(seed)
    sign = 1.0 if attacker_side == 0 else -1.0
    other = 1 - attacker_side
    if phase == PHASE_EASY_LAUNCH:
        height = (260.0, 480.0)
        forward = (300.0, 650.0)
        lateral = 120.0
        speed = 120.0
        ball_y_range = (-1200.0, 2200.0)
    elif phase == PHASE_MOVING_INTERCEPT:
        height = (400.0, 850.0)
        forward = (500.0, 1100.0)
        lateral = 400.0
        speed = 450.0
        ball_y_range = (-800.0, 2800.0)
    else:
        height = (450.0, 1000.0)
        forward = (650.0, 1400.0)
        lateral = 650.0
        speed = 600.0
        ball_y_range = (2500.0, 3900.0)

    state = StateSnapshot.empty(worlds)
    state.car_pos[..., 2] = 17.0
    state.on_ground.fill(1)
    state.boost.fill(100.0)
    forward_quat = _yaw_quat(sign * math.pi / 2.0)
    reverse_quat = _yaw_quat(-sign * math.pi / 2.0)
    for world in range(worlds):
        ball_y_normalized = float(rng.uniform(*ball_y_range))
        forward_offset = float(rng.uniform(*forward))
        car_x = float(rng.uniform(-1800.0, 1800.0))
        lateral_offset = float(rng.uniform(-lateral, lateral))
        state.ball_pos[world] = (
            np.clip(car_x + lateral_offset, -3600.0, 3600.0),
            sign * ball_y_normalized,
            float(rng.uniform(*height)),
        )
        state.car_pos[world, attacker_side] = (
            car_x,
            sign * (ball_y_normalized - forward_offset),
            17.0,
        )
        state.car_quat[world, attacker_side] = forward_quat
        state.car_vel[world, attacker_side] = (
            float(rng.uniform(-60.0, 60.0)),
            sign * float(rng.uniform(350.0, 850.0)),
            0.0,
        )
        # Keep the defender authoritative but outside the acquisition route.
        state.car_pos[world, other] = (
            -car_x * 0.25,
            -sign * float(rng.uniform(3500.0, 4500.0)),
            17.0,
        )
        state.car_quat[world, other] = reverse_quat
        state.ball_vel[world] = (
            float(rng.uniform(-speed, speed)),
            sign * float(rng.uniform(-0.25 * speed, speed)),
            float(rng.uniform(-0.3 * speed, 0.2 * speed)),
        )
    state.validate()
    return AerialScenarioBatch(
        state=state,
        attacker_side=np.full(worlds, attacker_side, dtype=np.int32),
        phase=phase,
    )


@dataclass(slots=True)
class AerialTelemetry:
    episodes: int = 0
    launches: int = 0
    reached_100uu: int = 0
    elevated_contacts: int = 0
    high_contacts: int = 0
    forward_high_contacts: int = 0
    aerial_origin_goals: int = 0
    missed_windows: int = 0
    reward_sum: float = 0.0


class AerialRewardTracker:
    """Training-only physical potentials and event latches for one team side."""

    def __init__(self, worlds: int, *, attacker_side: int, phase: int):
        if worlds <= 0 or attacker_side not in (0, 1):
            raise ValueError("invalid aerial tracker request")
        if phase not in range(len(PHASE_NAMES)):
            raise ValueError("invalid aerial tracker phase")
        self.worlds = int(worlds)
        self.attacker_side = int(attacker_side)
        self.phase = int(phase)
        self.device: torch.device | None = None
        self.launched: torch.Tensor | None = None
        self.reached_100: torch.Tensor | None = None
        self.elevated_contact: torch.Tensor | None = None
        self.high_contact: torch.Tensor | None = None
        self.aerial_goal_paid: torch.Tensor | None = None
        self.launch_tick: torch.Tensor | None = None
        self.positive_budget: torch.Tensor | None = None
        self.telemetry = AerialTelemetry(episodes=worlds)

    def _initialize(self, device: torch.device) -> None:
        self.device = device
        self.launched = torch.zeros(self.worlds, dtype=torch.bool, device=device)
        self.reached_100 = torch.zeros(self.worlds, dtype=torch.bool, device=device)
        self.elevated_contact = torch.zeros(
            self.worlds, dtype=torch.bool, device=device
        )
        self.high_contact = torch.zeros(self.worlds, dtype=torch.bool, device=device)
        self.aerial_goal_paid = torch.zeros(
            self.worlds, dtype=torch.bool, device=device
        )
        self.launch_tick = torch.full(
            (self.worlds,), -10_000, dtype=torch.int64, device=device
        )
        self.positive_budget = torch.zeros(
            self.worlds, dtype=torch.float32, device=device
        )

    def _self(self, observation: torch.Tensor, field: str) -> torch.Tensor:
        return observation[:, self.attacker_side, FIELD[field]]

    def _vector(self, observation: torch.Tensor, prefix: str) -> torch.Tensor:
        return torch.stack(
            [self._self(observation, f"{prefix}.{axis}") for axis in "xyz"],
            dim=-1,
        )

    def step(
        self,
        before: torch.Tensor,
        after: torch.Tensor,
        *,
        tick: int,
        goal_for_attacker: torch.Tensor,
        active: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if before.shape != after.shape or before.shape != (self.worlds, 2, 182):
            raise ValueError("aerial observations must align as [N,2,182]")
        if goal_for_attacker.shape != (self.worlds,) or active.shape != (
            self.worlds,
        ):
            raise ValueError("aerial tracker masks must align with worlds")
        if self.device is None:
            self._initialize(before.device)
        assert self.launched is not None
        assert self.reached_100 is not None
        assert self.elevated_contact is not None
        assert self.high_contact is not None
        assert self.aerial_goal_paid is not None
        assert self.launch_tick is not None
        assert self.positive_budget is not None

        ground_before = self._self(before, "self.on_ground") >= 0.5
        ground_after = self._self(after, "self.on_ground") >= 0.5
        launch = active & ground_before & ~ground_after & ~self.launched
        self.launched |= launch
        self.launch_tick.copy_(
            torch.where(launch, torch.full_like(self.launch_tick, tick), self.launch_tick)
        )

        car_height_before = self._self(before, "self.position.z") * POSITION_SCALE[2]
        car_height_after = self._self(after, "self.position.z") * POSITION_SCALE[2]
        ball_height = self._self(after, "ball.position.z") * POSITION_SCALE[2]
        airborne = ~ground_after
        reached = active & airborne & (car_height_after >= 100.0) & ~self.reached_100
        self.reached_100 |= reached

        relative_before = self._vector(before, "relative.ball_position") * torch.as_tensor(
            POSITION_SCALE, dtype=torch.float32, device=before.device
        )
        relative_after = self._vector(after, "relative.ball_position") * torch.as_tensor(
            POSITION_SCALE, dtype=torch.float32, device=before.device
        )
        distance_before = torch.linalg.vector_norm(relative_before, dim=-1)
        distance_after = torch.linalg.vector_norm(relative_after, dim=-1)
        distance_progress = (distance_before - distance_after).clamp(0.0, 30.0)
        height_progress = (car_height_after - car_height_before).clamp(0.0, 20.0)
        direction_before = relative_before / distance_before[:, None].clamp_min(1.0)
        direction_after = relative_after / distance_after[:, None].clamp_min(1.0)
        forward_before = self._vector(before, "self.forward")
        forward_after = self._vector(after, "self.forward")
        alignment_before = (forward_before * direction_before).sum(dim=-1)
        alignment_after = (forward_after * direction_after).sum(dim=-1)
        alignment_progress = (alignment_after - alignment_before).clamp(0.0, 0.2)

        reward = launch.to(torch.float32) * 0.1
        shaping_mask = active & airborne
        reward += torch.where(shaping_mask, height_progress / 20.0 * 0.01, 0.0)
        reward += torch.where(shaping_mask, distance_progress / 30.0 * 0.01, 0.0)
        reward += torch.where(shaping_mask, alignment_progress / 0.2 * 0.02, 0.0)

        touch = self._self(after, "lifecycle.self_touch_event") >= 0.5
        elevated = (
            active
            & touch
            & airborne
            & (car_height_after >= 150.0)
            & (ball_height >= 250.0)
            & ~self.elevated_contact
        )
        high = (
            elevated
            & (car_height_after >= 300.0)
            & (ball_height >= 300.0)
            & ~self.high_contact
        )
        self.elevated_contact |= elevated
        self.high_contact |= high
        before_forward_velocity = (
            self._self(before, "ball.linear_velocity.y") * BALL_LINEAR_SPEED_SCALE
        )
        after_forward_velocity = (
            self._self(after, "ball.linear_velocity.y") * BALL_LINEAR_SPEED_SCALE
        )
        forward_transfer = (after_forward_velocity - before_forward_velocity).clamp(
            0.0, 1600.0
        )
        forward_high = high & (forward_transfer >= 150.0)
        aerial_goal = (
            active
            & goal_for_attacker
            & self.high_contact
            & ~self.aerial_goal_paid
        )
        self.aerial_goal_paid |= aerial_goal
        reward += elevated.to(torch.float32) * 2.0
        reward += high.to(torch.float32)
        reward += torch.where(elevated, forward_transfer / 1600.0, 0.0)
        reward += aerial_goal.to(torch.float32) * 5.0

        no_launch = active & (tick >= 54) & ~self.launched
        post_launch_landing = (
            active
            & self.launched
            & ground_after
            & ((tick - self.launch_tick) > 12)
            & ~self.elevated_contact
        )
        missed_low_ball = active & (tick >= 30) & (ball_height < 210.0) & ~self.elevated_contact
        success = (
            aerial_goal | elevated
            if self.phase != PHASE_GOAL_DIRECTED
            else aerial_goal.clone()
        )
        missed = no_launch | post_launch_landing | missed_low_ball
        done = success | missed
        reward += missed.to(torch.float32) * -0.1
        positive = reward.clamp_min(0.0)
        remaining = (10.0 - self.positive_budget).clamp_min(0.0)
        paid_positive = torch.minimum(positive, remaining)
        self.positive_budget += paid_positive
        reward = paid_positive + reward.clamp_max(0.0)
        reward = torch.where(active, reward, 0.0)

        self.telemetry.launches += int(launch.sum())
        self.telemetry.reached_100uu += int(reached.sum())
        self.telemetry.elevated_contacts += int(elevated.sum())
        self.telemetry.high_contacts += int(high.sum())
        self.telemetry.forward_high_contacts += int(forward_high.sum())
        self.telemetry.aerial_origin_goals += int(aerial_goal.sum())
        self.telemetry.missed_windows += int(missed.sum())
        self.telemetry.reward_sum += float(reward.sum())
        return reward, done


__all__ = [
    "FIELD",
    "PHASE_EASY_LAUNCH",
    "PHASE_GOAL_DIRECTED",
    "PHASE_MOVING_INTERCEPT",
    "PHASE_NAMES",
    "AerialRewardTracker",
    "AerialScenarioBatch",
    "AerialTelemetry",
    "build_aerial_scenarios",
]
