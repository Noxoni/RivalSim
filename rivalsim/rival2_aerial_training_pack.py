"""Discrete physical aerial shot packs with explicit staged outcomes."""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import torch

from rivalsim.rival2_aerial_option import FIELD
from rivalsim.rival2_contracts import BALL_LINEAR_SPEED_SCALE, POSITION_SCALE
from rivalsim.state import StateSnapshot

PACK_CENTER_POP = 0
PACK_LATERAL_POP = 1
PACK_AIRBORNE_POSSESSION = 2
PACK_NAMES = ("center_pop", "lateral_pop", "airborne_possession")


def _yaw_quat(yaw: float) -> np.ndarray:
    return np.asarray(
        (0.0, 0.0, math.sin(yaw * 0.5), math.cos(yaw * 0.5)), dtype=np.float32
    )


@dataclass(frozen=True, slots=True)
class AerialTrainingPackBatch:
    state: StateSnapshot
    attacker_side: int
    pack: int


def build_training_pack_scenarios(
    worlds: int, *, seed: int, attacker_side: int, pack: int
) -> AerialTrainingPackBatch:
    if worlds <= 0 or attacker_side not in (0, 1):
        raise ValueError("invalid aerial training-pack request")
    if pack not in range(len(PACK_NAMES)):
        raise ValueError("invalid aerial training pack")
    rng = np.random.default_rng(seed)
    sign = 1.0 if attacker_side == 0 else -1.0
    other = 1 - attacker_side
    state = StateSnapshot.empty(worlds)
    state.car_pos[..., 2] = 17.0
    state.on_ground.fill(1)
    state.boost.fill(100.0)
    forward = _yaw_quat(sign * math.pi / 2.0)
    reverse = _yaw_quat(-sign * math.pi / 2.0)
    for world in range(worlds):
        if pack == PACK_CENTER_POP:
            # Beginner fast-aerial pack.  At the end of the fixed 29-tick
            # launch the ball is close enough for the learned controller to
            # have a non-zero physical interception basin, while still
            # requiring an airborne correction and a high touch.
            car_x = float(rng.uniform(-450.0, 450.0))
            car_y = float(rng.uniform(700.0, 2000.0))
            forward_offset = float(rng.uniform(300.0, 600.0))
            lateral_offset = float(rng.uniform(-80.0, 80.0))
            ball_z = float(rng.uniform(280.0, 450.0))
            ball_vx = float(rng.uniform(-60.0, 60.0))
            ball_vy = float(rng.uniform(-30.0, 120.0))
            ball_vz = float(rng.uniform(80.0, 200.0))
        elif pack == PACK_LATERAL_POP:
            car_x = float(rng.uniform(-1300.0, 1300.0))
            car_y = float(rng.uniform(650.0, 2100.0))
            forward_offset = float(rng.uniform(400.0, 800.0))
            lateral_offset = float(rng.uniform(120.0, 280.0))
            if bool(rng.integers(0, 2)):
                lateral_offset *= -1.0
            ball_z = float(rng.uniform(350.0, 650.0))
            ball_vx = float(-0.25 * lateral_offset + rng.uniform(-100.0, 100.0))
            ball_vy = float(rng.uniform(-50.0, 250.0))
            ball_vz = float(rng.uniform(80.0, 250.0))
        else:
            car_x = float(rng.uniform(-1500.0, 1500.0))
            car_y = float(rng.uniform(750.0, 2300.0))
            forward_offset = float(rng.uniform(500.0, 1000.0))
            lateral_offset = float(rng.uniform(220.0, 420.0))
            if bool(rng.integers(0, 2)):
                lateral_offset *= -1.0
            ball_z = float(rng.uniform(450.0, 800.0))
            ball_vx = float(-0.30 * lateral_offset + rng.uniform(-160.0, 160.0))
            ball_vy = float(rng.uniform(-80.0, 350.0))
            ball_vz = float(rng.uniform(100.0, 300.0))
        state.car_pos[world, attacker_side] = (car_x, sign * car_y, 17.0)
        state.car_quat[world, attacker_side] = forward
        state.car_vel[world, attacker_side] = (
            float(rng.uniform(-30.0, 30.0)),
            sign * float(rng.uniform(250.0, 500.0)),
            0.0,
        )
        state.ball_pos[world] = (
            np.clip(car_x + lateral_offset, -3600.0, 3600.0),
            sign * (car_y + forward_offset),
            ball_z,
        )
        state.ball_vel[world] = (ball_vx, sign * ball_vy, ball_vz)
        state.car_pos[world, other] = (
            float(rng.uniform(-1500.0, 1500.0)),
            -sign * float(rng.uniform(3600.0, 4500.0)),
            17.0,
        )
        state.car_quat[world, other] = reverse
    state.validate()
    return AerialTrainingPackBatch(state=state, attacker_side=attacker_side, pack=pack)


@dataclass(slots=True)
class AerialTrainingPackTelemetry:
    attempts: int = 0
    launches: int = 0
    first_high_touches: int = 0
    goalward_first_touches: int = 0
    second_airborne_touches: int = 0
    goals: int = 0
    no_launch_failures: int = 0
    missed_intercept_failures: int = 0
    ball_ground_failures: int = 0
    timeout_after_touch_failures: int = 0
    untouched_goal_failures: int = 0
    reward_sum: float = 0.0


class AerialTrainingPackTracker:
    def __init__(
        self,
        worlds: int,
        *,
        attacker_side: int,
        pack: int,
        first_touch_deadline: int,
        horizon: int,
    ):
        if worlds <= 0 or attacker_side not in (0, 1):
            raise ValueError("invalid training-pack tracker request")
        self.worlds = worlds
        self.attacker_side = attacker_side
        self.pack = pack
        self.first_touch_deadline = int(first_touch_deadline)
        self.horizon = int(horizon)
        self.initialized = False
        self.telemetry = AerialTrainingPackTelemetry(attempts=worlds)

    def _initialize(self, device: torch.device) -> None:
        self.launched = torch.zeros(self.worlds, dtype=torch.bool, device=device)
        self.ball_was_airborne = torch.zeros(self.worlds, dtype=torch.bool, device=device)
        self.first_high_touch = torch.zeros(self.worlds, dtype=torch.bool, device=device)
        self.second_airborne_touch = torch.zeros(
            self.worlds, dtype=torch.bool, device=device
        )
        self.first_touch_tick = torch.full(
            (self.worlds,), -10_000, dtype=torch.int64, device=device
        )
        self.last_touch_tick = torch.full(
            (self.worlds,), -10_000, dtype=torch.int64, device=device
        )
        self.goal_paid = torch.zeros(self.worlds, dtype=torch.bool, device=device)
        self.positive_budget = torch.zeros(self.worlds, device=device)
        self.initialized = True

    def _self(self, observation: torch.Tensor, field: str) -> torch.Tensor:
        return observation[:, self.attacker_side, FIELD[field]]

    def _vector(self, observation: torch.Tensor, prefix: str) -> torch.Tensor:
        return torch.stack(
            [self._self(observation, f"{prefix}.{axis}") for axis in "xyz"], dim=-1
        )

    def step(
        self,
        before: torch.Tensor,
        after: torch.Tensor,
        *,
        tick: int,
        goal_for_attacker: torch.Tensor,
        any_goal: torch.Tensor,
        active: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if before.shape != after.shape or before.shape != (self.worlds, 2, 182):
            raise ValueError("training-pack observations must align as [N,2,182]")
        if not self.initialized:
            self._initialize(before.device)
        ground_before = self._self(before, "self.on_ground") >= 0.5
        ground_after = self._self(after, "self.on_ground") >= 0.5
        launch = active & ground_before & ~ground_after & ~self.launched
        self.launched |= launch
        car_height = self._self(after, "self.position.z") * POSITION_SCALE[2]
        ball_height_before = self._self(before, "ball.position.z") * POSITION_SCALE[2]
        ball_height = self._self(after, "ball.position.z") * POSITION_SCALE[2]
        self.ball_was_airborne |= active & (ball_height >= 150.0)

        scale = torch.as_tensor(POSITION_SCALE, device=before.device)
        relative_before = self._vector(before, "relative.ball_position") * scale
        relative_after = self._vector(after, "relative.ball_position") * scale
        distance_before = torch.linalg.vector_norm(relative_before, dim=-1)
        distance_after = torch.linalg.vector_norm(relative_after, dim=-1)
        distance_progress = (distance_before - distance_after).clamp(0.0, 40.0)
        vertical_progress = (
            relative_before[:, 2].abs() - relative_after[:, 2].abs()
        ).clamp(0.0, 30.0)
        direction_before = relative_before / distance_before[:, None].clamp_min(1.0)
        direction_after = relative_after / distance_after[:, None].clamp_min(1.0)
        alignment_before = (self._vector(before, "self.forward") * direction_before).sum(-1)
        alignment_after = (self._vector(after, "self.forward") * direction_after).sum(-1)
        alignment_progress = (alignment_after - alignment_before).clamp(0.0, 0.2)
        pursuit = active & ~self.first_high_touch & (tick > 28)
        reward = torch.where(pursuit, distance_progress / 40.0 * 0.03, 0.0)
        reward += torch.where(pursuit, vertical_progress / 30.0 * 0.02, 0.0)
        reward += torch.where(pursuit, alignment_progress / 0.2 * 0.02, 0.0)

        touch = self._self(after, "lifecycle.self_touch_event") >= 0.5
        first = (
            active
            & touch
            & (car_height >= 300.0)
            & (ball_height >= 300.0)
            & ~self.first_high_touch
        )
        self.first_high_touch |= first
        self.first_touch_tick.copy_(
            torch.where(first, torch.full_like(self.first_touch_tick, tick), self.first_touch_tick)
        )
        before_forward = (
            self._self(before, "ball.linear_velocity.y") * BALL_LINEAR_SPEED_SCALE
        )
        after_forward = (
            self._self(after, "ball.linear_velocity.y") * BALL_LINEAR_SPEED_SCALE
        )
        forward_transfer = (after_forward - before_forward).clamp(0.0, 1600.0)
        goalward = first & (forward_transfer >= 150.0)
        second = (
            active
            & touch
            & self.first_high_touch
            & ~first
            & ~self.second_airborne_touch
            & ((tick - self.first_touch_tick) >= 4)
            & (car_height >= 150.0)
            & (ball_height >= 250.0)
        )
        self.second_airborne_touch |= second
        self.last_touch_tick.copy_(
            torch.where(touch, torch.full_like(self.last_touch_tick, tick), self.last_touch_tick)
        )
        possession = active & self.first_high_touch & ~self.second_airborne_touch
        height_progress = (ball_height - ball_height_before).clamp(0.0, 20.0)
        reward += torch.where(possession, height_progress / 20.0 * 0.01, 0.0)
        reward += first.to(torch.float32) * 3.0
        reward += torch.where(first, forward_transfer / 1600.0, 0.0)
        reward += second.to(torch.float32) * 2.0

        goal = (
            active
            & goal_for_attacker
            & self.first_high_touch
            & ~self.goal_paid
        )
        untouched_goal = active & any_goal & ~self.first_high_touch
        self.goal_paid |= goal
        reward += goal.to(torch.float32) * 8.0
        no_launch = active & (tick >= 36) & ~self.launched
        no_launch &= ~goal & ~untouched_goal
        missed = (
            active
            & (tick >= self.first_touch_deadline)
            & ~self.first_high_touch
        )
        missed &= ~goal & ~untouched_goal & ~no_launch
        ball_ground = (
            active
            & self.ball_was_airborne
            & (ball_height <= 100.0)
            & ~goal
        )
        ball_ground &= ~untouched_goal & ~no_launch & ~missed
        horizon_failure = active & (tick >= self.horizon - 1) & ~goal
        timeout_after_touch = (
            horizon_failure
            & self.first_high_touch
            & ~untouched_goal
            & ~ball_ground
        )
        timeout_without_touch = (
            horizon_failure
            & ~self.first_high_touch
            & ~untouched_goal
            & ~no_launch
            & ~ball_ground
        )
        missed |= timeout_without_touch
        reward += no_launch.to(torch.float32) * -1.0
        reward += missed.to(torch.float32) * -1.0
        reward += ball_ground.to(torch.float32) * -1.0
        reward += timeout_after_touch.to(torch.float32) * -0.5
        reward += untouched_goal.to(torch.float32) * -1.0
        failure = no_launch | missed | ball_ground | timeout_after_touch | untouched_goal
        done = goal | failure
        positive = reward.clamp_min(0.0)
        remaining = (15.0 - self.positive_budget).clamp_min(0.0)
        paid = torch.minimum(positive, remaining)
        self.positive_budget += paid
        reward = torch.where(active, paid + reward.clamp_max(0.0), 0.0)

        self.telemetry.launches += int(launch.sum())
        self.telemetry.first_high_touches += int(first.sum())
        self.telemetry.goalward_first_touches += int(goalward.sum())
        self.telemetry.second_airborne_touches += int(second.sum())
        self.telemetry.goals += int(goal.sum())
        self.telemetry.no_launch_failures += int(no_launch.sum())
        self.telemetry.missed_intercept_failures += int(missed.sum())
        self.telemetry.ball_ground_failures += int(ball_ground.sum())
        self.telemetry.timeout_after_touch_failures += int(timeout_after_touch.sum())
        self.telemetry.untouched_goal_failures += int(untouched_goal.sum())
        self.telemetry.reward_sum += float(reward.sum())
        return reward, done


__all__ = [
    "PACK_AIRBORNE_POSSESSION",
    "PACK_CENTER_POP",
    "PACK_LATERAL_POP",
    "PACK_NAMES",
    "AerialTrainingPackBatch",
    "AerialTrainingPackTelemetry",
    "AerialTrainingPackTracker",
    "build_training_pack_scenarios",
]
