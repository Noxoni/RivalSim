"""Prospective physical-outcome curriculum for Rival capability V2.

The module does not classify named mechanics.  It creates bounded physical
states and reports literal outcomes: a ground-origin high car-ball contact,
flip-to-surface-contact tangent acceleration, authoritative demolitions, and
subsequent touches/goals.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import torch

from rivalsim.rival2_contracts import (
    BALL_LINEAR_SPEED_SCALE,
    CAR_LINEAR_SPEED_SCALE,
    OBS_FIELD_NAMES,
    POSITION_SCALE,
)
from rivalsim.state import StateSnapshot

SCENARIO_HIGH_BALL = 0
SCENARIO_FLOOR_LANDING = 1
SCENARIO_WALL_LANDING = 2
SCENARIO_OFFENSIVE_DEMO = 3
SCENARIO_COUNT = 4
SCENARIO_NAMES = (
    "ground_origin_high_ball_offense",
    "floor_landing_acceleration",
    "wall_landing_acceleration",
    "offensive_demo_route",
)

FIELD = {name: index for index, name in enumerate(OBS_FIELD_NAMES)}


def _yaw_quat(yaw: float) -> np.ndarray:
    return np.asarray((0.0, 0.0, math.sin(yaw * 0.5), math.cos(yaw * 0.5)), dtype=np.float32)


def _quat_from_basis(forward: np.ndarray, up: np.ndarray) -> np.ndarray:
    forward = np.asarray(forward, dtype=np.float64)
    up = np.asarray(up, dtype=np.float64)
    forward /= np.linalg.norm(forward)
    up -= forward * np.dot(forward, up)
    up /= np.linalg.norm(up)
    left = np.cross(up, forward)
    matrix = np.stack((forward, left, up), axis=1)
    trace = float(np.trace(matrix))
    if trace > 0.0:
        scale = math.sqrt(trace + 1.0) * 2.0
        result = np.asarray(
            (
                (matrix[2, 1] - matrix[1, 2]) / scale,
                (matrix[0, 2] - matrix[2, 0]) / scale,
                (matrix[1, 0] - matrix[0, 1]) / scale,
                0.25 * scale,
            ),
            dtype=np.float32,
        )
    else:
        diagonal = int(np.argmax(np.diag(matrix)))
        if diagonal == 0:
            scale = math.sqrt(1.0 + matrix[0, 0] - matrix[1, 1] - matrix[2, 2]) * 2.0
            result = np.asarray(
                (
                    0.25 * scale,
                    (matrix[0, 1] + matrix[1, 0]) / scale,
                    (matrix[0, 2] + matrix[2, 0]) / scale,
                    (matrix[2, 1] - matrix[1, 2]) / scale,
                ),
                dtype=np.float32,
            )
        elif diagonal == 1:
            scale = math.sqrt(1.0 + matrix[1, 1] - matrix[0, 0] - matrix[2, 2]) * 2.0
            result = np.asarray(
                (
                    (matrix[0, 1] + matrix[1, 0]) / scale,
                    0.25 * scale,
                    (matrix[1, 2] + matrix[2, 1]) / scale,
                    (matrix[0, 2] - matrix[2, 0]) / scale,
                ),
                dtype=np.float32,
            )
        else:
            scale = math.sqrt(1.0 + matrix[2, 2] - matrix[0, 0] - matrix[1, 1]) * 2.0
            result = np.asarray(
                (
                    (matrix[0, 2] + matrix[2, 0]) / scale,
                    (matrix[1, 2] + matrix[2, 1]) / scale,
                    0.25 * scale,
                    (matrix[1, 0] - matrix[0, 1]) / scale,
                ),
                dtype=np.float32,
            )
    return result / np.linalg.norm(result)


@dataclass(frozen=True, slots=True)
class CapabilityScenarioBatchV2:
    state: StateSnapshot
    scenario: np.ndarray
    attacker_side: np.ndarray
    scripted_action: np.ndarray


def build_capability_scenarios_v2(
    worlds: int,
    *,
    seed: int,
    attacker_side: int,
    mix: tuple[float, float, float, float] = (0.45, 0.20, 0.10, 0.25),
) -> CapabilityScenarioBatchV2:
    """Build deterministic side-fixed states from the frozen V2 mixture."""

    if worlds <= 0 or attacker_side not in (0, 1):
        raise ValueError("invalid V2 scenario request")
    if len(mix) != SCENARIO_COUNT or any(value < 0.0 for value in mix):
        raise ValueError("invalid V2 scenario mixture")
    if not math.isclose(sum(mix), 1.0, abs_tol=1.0e-12):
        raise ValueError("V2 scenario mixture must sum to one")
    rng = np.random.default_rng(seed)
    counts = np.floor(np.asarray(mix, dtype=np.float64) * worlds).astype(np.int64)
    counts[-1] += worlds - int(counts.sum())
    scenario = np.concatenate(
        [np.full(int(count), index, dtype=np.int32) for index, count in enumerate(counts)]
    )
    rng.shuffle(scenario)
    attacker = np.full(worlds, attacker_side, dtype=np.int32)
    other = 1 - attacker_side
    sign = 1.0 if attacker_side == 0 else -1.0
    state = StateSnapshot.empty(worlds)
    state.car_pos[..., 2] = 17.0
    state.on_ground.fill(1)
    state.boost.fill(100.0)
    state.ball_pos[:] = np.asarray((0.0, 0.0, 93.15), dtype=np.float32)
    scripted = np.zeros((worlds, 2, 8), dtype=np.float32)
    forward_quat = _yaw_quat(sign * math.pi / 2.0)
    reverse_quat = _yaw_quat(-sign * math.pi / 2.0)

    for world, kind_value in enumerate(scenario.tolist()):
        lateral = float(rng.uniform(-850.0, 850.0))
        if kind_value == SCENARIO_HIGH_BALL:
            ball_y = sign * float(rng.uniform(2450.0, 3500.0))
            car_y = ball_y - sign * float(rng.uniform(650.0, 1050.0))
            state.car_pos[world, attacker_side] = (lateral, car_y, 17.0)
            state.car_vel[world, attacker_side] = (
                float(rng.uniform(-80.0, 80.0)),
                sign * float(rng.uniform(850.0, 1350.0)),
                0.0,
            )
            state.car_quat[world, attacker_side] = forward_quat
            state.ball_pos[world] = (
                lateral + float(rng.uniform(-180.0, 180.0)),
                ball_y,
                float(rng.uniform(430.0, 820.0)),
            )
            state.ball_vel[world] = (
                float(rng.uniform(-100.0, 100.0)),
                sign * float(rng.uniform(50.0, 450.0)),
                float(rng.uniform(-140.0, 100.0)),
            )
            state.car_pos[world, other] = (
                -lateral * 0.25,
                sign * float(rng.uniform(4100.0, 4700.0)),
                17.0,
            )
            state.car_quat[world, other] = reverse_quat
        elif kind_value == SCENARIO_FLOOR_LANDING:
            car_y = sign * float(rng.uniform(-2700.0, 1200.0))
            state.car_pos[world, attacker_side] = (
                lateral,
                car_y,
                float(rng.uniform(58.0, 115.0)),
            )
            state.car_vel[world, attacker_side] = (
                float(rng.uniform(-80.0, 80.0)),
                sign * float(rng.uniform(800.0, 1550.0)),
                float(rng.uniform(-330.0, -90.0)),
            )
            state.car_quat[world, attacker_side] = forward_quat
            state.on_ground[world, attacker_side] = 0
            state.has_jumped[world, attacker_side] = 1
            state.air_time[world, attacker_side] = float(rng.uniform(0.09, 0.28))
            state.air_time_since_jump[world, attacker_side] = state.air_time[world, attacker_side]
            state.ball_pos[world] = (lateral, car_y + sign * 1900.0, 93.15)
            state.car_pos[world, other] = (0.0, sign * 4300.0, 17.0)
            state.car_quat[world, other] = reverse_quat
        elif kind_value == SCENARIO_WALL_LANDING:
            wall_sign = 1.0 if rng.random() < 0.5 else -1.0
            car_y = sign * float(rng.uniform(-2500.0, 1700.0))
            forward = np.asarray((0.0, sign, 0.0), dtype=np.float64)
            up = np.asarray((-wall_sign, 0.0, 0.0), dtype=np.float64)
            state.car_pos[world, attacker_side] = (
                wall_sign * float(rng.uniform(4000.0, 4060.0)),
                car_y,
                float(rng.uniform(220.0, 1000.0)),
            )
            state.car_vel[world, attacker_side] = (
                -wall_sign * float(rng.uniform(20.0, 100.0)),
                sign * float(rng.uniform(850.0, 1550.0)),
                float(rng.uniform(-100.0, 100.0)),
            )
            state.car_quat[world, attacker_side] = _quat_from_basis(forward, up)
            state.on_ground[world, attacker_side] = 0
            state.has_jumped[world, attacker_side] = 1
            state.air_time[world, attacker_side] = float(rng.uniform(0.05, 0.24))
            state.air_time_since_jump[world, attacker_side] = state.air_time[world, attacker_side]
            state.ball_pos[world] = (0.0, car_y + sign * 2000.0, 500.0)
            state.car_pos[world, other] = (0.0, sign * 4300.0, 17.0)
            state.car_quat[world, other] = reverse_quat
        else:
            route_x = float(rng.uniform(-650.0, 650.0))
            attacker_y = sign * float(rng.uniform(100.0, 1800.0))
            victim_y = attacker_y + sign * float(rng.uniform(650.0, 1000.0))
            state.car_pos[world, attacker_side] = (
                route_x + float(rng.uniform(-100.0, 100.0)),
                attacker_y,
                17.0,
            )
            state.car_pos[world, other] = (route_x, victim_y, 17.0)
            state.car_vel[world, attacker_side] = (
                0.0,
                sign * float(rng.uniform(1900.0, 2200.0)),
                0.0,
            )
            state.car_vel[world, other] = (
                float(rng.uniform(-80.0, 80.0)),
                sign * float(rng.uniform(450.0, 850.0)),
                0.0,
            )
            state.car_quat[world, attacker_side] = forward_quat
            state.car_quat[world, other] = forward_quat
            state.is_supersonic[world, attacker_side] = 1
            state.supersonic_time[world, attacker_side] = 1.0
            state.ball_pos[world] = (
                route_x + float(rng.uniform(-300.0, 300.0)),
                victim_y + sign * float(rng.uniform(700.0, 1150.0)),
                93.15,
            )
            state.ball_vel[world] = (0.0, sign * float(rng.uniform(100.0, 350.0)), 0.0)
            scripted[world, other, 0] = 1.0

    state.validate()
    return CapabilityScenarioBatchV2(state, scenario, attacker, scripted)


@dataclass(slots=True)
class CapabilityTelemetryV2:
    ground_origin_high_contacts: int = 0
    high_forward_contacts: int = 0
    high_contact_goals: int = 0
    productive_floor_landings: int = 0
    productive_wall_landings: int = 0
    productive_landing_chains: int = 0
    actual_demos: int = 0
    offensive_context_demos: int = 0
    demo_followup_touches: int = 0
    demo_followup_goals: int = 0
    overlay_reward_sum: float = 0.0


class CapabilityRewardTrackerV2:
    """Device-resident physical event memory for a side-fixed V2 rollout."""

    def __init__(self, scenario: torch.Tensor, attacker_side: int):
        if scenario.ndim != 1 or attacker_side not in (0, 1):
            raise ValueError("invalid V2 tracker inputs")
        self.scenario = scenario.to(torch.int64)
        self.attacker_side = int(attacker_side)
        self.device = scenario.device
        count = int(scenario.numel())
        self.rows = torch.arange(count, device=self.device)
        self.flip_tick = torch.full((count,), -10_000, dtype=torch.int64, device=self.device)
        self.flip_velocity = torch.zeros((count, 3), dtype=torch.float32, device=self.device)
        self.last_productive_tick = torch.full(
            (count,), -10_000, dtype=torch.int64, device=self.device
        )
        self.high_contact_latched = torch.zeros(count, dtype=torch.bool, device=self.device)
        self.demo_tick = torch.full((count,), -10_000, dtype=torch.int64, device=self.device)
        self.demo_touch_paid = torch.zeros(count, dtype=torch.bool, device=self.device)
        self.demo_goal_paid = torch.zeros(count, dtype=torch.bool, device=self.device)
        self.episode_budget = torch.zeros(count, dtype=torch.float32, device=self.device)
        self.telemetry = CapabilityTelemetryV2()

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
        world_contact_normal: torch.Tensor,
        goal_for_attacker: torch.Tensor,
    ) -> torch.Tensor:
        if before.shape != after.shape or before.ndim != 3:
            raise ValueError("V2 observations must align as [N,2,182]")
        if world_contact_normal.shape != (before.shape[0], 3):
            raise ValueError("V2 world-contact normal shape mismatch")
        reward = torch.zeros(before.shape[0], dtype=torch.float32, device=self.device)
        position_scale = torch.as_tensor(POSITION_SCALE, dtype=torch.float32, device=self.device)
        direction = 1.0 if self.attacker_side == 0 else -1.0

        high_scenario = self.scenario == SCENARIO_HIGH_BALL
        relative_before = self._vector(before, "relative.ball_position") * position_scale
        relative_after = self._vector(after, "relative.ball_position") * position_scale
        distance_gain = (
            torch.linalg.vector_norm(relative_before, dim=-1)
            - torch.linalg.vector_norm(relative_after, dim=-1)
        ).clamp(0.0, 40.0)
        car_height = self._self(after, "self.position.z") * POSITION_SCALE[2]
        ball_height = self._self(after, "ball.position.z") * POSITION_SCALE[2]
        elevated_approach = high_scenario & (car_height >= 150.0) & (ball_height >= 350.0)
        reward += torch.where(elevated_approach, distance_gain / 40.0 * 0.004, 0.0)
        touch = self._self(after, "lifecycle.self_touch_event") >= 0.5
        high_contact = high_scenario & touch & (car_height >= 300.0) & (ball_height >= 350.0)
        forward_speed = direction * self._self(after, "ball.linear_velocity.y") * BALL_LINEAR_SPEED_SCALE
        forward_bonus = (forward_speed / 2000.0).clamp(0.0, 1.0) * 0.25
        reward += high_contact.to(torch.float32)
        reward += torch.where(high_contact, forward_bonus, 0.0)
        self.high_contact_latched |= high_contact
        high_goal = high_scenario & goal_for_attacker & self.high_contact_latched
        reward += high_goal.to(torch.float32) * 2.0

        velocity_before = self._vector(before, "self.linear_velocity") * CAR_LINEAR_SPEED_SCALE
        velocity_after = self._vector(after, "self.linear_velocity") * CAR_LINEAR_SPEED_SCALE
        flipping_before = self._self(before, "self.is_flipping") >= 0.5
        flipping_after = self._self(after, "self.is_flipping") >= 0.5
        has_flipped_before = self._self(before, "self.has_flipped") >= 0.5
        has_flipped_after = self._self(after, "self.has_flipped") >= 0.5
        flip_onset = (flipping_after & ~flipping_before) | (has_flipped_after & ~has_flipped_before)
        self.flip_tick.copy_(torch.where(flip_onset, torch.full_like(self.flip_tick, tick), self.flip_tick))
        self.flip_velocity.copy_(torch.where(flip_onset[:, None], velocity_before, self.flip_velocity))
        ground_before = self._self(before, "self.on_ground") >= 0.5
        ground_after = self._self(after, "self.on_ground") >= 0.5
        wheel_after = torch.zeros_like(ground_after)
        for wheel in ("front_left", "front_right", "back_left", "back_right"):
            wheel_after |= self._self(after, f"self.wheel_contact.{wheel}") >= 0.5
        landing = (~ground_before) & (ground_after | wheel_after)
        normal_norm = torch.linalg.vector_norm(world_contact_normal, dim=-1, keepdim=True).clamp_min(1e-6)
        normal = world_contact_normal / normal_norm
        flip_tangent = self.flip_velocity - (self.flip_velocity * normal).sum(dim=-1, keepdim=True) * normal
        after_tangent = velocity_after - (velocity_after * normal).sum(dim=-1, keepdim=True) * normal
        tangent_gain = torch.linalg.vector_norm(after_tangent, dim=-1) - torch.linalg.vector_norm(
            flip_tangent, dim=-1
        )
        recent_flip = ((tick - self.flip_tick) >= 0) & ((tick - self.flip_tick) <= 36)
        dash_scenario = (self.scenario == SCENARIO_FLOOR_LANDING) | (
            self.scenario == SCENARIO_WALL_LANDING
        )
        productive = dash_scenario & landing & recent_flip & (tangent_gain >= 100.0)
        reward += torch.where(productive, (tangent_gain / 500.0).clamp(0.0, 1.0) * 0.5, 0.0)
        chain = productive & ((tick - self.last_productive_tick) <= 90)
        reward += chain.to(torch.float32) * 0.75
        self.last_productive_tick.copy_(
            torch.where(productive, torch.full_like(self.last_productive_tick, tick), self.last_productive_tick)
        )
        normal_z = world_contact_normal[:, 2]
        floor_productive = productive & (normal_z >= 0.70)
        wall_productive = productive & (normal_z.abs() <= 0.30)

        demo_scenario = self.scenario == SCENARIO_OFFENSIVE_DEMO
        opponent_before = self._vector(before, "relative.opponent_position") * position_scale
        opponent_after = self._vector(after, "relative.opponent_position") * position_scale
        opponent_gain = (
            torch.linalg.vector_norm(opponent_before, dim=-1)
            - torch.linalg.vector_norm(opponent_after, dim=-1)
        ).clamp(0.0, 40.0)
        reward += torch.where(demo_scenario, opponent_gain / 40.0 * 0.002, 0.0)
        demo = demo_scenario & (self._self(after, "lifecycle.opponent_demoed_event") >= 0.5)
        ball_forward = direction * self._self(after, "ball.position.y") > 0.0
        reward += demo.to(torch.float32)
        reward += (demo & ball_forward).to(torch.float32) * 0.5
        self.demo_tick.copy_(torch.where(demo, torch.full_like(self.demo_tick, tick), self.demo_tick))
        pending_demo = ((tick - self.demo_tick) >= 0) & ((tick - self.demo_tick) <= 5 * 120)
        follow_touch = pending_demo & touch & ~self.demo_touch_paid
        follow_goal = pending_demo & goal_for_attacker & ~self.demo_goal_paid
        reward += follow_touch.to(torch.float32) * 0.5
        reward += follow_goal.to(torch.float32)
        self.demo_touch_paid |= follow_touch
        self.demo_goal_paid |= follow_goal

        remaining = (8.0 - self.episode_budget).clamp_min(0.0)
        paid = torch.minimum(reward.clamp_min(0.0), remaining)
        self.episode_budget += paid
        self.telemetry.ground_origin_high_contacts += int(high_contact.sum())
        self.telemetry.high_forward_contacts += int((high_contact & (forward_bonus > 0.05)).sum())
        self.telemetry.high_contact_goals += int(high_goal.sum())
        self.telemetry.productive_floor_landings += int(floor_productive.sum())
        self.telemetry.productive_wall_landings += int(wall_productive.sum())
        self.telemetry.productive_landing_chains += int(chain.sum())
        self.telemetry.actual_demos += int(demo.sum())
        self.telemetry.offensive_context_demos += int((demo & ball_forward).sum())
        self.telemetry.demo_followup_touches += int(follow_touch.sum())
        self.telemetry.demo_followup_goals += int(follow_goal.sum())
        self.telemetry.overlay_reward_sum += float(paid.sum())
        return paid


__all__ = [
    "SCENARIO_COUNT",
    "SCENARIO_NAMES",
    "CapabilityRewardTrackerV2",
    "CapabilityScenarioBatchV2",
    "CapabilityTelemetryV2",
    "build_capability_scenarios_v2",
]
