"""Physical scenario and reward overlay for Rival capability training.

This module deliberately does not classify named mechanics.  It exposes only
authoritative physical outcomes used by the V1 capability curriculum: elevated
car-ball contact, flip-to-wheel-contact tangent-speed gain, and actual demolition
events.  Human-readable names live in the campaign evidence, not in the reward
state machine.
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

SCENARIO_AIRBORNE_INTERCEPT = 0
SCENARIO_GROUND_TAKEOFF = 1
SCENARIO_FLOOR_RECOVERY = 2
SCENARIO_WALL_RECOVERY = 3
SCENARIO_OFFENSIVE_DEMO = 4
SCENARIO_COUNT = 5
SCENARIO_NAMES = (
    "airborne_intercept",
    "ground_takeoff",
    "floor_dash_recovery",
    "wall_dash_recovery",
    "offensive_demo_route",
)

FIELD = {name: index for index, name in enumerate(OBS_FIELD_NAMES)}


def _quat_from_basis(forward: np.ndarray, up: np.ndarray) -> np.ndarray:
    """Return xyzw quaternion for orthonormal local forward/up axes."""

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
        w = 0.25 * scale
        x = (matrix[2, 1] - matrix[1, 2]) / scale
        y = (matrix[0, 2] - matrix[2, 0]) / scale
        z = (matrix[1, 0] - matrix[0, 1]) / scale
    else:
        diagonal = int(np.argmax(np.diag(matrix)))
        if diagonal == 0:
            scale = math.sqrt(1.0 + matrix[0, 0] - matrix[1, 1] - matrix[2, 2]) * 2.0
            w = (matrix[2, 1] - matrix[1, 2]) / scale
            x = 0.25 * scale
            y = (matrix[0, 1] + matrix[1, 0]) / scale
            z = (matrix[0, 2] + matrix[2, 0]) / scale
        elif diagonal == 1:
            scale = math.sqrt(1.0 + matrix[1, 1] - matrix[0, 0] - matrix[2, 2]) * 2.0
            w = (matrix[0, 2] - matrix[2, 0]) / scale
            x = (matrix[0, 1] + matrix[1, 0]) / scale
            y = 0.25 * scale
            z = (matrix[1, 2] + matrix[2, 1]) / scale
        else:
            scale = math.sqrt(1.0 + matrix[2, 2] - matrix[0, 0] - matrix[1, 1]) * 2.0
            w = (matrix[1, 0] - matrix[0, 1]) / scale
            x = (matrix[0, 2] + matrix[2, 0]) / scale
            y = (matrix[1, 2] + matrix[2, 1]) / scale
            z = 0.25 * scale
    result = np.asarray((x, y, z, w), dtype=np.float32)
    result /= np.linalg.norm(result)
    return result


def _yaw_quat(yaw: np.ndarray) -> np.ndarray:
    result = np.zeros((yaw.shape[0], 4), dtype=np.float32)
    result[:, 2] = np.sin(yaw * 0.5)
    result[:, 3] = np.cos(yaw * 0.5)
    return result


@dataclass(frozen=True, slots=True)
class CapabilityScenarioBatch:
    state: StateSnapshot
    scenario: np.ndarray
    attacker_side: np.ndarray
    scripted_action: np.ndarray


def build_capability_scenarios(
    worlds: int,
    *,
    seed: int,
    mix: tuple[float, float, float, float, float] = (0.25, 0.25, 0.15, 0.10, 0.25),
) -> CapabilityScenarioBatch:
    """Create a deterministic, side-balanced physical training-state bank."""

    if worlds <= 0 or len(mix) != SCENARIO_COUNT or any(value < 0.0 for value in mix):
        raise ValueError("invalid capability scenario request")
    if not math.isclose(sum(mix), 1.0, abs_tol=1.0e-12):
        raise ValueError("scenario mixture must sum to one")
    rng = np.random.default_rng(seed)
    counts = np.floor(np.asarray(mix) * worlds).astype(np.int64)
    counts[-1] += worlds - int(counts.sum())
    scenario = np.concatenate(
        [np.full(count, index, dtype=np.int32) for index, count in enumerate(counts)]
    )
    rng.shuffle(scenario)
    attacker = rng.integers(0, 2, size=worlds, dtype=np.int32)
    direction = np.where(attacker == 0, 1.0, -1.0).astype(np.float32)
    state = StateSnapshot.empty(worlds)
    state.car_pos[..., 2] = 17.0
    state.on_ground.fill(1)
    state.boost.fill(100.0)
    state.ball_pos[:] = np.asarray((0.0, 0.0, 93.15), dtype=np.float32)
    scripted = np.zeros((worlds, 2, 8), dtype=np.float32)

    for world in range(worlds):
        side = int(attacker[world])
        other = 1 - side
        sign = float(direction[world])
        kind = int(scenario[world])
        lateral = float(rng.uniform(-900.0, 900.0))
        yaw = np.asarray((sign * math.pi / 2.0,), dtype=np.float32)
        attacker_quat = _yaw_quat(yaw)[0]
        other_quat = _yaw_quat(yaw)[0]
        if kind == SCENARIO_AIRBORNE_INTERCEPT:
            car_z = float(rng.uniform(260.0, 720.0))
            car_y = float(-sign * rng.uniform(1200.0, 2200.0))
            ball_forward = float(rng.uniform(350.0, 750.0))
            state.car_pos[world, side] = (lateral, car_y, car_z)
            state.car_vel[world, side] = (
                float(rng.uniform(-120.0, 120.0)),
                sign * float(rng.uniform(700.0, 1300.0)),
                float(rng.uniform(100.0, 420.0)),
            )
            state.car_quat[world, side] = attacker_quat
            state.on_ground[world, side] = 0
            state.has_jumped[world, side] = 1
            state.air_time[world, side] = float(rng.uniform(0.08, 0.45))
            state.air_time_since_jump[world, side] = state.air_time[world, side]
            state.ball_pos[world] = (
                lateral + float(rng.uniform(-130.0, 130.0)),
                car_y + sign * ball_forward,
                car_z + float(rng.uniform(80.0, 260.0)),
            )
            state.ball_vel[world] = (
                float(rng.uniform(-180.0, 180.0)),
                sign * float(rng.uniform(350.0, 950.0)),
                float(rng.uniform(-120.0, 180.0)),
            )
            state.car_pos[world, other] = (-lateral * 0.3, sign * 3400.0, 17.0)
            state.car_quat[world, other] = _yaw_quat(
                np.asarray((-sign * math.pi / 2.0,), dtype=np.float32)
            )[0]
        elif kind == SCENARIO_GROUND_TAKEOFF:
            car_y = float(-sign * rng.uniform(1500.0, 2600.0))
            state.car_pos[world, side] = (lateral, car_y, 17.0)
            state.car_vel[world, side] = (0.0, sign * float(rng.uniform(700.0, 1300.0)), 0.0)
            state.car_quat[world, side] = attacker_quat
            state.ball_pos[world] = (
                lateral + float(rng.uniform(-240.0, 240.0)),
                car_y + sign * float(rng.uniform(700.0, 1250.0)),
                float(rng.uniform(400.0, 1000.0)),
            )
            state.ball_vel[world] = (
                float(rng.uniform(-180.0, 180.0)),
                sign * float(rng.uniform(150.0, 650.0)),
                float(rng.uniform(-160.0, 120.0)),
            )
            state.car_pos[world, other] = (-lateral * 0.3, sign * 3600.0, 17.0)
            state.car_quat[world, other] = _yaw_quat(
                np.asarray((-sign * math.pi / 2.0,), dtype=np.float32)
            )[0]
        elif kind == SCENARIO_FLOOR_RECOVERY:
            car_y = float(-sign * rng.uniform(1800.0, 3200.0))
            state.car_pos[world, side] = (lateral, car_y, float(rng.uniform(55.0, 135.0)))
            state.car_vel[world, side] = (
                float(rng.uniform(-120.0, 120.0)),
                sign * float(rng.uniform(900.0, 1650.0)),
                float(rng.uniform(-420.0, -80.0)),
            )
            state.car_quat[world, side] = attacker_quat
            state.on_ground[world, side] = 0
            state.has_jumped[world, side] = 1
            state.air_time[world, side] = float(rng.uniform(0.08, 0.30))
            state.air_time_since_jump[world, side] = state.air_time[world, side]
            state.ball_pos[world] = (lateral, car_y + sign * 2200.0, 93.15)
            state.car_pos[world, other] = (-lateral, sign * 3900.0, 17.0)
            state.car_quat[world, other] = _yaw_quat(
                np.asarray((-sign * math.pi / 2.0,), dtype=np.float32)
            )[0]
        elif kind == SCENARIO_WALL_RECOVERY:
            wall_sign = 1.0 if rng.random() < 0.5 else -1.0
            car_y = float(-sign * rng.uniform(1400.0, 3000.0))
            forward = np.asarray((0.0, sign, 0.0), dtype=np.float64)
            up = np.asarray((-wall_sign, 0.0, 0.0), dtype=np.float64)
            state.car_pos[world, side] = (
                wall_sign * float(rng.uniform(4010.0, 4065.0)),
                car_y,
                float(rng.uniform(320.0, 1250.0)),
            )
            state.car_vel[world, side] = (
                0.0,
                sign * float(rng.uniform(900.0, 1650.0)),
                float(rng.uniform(-80.0, 80.0)),
            )
            state.car_quat[world, side] = _quat_from_basis(forward, up)
            state.on_ground[world, side] = 0
            state.has_jumped[world, side] = 1
            state.air_time[world, side] = float(rng.uniform(0.04, 0.22))
            state.air_time_since_jump[world, side] = state.air_time[world, side]
            state.ball_pos[world] = (0.0, car_y + sign * 2400.0, 600.0)
            state.car_pos[world, other] = (0.0, sign * 3900.0, 17.0)
            state.car_quat[world, other] = _yaw_quat(
                np.asarray((-sign * math.pi / 2.0,), dtype=np.float32)
            )[0]
        else:
            car_y = float(-sign * rng.uniform(2000.0, 3200.0))
            victim_y = car_y + sign * float(rng.uniform(700.0, 1200.0))
            route_x = float(rng.uniform(-550.0, 550.0))
            state.car_pos[world, side] = (route_x + float(rng.uniform(-120.0, 120.0)), car_y, 17.0)
            state.car_pos[world, other] = (route_x, victim_y, 17.0)
            state.car_vel[world, side] = (0.0, sign * float(rng.uniform(1950.0, 2250.0)), 0.0)
            state.car_vel[world, other] = (0.0, sign * float(rng.uniform(650.0, 1150.0)), 0.0)
            state.car_quat[world, side] = attacker_quat
            state.car_quat[world, other] = other_quat
            state.is_supersonic[world, side] = 1
            state.supersonic_time[world, side] = 1.0
            state.ball_pos[world] = (
                route_x + float(rng.uniform(-400.0, 400.0)),
                victim_y + sign * float(rng.uniform(850.0, 1500.0)),
                93.15,
            )
            state.ball_vel[world] = (0.0, sign * float(rng.uniform(100.0, 500.0)), 0.0)
            scripted[world, other, 0] = 1.0

    state.validate()
    return CapabilityScenarioBatch(state, scenario, attacker, scripted)


@dataclass(slots=True)
class CapabilityTelemetry:
    elevated_contacts: int = 0
    high_forward_contacts: int = 0
    productive_floor_landings: int = 0
    productive_wall_landings: int = 0
    productive_dash_chains: int = 0
    actual_demos: int = 0
    offensive_context_demos: int = 0
    overlay_reward_sum: float = 0.0


class CapabilityRewardTracker:
    """GPU-only per-world physical event memory for one scenario rollout."""

    def __init__(self, scenario: torch.Tensor, attacker_side: torch.Tensor):
        if scenario.shape != attacker_side.shape:
            raise ValueError("scenario and attacker side must align")
        self.scenario = scenario.to(torch.int64)
        self.attacker_side = attacker_side.to(torch.int64)
        self.device = scenario.device
        count = scenario.numel()
        self.rows = torch.arange(count, device=self.device)
        self.flip_tick = torch.full((count,), -10_000, dtype=torch.int64, device=self.device)
        self.flip_speed = torch.zeros(count, dtype=torch.float32, device=self.device)
        self.last_productive_landing_tick = torch.full(
            (count,), -10_000, dtype=torch.int64, device=self.device
        )
        self.episode_budget = torch.zeros(count, dtype=torch.float32, device=self.device)
        self.telemetry = CapabilityTelemetry()

    def _self(self, observation: torch.Tensor, field: str) -> torch.Tensor:
        return observation[self.rows, self.attacker_side, FIELD[field]]

    def step(
        self,
        before: torch.Tensor,
        after: torch.Tensor,
        *,
        tick: int,
        world_contact_normal: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Return bounded attacker reward overlay for one physics tick."""

        if before.shape != after.shape or before.ndim != 3:
            raise ValueError("capability reward observations must be aligned [N,2,182]")
        reward = torch.zeros(self.scenario.numel(), dtype=torch.float32, device=self.device)
        scenario = self.scenario
        position_scale = torch.as_tensor(POSITION_SCALE, dtype=torch.float32, device=self.device)
        relative_before = (
            torch.stack(
                [self._self(before, f"relative.ball_position.{axis}") for axis in "xyz"], dim=-1
            )
            * position_scale
        )
        relative_after = (
            torch.stack(
                [self._self(after, f"relative.ball_position.{axis}") for axis in "xyz"], dim=-1
            )
            * position_scale
        )
        distance_gain = (
            torch.linalg.vector_norm(relative_before, dim=-1)
            - torch.linalg.vector_norm(relative_after, dim=-1)
        ).clamp(0.0, 40.0)
        aerial = (scenario == SCENARIO_AIRBORNE_INTERCEPT) | (scenario == SCENARIO_GROUND_TAKEOFF)
        ball_height = self._self(after, "ball.position.z") * POSITION_SCALE[2]
        car_height = self._self(after, "self.position.z") * POSITION_SCALE[2]
        moving_to_elevated_ball = aerial & (ball_height >= 300.0) & (car_height >= 80.0)
        reward += torch.where(moving_to_elevated_ball, distance_gain / 40.0 * 0.004, 0.0)
        touch = self._self(after, "lifecycle.self_touch_event") >= 0.5
        elevated_contact = aerial & touch & (ball_height >= 300.0) & (car_height >= 250.0)
        reward += elevated_contact.to(torch.float32) * 0.75
        forward_ball_speed = self._self(after, "ball.linear_velocity.y") * BALL_LINEAR_SPEED_SCALE
        forward_bonus = (forward_ball_speed / 2000.0).clamp(0.0, 1.0) * 0.25
        reward += torch.where(elevated_contact, forward_bonus, 0.0)

        speed_before = (
            torch.linalg.vector_norm(
                torch.stack(
                    [self._self(before, f"self.linear_velocity.{axis}") for axis in "xyz"], dim=-1
                ),
                dim=-1,
            )
            * CAR_LINEAR_SPEED_SCALE
        )
        speed_after = (
            torch.linalg.vector_norm(
                torch.stack(
                    [self._self(after, f"self.linear_velocity.{axis}") for axis in "xyz"], dim=-1
                ),
                dim=-1,
            )
            * CAR_LINEAR_SPEED_SCALE
        )
        flipping_before = self._self(before, "self.is_flipping") >= 0.5
        flipping_after = self._self(after, "self.is_flipping") >= 0.5
        has_flipped_before = self._self(before, "self.has_flipped") >= 0.5
        has_flipped_after = self._self(after, "self.has_flipped") >= 0.5
        flip_onset = (flipping_after & ~flipping_before) | (has_flipped_after & ~has_flipped_before)
        self.flip_tick.copy_(
            torch.where(flip_onset, torch.full_like(self.flip_tick, tick), self.flip_tick)
        )
        self.flip_speed.copy_(torch.where(flip_onset, speed_before, self.flip_speed))
        ground_before = self._self(before, "self.on_ground") >= 0.5
        ground_after = self._self(after, "self.on_ground") >= 0.5
        wheel_after = torch.zeros_like(ground_after)
        for wheel in ("front_left", "front_right", "back_left", "back_right"):
            wheel_after |= self._self(after, f"self.wheel_contact.{wheel}") >= 0.5
        landing = (~ground_before) & (ground_after | wheel_after)
        dash_scenario = (scenario == SCENARIO_FLOOR_RECOVERY) | (scenario == SCENARIO_WALL_RECOVERY)
        recent_flip = (tick - self.flip_tick) >= 0
        recent_flip &= (tick - self.flip_tick) <= 30
        tangent_gain = speed_after - self.flip_speed
        productive = dash_scenario & landing & recent_flip & (tangent_gain >= 100.0)
        reward += torch.where(productive, (tangent_gain / 500.0).clamp(0.0, 1.0) * 0.25, 0.0)
        chain = productive & ((tick - self.last_productive_landing_tick) <= 90)
        reward += chain.to(torch.float32) * 0.5
        self.last_productive_landing_tick.copy_(
            torch.where(
                productive,
                torch.full_like(self.last_productive_landing_tick, tick),
                self.last_productive_landing_tick,
            )
        )

        demo_scenario = scenario == SCENARIO_OFFENSIVE_DEMO
        opponent_before = (
            torch.stack(
                [self._self(before, f"relative.opponent_position.{axis}") for axis in "xyz"], dim=-1
            )
            * position_scale
        )
        opponent_after = (
            torch.stack(
                [self._self(after, f"relative.opponent_position.{axis}") for axis in "xyz"], dim=-1
            )
            * position_scale
        )
        opponent_gain = (
            torch.linalg.vector_norm(opponent_before, dim=-1)
            - torch.linalg.vector_norm(opponent_after, dim=-1)
        ).clamp(0.0, 40.0)
        reward += torch.where(demo_scenario, opponent_gain / 40.0 * 0.002, 0.0)
        demo = demo_scenario & (self._self(after, "lifecycle.opponent_demoed_event") >= 0.5)
        ball_forward = self._self(after, "ball.position.y") > -0.1
        reward += demo.to(torch.float32)
        reward += (demo & ball_forward).to(torch.float32) * 0.5

        remaining = (8.0 - self.episode_budget).clamp_min(0.0)
        paid = torch.minimum(reward.clamp_min(0.0), remaining)
        self.episode_budget += paid
        self.telemetry.elevated_contacts += int(elevated_contact.sum().item())
        self.telemetry.high_forward_contacts += int(
            (elevated_contact & (forward_bonus > 0.05)).sum().item()
        )
        self.telemetry.productive_floor_landings += int(
            (productive & (scenario == SCENARIO_FLOOR_RECOVERY)).sum().item()
        )
        self.telemetry.productive_wall_landings += int(
            (productive & (scenario == SCENARIO_WALL_RECOVERY)).sum().item()
        )
        self.telemetry.productive_dash_chains += int(chain.sum().item())
        self.telemetry.actual_demos += int(demo.sum().item())
        self.telemetry.offensive_context_demos += int((demo & ball_forward).sum().item())
        self.telemetry.overlay_reward_sum += float(paid.sum().item())
        return paid


__all__ = [
    "SCENARIO_COUNT",
    "SCENARIO_NAMES",
    "CapabilityRewardTracker",
    "CapabilityScenarioBatch",
    "CapabilityTelemetry",
    "build_capability_scenarios",
]
