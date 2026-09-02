"""Observation-only ground-to-air possession option and physical telemetry.

The option creates its own aerial opportunity.  It approaches a low ball from
behind, uses a bounded jump/pop primitive, and then hands orientation and boost
control to the ballistic aerial-intercept teacher.  All gates and actions use
the ordinary 182-field Rival observation; the module does not read hidden
simulator state or change the production reward.
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from typing import Any

import numpy as np
import torch

from rivalsim.rival2_aerial_intercept_teacher import plan_aerial_intercept
from rivalsim.rival2_aerial_option import FIELD
from rivalsim.rival2_contracts import (
    BALL_LINEAR_SPEED_SCALE,
    POSITION_SCALE,
)
from rivalsim.state import StateSnapshot

GROUND_TO_AIR_OPTION_VERSION = "RIVAL2_GROUND_TO_AIR_OPTION_V1"


def _yaw_quat(yaw: float) -> np.ndarray:
    return np.asarray(
        (0.0, 0.0, math.sin(yaw * 0.5), math.cos(yaw * 0.5)),
        dtype=np.float32,
    )


@dataclass(frozen=True, slots=True)
class GroundToAirScenarioBatch:
    state: StateSnapshot
    attacker_side: int


def build_ground_to_air_scenarios(
    worlds: int,
    *,
    seed: int,
    attacker_side: int,
) -> GroundToAirScenarioBatch:
    """Build deterministic low-ball possession starts in the attacking half."""

    if worlds <= 0 or attacker_side not in (0, 1):
        raise ValueError("invalid ground-to-air scenario request")
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
        ball_x = float(rng.uniform(-1_400.0, 1_400.0))
        ball_y = float(rng.uniform(-1_200.0, 900.0))
        following_distance = float(rng.uniform(12.0, 45.0))
        lateral_offset = float(rng.uniform(-22.0, 22.0))
        ball_forward_speed = float(rng.uniform(950.0, 1_700.0))
        # The reviewed human ground-to-air examples catch the underside of a
        # slightly elevated, descending ball while all four wheels remain on
        # the ground. A fully resting 92.75-uu ball mostly produces a flat
        # push and is not the demonstrated launch geometry.
        state.ball_pos[world] = (
            ball_x,
            sign * ball_y,
            float(rng.uniform(142.0, 168.0)),
        )
        state.ball_vel[world] = (
            float(rng.uniform(-60.0, 60.0)),
            sign * ball_forward_speed,
            float(rng.uniform(-105.0, -20.0)),
        )
        state.car_pos[world, attacker_side] = (
            ball_x + lateral_offset,
            sign * (ball_y - following_distance),
            17.0,
        )
        state.car_quat[world, attacker_side] = forward
        state.car_vel[world, attacker_side] = (
            float(rng.uniform(-35.0, 35.0)),
            sign * float(ball_forward_speed + rng.uniform(-80.0, 120.0)),
            0.0,
        )
        # The first authority calibrates the possession/launch chain without
        # conflating it with defender quality.  Live-defender validation is a
        # later, separate gate.
        state.car_pos[world, other] = (
            float(rng.uniform(-1_000.0, 1_000.0)),
            -sign * float(rng.uniform(3_700.0, 4_500.0)),
            17.0,
        )
        state.car_quat[world, other] = reverse
    state.validate()
    return GroundToAirScenarioBatch(state=state, attacker_side=attacker_side)


@dataclass(frozen=True, slots=True)
class GroundToAirConfig:
    minimum_boost_fraction: float = 0.25
    minimum_ball_height_uu: float = 80.0
    maximum_ball_height_uu: float = 190.0
    minimum_planar_distance_uu: float = 5.0
    maximum_planar_distance_uu: float = 100.0
    minimum_forward_alignment: float = 0.50
    approach_steer_gain: float = 3.5
    launch_delay_ticks: int = 0
    first_jump_hold_ticks: int = 8
    jump_release_ticks: int = 8
    second_jump: bool = True
    pop_pitch: float = 0.5
    boost_during_pop: bool = False
    carry_ticks_after_second_jump: int = 36
    carry_pitch: float = 1.0
    carry_boost: bool = True
    carry_boost_min_forward_error_uu: float = 20.0
    carry_boost_min_relative_speed_uu_per_second: float = 20.0
    learned_after_second_jump: bool = False
    release_ball_height_uu: float = 105.0
    release_grounded_after_tick: int = 70
    maximum_option_ticks: int = 420
    cooldown_ticks: int = 120

    def __post_init__(self) -> None:
        if not 0.0 <= self.minimum_boost_fraction <= 1.0:
            raise ValueError("minimum boost fraction must be in [0,1]")
        if self.minimum_ball_height_uu >= self.maximum_ball_height_uu:
            raise ValueError("ball-height gate is empty")
        if self.minimum_planar_distance_uu >= self.maximum_planar_distance_uu:
            raise ValueError("planar-distance gate is empty")
        if not -1.0 <= self.minimum_forward_alignment <= 1.0:
            raise ValueError("minimum forward alignment must be in [-1,1]")
        if self.approach_steer_gain == 0.0:
            raise ValueError("approach steer gain cannot be zero")
        if self.launch_delay_ticks < 0:
            raise ValueError("launch delay cannot be negative")
        if self.first_jump_hold_ticks <= 0 or self.jump_release_ticks <= 0:
            raise ValueError("jump timing must be positive")
        if not -1.0 <= self.pop_pitch <= 1.0:
            raise ValueError("pop pitch must be in [-1,1]")
        if self.carry_ticks_after_second_jump < 0:
            raise ValueError("carry ticks cannot be negative")
        if not -1.0 <= self.carry_pitch <= 1.0:
            raise ValueError("carry pitch must be in [-1,1]")
        if self.carry_boost_min_forward_error_uu < 0.0:
            raise ValueError("carry boost forward error cannot be negative")
        if self.maximum_option_ticks <= self.release_grounded_after_tick:
            raise ValueError("maximum option ticks must follow grounded release")
        if self.cooldown_ticks < 0:
            raise ValueError("cooldown ticks cannot be negative")

    @property
    def second_jump_tick(self) -> int:
        return self.first_jump_hold_ticks + self.jump_release_ticks

    @property
    def pursuit_tick(self) -> int:
        return self.second_jump_tick + int(self.second_jump) + self.carry_ticks_after_second_jump

    @property
    def carry_tick(self) -> int:
        return self.second_jump_tick + int(self.second_jump)


@dataclass(frozen=True, slots=True)
class GroundToAirEligibility:
    eligible: torch.Tensor
    ball_height_uu: torch.Tensor
    planar_distance_uu: torch.Tensor
    forward_alignment: torch.Tensor
    local_right_error: torch.Tensor


@dataclass(frozen=True, slots=True)
class GroundToAirStep:
    action: torch.Tensor
    activated: torch.Tensor
    active: torch.Tensor
    pop_started: torch.Tensor
    approach: torch.Tensor
    waiting_to_launch: torch.Tensor
    pop_primitive: torch.Tensor
    carry: torch.Tensor
    pursuit: torch.Tensor
    learned_control: torch.Tensor
    released: torch.Tensor
    eligibility: GroundToAirEligibility


def _vector(observation: torch.Tensor, prefix: str) -> torch.Tensor:
    return torch.stack([observation[:, FIELD[f"{prefix}.{axis}"]] for axis in "xyz"], dim=-1)


def ground_to_air_eligibility(
    observation: torch.Tensor, config: GroundToAirConfig
) -> GroundToAirEligibility:
    if observation.ndim != 2 or observation.shape[1] != 182:
        raise ValueError("ground-to-air option expects [N,182] observations")
    if observation.device.type == "cpu" and not bool(torch.isfinite(observation).all()):
        raise ValueError("ground-to-air observation contains nonfinite values")
    scale = torch.as_tensor(POSITION_SCALE, dtype=observation.dtype, device=observation.device)
    relative = _vector(observation, "relative.ball_position") * scale
    planar = torch.linalg.vector_norm(relative[:, :2], dim=-1)
    planar_direction = relative[:, :2] / planar[:, None].clamp_min(1.0e-6)
    forward = _vector(observation, "self.forward")
    forward = forward / torch.linalg.vector_norm(forward, dim=-1, keepdim=True).clamp_min(1.0e-6)
    up = _vector(observation, "self.up")
    up = up / torch.linalg.vector_norm(up, dim=-1, keepdim=True).clamp_min(1.0e-6)
    right = torch.linalg.cross(forward, up, dim=-1)
    right = right / torch.linalg.vector_norm(right, dim=-1, keepdim=True).clamp_min(1.0e-6)
    alignment = (forward[:, :2] * planar_direction).sum(dim=-1)
    local_right = (right[:, :2] * planar_direction).sum(dim=-1)
    ball_height = observation[:, FIELD["ball.position.z"]] * POSITION_SCALE[2]
    finite = torch.isfinite(observation).all(dim=1)
    eligible = (
        (observation[:, FIELD["self.on_ground"]] >= 0.5)
        & (observation[:, FIELD["self.is_demoed"]] < 0.5)
        & (observation[:, FIELD["self.boost"]] >= config.minimum_boost_fraction)
        & (ball_height >= config.minimum_ball_height_uu)
        & (ball_height <= config.maximum_ball_height_uu)
        & (planar >= config.minimum_planar_distance_uu)
        & (planar <= config.maximum_planar_distance_uu)
        & (alignment >= config.minimum_forward_alignment)
        & finite
    )
    return GroundToAirEligibility(
        eligible=eligible,
        ball_height_uu=ball_height,
        planar_distance_uu=planar,
        forward_alignment=alignment,
        local_right_error=local_right,
    )


class GroundToAirController:
    """Vectorized latch for approach, pop/launch, and aerial pursuit."""

    def __init__(
        self,
        worlds: int,
        *,
        device: str | torch.device,
        config: GroundToAirConfig | None = None,
    ) -> None:
        if worlds <= 0:
            raise ValueError("world count must be positive")
        self.worlds = int(worlds)
        self.device = torch.device(device)
        self.config = config or GroundToAirConfig()
        self.active = torch.zeros(worlds, dtype=torch.bool, device=self.device)
        self.age = torch.zeros(worlds, dtype=torch.int64, device=self.device)
        self.pop_age = torch.full((worlds,), -1, dtype=torch.int64, device=self.device)
        self.ever_airborne = torch.zeros(worlds, dtype=torch.bool, device=self.device)
        self.cooldown = torch.zeros(worlds, dtype=torch.int64, device=self.device)
        self._counters = {
            name: torch.zeros((), dtype=torch.int64, device=self.device)
            for name in (
                "activations",
                "active_ticks",
                "approach_ticks",
                "pop_starts",
                "pop_primitive_ticks",
                "carry_ticks",
                "pursuit_ticks",
                "released_reset",
                "released_low_ball",
                "released_grounded",
                "released_timeout",
            )
        }

    def step(
        self,
        base_action: torch.Tensor,
        observation: torch.Tensor,
        *,
        kickoff_active: torch.Tensor,
        match_done: torch.Tensor,
    ) -> GroundToAirStep:
        if base_action.shape != (self.worlds, 8):
            raise ValueError("base action must have shape [worlds,8]")
        if observation.shape != (self.worlds, 182):
            raise ValueError("observation must have shape [worlds,182]")
        if kickoff_active.shape != (self.worlds,) or match_done.shape != (self.worlds,):
            raise ValueError("lifecycle masks must align with worlds")
        kickoff_active = kickoff_active.to(torch.bool)
        match_done = match_done.to(torch.bool)
        self.cooldown.sub_(1).clamp_min_(0)

        reset = self.active & (kickoff_active | match_done)
        ball_height = observation[:, FIELD["ball.position.z"]] * POSITION_SCALE[2]
        low_ball = (
            self.active
            & (self.pop_age >= self.config.launch_delay_ticks + self.config.pursuit_tick)
            & (ball_height <= self.config.release_ball_height_uu)
        )
        grounded = (
            self.active
            & self.ever_airborne
            & (self.age >= self.config.release_grounded_after_tick)
            & (observation[:, FIELD["self.on_ground"]] >= 0.5)
        )
        timeout = self.active & (self.age >= self.config.maximum_option_ticks)
        low_ball &= ~reset
        grounded &= ~(reset | low_ball)
        timeout &= ~(reset | low_ball | grounded)
        released = reset | low_ball | grounded | timeout
        self.active &= ~released
        self.age.masked_fill_(released, 0)
        self.pop_age.masked_fill_(released, -1)
        self.ever_airborne.masked_fill_(released, False)
        self.cooldown.copy_(
            torch.where(
                released & ~reset,
                torch.full_like(self.cooldown, self.config.cooldown_ticks),
                self.cooldown,
            )
        )
        self.cooldown.masked_fill_(reset, 0)

        eligibility = ground_to_air_eligibility(observation, self.config)
        activated = (
            ~self.active
            & (self.cooldown == 0)
            & ~kickoff_active
            & ~match_done
            & eligibility.eligible
        )
        self.active |= activated
        self.age.masked_fill_(activated, 0)
        self.pop_age.masked_fill_(activated, -1)

        pop_started = (
            self.active
            & (self.pop_age < 0)
            & (observation[:, FIELD["lifecycle.self_touch_event"]] >= 0.5)
            & (observation[:, FIELD["self.on_ground"]] >= 0.5)
            & (eligibility.ball_height_uu <= 205.0)
        )
        self.pop_age.masked_fill_(pop_started, 0)
        approach = self.active & (self.pop_age < 0)
        waiting_to_launch = (
            self.active & (self.pop_age >= 0) & (self.pop_age < self.config.launch_delay_ticks)
        )

        plan = plan_aerial_intercept(observation)
        emitted = base_action.clone()
        ground_control = approach | waiting_to_launch
        emitted[ground_control] = 0.0
        emitted[approach, 0] = 1.0
        emitted[waiting_to_launch, 0] = 0.45
        emitted[ground_control, 1] = (
            eligibility.local_right_error[ground_control] * self.config.approach_steer_gain
        ).clamp(-1.0, 1.0)
        approach_boost = (
            approach
            & (eligibility.forward_alignment >= 0.92)
            & (eligibility.planar_distance_uu >= 240.0)
        )
        emitted[approach_boost, 6] = 1.0

        launch_age = self.pop_age - self.config.launch_delay_ticks
        first_hold = (
            self.active & (launch_age >= 0) & (launch_age < self.config.first_jump_hold_ticks)
        )
        release_jump = (
            self.active
            & (launch_age >= self.config.first_jump_hold_ticks)
            & (launch_age < self.config.second_jump_tick)
        )
        second_jump = (
            self.active & self.config.second_jump & (launch_age == self.config.second_jump_tick)
        )
        pop_primitive = first_hold | release_jump | second_jump
        emitted[pop_primitive] = 0.0
        emitted[pop_primitive, 0] = 1.0
        emitted[first_hold, 2] = self.config.pop_pitch
        emitted[first_hold, 5] = 1.0
        emitted[first_hold, 6] = float(self.config.boost_during_pop)
        emitted[release_jump, 2] = self.config.pop_pitch
        emitted[release_jump, 5] = 0.0
        emitted[release_jump, 6] = float(self.config.boost_during_pop)
        emitted[second_jump, 5] = 1.0
        emitted[second_jump, 6] = float(self.config.boost_during_pop)

        carry = (
            self.active
            & (launch_age >= self.config.carry_tick)
            & (launch_age < self.config.pursuit_tick)
        )
        if not self.config.learned_after_second_jump:
            emitted[carry] = 0.0
            emitted[carry, 0] = 1.0
            emitted[carry, 2] = self.config.carry_pitch
            emitted[carry, 5] = 1.0
        relative = _vector(observation, "relative.ball_position") * torch.as_tensor(
            POSITION_SCALE,
            dtype=observation.dtype,
            device=observation.device,
        )
        relative_velocity = _vector(observation, "relative.ball_velocity") * BALL_LINEAR_SPEED_SCALE
        forward = _vector(observation, "self.forward")
        forward = forward / torch.linalg.vector_norm(forward, dim=-1, keepdim=True).clamp_min(
            1.0e-6
        )
        forward_error = (relative * forward).sum(dim=-1)
        forward_relative_speed = (relative_velocity * forward).sum(dim=-1)
        carry_boost = (
            carry
            & self.config.carry_boost
            & (forward_error >= self.config.carry_boost_min_forward_error_uu)
            & (forward_relative_speed >= self.config.carry_boost_min_relative_speed_uu_per_second)
        )
        if not self.config.learned_after_second_jump:
            emitted[carry_boost, 6] = 1.0

        pursuit = self.active & (launch_age >= self.config.pursuit_tick)
        if not self.config.learned_after_second_jump:
            emitted[pursuit] = plan.action[pursuit]
        learned_control = (carry | pursuit) & self.config.learned_after_second_jump
        airborne = observation[:, FIELD["self.on_ground"]] < 0.5
        self.ever_airborne |= self.active & airborne

        self._counters["activations"] += activated.sum()
        self._counters["active_ticks"] += self.active.sum()
        self._counters["approach_ticks"] += approach.sum()
        self._counters["pop_starts"] += pop_started.sum()
        self._counters["pop_primitive_ticks"] += pop_primitive.sum()
        self._counters["carry_ticks"] += carry.sum()
        self._counters["pursuit_ticks"] += pursuit.sum()
        self._counters["released_reset"] += reset.sum()
        self._counters["released_low_ball"] += low_ball.sum()
        self._counters["released_grounded"] += grounded.sum()
        self._counters["released_timeout"] += timeout.sum()

        self.age += self.active.to(torch.int64)
        self.pop_age += (self.active & (self.pop_age >= 0)).to(torch.int64)
        return GroundToAirStep(
            action=emitted,
            activated=activated,
            active=self.active.clone(),
            pop_started=pop_started,
            approach=approach,
            waiting_to_launch=waiting_to_launch,
            pop_primitive=pop_primitive,
            carry=carry,
            pursuit=pursuit,
            learned_control=learned_control,
            released=released,
            eligibility=eligibility,
        )

    def telemetry(self) -> dict[str, Any]:
        values = {name: int(value.item()) for name, value in self._counters.items()}
        values.update(
            {
                "currently_active": int(self.active.sum().item()),
                "currently_cooling_down": int((self.cooldown > 0).sum().item()),
                "config": asdict(self.config),
            }
        )
        return values


@dataclass(slots=True)
class GroundToAirTelemetry:
    attempts: int = 0
    pop_touches: int = 0
    qualified_pops: int = 0
    launches_near_pop: int = 0
    ball_rise_250: int = 0
    elevated_follow_touches: int = 0
    high_follow_touches: int = 0
    second_airborne_touches: int = 0
    goals_after_pop: int = 0
    misses: int = 0


class GroundToAirTracker:
    """Causal physical outcome tracker; it does not classify named mechanics."""

    def __init__(self, worlds: int, *, attacker_side: int, horizon: int) -> None:
        if worlds <= 0 or attacker_side not in (0, 1):
            raise ValueError("invalid ground-to-air tracker request")
        self.worlds = int(worlds)
        self.attacker_side = int(attacker_side)
        self.horizon = int(horizon)
        self.initialized = False
        self.telemetry = GroundToAirTelemetry(attempts=worlds)

    def _initialize(self, device: torch.device) -> None:
        self.pop_touch = torch.zeros(self.worlds, dtype=torch.bool, device=device)
        self.qualified_pop = torch.zeros(self.worlds, dtype=torch.bool, device=device)
        self.launch_near_pop = torch.zeros(self.worlds, dtype=torch.bool, device=device)
        self.rise_250 = torch.zeros(self.worlds, dtype=torch.bool, device=device)
        self.elevated_follow = torch.zeros(self.worlds, dtype=torch.bool, device=device)
        self.high_follow = torch.zeros(self.worlds, dtype=torch.bool, device=device)
        self.second_airborne = torch.zeros(self.worlds, dtype=torch.bool, device=device)
        self.pop_tick = torch.full((self.worlds,), -10_000, dtype=torch.int64, device=device)
        self.launch_tick = torch.full((self.worlds,), -10_000, dtype=torch.int64, device=device)
        self.last_air_touch_tick = torch.full(
            (self.worlds,), -10_000, dtype=torch.int64, device=device
        )
        self.initialized = True

    def _self(self, observation: torch.Tensor, field: str) -> torch.Tensor:
        return observation[:, self.attacker_side, FIELD[field]]

    def step(
        self,
        before: torch.Tensor,
        after: torch.Tensor,
        *,
        tick: int,
        goal_for_attacker: torch.Tensor,
        active: torch.Tensor,
    ) -> torch.Tensor:
        if before.shape != after.shape or before.shape != (self.worlds, 2, 182):
            raise ValueError("ground-to-air observations must align as [N,2,182]")
        if not self.initialized:
            self._initialize(before.device)
        touch = active & (self._self(after, "lifecycle.self_touch_event") >= 0.5)
        car_height = self._self(after, "self.position.z") * POSITION_SCALE[2]
        ball_height_before = self._self(before, "ball.position.z") * POSITION_SCALE[2]
        ball_height = self._self(after, "ball.position.z") * POSITION_SCALE[2]
        vertical_before = self._self(before, "ball.linear_velocity.z") * BALL_LINEAR_SPEED_SCALE
        vertical_after = self._self(after, "ball.linear_velocity.z") * BALL_LINEAR_SPEED_SCALE
        forward_before = self._self(before, "ball.linear_velocity.y") * BALL_LINEAR_SPEED_SCALE
        forward_after = self._self(after, "ball.linear_velocity.y") * BALL_LINEAR_SPEED_SCALE
        low_pop = touch & ~self.pop_touch & (ball_height_before <= 205.0) & (car_height <= 150.0)
        self.pop_touch |= low_pop
        self.pop_tick.copy_(
            torch.where(low_pop, torch.full_like(self.pop_tick, tick), self.pop_tick)
        )
        vertical_transfer = vertical_after - vertical_before
        forward_transfer = forward_after - forward_before
        qualified = (
            active
            & self.pop_touch
            & ~self.qualified_pop
            & ((tick - self.pop_tick) <= 60)
            & ((vertical_after >= 350.0) | (vertical_transfer >= 350.0) | (ball_height >= 250.0))
            & (forward_transfer >= -600.0)
        )
        self.qualified_pop |= qualified
        left_ground = (
            active
            & (self._self(before, "self.on_ground") >= 0.5)
            & (self._self(after, "self.on_ground") < 0.5)
        )
        self.launch_tick.copy_(
            torch.where(
                left_ground,
                torch.full_like(self.launch_tick, tick),
                self.launch_tick,
            )
        )
        launch_near_pop = (
            low_pop & ~self.launch_near_pop & ((tick - self.launch_tick).abs() <= 90)
        ) | (
            left_ground
            & self.pop_touch
            & ~self.launch_near_pop
            & ((tick - self.pop_tick).abs() <= 90)
        )
        self.launch_near_pop |= launch_near_pop
        rise = active & self.pop_touch & ~self.rise_250 & (ball_height >= 250.0)
        self.rise_250 |= rise
        elevated = (
            touch
            & self.pop_touch
            & ~low_pop
            & ~self.elevated_follow
            & (car_height >= 150.0)
            & (ball_height >= 250.0)
        )
        high = elevated & (car_height >= 300.0) & (ball_height >= 300.0)
        self.elevated_follow |= elevated
        self.high_follow |= high
        second = (
            touch
            & self.elevated_follow
            & ~elevated
            & ~self.second_airborne
            & (car_height >= 130.0)
            & (ball_height >= 220.0)
            & ((tick - self.last_air_touch_tick) >= 4)
        )
        self.second_airborne |= second
        self.last_air_touch_tick.copy_(
            torch.where(
                elevated | second,
                torch.full_like(self.last_air_touch_tick, tick),
                self.last_air_touch_tick,
            )
        )
        goal = active & goal_for_attacker & self.elevated_follow
        done = goal | (active & (tick >= self.horizon - 1))

        self.telemetry.pop_touches += int(low_pop.sum())
        self.telemetry.qualified_pops += int(qualified.sum())
        self.telemetry.launches_near_pop += int(launch_near_pop.sum())
        self.telemetry.ball_rise_250 += int(rise.sum())
        self.telemetry.elevated_follow_touches += int(elevated.sum())
        self.telemetry.high_follow_touches += int(high.sum())
        self.telemetry.second_airborne_touches += int(second.sum())
        self.telemetry.goals_after_pop += int(goal.sum())
        if tick >= self.horizon - 1:
            self.telemetry.misses += int((active & ~goal).sum())
        return done


__all__ = [
    "GROUND_TO_AIR_OPTION_VERSION",
    "GroundToAirConfig",
    "GroundToAirController",
    "GroundToAirEligibility",
    "GroundToAirScenarioBatch",
    "GroundToAirStep",
    "GroundToAirTelemetry",
    "GroundToAirTracker",
    "build_ground_to_air_scenarios",
    "ground_to_air_eligibility",
]
