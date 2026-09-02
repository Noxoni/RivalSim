"""State-gated deterministic aerial-intercept option for Rival.

The ordinary Rival policy remains responsible for gameplay whenever the option
is inactive.  A qualifying grounded high-ball opportunity latches a bounded
fast-aerial launch, after which the observation-only ballistic intercept
planner supplies orientation and boost controls.  The module never reads
simulator internals, mutates game state, or changes the production reward.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

import torch

from rivalsim.rival2_aerial_intercept_teacher import (
    BOOST_ACCELERATION,
    AerialInterceptPlan,
    plan_aerial_intercept,
)
from rivalsim.rival2_aerial_option import FIELD
from rivalsim.rival2_aerial_option_v2 import apply_fast_aerial_initiation
from rivalsim.rival2_contracts import BALL_LINEAR_SPEED_SCALE, POSITION_SCALE

HYBRID_VERSION = "RIVAL2_AERIAL_INTERCEPT_HYBRID_V1"


@dataclass(frozen=True, slots=True)
class AerialInterceptGateConfig:
    allow_ground_launch: bool = True
    allow_airborne_intercept: bool = False
    minimum_ball_height_uu: float = 300.0
    maximum_ball_height_uu: float = 1_200.0
    minimum_relative_distance_uu: float = 300.0
    maximum_relative_distance_uu: float = 2_500.0
    minimum_forward_alignment: float = 0.0
    minimum_boost_fraction: float = 0.15
    minimum_intercept_time_seconds: float = 0.30
    minimum_predicted_ball_height_uu: float = 300.0
    maximum_reach_residual_uu: float = 450.0
    airborne_minimum_car_height_uu: float = 80.0
    airborne_maximum_car_height_uu: float = 1_200.0
    airborne_minimum_ball_height_uu: float = 300.0
    airborne_maximum_ball_height_uu: float = 1_400.0
    airborne_minimum_relative_distance_uu: float = 150.0
    airborne_maximum_relative_distance_uu: float = 2_000.0
    airborne_minimum_forward_alignment: float = -0.25
    airborne_minimum_boost_fraction: float = 0.05
    airborne_minimum_intercept_time_seconds: float = 0.12
    airborne_minimum_predicted_ball_height_uu: float = 300.0
    airborne_maximum_reach_residual_uu: float = 450.0
    release_ball_height_uu: float = 190.0
    release_grounded_after_tick: int = 40
    maximum_option_ticks: int = 300
    cooldown_ticks: int = 90

    def __post_init__(self) -> None:
        if not 0.0 <= self.minimum_boost_fraction <= 1.0:
            raise ValueError("minimum boost fraction must be in [0,1]")
        if self.minimum_ball_height_uu >= self.maximum_ball_height_uu:
            raise ValueError("ball-height gate is empty")
        if self.minimum_relative_distance_uu >= self.maximum_relative_distance_uu:
            raise ValueError("distance gate is empty")
        if not -1.0 <= self.minimum_forward_alignment <= 1.0:
            raise ValueError("forward alignment must be in [-1,1]")
        if self.minimum_intercept_time_seconds <= 0.0:
            raise ValueError("minimum intercept time must be positive")
        if self.minimum_predicted_ball_height_uu <= 0.0:
            raise ValueError("minimum predicted ball height must be positive")
        if self.maximum_reach_residual_uu <= 0.0:
            raise ValueError("maximum reach residual must be positive")
        if self.airborne_minimum_car_height_uu >= self.airborne_maximum_car_height_uu:
            raise ValueError("airborne car-height gate is empty")
        if self.airborne_minimum_ball_height_uu >= self.airborne_maximum_ball_height_uu:
            raise ValueError("airborne ball-height gate is empty")
        if (
            self.airborne_minimum_relative_distance_uu
            >= self.airborne_maximum_relative_distance_uu
        ):
            raise ValueError("airborne distance gate is empty")
        if not -1.0 <= self.airborne_minimum_forward_alignment <= 1.0:
            raise ValueError("airborne forward alignment must be in [-1,1]")
        if not 0.0 <= self.airborne_minimum_boost_fraction <= 1.0:
            raise ValueError("airborne minimum boost fraction must be in [0,1]")
        if self.airborne_minimum_intercept_time_seconds <= 0.0:
            raise ValueError("airborne minimum intercept time must be positive")
        if self.airborne_minimum_predicted_ball_height_uu <= 0.0:
            raise ValueError("airborne predicted ball height must be positive")
        if self.airborne_maximum_reach_residual_uu <= 0.0:
            raise ValueError("airborne maximum reach residual must be positive")
        if self.release_grounded_after_tick < 29:
            raise ValueError("grounded release cannot precede the launch primitive")
        if self.maximum_option_ticks <= self.release_grounded_after_tick:
            raise ValueError("maximum option ticks must follow grounded release")
        if self.cooldown_ticks < 0:
            raise ValueError("cooldown ticks cannot be negative")


@dataclass(frozen=True, slots=True)
class AerialInterceptEligibility:
    eligible: torch.Tensor
    ground_eligible: torch.Tensor
    airborne_eligible: torch.Tensor
    ball_height_uu: torch.Tensor
    car_height_uu: torch.Tensor
    relative_distance_uu: torch.Tensor
    forward_alignment: torch.Tensor
    predicted_ball_height_uu: torch.Tensor
    reach_residual_uu: torch.Tensor
    plan: AerialInterceptPlan


@dataclass(frozen=True, slots=True)
class AerialInterceptStep:
    action: torch.Tensor
    activated: torch.Tensor
    ground_activated: torch.Tensor
    airborne_activated: torch.Tensor
    active: torch.Tensor
    released_touch: torch.Tensor
    released_grounded: torch.Tensor
    released_low_ball: torch.Tensor
    released_timeout: torch.Tensor
    released_reset: torch.Tensor
    primitive: torch.Tensor
    eligibility: AerialInterceptEligibility


def _vector(observation: torch.Tensor, prefix: str) -> torch.Tensor:
    return torch.stack(
        [observation[:, FIELD[f"{prefix}.{axis}"]] for axis in "xyz"], dim=-1
    )


def aerial_intercept_eligibility(
    observation: torch.Tensor,
    config: AerialInterceptGateConfig,
) -> AerialInterceptEligibility:
    """Return the policy-visible physical gate and its diagnostic signals."""

    if observation.ndim != 2 or observation.shape[1] != 182:
        raise ValueError("aerial hybrid expects [N,182] observations")
    if observation.device.type == "cpu" and not bool(torch.isfinite(observation).all()):
        raise ValueError("aerial hybrid observation contains nonfinite values")
    plan = plan_aerial_intercept(observation)
    scale = torch.as_tensor(
        POSITION_SCALE, dtype=observation.dtype, device=observation.device
    )
    relative = _vector(observation, "relative.ball_position") * scale
    relative_distance = torch.linalg.vector_norm(relative, dim=-1)
    direction = relative / relative_distance[:, None].clamp_min(1.0e-6)
    forward = _vector(observation, "self.forward")
    forward = forward / torch.linalg.vector_norm(
        forward, dim=-1, keepdim=True
    ).clamp_min(1.0e-6)
    forward_alignment = (forward * direction).sum(dim=-1)
    ball_height = observation[:, FIELD["ball.position.z"]] * POSITION_SCALE[2]
    ball_vertical_velocity = (
        observation[:, FIELD["ball.linear_velocity.z"]] * BALL_LINEAR_SPEED_SCALE
    )
    predicted_ball_height = (
        ball_height
        + ball_vertical_velocity * plan.intercept_time
        - 0.5 * 650.0 * plan.intercept_time.square()
    )
    reachable = 0.5 * BOOST_ACCELERATION * plan.intercept_time.square()
    reach_residual = (plan.predicted_distance - reachable).abs()
    finite = torch.isfinite(observation).all(dim=1)
    ground_eligible = (
        config.allow_ground_launch
        & (observation[:, FIELD["self.on_ground"]] >= 0.5)
        & (observation[:, FIELD["self.is_demoed"]] < 0.5)
        & (observation[:, FIELD["self.boost"]] >= config.minimum_boost_fraction)
        & (ball_height >= config.minimum_ball_height_uu)
        & (ball_height <= config.maximum_ball_height_uu)
        & (relative_distance >= config.minimum_relative_distance_uu)
        & (relative_distance <= config.maximum_relative_distance_uu)
        & (forward_alignment >= config.minimum_forward_alignment)
        & (plan.intercept_time >= config.minimum_intercept_time_seconds)
        & (predicted_ball_height >= config.minimum_predicted_ball_height_uu)
        & (reach_residual <= config.maximum_reach_residual_uu)
        & finite
    )
    car_height = observation[:, FIELD["self.position.z"]] * POSITION_SCALE[2]
    airborne_eligible = (
        config.allow_airborne_intercept
        & (observation[:, FIELD["self.on_ground"]] < 0.5)
        & (observation[:, FIELD["self.is_demoed"]] < 0.5)
        & (
            observation[:, FIELD["self.boost"]]
            >= config.airborne_minimum_boost_fraction
        )
        & (car_height >= config.airborne_minimum_car_height_uu)
        & (car_height <= config.airborne_maximum_car_height_uu)
        & (ball_height >= config.airborne_minimum_ball_height_uu)
        & (ball_height <= config.airborne_maximum_ball_height_uu)
        & (relative_distance >= config.airborne_minimum_relative_distance_uu)
        & (relative_distance <= config.airborne_maximum_relative_distance_uu)
        & (forward_alignment >= config.airborne_minimum_forward_alignment)
        & (plan.intercept_time >= config.airborne_minimum_intercept_time_seconds)
        & (
            predicted_ball_height
            >= config.airborne_minimum_predicted_ball_height_uu
        )
        & (reach_residual <= config.airborne_maximum_reach_residual_uu)
        & finite
    )
    eligible = ground_eligible | airborne_eligible
    return AerialInterceptEligibility(
        eligible=eligible,
        ground_eligible=ground_eligible,
        airborne_eligible=airborne_eligible,
        ball_height_uu=ball_height,
        car_height_uu=car_height,
        relative_distance_uu=relative_distance,
        forward_alignment=forward_alignment,
        predicted_ball_height_uu=predicted_ball_height,
        reach_residual_uu=reach_residual,
        plan=plan,
    )


class AerialInterceptHybridController:
    """Per-world latch joining an immutable base action to the aerial option."""

    def __init__(
        self,
        worlds: int,
        *,
        device: str | torch.device,
        config: AerialInterceptGateConfig | None = None,
    ) -> None:
        if worlds <= 0:
            raise ValueError("world count must be positive")
        self.worlds = int(worlds)
        self.device = torch.device(device)
        self.config = config or AerialInterceptGateConfig()
        self.active = torch.zeros(worlds, dtype=torch.bool, device=self.device)
        self.age = torch.zeros(worlds, dtype=torch.int64, device=self.device)
        self.cooldown = torch.zeros(worlds, dtype=torch.int64, device=self.device)
        self.ever_airborne = torch.zeros(worlds, dtype=torch.bool, device=self.device)
        self._counters = {
            name: torch.zeros((), dtype=torch.int64, device=self.device)
            for name in (
                "activations",
                "ground_activations",
                "airborne_activations",
                "active_ticks",
                "primitive_ticks",
                "teacher_ticks",
                "released_touch",
                "released_grounded",
                "released_low_ball",
                "released_timeout",
                "released_reset",
            )
        }

    def step(
        self,
        base_action: torch.Tensor,
        observation: torch.Tensor,
        *,
        kickoff_active: torch.Tensor,
        match_done: torch.Tensor,
    ) -> AerialInterceptStep:
        if base_action.shape != (self.worlds, 8):
            raise ValueError("base action must have shape [worlds,8]")
        if observation.shape != (self.worlds, 182):
            raise ValueError("observation must have shape [worlds,182]")
        if kickoff_active.shape != (self.worlds,) or match_done.shape != (self.worlds,):
            raise ValueError("lifecycle masks must align with worlds")
        if base_action.device != self.device or observation.device != self.device:
            raise ValueError("aerial hybrid tensors must share the controller device")
        kickoff_active = kickoff_active.to(torch.bool)
        match_done = match_done.to(torch.bool)
        self.cooldown.sub_(1).clamp_min_(0)

        reset = self.active & (kickoff_active | match_done)
        touch = (
            self.active
            & (self.age > 29)
            & (observation[:, FIELD["lifecycle.self_touch_event"]] >= 0.5)
        )
        grounded = (
            self.active
            & self.ever_airborne
            & (self.age > self.config.release_grounded_after_tick)
            & (observation[:, FIELD["self.on_ground"]] >= 0.5)
        )
        ball_height = observation[:, FIELD["ball.position.z"]] * POSITION_SCALE[2]
        low_ball = (
            self.active
            & (self.age > 29)
            & (ball_height < self.config.release_ball_height_uu)
        )
        timeout = self.active & (self.age >= self.config.maximum_option_ticks)

        # Lifecycle reset has priority, then the first physical release reason.
        touch &= ~reset
        grounded &= ~(reset | touch)
        low_ball &= ~(reset | touch | grounded)
        timeout &= ~(reset | touch | grounded | low_ball)
        released = reset | touch | grounded | low_ball | timeout
        self.active &= ~released
        self.age.masked_fill_(released, 0)
        self.ever_airborne.masked_fill_(released, False)
        self.cooldown.copy_(
            torch.where(
                released & ~reset,
                torch.full_like(self.cooldown, self.config.cooldown_ticks),
                self.cooldown,
            )
        )
        self.cooldown.masked_fill_(reset, 0)

        eligibility = aerial_intercept_eligibility(observation, self.config)
        activated = (
            ~self.active
            & (self.cooldown == 0)
            & ~kickoff_active
            & ~match_done
            & eligibility.eligible
        )
        ground_activated = activated & eligibility.ground_eligible
        airborne_activated = activated & eligibility.airborne_eligible
        self.active |= activated
        self.age.masked_fill_(ground_activated, 0)
        # Starting at 29 bypasses the ground-only fast-aerial primitive.
        self.age.masked_fill_(airborne_activated, 29)
        self.ever_airborne.masked_fill_(ground_activated, False)
        self.ever_airborne.masked_fill_(airborne_activated, True)

        option_action, primitive = apply_fast_aerial_initiation(
            eligibility.plan.action, self.age, self.active
        )
        action = torch.where(self.active[:, None], option_action, base_action)
        airborne = observation[:, FIELD["self.on_ground"]] < 0.5
        self.ever_airborne |= self.active & airborne

        self._counters["activations"] += activated.sum()
        self._counters["ground_activations"] += ground_activated.sum()
        self._counters["airborne_activations"] += airborne_activated.sum()
        self._counters["active_ticks"] += self.active.sum()
        self._counters["primitive_ticks"] += (self.active & primitive).sum()
        self._counters["teacher_ticks"] += (self.active & ~primitive).sum()
        self._counters["released_touch"] += touch.sum()
        self._counters["released_grounded"] += grounded.sum()
        self._counters["released_low_ball"] += low_ball.sum()
        self._counters["released_timeout"] += timeout.sum()
        self._counters["released_reset"] += reset.sum()
        self.age += self.active.to(torch.int64)

        return AerialInterceptStep(
            action=action,
            activated=activated,
            ground_activated=ground_activated,
            airborne_activated=airborne_activated,
            active=self.active.clone(),
            released_touch=touch,
            released_grounded=grounded,
            released_low_ball=low_ball,
            released_timeout=timeout,
            released_reset=reset,
            primitive=primitive,
            eligibility=eligibility,
        )

    def telemetry(self) -> dict[str, Any]:
        values = {name: int(value.item()) for name, value in self._counters.items()}
        values.update(
            {
                "currently_active": int(self.active.sum().item()),
                "currently_cooling_down": int((self.cooldown > 0).sum().item()),
                "gate_config": asdict(self.config),
            }
        )
        return values


__all__ = [
    "HYBRID_VERSION",
    "AerialInterceptEligibility",
    "AerialInterceptGateConfig",
    "AerialInterceptHybridController",
    "AerialInterceptStep",
    "aerial_intercept_eligibility",
]
