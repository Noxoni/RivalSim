"""Training-only bridge from a low air touch to the human contact envelope.

The component is causal and physical: it becomes active only after the same
car has produced the existing separated native prompt touch.  It rewards
bounded potential progress toward source-measured height, vertical speed, and
car-ball geometry.  It does not reward raw airtime, classify a named mechanic,
modify production gameplay reward, or synthesize a contact.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

import torch

from rivalsim.rival2_contracts import CAR_LINEAR_SPEED_SCALE, POSITION_SCALE
from rivalsim.rival2_ground_to_air_option import FIELD
from rivalsim.rival2_ground_to_air_prompt_follow_v10 import (
    PromptAerialFollowTrainingTracker,
)

GROUND_TO_AIR_HUMAN_BRIDGE_V11_VERSION = (
    "RIVAL2_GROUND_TO_AIR_HUMAN_ENVELOPE_BRIDGE_V11"
)


@dataclass(frozen=True, slots=True)
class HumanAerialEnvelopeConfig:
    """Prospectively frozen physical envelope and potential weights."""

    target_car_height_uu: float
    target_ball_height_uu: float
    target_car_vertical_speed_uu_per_second: float
    target_distance_uu: float
    target_vertical_standoff_uu: float
    distance_tolerance_uu: float
    vertical_standoff_tolerance_uu: float
    minimum_event_car_height_uu: float
    minimum_event_ball_height_uu: float
    minimum_event_car_vertical_speed_uu_per_second: float
    maximum_event_distance_uu: float
    maximum_bridge_ticks: int
    car_height_weight: float
    ball_height_weight: float
    car_vertical_speed_weight: float
    distance_weight: float
    vertical_standoff_weight: float

    @classmethod
    def from_authority(cls, authority: dict[str, Any]) -> HumanAerialEnvelopeConfig:
        return cls(**authority["human_bridge"])

    def validate(self) -> None:
        positive = (
            self.target_car_height_uu,
            self.target_ball_height_uu,
            self.target_car_vertical_speed_uu_per_second,
            self.target_distance_uu,
            self.target_vertical_standoff_uu,
            self.distance_tolerance_uu,
            self.vertical_standoff_tolerance_uu,
            self.minimum_event_car_height_uu,
            self.minimum_event_ball_height_uu,
            self.minimum_event_car_vertical_speed_uu_per_second,
            self.maximum_event_distance_uu,
            self.car_height_weight,
            self.ball_height_weight,
            self.car_vertical_speed_weight,
            self.distance_weight,
            self.vertical_standoff_weight,
        )
        if any(value <= 0.0 for value in positive) or self.maximum_bridge_ticks <= 0:
            raise ValueError("human aerial envelope values must be positive")


def _vector(observation: torch.Tensor, side: int, prefix: str) -> torch.Tensor:
    return torch.stack(
        [observation[:, side, FIELD[f"{prefix}.{axis}"]] for axis in "xyz"],
        dim=-1,
    )


def human_envelope_features(
    observation: torch.Tensor,
    *,
    side: int,
    config: HumanAerialEnvelopeConfig,
) -> dict[str, torch.Tensor]:
    """Return physical features, a bounded potential, and event eligibility."""

    if observation.ndim != 3 or observation.shape[1:] != (2, 182):
        raise ValueError("human bridge observations must be [N,2,182]")
    if side not in (0, 1):
        raise ValueError("attacker side must be 0 or 1")
    config.validate()
    scale = torch.as_tensor(
        POSITION_SCALE,
        dtype=observation.dtype,
        device=observation.device,
    )
    relative = _vector(observation, side, "relative.ball_position") * scale
    distance = torch.linalg.vector_norm(relative, dim=-1)
    car_height = (
        observation[:, side, FIELD["self.position.z"]] * POSITION_SCALE[2]
    )
    ball_height = (
        observation[:, side, FIELD["ball.position.z"]] * POSITION_SCALE[2]
    )
    car_vertical_speed = (
        observation[:, side, FIELD["self.linear_velocity.z"]]
        * CAR_LINEAR_SPEED_SCALE
    )
    relative_z = relative[:, 2]
    car_height_score = (car_height / config.target_car_height_uu).clamp(0.0, 1.0)
    ball_height_score = (ball_height / config.target_ball_height_uu).clamp(0.0, 1.0)
    vertical_speed_score = (
        car_vertical_speed / config.target_car_vertical_speed_uu_per_second
    ).clamp(0.0, 1.0)
    distance_score = (
        1.0
        - (distance - config.target_distance_uu).abs()
        / config.distance_tolerance_uu
    ).clamp(0.0, 1.0)
    standoff_score = (
        1.0
        - (relative_z - config.target_vertical_standoff_uu).abs()
        / config.vertical_standoff_tolerance_uu
    ).clamp(0.0, 1.0)
    weights = (
        config.car_height_weight
        + config.ball_height_weight
        + config.car_vertical_speed_weight
        + config.distance_weight
        + config.vertical_standoff_weight
    )
    potential = (
        car_height_score * config.car_height_weight
        + ball_height_score * config.ball_height_weight
        + vertical_speed_score * config.car_vertical_speed_weight
        + distance_score * config.distance_weight
        + standoff_score * config.vertical_standoff_weight
    ) / weights
    envelope = (
        (car_height >= config.minimum_event_car_height_uu)
        & (ball_height >= config.minimum_event_ball_height_uu)
        & (
            car_vertical_speed
            >= config.minimum_event_car_vertical_speed_uu_per_second
        )
        & (distance <= config.maximum_event_distance_uu)
        & (relative_z > 0.0)
    )
    return {
        "car_height_uu": car_height,
        "ball_height_uu": ball_height,
        "car_vertical_speed_uu_per_second": car_vertical_speed,
        "distance_uu": distance,
        "vertical_standoff_uu": relative_z,
        "potential": potential,
        "envelope": envelope,
    }


class HumanEnvelopeBridgeTrainingTracker(PromptAerialFollowTrainingTracker):
    """Add source-measured post-prompt progress to the isolated curriculum."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self.human_envelope_config = HumanAerialEnvelopeConfig.from_authority(
            self.authority
        )
        self.human_envelope_config.validate()
        self.bridge_initialized = False
        self.human_bridge_progress_reward_sum = 0.0
        self.human_bridge_event_reward_sum = 0.0
        self.human_bridge_active_ticks = 0
        self.human_bridge_envelope_reached = 0

    def _initialize_bridge(self, observation: torch.Tensor) -> None:
        self.human_bridge_prompt_tick = torch.full(
            (self.worlds,),
            -10_000,
            dtype=torch.int64,
            device=observation.device,
        )
        self.human_bridge_envelope_seen = torch.zeros(
            self.worlds,
            dtype=torch.bool,
            device=observation.device,
        )
        self.bridge_initialized = True

    def step(
        self,
        before: torch.Tensor,
        after: torch.Tensor,
        **kwargs: Any,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if not self.bridge_initialized:
            self._initialize_bridge(before)
        if self.geometry_probe.initialized:
            prompt_seen_before = (
                self.geometry_probe.prompt_airborne_follow_seen.clone()
            )
        else:
            prompt_seen_before = torch.zeros(
                self.worlds,
                dtype=torch.bool,
                device=before.device,
            )
        reward, done = super().step(before, after, **kwargs)
        prompt_seen = self.geometry_probe.prompt_airborne_follow_seen
        prompt_event = prompt_seen & ~prompt_seen_before
        tick = int(kwargs["tick"])
        self.human_bridge_prompt_tick.copy_(
            torch.where(
                prompt_event,
                torch.full_like(self.human_bridge_prompt_tick, tick),
                self.human_bridge_prompt_tick,
            )
        )
        age = tick - self.human_bridge_prompt_tick
        bridge_active = (
            kwargs["active"]
            & ~done
            & prompt_seen
            & (age >= 0)
            & (age <= self.human_envelope_config.maximum_bridge_ticks)
        )
        before_features = human_envelope_features(
            before,
            side=self.side,
            config=self.human_envelope_config,
        )
        after_features = human_envelope_features(
            after,
            side=self.side,
            config=self.human_envelope_config,
        )
        delta = (
            after_features["potential"] - before_features["potential"]
        ).clamp(-0.1, 0.1)
        delta_mask = bridge_active & ~prompt_event
        progress_bonus = (
            delta_mask.to(reward.dtype)
            * delta
            * float(
                self.authority["reward"][
                    "human_bridge_progress_per_potential_unit"
                ]
            )
        )
        envelope_event = (
            bridge_active
            & after_features["envelope"]
            & ~self.human_bridge_envelope_seen
        )
        self.human_bridge_envelope_seen |= envelope_event
        event_bonus = envelope_event.to(reward.dtype) * float(
            self.authority["reward"]["human_bridge_envelope_event"]
        )
        total_bonus = progress_bonus + event_bonus
        reward = reward + total_bonus
        bonus_sum = float(total_bonus.sum())
        self.reward_sum += bonus_sum
        self.human_bridge_progress_reward_sum += float(progress_bonus.sum())
        self.human_bridge_event_reward_sum += float(event_bonus.sum())
        self.human_bridge_active_ticks += int(bridge_active.sum())
        self.human_bridge_envelope_reached += int(envelope_event.sum())
        return reward, done

    def telemetry(self) -> dict[str, Any]:
        result = super().telemetry()
        result.update(
            {
                "human_bridge_identity": GROUND_TO_AIR_HUMAN_BRIDGE_V11_VERSION,
                "human_bridge_config": asdict(self.human_envelope_config),
                "human_bridge_active_ticks": self.human_bridge_active_ticks,
                "human_bridge_envelope_reached": (
                    self.human_bridge_envelope_reached
                ),
                "human_bridge_envelope_fraction": (
                    self.human_bridge_envelope_reached / self.worlds
                ),
                "human_bridge_progress_reward_sum": (
                    self.human_bridge_progress_reward_sum
                ),
                "human_bridge_event_reward_sum": (
                    self.human_bridge_event_reward_sum
                ),
            }
        )
        return result


__all__ = [
    "GROUND_TO_AIR_HUMAN_BRIDGE_V11_VERSION",
    "HumanAerialEnvelopeConfig",
    "HumanEnvelopeBridgeTrainingTracker",
    "human_envelope_features",
]
