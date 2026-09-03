"""Ground-to-air option control with learned pop orientation.

The base option keeps the deterministic approach and jump-button schedule that
made the controlled scorer reliable.  During the jump hold/release/second-jump
phase, however, V6 exposes steer, pitch, yaw, and roll to the learned latent
action.  This permits either a plain double jump or a partial tornado/front-
corner alignment without prescribing or rewarding a named animation.
"""

from __future__ import annotations

from dataclasses import dataclass, replace

import torch

from rivalsim.rival2_ground_to_air_option import (
    GroundToAirConfig,
    GroundToAirController,
    GroundToAirStep,
)

GROUND_TO_AIR_POP_CONTROL_V6_VERSION = "RIVAL2_GROUND_TO_AIR_POP_CONTROL_V6"
POP_ORIENTATION_CHANNEL_INDICES = (1, 2, 3, 4)


@dataclass(frozen=True, slots=True)
class LearnedPopOrientationConfig:
    pitch_center: float = 0.5
    pitch_residual_scale: float = 0.5
    steer_scale: float = 1.0
    yaw_scale: float = 1.0
    roll_scale: float = 1.0

    def __post_init__(self) -> None:
        if not -1.0 <= self.pitch_center <= 1.0:
            raise ValueError("pitch center must be in [-1,1]")
        for name in (
            "pitch_residual_scale",
            "steer_scale",
            "yaw_scale",
            "roll_scale",
        ):
            if not 0.0 <= getattr(self, name) <= 1.0:
                raise ValueError(f"{name} must be in [0,1]")


class LearnedPopOrientationController(GroundToAirController):
    """Apply learned orientation residuals inside the scripted jump sequence."""

    def __init__(
        self,
        worlds: int,
        *,
        device: str | torch.device,
        config: GroundToAirConfig,
        orientation: LearnedPopOrientationConfig,
    ) -> None:
        super().__init__(worlds, device=device, config=config)
        self.orientation = orientation

    def step(
        self,
        base_action: torch.Tensor,
        observation: torch.Tensor,
        *,
        kickoff_active: torch.Tensor,
        match_done: torch.Tensor,
    ) -> GroundToAirStep:
        step = super().step(
            base_action,
            observation,
            kickoff_active=kickoff_active,
            match_done=match_done,
        )
        pop = step.pop_primitive
        if not bool(pop.any()):
            return step
        action = step.action.clone()
        action[pop, 1] = (
            base_action[pop, 1] * self.orientation.steer_scale
        ).clamp(-1.0, 1.0)
        action[pop, 2] = (
            self.orientation.pitch_center
            + base_action[pop, 2] * self.orientation.pitch_residual_scale
        ).clamp(-1.0, 1.0)
        action[pop, 3] = (
            base_action[pop, 3] * self.orientation.yaw_scale
        ).clamp(-1.0, 1.0)
        action[pop, 4] = (
            base_action[pop, 4] * self.orientation.roll_scale
        ).clamp(-1.0, 1.0)
        # Throttle, jump, boost, and handbrake remain the deterministic option
        # primitive.  Only the four orientation channels are learned here.
        return replace(
            step,
            action=action,
            learned_control=step.learned_control | pop,
        )


def pop_orientation_channel_mask(step: GroundToAirStep) -> torch.Tensor:
    """Return the latent channels whose sampled values reached the simulator."""

    mask = torch.zeros(
        (step.action.shape[0], step.action.shape[1]),
        dtype=torch.bool,
        device=step.action.device,
    )
    mask[step.learned_control & ~step.pop_primitive] = True
    for channel in POP_ORIENTATION_CHANNEL_INDICES:
        mask[step.pop_primitive, channel] = True
    return mask


__all__ = [
    "GROUND_TO_AIR_POP_CONTROL_V6_VERSION",
    "POP_ORIENTATION_CHANNEL_INDICES",
    "LearnedPopOrientationConfig",
    "LearnedPopOrientationController",
    "pop_orientation_channel_mask",
]
