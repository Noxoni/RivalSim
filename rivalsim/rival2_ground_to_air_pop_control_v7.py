"""Causal likelihood mask for narrow learned pop-orientation residuals."""

from __future__ import annotations

import torch

from rivalsim.rival2_ground_to_air_option import GroundToAirStep
from rivalsim.rival2_ground_to_air_pop_control_v6 import LearnedPopOrientationConfig

GROUND_TO_AIR_POP_CONTROL_V7_VERSION = "RIVAL2_GROUND_TO_AIR_POP_CONTROL_V7"


def active_pop_orientation_channel_mask(
    step: GroundToAirStep,
    orientation: LearnedPopOrientationConfig,
) -> torch.Tensor:
    """Mask only latent channels whose configured residual reaches physics."""

    mask = torch.zeros_like(step.action, dtype=torch.bool)
    mask[step.learned_control & ~step.pop_primitive] = True
    scale_by_channel = {
        1: orientation.steer_scale,
        2: orientation.pitch_residual_scale,
        3: orientation.yaw_scale,
        4: orientation.roll_scale,
    }
    for channel, scale in scale_by_channel.items():
        if scale > 0.0:
            mask[step.pop_primitive, channel] = True
    return mask


__all__ = [
    "GROUND_TO_AIR_POP_CONTROL_V7_VERSION",
    "active_pop_orientation_channel_mask",
]
