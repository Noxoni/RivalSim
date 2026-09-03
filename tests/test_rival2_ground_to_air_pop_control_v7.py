from __future__ import annotations

import torch

from rivalsim.rival2_ground_to_air_option import GroundToAirStep
from rivalsim.rival2_ground_to_air_pop_control_v6 import LearnedPopOrientationConfig
from rivalsim.rival2_ground_to_air_pop_control_v7 import (
    active_pop_orientation_channel_mask,
)


def test_narrow_pop_mask_excludes_fixed_pitch_and_scripted_channels() -> None:
    worlds = 2
    false = torch.zeros(worlds, dtype=torch.bool)
    true = torch.ones(worlds, dtype=torch.bool)
    step = GroundToAirStep(
        action=torch.zeros((worlds, 8)),
        activated=false,
        active=true,
        pop_started=false,
        approach=false,
        waiting_to_launch=false,
        pop_primitive=true,
        carry=false,
        pursuit=false,
        learned_control=true,
        released=false,
        eligibility=None,  # type: ignore[arg-type]
    )
    mask = active_pop_orientation_channel_mask(
        step,
        LearnedPopOrientationConfig(
            pitch_residual_scale=0.0,
            steer_scale=0.25,
            yaw_scale=0.35,
            roll_scale=0.35,
        ),
    )
    assert mask.tolist() == [
        [False, True, False, True, True, False, False, False],
        [False, True, False, True, True, False, False, False],
    ]


def test_post_pop_learned_control_keeps_all_channels() -> None:
    worlds = 1
    false = torch.zeros(worlds, dtype=torch.bool)
    true = torch.ones(worlds, dtype=torch.bool)
    step = GroundToAirStep(
        action=torch.zeros((worlds, 8)),
        activated=false,
        active=true,
        pop_started=false,
        approach=false,
        waiting_to_launch=false,
        pop_primitive=false,
        carry=true,
        pursuit=false,
        learned_control=true,
        released=false,
        eligibility=None,  # type: ignore[arg-type]
    )
    mask = active_pop_orientation_channel_mask(
        step,
        LearnedPopOrientationConfig(pitch_residual_scale=0.0),
    )
    assert mask.all()
