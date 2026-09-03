from __future__ import annotations

import torch

from rivalsim.rival2_ground_to_air_option import GroundToAirConfig
from rivalsim.rival2_ground_to_air_pop_control_v6 import (
    LearnedPopOrientationConfig,
    LearnedPopOrientationController,
    pop_orientation_channel_mask,
)


def _observation(worlds: int) -> torch.Tensor:
    observation = torch.zeros((worlds, 182), dtype=torch.float32)
    # The test drives internal pop age directly, so only on-ground is needed.
    from rivalsim.rival2_aerial_option import FIELD

    observation[:, FIELD["self.on_ground"]] = 1.0
    observation[:, FIELD["ball.position.z"]] = 0.03
    return observation


def test_pop_orientation_is_learned_while_buttons_remain_scripted() -> None:
    controller = LearnedPopOrientationController(
        2,
        device="cpu",
        config=GroundToAirConfig(
            minimum_boost_fraction=0.0,
            learned_after_second_jump=True,
        ),
        orientation=LearnedPopOrientationConfig(),
    )
    controller.active[:] = True
    controller.pop_age[:] = 0
    action = torch.tensor(
        [
            [-0.8, 0.4, -0.6, 0.7, -0.9, 0.0, 1.0, 1.0],
            [0.2, -0.3, 0.8, -0.5, 0.6, 0.0, 1.0, 1.0],
        ]
    )
    result = controller.step(
        action,
        _observation(2),
        kickoff_active=torch.zeros(2, dtype=torch.bool),
        match_done=torch.zeros(2, dtype=torch.bool),
    )
    assert result.pop_primitive.all()
    assert torch.allclose(result.action[:, 1], action[:, 1])
    assert torch.allclose(result.action[:, 2], 0.5 + 0.5 * action[:, 2])
    assert torch.allclose(result.action[:, 3:5], action[:, 3:5])
    assert torch.all(result.action[:, 0] == 1.0)
    assert torch.all(result.action[:, 5] == 1.0)
    assert torch.all(result.action[:, 6] == 0.0)
    assert torch.all(result.action[:, 7] == 0.0)


def test_pop_likelihood_mask_contains_only_orientation_channels() -> None:
    controller = LearnedPopOrientationController(
        1,
        device="cpu",
        config=GroundToAirConfig(
            minimum_boost_fraction=0.0,
            learned_after_second_jump=True,
        ),
        orientation=LearnedPopOrientationConfig(),
    )
    controller.active[:] = True
    controller.pop_age[:] = 0
    result = controller.step(
        torch.zeros((1, 8)),
        _observation(1),
        kickoff_active=torch.zeros(1, dtype=torch.bool),
        match_done=torch.zeros(1, dtype=torch.bool),
    )
    assert pop_orientation_channel_mask(result).tolist() == [
        [False, True, True, True, True, False, False, False]
    ]
