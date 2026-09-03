from __future__ import annotations

import pytest
import torch

from rivalsim.rival2_aerial_option import FIELD
from rivalsim.rival2_contracts import POSITION_SCALE
from rivalsim.rival2_ground_ball_pop import (
    PrecontactPopConfig,
    PrecontactPopController,
)


def _observation(distance: float, worlds: int = 1) -> torch.Tensor:
    observation = torch.zeros((worlds, 182), dtype=torch.float32)
    observation[:, FIELD["relative.ball_position.y"]] = distance / POSITION_SCALE[1]
    observation[:, FIELD["self.forward.y"]] = 1.0
    observation[:, FIELD["self.up.z"]] = 1.0
    return observation


def test_precontact_launch_starts_before_ordinary_contact() -> None:
    config = PrecontactPopConfig(trigger_distance_uu=175.0)
    controller = PrecontactPopController(1, device="cpu", config=config)
    learned = torch.zeros((1, 8))
    far = controller.step(learned, _observation(220.0))
    assert far.approaching.item()
    assert not far.launch_started.item()
    near = controller.step(learned, _observation(170.0))
    assert near.launch_started.item()
    assert near.primitive.item()
    assert near.action[0, 5].item() == 1.0


def test_precontact_sequence_releases_jump_before_second_jump() -> None:
    config = PrecontactPopConfig(
        trigger_distance_uu=175.0,
        first_jump_hold_ticks=2,
        jump_release_ticks=2,
        second_jump=True,
    )
    controller = PrecontactPopController(1, device="cpu", config=config)
    learned = torch.zeros((1, 8))
    actions = [controller.step(learned, _observation(170.0)).action.clone()]
    actions.extend(controller.step(learned, _observation(150.0)).action.clone() for _ in range(5))
    jumps = [int(action[0, 5].item()) for action in actions]
    assert jumps == [1, 1, 0, 0, 1, 0]


def test_precontact_config_rejects_post_contact_trigger() -> None:
    with pytest.raises(ValueError, match="precede ordinary contact"):
        PrecontactPopConfig(trigger_distance_uu=120.0)
