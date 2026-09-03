from __future__ import annotations

import torch

from rivalsim.rival2_ground_to_air_self_imitation_v13 import (
    SelfImitationConfig,
    successful_history_mask,
)


def test_successful_history_mask_unions_multiple_event_windows() -> None:
    event = torch.zeros((8, 1, 2), dtype=torch.bool)
    active = torch.ones_like(event)
    event[2, 0, 0] = True
    event[6, 0, 0] = True
    selected = successful_history_mask(event, active, history_ticks=3)
    assert selected[:, 0, 0].tolist() == [True, True, True, False, True, True, True, False]
    assert not bool(selected[:, 0, 1].any())


def test_successful_history_mask_never_selects_inactive_actions() -> None:
    event = torch.zeros((5, 1, 2), dtype=torch.bool)
    active = torch.ones_like(event)
    event[4, 0, 1] = True
    active[2, 0, 1] = False
    selected = successful_history_mask(event, active, history_ticks=4)
    assert selected[:, 0, 1].tolist() == [False, True, False, True, True]


def test_self_imitation_config_keeps_positive_bounded_history() -> None:
    config = SelfImitationConfig()
    assert config.history_ticks == 96
    assert config.maximum_success_samples == 65_536
