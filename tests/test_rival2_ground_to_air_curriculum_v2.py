from __future__ import annotations

import pytest
import torch

from rivalsim.rival2_ground_to_air_curriculum_v2 import (
    HandoffStage,
    handoff_tick_for_block,
    learned_handoff_mask,
    successful_rehearsal_mask,
    validate_handoff_stages,
)


def test_handoff_schedule_moves_credit_assignment_earlier() -> None:
    stages = (
        HandoffStage(1, (40, 36)),
        HandoffStage(5, (32, 28)),
        HandoffStage(9, (24, 17)),
    )
    assert [handoff_tick_for_block(block, stages) for block in range(1, 13)] == [
        40,
        36,
        40,
        36,
        32,
        28,
        32,
        28,
        24,
        17,
        24,
        17,
    ]


def test_handoff_mask_requires_active_post_pop_world() -> None:
    active = torch.tensor([True, True, False, True])
    pop_age = torch.tensor([-1, 31, 40, 32])
    assert learned_handoff_mask(
        active=active, pop_age=pop_age, handoff_tick=32
    ).tolist() == [False, False, False, True]


def test_success_rehearsal_keeps_only_learned_precontact_window() -> None:
    reward = torch.zeros((8, 3))
    reward[5, 0] = 4.0
    reward[2, 1] = 4.0
    reward[6, 1] = 4.0
    learned = torch.ones_like(reward, dtype=torch.bool)
    learned[:3, 0] = False
    selected = successful_rehearsal_mask(
        reward,
        learned,
        event_reward_threshold=3.5,
        history_ticks=4,
    )
    assert torch.nonzero(selected[:, 0]).flatten().tolist() == [3, 4, 5]
    assert torch.nonzero(selected[:, 1]).flatten().tolist() == [0, 1, 2]
    assert not bool(selected[:, 2].any())


def test_invalid_handoff_schedule_fails_closed() -> None:
    with pytest.raises(ValueError):
        validate_handoff_stages((HandoffStage(2, (20,)),))
