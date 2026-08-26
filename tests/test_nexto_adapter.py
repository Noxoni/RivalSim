from __future__ import annotations

import numpy as np
import torch

from third_party.nexto.adapter import (
    KICKOFF_LENGTH,
    NEXTO_ACTION_COUNT,
    build_action_table,
    build_kickoff_sequence,
    nexto_pad_mapping,
)


def test_nexto_action_and_kickoff_tables_are_source_exact() -> None:
    action = build_action_table("cpu")
    kickoff = build_kickoff_sequence("cpu")
    assert action.shape == (NEXTO_ACTION_COUNT, 8)
    assert kickoff.shape == (KICKOFF_LENGTH, 8)
    assert NEXTO_ACTION_COUNT == 90
    assert KICKOFF_LENGTH == 168
    assert set(torch.unique(action).tolist()) == {-1.0, 0.0, 1.0}
    assert torch.equal(kickoff[:44], torch.tensor([[1, 0, 0, 0, 0, 0, 1, 0]]).repeat(44, 1))


def test_nexto_pad_mapping_is_unique_and_bounded() -> None:
    mapping, residual = nexto_pad_mapping()
    assert mapping.shape == (34,)
    assert len(np.unique(mapping)) == 34
    assert float(residual.max()) == 2.0
    assert np.flatnonzero(residual).tolist() == [27]
