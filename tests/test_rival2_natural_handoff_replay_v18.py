from __future__ import annotations

from dataclasses import fields

import numpy as np
import pytest
import torch

from rivalsim.lifecycle_state import LifecycleSnapshot
from rivalsim.rival2_natural_handoff_replay_v18 import (
    NATURAL_HANDOFF_CORPUS_V18_FORMAT,
    load_natural_handoff_corpus,
    normalized_indices,
    state_snapshot_from_corpus,
)
from rivalsim.state import StateSnapshot


def _payload(count: int = 3):
    state = StateSnapshot.empty(count)
    lifecycle = {
        item.name: np.zeros((count,), dtype=np.int32)
        for item in fields(LifecycleSnapshot)
    }
    # The loader validates presence; exact lifecycle shapes are validated by
    # the live replay because fields have different per-world widths.
    return {
        "format": NATURAL_HANDOFF_CORPUS_V18_FORMAT,
        "count": count,
        "state": {item.name: getattr(state, item.name) for item in fields(state)},
        "lifecycle": lifecycle,
        "bridge_views": {},
        "observation": torch.zeros((count, 2, 182)),
    }


def test_corpus_selection_preserves_exact_physical_rows() -> None:
    payload = _payload()
    payload["state"]["ball_pos"][:, 0] = np.arange(3, dtype=np.float32)
    selected = state_snapshot_from_corpus(payload, [2, 0])
    assert selected.num_envs == 2
    assert np.array_equal(selected.ball_pos[:, 0], np.asarray((2.0, 0.0)))


def test_selection_rejects_empty_and_out_of_range() -> None:
    payload = _payload()
    with pytest.raises(ValueError):
        normalized_indices(payload, [])
    with pytest.raises(ValueError):
        normalized_indices(payload, [3])


def test_loader_rejects_wrong_format(tmp_path) -> None:
    path = tmp_path / "bad.pt"
    payload = _payload()
    payload["format"] = "WRONG"
    torch.save(payload, path)
    with pytest.raises(ValueError):
        load_natural_handoff_corpus(path)
