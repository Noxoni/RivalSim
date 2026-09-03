from __future__ import annotations

import json
from pathlib import Path

import torch

from benchmarks.run_rival2_unified_capability_distillation_v5 import (
    FAMILIES,
    PrefixSequencePool,
)

ROOT = Path(__file__).resolve().parents[1]


def test_prefix_pool_replays_from_episode_start_and_zeros_terminal_padding() -> None:
    observation = torch.ones((8, 3, 182), dtype=torch.float32)
    valid = torch.zeros((8, 3), dtype=torch.bool)
    valid[:5, 0] = True
    valid[:, 1:] = True
    supervision = torch.zeros_like(valid)
    supervision[2:5, 0] = True
    supervision[4, 1] = True
    supervision[2:5, 2] = True
    payload = {
        "observation": observation,
        "valid": valid,
        "supervision": supervision,
        "side": torch.tensor([0, 1, 1]),
        "scenario": torch.tensor([3, 2, 3]),
    }
    pool = PrefixSequencePool.from_payload(
        payload,
        sequence_ticks=6,
        burn_in_ticks=2,
        minimum_supervision_ticks=2,
        scenario=3,
    )
    assert torch.equal(pool.candidates, torch.tensor([0, 2]))
    assert torch.equal(pool.observation[5, 0], torch.zeros(182))
    sampled, side, mask = pool.sample(8, generator=torch.Generator().manual_seed(4), device="cpu")
    assert sampled.shape == (8, 6, 182)
    assert side.shape == (8,)
    assert mask.shape == (8, 6)
    assert bool((mask.sum(dim=1) >= 3).all())


def test_v5_authority_has_one_weight_for_every_family() -> None:
    authority = json.loads(
        (ROOT / "results/rival2/unified_capability_distillation_v5/authority.json").read_text(
            encoding="utf-8"
        )
    )
    weights = authority["optimization"]["family_weights"]
    assert set(weights) == set(FAMILIES)
    assert abs(sum(weights.values()) - 1.0) < 1.0e-9
    assert authority["corpora"]["prefix_sequence_ticks"] == 192
    assert authority["integrity"]["optimizer_steps_before_authority_commit"] == 0
