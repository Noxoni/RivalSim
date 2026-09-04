from __future__ import annotations

import json
from pathlib import Path

import torch

from benchmarks.evaluate_rival2_unified_capability_v2 import load_unified, report_path
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


def test_physical_evaluator_accepts_ground_curriculum_checkpoint(
    tmp_path: Path,
) -> None:
    source = (
        ROOT
        / "checkpoints/rival2/unified_capability_distillation_v5"
        / "rival2_unified_capability_v5.pt"
    )
    payload = torch.load(source, map_location="cpu", weights_only=False)
    payload["format"] = "RIVAL2_UNIFIED_GROUND_CURRICULUM_PPO_V2_CHECKPOINT"
    payload["accepted_updates_total"] = 174
    target = tmp_path / "outside-repo-ground-checkpoint.pt"
    torch.save(payload, target)

    loaded, model = load_unified(target.resolve(), "cpu")

    assert loaded["accepted_updates_total"] == 174
    assert report_path(target.resolve()) == target.resolve().as_posix()
    assert sum(parameter.numel() for parameter in model.parameters()) == 1_071_131
