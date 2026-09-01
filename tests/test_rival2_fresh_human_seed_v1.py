from __future__ import annotations

import json
from pathlib import Path

import torch

from benchmarks import run_rival2_fresh_human_seed_v1 as fresh
from rivalsim.rival2_120hz_transition import tensor_tree_sha256

ROOT = Path(__file__).resolve().parents[1]


def test_frozen_temporal_split_is_disjoint_and_complete() -> None:
    manifest = json.loads(fresh.SPLIT_MANIFEST.read_text(encoding="utf-8"))
    train = manifest["splits"]["train"]
    validation = manifest["splits"]["validation"]
    test = manifest["splits"]["test"]
    assert train["frame_count"] == 46_644
    assert validation["frame_count"] == 5_831
    assert test["frame_count"] == 5_831
    assert train["end_index_exclusive"] == validation["start_index_inclusive"]
    assert validation["end_index_exclusive"] == test["start_index_inclusive"]
    assert sum(row["frame_count"] for row in (train, validation, test)) == 58_306
    assert manifest["disjoint"]
    assert manifest["complete_coverage"]


def test_fresh_initialization_is_deterministic_and_has_fixed_stage1_log_std() -> None:
    first = fresh.fresh_model()
    second = fresh.fresh_model()
    assert tensor_tree_sha256(first.state_dict()) == tensor_tree_sha256(second.state_dict())
    assert torch.count_nonzero(first.actor.weight[5:10]) == 0
    assert torch.equal(first.actor.bias[5:10], torch.full((5,), -1.0))
    assert all(not parameter.requires_grad for parameter in first.critic.parameters())
    assert all(parameter.requires_grad for parameter in first.trunk.parameters())


def test_authority_forbids_old_lineage_and_external_opponents() -> None:
    authority = json.loads(fresh.AUTHORITY.read_text(encoding="utf-8"))
    assert authority["lineage"]["fresh_random_initialization"]
    assert not authority["lineage"]["prior_rival_checkpoint_loaded"]
    assert authority["forbidden"]["previous_rival_checkpoint_load"]
    assert authority["stage2"]["opponents"] == {
        "current": 1.0,
        "historical": 0.0,
        "nexto": 0.0,
        "wisp": 0.0,
    }
    assert authority["stage2"]["accepted_updates"] == 600


def test_stage1_checkpoint_path_is_new_lineage_only() -> None:
    assert fresh.CHECKPOINT.relative_to(ROOT).as_posix() == (
        "checkpoints/rival2/fresh_human_seed_v1/rival2_fresh_human_seed_v1.pt"
    )
    assert "human_bc_v5" not in Path(fresh.__file__).read_text(encoding="utf-8")

