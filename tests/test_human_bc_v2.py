from __future__ import annotations

import json
from pathlib import Path

import pytest
import torch

from benchmarks.run_rival2_human_bc_continuation_v1 import (
    _actor_only_mode,
    _actor_only_selection_score,
    _configure_trainable_parameters,
    _mechanic_label_comparison,
    _model_partition_hashes,
)
from rivalsim.rival2_policy import Rival2ActorCritic, Rival2PolicyConfig

ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = ROOT / "results/rival2/human_bc_v2/frozen_config.json"


def _model() -> Rival2ActorCritic:
    return Rival2ActorCritic(Rival2PolicyConfig())


def test_actor_only_partition_and_fresh_optimizer() -> None:
    config = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
    model = _model()
    before = _model_partition_hashes(model)
    parameters, names = _configure_trainable_parameters(model, config)
    assert _actor_only_mode(config)
    assert names == ("actor.weight", "actor.bias")
    assert [name for name, parameter in model.named_parameters() if parameter.requires_grad] == [
        "actor.weight",
        "actor.bias",
    ]
    optimizer = torch.optim.AdamW(parameters, lr=3e-5)
    assert optimizer.state_dict()["state"] == {}
    assert before == _model_partition_hashes(model)


def test_mechanic_label_comparison_is_broad_and_deterministic() -> None:
    parent = {
        "per_mechanic_label": {
            "a": {"complete_action_rmse": 1.0},
            "b": {"complete_action_rmse": 2.0},
        }
    }
    candidate = {
        "per_mechanic_label": {
            "a": {"complete_action_rmse": 0.9},
            "b": {"complete_action_rmse": 2.02},
        }
    }
    result = _mechanic_label_comparison(
        parent, candidate, nonregression_relative_tolerance=0.02
    )
    assert result["total_labels"] == 2
    assert result["improved_labels"] == 1
    assert result["nonregressed_labels"] == 2
    assert result["per_label_rmse_ratio_to_parent"] == {"a": 0.9, "b": 1.01}


def test_actor_only_score_includes_families_labels_and_retention() -> None:
    config = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
    parent = {
        "families": {
            "gameplay": {"complete_action_rmse": 1.0},
            "mechanic": {"complete_action_rmse": 1.0},
        }
    }
    candidate = {
        "families": {
            "gameplay": {"complete_action_rmse": 0.9},
            "mechanic": {"complete_action_rmse": 0.8},
        }
    }
    score = _actor_only_selection_score(
        candidate,
        parent,
        {"actor_mean_kl": 0.005},
        {"mean_rmse_ratio_to_parent": 0.85},
        config,
        {"retention": {"soft_actor_mean_kl": 0.01}},
    )
    assert score == pytest.approx(0.815)
