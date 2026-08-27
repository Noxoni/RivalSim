from __future__ import annotations

import copy
from typing import Any

import torch

from rivalsim.rival2_mixed_ppo import (
    Rival2MixedPPOSafetyConfig,
    migrate_adam_to_mixed_groups,
    mixed_optimizer_learning_rates,
)
from rivalsim.rival2_policy import Rival2ActorCritic, Rival2PolicyConfig


def _exact(left: Any, right: Any) -> bool:
    if isinstance(left, torch.Tensor) and isinstance(right, torch.Tensor):
        return left.dtype == right.dtype and left.shape == right.shape and torch.equal(left, right)
    if isinstance(left, dict) and isinstance(right, dict):
        return left.keys() == right.keys() and all(_exact(left[key], right[key]) for key in left)
    if isinstance(left, (list, tuple)) and isinstance(right, type(left)):
        return len(left) == len(right) and all(
            _exact(a, b) for a, b in zip(left, right, strict=True)
        )
    return left == right


def test_adam_group_migration_preserves_every_parameter_state_and_step() -> None:
    torch.manual_seed(20260827)
    model = Rival2ActorCritic(Rival2PolicyConfig(hidden_dim=16, hidden_layers=2))
    optimizer = torch.optim.Adam(model.parameters(), lr=3.0e-4)
    observation = torch.randn(32, model.config.obs_dim)
    actor, value = model(observation)
    (actor.square().mean() + value.square().mean()).backward()
    optimizer.step()

    state_before = {
        name: copy.deepcopy(optimizer.state[parameter])
        for name, parameter in model.named_parameters()
    }
    migrated, proof = migrate_adam_to_mixed_groups(
        model,
        optimizer,
        Rival2MixedPPOSafetyConfig(),
    )
    state_after = {
        name: copy.deepcopy(migrated.state[parameter])
        for name, parameter in model.named_parameters()
    }

    assert proof["verdict"] == "PASS_GREEN"
    assert proof["parameter_count"] == len(list(model.parameters()))
    assert _exact(state_before, state_after)
    assert mixed_optimizer_learning_rates(migrated) == {
        "policy": 1.0e-4,
        "critic": 3.0e-4,
    }
    assert len(migrated.param_groups[0]["params"]) == 6
    assert len(migrated.param_groups[1]["params"]) == 2
