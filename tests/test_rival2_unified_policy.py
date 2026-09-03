from __future__ import annotations

import torch

from rivalsim.rival2_policy import Rival2ActorCritic, deterministic_hybrid_action
from rivalsim.rival2_unified_policy import (
    Rival2UnifiedActorCritic,
    Rival2UnifiedPolicyConfig,
    deterministic_unified_action,
)


def make_pair() -> tuple[Rival2ActorCritic, Rival2UnifiedActorCritic]:
    torch.manual_seed(7)
    parent = Rival2ActorCritic()
    unified = Rival2UnifiedActorCritic()
    unified.load_feedforward_parent(parent)
    return parent, unified


def test_zero_residual_is_exact_feedforward_parent_parity() -> None:
    parent, unified = make_pair()
    observation = torch.randn(11, 182)
    parent_actor, parent_value = parent(observation)
    actor, value, hidden = unified(observation)
    assert torch.equal(actor, parent_actor)
    assert torch.equal(value, parent_value)
    assert torch.equal(
        deterministic_unified_action(actor), deterministic_hybrid_action(parent_actor)
    )
    assert hidden.shape == (1, 11, 256)


def test_zero_residual_sequence_parity_and_reset_semantics() -> None:
    parent, unified = make_pair()
    observation = torch.randn(4, 9, 182)
    reset = torch.zeros((4, 9), dtype=torch.bool)
    reset[1, 4] = True
    parent_actor, parent_value = parent(observation.reshape(-1, 182))
    actor, value, _hidden = unified(observation, reset_before=reset)
    assert torch.equal(actor, parent_actor.reshape(4, 9, 13))
    assert torch.equal(value, parent_value.reshape(4, 9))


def test_freeze_base_exposes_only_recurrent_context_parameters() -> None:
    _parent, unified = make_pair()
    unified.freeze_base()
    trainable = {name for name, value in unified.named_parameters() if value.requires_grad}
    assert trainable == {
        "context_encoder.weight",
        "context_encoder.bias",
        "context_gru.weight_ih_l0",
        "context_gru.weight_hh_l0",
        "context_gru.bias_ih_l0",
        "context_gru.bias_hh_l0",
        "context_actor.weight",
        "context_actor.bias",
    }


def test_policy_config_hash_is_stable_and_versioned() -> None:
    left = Rival2UnifiedPolicyConfig()
    right = Rival2UnifiedPolicyConfig()
    assert left.content_hash == right.content_hash
    assert len(left.content_hash) == 64


def test_isolated_value_trains_critic_without_trunk_gradient() -> None:
    _parent, unified = make_pair()
    unified.zero_grad(set_to_none=True)
    value = unified.isolated_value(torch.randn(5, 182))
    value.sum().backward()
    assert any(parameter.grad is not None for parameter in unified.critic.parameters())
    assert all(parameter.grad is None for parameter in unified.trunk.parameters())
