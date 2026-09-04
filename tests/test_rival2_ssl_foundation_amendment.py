from __future__ import annotations

import copy
from dataclasses import asdict

import pytest
import torch

from benchmarks import run_rival2_ssl_foundation_ppo_v2_amended as campaign
from rivalsim.rival2_independent_critic import (
    IndependentCriticActorCritic,
    IndependentCriticPolicyConfig,
    upgrade_state_dict,
)
from rivalsim.rival2_unified_policy import Rival2UnifiedActorCritic, Rival2UnifiedPolicyConfig


@pytest.fixture(scope="module")
def parent():
    prior_threads = torch.get_num_threads()
    torch.set_num_threads(2)
    assert campaign.engine.sha256_file(campaign.PARENT) == campaign.PARENT_SHA256
    payload = torch.load(campaign.PARENT, map_location="cpu", weights_only=False)
    yield payload
    torch.set_num_threads(prior_threads)


def upgraded_model(parent):
    model = IndependentCriticActorCritic(IndependentCriticPolicyConfig(**parent["policy_config"]))
    model.load_state_dict(upgrade_state_dict(parent["model"]), strict=True)
    return model


def test_real_checkpoint_actor_value_and_hidden_exact_parity(parent):
    legacy = Rival2UnifiedActorCritic(Rival2UnifiedPolicyConfig(**parent["policy_config"]))
    legacy.load_state_dict(parent["model"], strict=True)
    upgraded = upgraded_model(parent)
    generator = torch.Generator().manual_seed(9026)
    observation = torch.randn(7, 9, 182, generator=generator)
    hidden = torch.randn(1, 7, 256, generator=generator)
    reset = torch.zeros(7, 9, dtype=torch.bool)
    reset[2, 3] = True
    reset[5, 7] = True
    with torch.no_grad():
        before = legacy(observation, hidden, reset_before=reset)
        after = upgraded(observation, hidden, reset_before=reset)
    for left, right in zip(before, after, strict=True):
        torch.testing.assert_close(left, right, rtol=0, atol=0)
    for name, value in parent["model"].items():
        if not name.startswith("critic."):
            assert torch.equal(value, upgraded.state_dict()[name])


def test_value_loss_trains_every_critic_layer_without_actor_gradient(parent):
    model = upgraded_model(parent)
    observation = torch.randn(16, 182, generator=torch.Generator().manual_seed(12))
    actor_before = {
        n: p.detach().clone() for n, p in model.named_parameters() if not n.startswith("critic.")
    }
    with torch.no_grad():
        value_before = model.isolated_value(observation).clone()
        actions_before = model(observation)[0].clone()
    optimizer = torch.optim.Adam(model.critic.parameters(), lr=3e-4)
    model.isolated_value(observation).sub(3).square().mean().backward()
    for name, parameter in model.named_parameters():
        if name.startswith("critic."):
            assert parameter.grad is not None and torch.isfinite(parameter.grad).all()
            assert parameter.grad.abs().sum() > 0
        else:
            assert parameter.grad is None
    optimizer.step()
    for name, before in actor_before.items():
        assert torch.equal(before, dict(model.named_parameters())[name])
    with torch.no_grad():
        assert torch.equal(actions_before, model(observation)[0])
        assert not torch.equal(value_before, model.isolated_value(observation))


def test_actor_loss_has_no_gradient_into_independent_critic(parent):
    model = upgraded_model(parent)
    actor, _value, _hidden = model(torch.randn(3, 182))
    actor.square().mean().backward()
    assert all(p.grad is None for p in model.critic.parameters())
    assert any(p.grad is not None for p in model.actor.parameters())


def test_migration_preserves_actor_and_existing_adam_state(parent):
    payload = campaign.migrate_payload(parent, "A" * 64, "B" * 64)
    model = upgraded_model(parent)
    model.load_state_dict(payload["model"], strict=True)
    critic_parameters = tuple(model.critic.parameters())
    critic_ids = {id(p) for p in critic_parameters}
    policy_parameters = tuple(p for p in model.parameters() if id(p) not in critic_ids)
    optimizer = torch.optim.Adam(
        [
            {"name": "policy", "params": policy_parameters, "lr": 1e-4},
            {"name": "critic", "params": critic_parameters, "lr": 3e-4},
        ]
    )
    optimizer.load_state_dict(payload["optimizer"])
    old_groups = parent["optimizer"]["param_groups"]
    for parameter, old_id in zip(policy_parameters, old_groups[0]["params"], strict=True):
        for key, value in parent["optimizer"]["state"][old_id].items():
            assert torch.equal(value, optimizer.state[parameter][key])
    for parameter in model.critic.features.parameters():
        assert not optimizer.state.get(parameter)
    for parameter, old_id in zip(
        model.critic.head.parameters(), old_groups[1]["params"], strict=True
    ):
        for key, value in parent["optimizer"]["state"][old_id].items():
            assert torch.equal(value, optimizer.state[parameter][key])
    assert payload["accepted_updates_total"] == parent["accepted_updates_total"] == 600
    assert payload["total_agent_samples"] == parent["total_agent_samples"]
    assert payload["ppo_config"] == asdict(campaign.new_ppo_config())
    assert payload["optimizer_group_lrs"] == {"policy": 1e-4, "critic": 3e-4}
    assert payload["policy_config_sha256"] == model.config.content_hash
    for key in (
        "policy_generator_state",
        "shuffle_generator_state",
        "torch_cpu_rng_state",
        "torch_cuda_rng_state",
    ):
        assert torch.equal(payload[key], parent[key])
    assert campaign.engine.sha256_file(campaign.PARENT) == campaign.PARENT_SHA256


def test_migration_rejects_wrong_lineage_and_wrong_config(parent):
    wrong = copy.copy(parent)
    wrong["accepted_updates_total"] = 599
    with pytest.raises(ValueError, match="parent rejected"):
        campaign.migrate_payload(wrong, "A" * 64, "B" * 64)
    wrong = copy.copy(parent)
    wrong["ppo_config"] = {**parent["ppo_config"], "epochs": 7}
    with pytest.raises(ValueError, match="parent rejected"):
        campaign.migrate_payload(wrong, "A" * 64, "B" * 64)


def test_authority_changes_only_authorized_settings():
    payload = campaign.authority_payload()
    original = __import__("json").loads(campaign.original.AUTHORITY.read_text())
    for key in ("reward", "reset_curriculum", "exploration", "integrity", "source"):
        assert payload[key] == original[key]
    assert payload["ppo"]["policy_learning_rate"] == 1e-4
    assert payload["ppo"]["epochs"] == 2
    assert payload["campaign"]["evaluation_ticks"] == 3600
    assert payload["opponents"]["current_probability"] == 0.4
    assert payload["opponents"]["nexto_probability"] == 0.4
    assert payload["opponents"]["frozen_v5_probability"] == 0.2
    assert payload["amendment"]["critic"]["architecture"] == [182, 512, 512, 512, 1]
