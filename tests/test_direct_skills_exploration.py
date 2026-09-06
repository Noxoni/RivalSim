from dataclasses import asdict

import pytest
import torch

from benchmarks.run_direct_skills_exploration_v1 import REVIEW, SOURCE, SOURCE_SHA, START, amendment
from benchmarks.run_rival2_fresh_ground_30hz_v1 import sha
from rivalsim.direct_skills_exploration_v1 import TEMPERATURE, TrainingExplorationPolicy
from rivalsim.fresh_ground_30hz import ppo_config
from rivalsim.ssl_entity_policy import EntityJointControlActorCritic
from rivalsim.ssl_joint_control_policy import categorical_statistics


@pytest.fixture(scope="module")
def policies():
    torch.set_num_threads(4)
    assert sha(SOURCE) == SOURCE_SHA
    payload = torch.load(SOURCE, map_location="cpu", weights_only=False)
    old, new = EntityJointControlActorCritic(), TrainingExplorationPolicy()
    for model in (old, new):
        model.load_state_dict(payload["model"], strict=True)
        model.eval()
    return old, new


def test_scalar_temperature_has_no_parameters_or_state_dict_change(policies):
    old, new = policies
    assert old.config.content_hash == new.config.content_hash
    assert dict(old.named_parameters()).keys() == dict(new.named_parameters()).keys()
    assert all(torch.equal(old.state_dict()[k], v) for k, v in new.state_dict().items())


def test_actor_scaling_preserves_deterministic_action_value_and_recurrence(policies):
    old, new = policies
    obs = torch.randn(5, 7, 182, generator=torch.Generator().manual_seed(104)) * 0.1
    reset = torch.zeros(5, 7, dtype=torch.bool)
    reset[:, 0] = True
    reset[2, 3] = True
    with torch.no_grad():
        a, v, h = old(obs, reset_before=reset)
        b, w, g = new(obs, reset_before=reset)
        assert torch.equal(b, a / TEMPERATURE)
        assert torch.equal(old.deterministic(a), new.deterministic(b))
        assert torch.equal(v, w) and torch.equal(h, g)
        _, before = categorical_statistics(a, a.argmax(-1))
        _, after = categorical_statistics(b, b.argmax(-1))
        assert bool((after >= before - 1e-5).all())
        actor, hidden = new.forward_actor(obs, reset_before=reset)
        assert torch.equal(actor, b) and torch.equal(hidden, g)


def test_sample_and_ppo_likelihood_refer_to_the_same_softened_distribution(policies):
    _, model = policies
    logits = torch.tensor([[8.0, 4.0, 0.0], [-2.0, 4.0, 1.0]]) / TEMPERATURE
    index, action, logp = model.sample(logits, torch.Generator().manual_seed(83))
    replay, _ = categorical_statistics(logits, index)
    assert torch.equal(logp, replay)
    assert torch.equal(action, model.action_table[index])
    assert torch.equal((replay - logp).exp(), torch.ones_like(logp))


def test_critic_loss_still_cannot_change_actor(policies):
    _, model = policies
    model.zero_grad(set_to_none=True)
    model.train()
    model.isolated_value(torch.ones(4, 182) * 0.1).sum().backward()
    assert all(
        p.grad is None or not bool(p.grad.any())
        for n, p in model.named_parameters()
        if not n.startswith("critic.")
    )
    assert any(p.grad is not None and bool(p.grad.any()) for p in model.critic.parameters())
    model.zero_grad(set_to_none=True)
    model.eval()


def test_amendment_is_one_setting_with_fixed_review_and_same_ppo():
    a = amendment()
    assert a["training_temperature"] == 2.0
    assert a["ppo"] == asdict(ppo_config())
    assert a["ppo"]["entropy_coefficient"] == 0.001
    assert START == 554 and REVIEW == 600
    assert not a["reward_change"] and not a["critic_change"] and not a["architecture_change"]
