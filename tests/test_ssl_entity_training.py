from dataclasses import replace

import pytest
import torch

from rivalsim.fresh_ground_30hz import ppo_config
from rivalsim.ssl_entity_policy import EntityJointControlActorCritic
from rivalsim.ssl_entity_training import (
    JointRollout,
    fresh_entity_optimizer,
    joint_ppo_update,
    joint_sequence_loss,
    sequence_data,
)
from rivalsim.ssl_joint_control_policy import categorical_statistics


def fixture():
    torch.set_num_threads(4)
    torch.manual_seed(18)
    model = EntityJointControlActorCritic()
    config = replace(ppo_config(), rollout_horizon=3, minibatch_size=6)
    hidden = model.initial_hidden(4).reshape(1, 2, 2, -1)
    rollout = JointRollout(3, 2, hidden, "cpu")
    gen = torch.Generator().manual_seed(10)
    with torch.no_grad():
        for t in range(3):
            obs = torch.randn(2, 2, 182)
            reset = torch.full((2, 2), t == 0)
            if t == 2:
                reset[0] = True
            logits, value, h = model(
                obs.reshape(4, 182), hidden.reshape(1, 4, -1), reset_before=reset.flatten()
            )
            index, action, logp = model.sample(logits, gen)
            rollout.action_indices[t].copy_(index.reshape(2, 2))
            terminated = torch.zeros(2, 2, dtype=torch.bool)
            truncated = terminated.clone()
            truncated[0] = t == 1
            rollout.add(
                observation=obs,
                action=action.reshape(2, 2, 8),
                pre_tanh=torch.zeros(2, 2, 5),
                old_log_probability=logp.reshape(2, 2),
                value=value.reshape(2, 2),
                reward=torch.randn(2, 2),
                terminated=terminated,
                truncated=truncated,
                next_value=torch.randn(2, 2),
                train_mask=torch.ones_like(terminated),
                reset_before=reset,
            )
            hidden = h.reshape_as(hidden).masked_fill(truncated[None, ..., None], 0)
    return model, config, rollout


def test_categorical_rollout_recompute_and_loss_matches_reference():
    model, config, rollout = fixture()
    data = sequence_data(rollout, config)
    index = torch.arange(4)
    logits, values, _ = model(
        data["observations"], data["initial_hidden"], reset_before=data["reset_before"]
    )
    logp, entropy = categorical_statistics(logits, data["action_indices"])
    torch.testing.assert_close(logp, data["old_log_probability"], atol=1e-6, rtol=1e-6)
    assert torch.equal(model.action_table[rollout.action_indices], rollout.actions)
    loss, _metrics = joint_sequence_loss(model, data, index, config)
    ratio = (logp - data["old_log_probability"]).exp()
    adv = data["normalized_advantage"]
    expected = -torch.minimum(ratio * adv, ratio.clamp(0.8, 1.2) * adv).mean()
    expected += config.value_loss_coefficient * 0.5 * (values - data["returns"]).square().mean()
    expected -= config.entropy_coefficient * entropy.mean()
    torch.testing.assert_close(loss, expected, atol=0, rtol=0)
    loss.backward()
    assert model.entity_actor.weight.grad.abs().sum() > 0
    assert all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None)


def test_complete_update_counts_and_corruption_rolls_back(monkeypatch):
    model, config, rollout = fixture()
    optimizer = fresh_entity_optimizer(model)
    generator = torch.Generator().manual_seed(9)
    metrics = joint_ppo_update(model, optimizer, rollout, config, generator)
    assert metrics["optimizer_steps"] == 4
    assert metrics["kl_rejections"] == 0
    assert all(float(s["step"]) == 4 for s in optimizer.state.values())
    state = {n: v.clone() for n, v in model.state_dict().items()}
    moments = {
        p: {k: v.clone() if torch.is_tensor(v) else v for k, v in s.items()}
        for p, s in optimizer.state.items()
    }
    rng = generator.get_state().clone()
    real_step = optimizer.step

    def failing_step(*args, **kwargs):
        real_step(*args, **kwargs)
        with torch.no_grad():
            model.actor.weight.fill_(float("nan"))

    monkeypatch.setattr(optimizer, "step", failing_step)
    with pytest.raises(RuntimeError, match="nonfinite_entity_parameter_or_adam"):
        joint_ppo_update(model, optimizer, rollout, config, generator)
    assert all(torch.equal(v, model.state_dict()[n]) for n, v in state.items())
    for p, s in moments.items():
        for k, v in s.items():
            assert torch.equal(v, optimizer.state[p][k])
    assert torch.equal(rng, generator.get_state())
