import torch

from rivalsim.fresh_ground_30hz import fresh_model
from rivalsim.ssl_joint_control_policy import (
    PROJECTION_SIGMA,
    JointControlActorCritic,
    categorical_statistics,
)


def test_joint_table_probability_projection_and_frozen_backbone():
    torch.set_num_threads(4)
    old = fresh_model(123)
    new = JointControlActorCritic()
    new.initialize_from_hybrid(old.state_dict())
    assert new.action_table.shape == (90, 8)
    assert len(torch.unique(new.action_table, dim=0)) == 90
    assert set(new.action_table[:, 5:].flatten().tolist()) == {0.0, 1.0}
    assert (new.action_table[new.action_table[:, 6] > 0, 0] == 1).all()
    obs = torch.randn(11, 182)
    initial = old.initial_hidden(11)
    a, v, h = old(obs, initial)
    logits, v2, h2 = new(obs, initial)
    expected = (
        a[:, :5] @ new.action_table[:, :5].T / PROJECTION_SIGMA**2
        - new.action_table[:, :5].square().sum(-1) / (2 * PROJECTION_SIGMA**2)
        + a[:, 10:] @ new.action_table[:, 5:].T
    )
    torch.testing.assert_close(logits, expected, atol=2e-6, rtol=2e-6)
    torch.testing.assert_close(v, v2, atol=0, rtol=0)
    torch.testing.assert_close(h, h2, atol=0, rtol=0)
    for name, value in old.state_dict().items():
        if not name.startswith(("actor.", "context_actor.")):
            assert torch.equal(value, new.state_dict()[name])
    index, action, sampled = new.sample(logits, torch.Generator().manual_seed(42))
    logp, entropy = categorical_statistics(logits, index)
    torch.testing.assert_close(sampled, logp, atol=0, rtol=0)
    distribution = torch.distributions.Categorical(logits=logits)
    torch.testing.assert_close(logp, distribution.log_prob(index), atol=2e-6, rtol=2e-6)
    torch.testing.assert_close(entropy, distribution.entropy(), atol=2e-6, rtol=2e-6)
    assert torch.equal(action, new.action_table[index])
    assert torch.equal(new.deterministic(logits), new.action_table[logits.argmax(-1)])


def test_joint_sequence_reset_parity_and_critic_isolation():
    model = JointControlActorCritic()
    obs = torch.randn(4, 7, 182)
    reset = torch.zeros(4, 7, dtype=torch.bool)
    reset[:, 0] = True
    reset[1, 3] = True
    reset[2, 5] = True
    initial = model.initial_hidden(4)
    logits, _, last = model(obs, initial, reset_before=reset)
    h = initial
    pieces = []
    for i in range(7):
        a, h = model.forward_actor(obs[:, i], h, reset_before=reset[:, i])
        pieces.append(a)
    torch.testing.assert_close(logits, torch.stack(pieces, 1), atol=1e-6, rtol=1e-5)
    torch.testing.assert_close(last, h, atol=1e-6, rtol=1e-5)
    model.isolated_value(obs).square().mean().backward()
    assert all(p.grad is None for n, p in model.named_parameters() if not n.startswith("critic."))
