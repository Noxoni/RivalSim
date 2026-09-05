import math

import pytest
import torch

from rivalsim.fresh_ground_30hz import policy_config
from rivalsim.fresh_ground_exploration_comparison import ScaledNoiseActorCritic
from rivalsim.rival2_independent_critic import IndependentCriticActorCritic
from rivalsim.rival2_policy import (
    deterministic_hybrid_action,
    hybrid_distribution_parameters,
    hybrid_log_probability,
    sample_hybrid_action,
)


@pytest.mark.parametrize("scale", [1.0, 0.5])
def test_same_weights_exact_means_buttons_values_and_scaled_distribution(scale):
    torch.set_num_threads(2)
    torch.manual_seed(20260905)
    base = IndependentCriticActorCritic(policy_config())
    test = ScaledNoiseActorCritic(policy_config())
    test.load_state_dict(base.state_dict())
    test.set_sigma_scale(scale)
    x = torch.randn(4, 9, 182)
    reset = torch.zeros(4, 9, dtype=torch.bool)
    reset[:, 0] = True
    reset[1, 4] = True
    a, v, h = base(x, reset_before=reset)
    b, w, j = test(x, reset_before=reset)
    for left, right in [(a[..., :5], b[..., :5]), (a[..., 10:], b[..., 10:]), (v, w), (h, j)]:
        torch.testing.assert_close(left, right, rtol=0, atol=0)
    torch.testing.assert_close(
        deterministic_hybrid_action(a, base.config),
        deterministic_hybrid_action(b, test.config),
        rtol=0,
        atol=0,
    )
    _, old_std, _ = hybrid_distribution_parameters(a, base.config)
    _, new_std, _ = hybrid_distribution_parameters(b, test.config)
    torch.testing.assert_close(new_std, old_std + math.log(scale), rtol=0, atol=2e-7)
    sample = sample_hybrid_action(
        b, generator=torch.Generator().manual_seed(100), config=test.config
    )
    recomputed = hybrid_log_probability(
        b, sample.action, config=test.config, pre_tanh=sample.pre_tanh
    )
    torch.testing.assert_close(sample.log_probability, recomputed, rtol=0, atol=0)
    assert all(torch.equal(t, test.state_dict()[k]) for k, t in base.state_dict().items())
    assert test.config.log_std_min == pytest.approx(math.log(0.2 * scale))
    assert test.config.log_std_max == pytest.approx(math.log(0.9 * scale))


def test_step_and_sequence_actor_likelihood_agree_across_resets():
    torch.set_num_threads(2)
    model = ScaledNoiseActorCritic(policy_config())
    model.set_sigma_scale(0.5)
    x = torch.randn(6, 11, 182)
    reset = torch.zeros(6, 11, dtype=torch.bool)
    reset[:, 0] = True
    reset[2, 5] = True
    reset[3, 8] = True
    hidden = model.initial_hidden(6)
    actors = []
    for t in range(11):
        actor, hidden = model.forward_actor(x[:, t], hidden, reset_before=reset[:, t])
        actors.append(actor)
    step = torch.stack(actors, 1)
    sequence, _, h = model(x, reset_before=reset)
    torch.testing.assert_close(step, sequence, atol=2e-6, rtol=2e-5)
    torch.testing.assert_close(hidden, h, atol=2e-6, rtol=2e-5)
    with pytest.raises(ValueError):
        model.set_sigma_scale(0.1)
