import torch

from benchmarks.diagnose_rival2_exploration_gradients import grouped
from rivalsim.fresh_ground_30hz import policy_config
from rivalsim.fresh_ground_exploration_comparison import ScaledNoiseActorCritic
from rivalsim.rival2_policy import hybrid_entropy, hybrid_log_probability, sample_hybrid_action


def test_separate_entropy_gradient_groups_do_not_claim_direct_mean_payment():
    torch.set_num_threads(2)
    model = ScaledNoiseActorCritic(policy_config())
    with torch.no_grad():
        model.actor.bias[5:10].fill_(-0.5)
    model.set_sigma_scale(0.5)
    actor, _ = model.forward_actor(torch.randn(5, 4, 182))
    sample = sample_hybrid_action(
        actor.detach(), config=model.config, generator=torch.Generator().manual_seed(17)
    )
    logp = hybrid_log_probability(
        actor, sample.action, pre_tanh=sample.pre_tanh, config=model.config
    )
    objective = -(torch.exp(logp - sample.log_probability) * torch.randn(5, 4)).mean()
    entropy = -0.001 * hybrid_entropy(actor, model.config).mean()
    named = [(n, p) for n, p in model.named_parameters() if not n.startswith("critic.")]
    p = grouped(named, torch.autograd.grad(objective, [v for _, v in named], retain_graph=True))
    e = grouped(named, torch.autograd.grad(entropy, [v for _, v in named]))
    assert set(p) == set(e) == {"mean_heads", "log_std_heads", "button_heads", "actor_features"}
    assert all(torch.isfinite(v).all() for v in [*p.values(), *e.values()])
    assert e["mean_heads"].count_nonzero() == 0
    assert e["log_std_heads"].norm() > 0 and p["mean_heads"].norm() > 0
    assert all(v.grad is None for v in model.parameters())
