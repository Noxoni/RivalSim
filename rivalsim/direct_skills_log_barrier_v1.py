"""Opt-in categorical exploration regularizer; no environment reward changes.

Unlike ordinary entropy, the derivative of KL(uniform || policy) does not vanish
for an almost-deterministic softmax. It is not old-policy retention or a KL gate.
"""
from __future__ import annotations

import math
import torch

VERSION = 'RIVAL2_DIRECT_SKILLS_UNIFORM_LOG_BARRIER_V1'


def uniform_log_barrier(logits):
    """Per-state KL(U_K || softmax(logits)), differentiating actual logits."""
    return -logits.log_softmax(-1).mean(-1) - math.log(logits.shape[-1])


def sequence_loss(model, data, index, config, *, coefficient):
    if not 0 <= coefficient <= .1:
        raise ValueError('Explicit bounded exploration coefficient required')
    logits, value, _ = model(data['observations'][index], data['initial_hidden'][:, index],
                             reset_before=data['reset_before'][index])
    log_prob = logits.log_softmax(-1)
    logp = log_prob.gather(-1, data['action_indices'][index, :, None]).squeeze(-1)
    mask = data['train_mask'][index]
    log_ratio = logp[mask] - data['old_log_probability'][index][mask]
    ratio = log_ratio.exp()
    advantage = data['normalized_advantage'][index][mask]
    policy = -torch.minimum(ratio * advantage,
        ratio.clamp(1 - config.clip_range, 1 + config.clip_range) * advantage).mean()
    value_loss = .5 * (value[mask] - data['returns'][index][mask]).square().mean()
    entropy = -(log_prob.exp() * log_prob).sum(-1)[mask].mean()
    barrier = uniform_log_barrier(logits)[mask].mean()
    total = policy + config.value_loss_coefficient * value_loss - config.entropy_coefficient * entropy + coefficient * barrier
    return total, dict(policy_loss=policy.detach(), value_loss=value_loss.detach(),
        entropy=entropy.detach(), approx_kl=((ratio - 1) - log_ratio).mean().detach(),
        clip_fraction=((ratio - 1).abs() > config.clip_range).float().mean().detach(),
        exploration_barrier=barrier.detach(), exploration_coefficient=logits.new_tensor(coefficient),
        weighted_exploration_barrier=(coefficient * barrier).detach())
