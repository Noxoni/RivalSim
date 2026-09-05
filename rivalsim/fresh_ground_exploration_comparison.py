"""Isolated same-parent exploration comparison; no frozen production edits."""

from __future__ import annotations

import math
from dataclasses import replace

import torch

from rivalsim.fresh_ground_30hz import SEED, FreshGroundEnv, policy_config, scenarios
from rivalsim.fresh_ground_30hz_training import FreshGroundTrainer
from rivalsim.rival2_independent_critic import IndependentCriticActorCritic

VERSION = "RIVAL2_SSL_DEVELOPMENT_EXPLORATION_V1"
ARMS = {"control": 1.0, "half_sigma": 0.5}


class ScaledNoiseActorCritic(IndependentCriticActorCritic):
    """Same learned tensors; multiply the learned effective sigma, not actions.

    Both rollout and sequence-PPO call this exact forward path. Bounds are scaled
    with sigma, preserving the existing clamp's gradient and avoiding a hidden
    floor. Means, Bernoulli logits, hidden state and values are untouched.
    The scale is checkpoint metadata, never an unrecorded deployment default.
    """

    analog_sigma_scale = 1.0

    def set_sigma_scale(self, scale):
        if scale not in ARMS.values():
            raise ValueError("sigma scale must be a prospectively defined arm")
        self.analog_sigma_scale = float(scale)
        base = policy_config()
        self.config = replace(
            base,
            log_std_min=base.log_std_min + math.log(scale),
            log_std_max=base.log_std_max + math.log(scale),
        )

    def _forward(self, *args, **kwargs):
        actor, value, hidden = super()._forward(*args, **kwargs)
        if self.analog_sigma_scale != 1.0:
            base = policy_config()
            log_std = actor[..., 5:10].clamp(base.log_std_min, base.log_std_max)
            actor = torch.cat(
                (actor[..., :5], log_std + math.log(self.analog_sigma_scale), actor[..., 10:]),
                dim=-1,
            )
        return actor, value, hidden


def make_comparison_trainer(parent, arm, collision_root, *, worlds=32768):
    if arm not in ARMS:
        raise ValueError("unknown exploration arm")
    bank = scenarios(worlds)
    env = FreshGroundEnv(
        worlds, collision_root, device="cuda:0", seed=SEED, ssl_foundation_scenarios=bank
    )
    model = ScaledNoiseActorCritic(policy_config())
    trainer = FreshGroundTrainer(env, model=model)
    # Load with unchanged source contracts, then explicitly activate the new
    # experiment distribution. Adam, model tensors and RNG remain the parent's.
    trainer.load_checkpoint(parent)
    model.set_sigma_scale(ARMS[arm])
    trainer.policy_config = model.config
    return trainer, bank
