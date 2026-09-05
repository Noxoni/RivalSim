"""One recurrent actor over 90 joint controller choices, never a model router.

Prepared for a prospective action-parameterization experiment. Existing hybrid
checkpoints are immutable. The linear head projection is only initialization;
it is not an exact stochastic/deterministic policy migration and is not BC.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass

import torch
from torch import nn

from rivalsim.fresh_ground_30hz import policy_config
from rivalsim.rival2_independent_critic import (
    IndependentCriticActorCritic,
    IndependentCriticPolicyConfig,
)
from third_party.nexto.adapter import build_action_table

POLICY_VERSION = "RIVAL2_JOINT_CONTROL90_RECURRENT_V1"
PROJECTION_SIGMA = 0.65


@dataclass(frozen=True, slots=True)
class JointControlConfig(IndependentCriticPolicyConfig):
    actor_outputs: int = 90
    version: str = POLICY_VERSION

    def __post_init__(self):
        # Validate every inherited backbone field using its original validator.
        values = asdict(self)
        assert values.pop("version") == POLICY_VERSION
        assert values.pop("actor_outputs") == 90
        IndependentCriticPolicyConfig(**values)

    @property
    def content_hash(self):
        return hashlib.sha256(json.dumps(asdict(self), sort_keys=True).encode()).hexdigest().upper()


class JointControlActorCritic(IndependentCriticActorCritic):
    def __init__(self):
        # Construct the unchanged backbone with its valid 13-output config,
        # then replace only the two action-producing linear heads.
        super().__init__(policy_config())
        self.actor = nn.Linear(self.config.base_hidden_dim, 90)
        self.context_actor = nn.Linear(self.config.context_hidden_dim, 90)
        nn.init.orthogonal_(self.actor.weight, 0.01)
        nn.init.zeros_(self.actor.bias)
        nn.init.orthogonal_(self.context_actor.weight, 0.01)
        nn.init.zeros_(self.context_actor.bias)
        self.config = JointControlConfig(
            log_std_min=self.config.log_std_min, log_std_max=self.config.log_std_max
        )
        self.register_buffer("action_table", build_action_table("cpu"))

    @torch.no_grad()
    def initialize_from_hybrid(self, source):
        """Preserve all features/recurrent/critic tensors; project action preferences.

        For table action a, initial score = a_analog dot raw_mean/sigma^2
        - ||a_analog||^2/(2*sigma^2) + a_buttons dot button_logits. Shared
        Gaussian/Bernoulli normalizers cancel between choices. This is a fixed
        preference projection, not integration of Gaussian probability mass;
        it ignores source learned variance and does not claim policy parity.
        After initialization all 90 head rows are freely trainable.
        """
        current = self.state_dict()
        head_names = {"actor.weight", "actor.bias", "context_actor.weight", "context_actor.bias"}
        expected = {
            name
            for name in current
            if name != "action_table"
            and not name.startswith(("entities.", "entity_actor.", "entity_context."))
        }
        if set(source) != expected:
            raise ValueError("hybrid source tensor identity mismatch")
        for name, value in source.items():
            shape = list(current[name].shape)
            if name in head_names:
                shape[0] = 13
            if list(value.shape) != shape or not bool(torch.isfinite(value).all()):
                raise ValueError(f"invalid hybrid source tensor: {name}")
        for name, value in source.items():
            if name not in head_names:
                current[name].copy_(value)
        mapping = torch.zeros(90, 13, device=self.action_table.device)
        mapping[:, :5] = self.action_table[:, :5] / PROJECTION_SIGMA**2
        mapping[:, 10:] = self.action_table[:, 5:]
        bias = -self.action_table[:, :5].square().sum(-1) / (2 * PROJECTION_SIGMA**2)
        for prefix in ("actor", "context_actor"):
            layer = getattr(self, prefix)
            layer.weight.copy_(mapping @ source[prefix + ".weight"])
            layer.bias.copy_(mapping @ source[prefix + ".bias"])
        self.actor.bias.add_(bias)

    def sample(self, logits, generator):
        logp = logits.log_softmax(-1)
        index = torch.multinomial(logp.exp(), 1, generator=generator).squeeze(-1)
        return index, self.action_table[index], logp.gather(-1, index[..., None]).squeeze(-1)

    def deterministic(self, logits):
        return self.action_table[logits.argmax(-1)]


def categorical_statistics(logits, index):
    logp = logits.log_softmax(-1)
    return logp.gather(-1, index[..., None]).squeeze(-1), -(logp.exp() * logp).sum(-1)
