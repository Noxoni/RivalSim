"""Independent nonlinear critic with exact initialization from Unified V5 descendants."""

from __future__ import annotations

import copy
from collections import OrderedDict
from dataclasses import dataclass

import torch
from torch import nn

from rivalsim.rival2_unified_policy import Rival2UnifiedActorCritic, Rival2UnifiedPolicyConfig

CRITIC_VERSION = "RIVAL2_INDEPENDENT_CRITIC_MLP_3X512_V1"


@dataclass(frozen=True, slots=True)
class IndependentCriticPolicyConfig(Rival2UnifiedPolicyConfig):
    critic_architecture: str = CRITIC_VERSION

    def __post_init__(self) -> None:
        Rival2UnifiedPolicyConfig.__post_init__(self)
        if self.critic_architecture != CRITIC_VERSION:
            raise ValueError("unsupported independent critic architecture")


def upgrade_state_dict(legacy: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    """Copy actor tensors verbatim; initialize separate value features from the trunk."""
    upgraded = {
        name: value.clone() for name, value in legacy.items() if not name.startswith("critic.")
    }
    for name, value in legacy.items():
        if name.startswith("trunk."):
            upgraded["critic.features." + name.removeprefix("trunk.")] = value.clone()
        elif name.startswith("critic."):
            upgraded["critic.head." + name.removeprefix("critic.")] = value.clone()
    return upgraded


class IndependentCriticActorCritic(Rival2UnifiedActorCritic):
    """Unchanged recurrent actor; separately trainable 182 -> 512x3 -> 1 critic."""

    critic_is_independent = True

    def __init__(self, config: IndependentCriticPolicyConfig | None = None):
        super().__init__(config or IndependentCriticPolicyConfig())
        self.critic = nn.Sequential(
            OrderedDict(
                [
                    ("features", copy.deepcopy(self.trunk)),
                    ("head", self.critic),
                ]
            )
        )

    def _value_from_features(
        self, observation: torch.Tensor, base_features: torch.Tensor
    ) -> torch.Tensor:
        return self.critic(observation)

    def isolated_value(self, observation: torch.Tensor) -> torch.Tensor:
        if observation.ndim not in {2, 3} or observation.shape[-1] != self.config.obs_dim:
            raise ValueError("independent critic expects [B,182] or [B,T,182]")
        return self.critic(observation.reshape(-1, self.config.obs_dim)).reshape(
            observation.shape[:-1]
        )
