"""Single-policy recurrent consolidation of Rival 2 capability teachers.

The module deliberately contains no policy router.  A copied feed-forward Rival
policy supplies an exactly parity-preserving base path while one recurrent
residual path learns temporal capability context.  The residual actor is zero
initialized, so a freshly constructed unified model is action- and value-exact
to its feed-forward parent for every observation and every hidden state.
"""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import asdict, dataclass

import torch
from torch import nn

from rivalsim.rival2_contracts import OBS_DIM
from rivalsim.rival2_policy import (
    Rival2ActorCritic,
    Rival2PolicyConfig,
    deterministic_hybrid_action,
)

UNIFIED_CAPABILITY_POLICY_VERSION = "RIVAL2_UNIFIED_CAPABILITY_POLICY_V1"


@dataclass(frozen=True, slots=True)
class Rival2UnifiedPolicyConfig:
    obs_dim: int = OBS_DIM
    base_hidden_dim: int = 512
    base_hidden_layers: int = 3
    context_encoder_dim: int = 256
    context_hidden_dim: int = 256
    context_layers: int = 1
    actor_outputs: int = 13
    activation: str = "silu"
    log_std_min: float = -5.0
    log_std_max: float = 1.0
    dtype: str = "float32"
    version: str = UNIFIED_CAPABILITY_POLICY_VERSION

    def __post_init__(self) -> None:
        if self.obs_dim != OBS_DIM:
            raise ValueError("unified policy must preserve the 182-field observation")
        if self.base_hidden_dim != 512 or self.base_hidden_layers != 3:
            raise ValueError("unified V1 must preserve the V23 3x512 base trunk")
        if self.context_layers != 1:
            raise ValueError("unified V1 freezes one recurrent context layer")
        if self.actor_outputs != 13:
            raise ValueError("unified V1 must preserve the 13-output hybrid actor")
        if self.activation != "silu":
            raise ValueError("unified V1 freezes SiLU activation")
        if self.version != UNIFIED_CAPABILITY_POLICY_VERSION:
            raise ValueError("unified policy version mismatch")

    @property
    def content_hash(self) -> str:
        payload = json.dumps(asdict(self), sort_keys=True, separators=(",", ":")).encode(
            "ascii"
        )
        return hashlib.sha256(payload).hexdigest().upper()

    @property
    def base_policy_config(self) -> Rival2PolicyConfig:
        return Rival2PolicyConfig(
            obs_dim=self.obs_dim,
            hidden_dim=self.base_hidden_dim,
            hidden_layers=self.base_hidden_layers,
            activation=self.activation,
            actor_outputs=self.actor_outputs,
            log_std_min=self.log_std_min,
            log_std_max=self.log_std_max,
            dtype=self.dtype,
            autocast=False,
            initialization="orthogonal_sqrt2_actor0.01_critic0.01",
            zero_previous_action_inputs=False,
        )

    @property
    def recurrent_layers(self) -> int:
        """Compatibility alias for the shared sequence-PPO infrastructure."""

        return self.context_layers

    @property
    def hidden_dim(self) -> int:
        """Compatibility alias for the recurrent residual state width."""

        return self.context_hidden_dim


class Rival2UnifiedActorCritic(nn.Module):
    """One actor with a parity-preserving V23 base and recurrent residual.

    ``trunk``, ``actor``, and ``critic`` intentionally retain the names used by
    :class:`Rival2ActorCritic`.  This makes the parent import auditable.  The
    output is one actor tensor; there is no expert index, route mode, task id,
    or action selection between component policies.
    """

    def __init__(self, config: Rival2UnifiedPolicyConfig | None = None):
        super().__init__()
        self.config = config or Rival2UnifiedPolicyConfig()
        base = Rival2ActorCritic(self.config.base_policy_config)
        self.trunk = base.trunk
        self.actor = base.actor
        self.critic = base.critic
        self.context_encoder = nn.Linear(
            self.config.obs_dim, self.config.context_encoder_dim
        )
        self.context_activation = nn.SiLU()
        self.context_gru = nn.GRU(
            self.config.context_encoder_dim,
            self.config.context_hidden_dim,
            num_layers=self.config.context_layers,
            batch_first=True,
        )
        self.context_actor = nn.Linear(
            self.config.context_hidden_dim, self.config.actor_outputs
        )
        self._initialize_context()

    def _initialize_context(self) -> None:
        nn.init.orthogonal_(self.context_encoder.weight, gain=math.sqrt(2.0))
        nn.init.zeros_(self.context_encoder.bias)
        for name, parameter in self.context_gru.named_parameters():
            if "weight" in name:
                for gate in parameter.chunk(3, dim=0):
                    nn.init.orthogonal_(gate, gain=1.0)
            else:
                nn.init.zeros_(parameter)
        # Exact parent parity.  This also prevents random recurrent state from
        # changing the actor before any consolidation step is accepted.
        nn.init.zeros_(self.context_actor.weight)
        nn.init.zeros_(self.context_actor.bias)

    def load_feedforward_parent(self, parent: Rival2ActorCritic) -> None:
        expected = self.config.base_policy_config
        if parent.config != expected:
            raise ValueError("feed-forward parent policy configuration mismatch")
        self.trunk.load_state_dict(parent.trunk.state_dict(), strict=True)
        self.actor.load_state_dict(parent.actor.state_dict(), strict=True)
        self.critic.load_state_dict(parent.critic.state_dict(), strict=True)

    def freeze_base(self) -> None:
        self.trunk.requires_grad_(False)
        self.actor.requires_grad_(False)
        self.critic.requires_grad_(False)
        self.context_encoder.requires_grad_(True)
        self.context_gru.requires_grad_(True)
        self.context_actor.requires_grad_(True)

    def initial_hidden(
        self,
        batch_size: int,
        *,
        device: torch.device | str | None = None,
        dtype: torch.dtype | None = None,
    ) -> torch.Tensor:
        parameter = next(self.parameters())
        return torch.zeros(
            (self.config.context_layers, int(batch_size), self.config.context_hidden_dim),
            device=parameter.device if device is None else device,
            dtype=parameter.dtype if dtype is None else dtype,
        )

    def _context_with_resets(
        self,
        encoded: torch.Tensor,
        hidden: torch.Tensor,
        reset_before: torch.Tensor | None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if reset_before is None or not bool(torch.any(reset_before)):
            return self.context_gru(encoded, hidden)
        if reset_before.shape != encoded.shape[:2]:
            raise ValueError("reset_before must have shape [batch, sequence]")
        outputs: list[torch.Tensor] = []
        current = hidden
        for tick in range(encoded.shape[1]):
            reset = reset_before[:, tick].to(torch.bool)
            if bool(torch.any(reset)):
                current = current.masked_fill(reset.view(1, -1, 1), 0.0)
            output, current = self.context_gru(encoded[:, tick : tick + 1], current)
            outputs.append(output)
        return torch.cat(outputs, dim=1), current

    def forward(
        self,
        observation: torch.Tensor,
        hidden: torch.Tensor | None = None,
        *,
        reset_before: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        step_input = observation.ndim == 2
        if step_input:
            observation = observation.unsqueeze(1)
            if reset_before is not None and reset_before.ndim == 1:
                reset_before = reset_before.unsqueeze(1)
        if observation.ndim != 3 or observation.shape[-1] != self.config.obs_dim:
            raise ValueError("unified observation must have shape [B,T,182] or [B,182]")
        if hidden is None:
            hidden = self.initial_hidden(
                observation.shape[0], device=observation.device, dtype=observation.dtype
            )
        expected_hidden = (
            self.config.context_layers,
            observation.shape[0],
            self.config.context_hidden_dim,
        )
        if hidden.shape != expected_hidden:
            raise ValueError(f"hidden-state shape mismatch: {hidden.shape} != {expected_hidden}")

        flat = observation.reshape(-1, self.config.obs_dim)
        base_features = self.trunk(flat)
        base_actor = self.actor(base_features).reshape(
            observation.shape[0], observation.shape[1], self.config.actor_outputs
        )
        value = self.critic(base_features).reshape(observation.shape[:2])
        encoded = self.context_activation(self.context_encoder(observation))
        context, next_hidden = self._context_with_resets(encoded, hidden, reset_before)
        actor = base_actor + self.context_actor(context)
        if step_input:
            actor = actor[:, 0]
            value = value[:, 0]
        return actor, value, next_hidden

    @property
    def context_parameters(self) -> tuple[nn.Parameter, ...]:
        return tuple(
            parameter
            for module in (self.context_encoder, self.context_gru, self.context_actor)
            for parameter in module.parameters()
        )

    @property
    def parameter_count(self) -> int:
        return sum(parameter.numel() for parameter in self.parameters())


def deterministic_unified_action(actor_output: torch.Tensor) -> torch.Tensor:
    return deterministic_hybrid_action(actor_output)


__all__ = [
    "UNIFIED_CAPABILITY_POLICY_VERSION",
    "Rival2UnifiedActorCritic",
    "Rival2UnifiedPolicyConfig",
    "deterministic_unified_action",
]
