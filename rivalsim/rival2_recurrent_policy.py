"""Fresh recurrent Rival policy used only by Human Sequence Seed v1."""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import asdict, dataclass

import torch
from torch import nn

from rivalsim.rival2_contracts import OBS_DIM
from rivalsim.rival2_human_sequence import (
    HUMAN_SEQUENCE_OBS_VIEW_CONTRACT_SHA256,
    HUMAN_SEQUENCE_OBS_VIEW_VERSION,
    project_human_sequence_observation,
)
from rivalsim.rival2_policy import deterministic_hybrid_action

FROZEN_STAGE1_LOG_STD = -1.0


@dataclass(frozen=True, slots=True)
class Rival2RecurrentPolicyConfig:
    obs_dim: int = OBS_DIM
    encoder_dim: int = 512
    hidden_dim: int = 512
    recurrent_layers: int = 1
    post_dim: int = 512
    actor_outputs: int = 13
    activation: str = "silu"
    log_std_min: float = -5.0
    log_std_max: float = 2.0
    dtype: str = "float32"
    initialization: str = "orthogonal"
    observation_view_version: str = HUMAN_SEQUENCE_OBS_VIEW_VERSION
    observation_view_sha256: str = HUMAN_SEQUENCE_OBS_VIEW_CONTRACT_SHA256

    def __post_init__(self) -> None:
        if self.obs_dim != OBS_DIM:
            raise ValueError("Human Sequence policy must preserve the 182-slot contract")
        if self.recurrent_layers != 1:
            raise ValueError("Human Sequence v1 freezes one recurrent layer")
        if self.activation != "silu":
            raise ValueError("Human Sequence v1 freezes SiLU")
        if self.observation_view_version != HUMAN_SEQUENCE_OBS_VIEW_VERSION:
            raise ValueError("Human Sequence observation-view version mismatch")
        if self.observation_view_sha256 != HUMAN_SEQUENCE_OBS_VIEW_CONTRACT_SHA256:
            raise ValueError("Human Sequence observation-view hash mismatch")

    @property
    def content_hash(self) -> str:
        payload = json.dumps(asdict(self), sort_keys=True, separators=(",", ":")).encode("ascii")
        return hashlib.sha256(payload).hexdigest().upper()


class Rival2RecurrentActorCritic(nn.Module):
    """Linear/GRU/post recurrent actor-critic with an enforced shared input view."""

    def __init__(self, config: Rival2RecurrentPolicyConfig | None = None):
        super().__init__()
        self.config = config or Rival2RecurrentPolicyConfig()
        self.encoder = nn.Linear(self.config.obs_dim, self.config.encoder_dim)
        self.encoder_activation = nn.SiLU()
        self.gru = nn.GRU(
            input_size=self.config.encoder_dim,
            hidden_size=self.config.hidden_dim,
            num_layers=self.config.recurrent_layers,
            batch_first=True,
        )
        self.post = nn.Linear(self.config.hidden_dim, self.config.post_dim)
        self.post_activation = nn.SiLU()
        self.actor = nn.Linear(self.config.post_dim, self.config.actor_outputs)
        self.critic = nn.Linear(self.config.post_dim, 1)
        self._initialize()

    def _initialize(self) -> None:
        nn.init.orthogonal_(self.encoder.weight, gain=math.sqrt(2.0))
        nn.init.zeros_(self.encoder.bias)
        for name, parameter in self.gru.named_parameters():
            if "weight" in name:
                for gate in parameter.chunk(3, dim=0):
                    nn.init.orthogonal_(gate, gain=1.0)
            else:
                nn.init.zeros_(parameter)
        nn.init.orthogonal_(self.post.weight, gain=math.sqrt(2.0))
        nn.init.zeros_(self.post.bias)
        nn.init.orthogonal_(self.actor.weight, gain=0.01)
        nn.init.zeros_(self.actor.bias)
        nn.init.orthogonal_(self.critic.weight, gain=0.01)
        nn.init.zeros_(self.critic.bias)
        self.freeze_log_std_value(FROZEN_STAGE1_LOG_STD)

    def freeze_log_std_value(self, value: float = FROZEN_STAGE1_LOG_STD) -> None:
        with torch.no_grad():
            self.actor.weight[5:10].zero_()
            self.actor.bias[5:10].fill_(float(value))

    def initial_hidden(
        self,
        batch_size: int,
        *,
        device: torch.device | str | None = None,
        dtype: torch.dtype | None = None,
    ) -> torch.Tensor:
        parameter = next(self.parameters())
        return torch.zeros(
            (self.config.recurrent_layers, int(batch_size), self.config.hidden_dim),
            device=parameter.device if device is None else device,
            dtype=parameter.dtype if dtype is None else dtype,
        )

    def _gru_with_resets(
        self,
        encoded: torch.Tensor,
        hidden: torch.Tensor,
        reset_before: torch.Tensor | None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if reset_before is None or not bool(torch.any(reset_before)):
            return self.gru(encoded, hidden)
        if reset_before.shape != encoded.shape[:2]:
            raise ValueError("reset_before must have shape [batch, sequence]")
        outputs: list[torch.Tensor] = []
        current = hidden
        for tick in range(encoded.shape[1]):
            reset = reset_before[:, tick].to(torch.bool)
            if bool(torch.any(reset)):
                current = current.masked_fill(reset.view(1, -1, 1), 0.0)
            output, current = self.gru(encoded[:, tick : tick + 1], current)
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
            raise ValueError("recurrent observation must have shape [B,T,182] or [B,182]")
        if hidden is None:
            hidden = self.initial_hidden(
                observation.shape[0], device=observation.device, dtype=observation.dtype
            )
        expected_hidden = (
            self.config.recurrent_layers,
            observation.shape[0],
            self.config.hidden_dim,
        )
        if hidden.shape != expected_hidden:
            raise ValueError(f"hidden-state shape mismatch: {hidden.shape} != {expected_hidden}")
        projected = project_human_sequence_observation(observation)
        assert isinstance(projected, torch.Tensor)
        encoded = self.encoder_activation(self.encoder(projected))
        recurrent, next_hidden = self._gru_with_resets(encoded, hidden, reset_before)
        post = self.post_activation(self.post(recurrent))
        actor = self.actor(post)
        value = self.critic(post).squeeze(-1)
        if step_input:
            actor = actor[:, 0]
            value = value[:, 0]
        return actor, value, next_hidden

    @property
    def parameter_count(self) -> int:
        return sum(parameter.numel() for parameter in self.parameters())


def deterministic_recurrent_action(actor_output: torch.Tensor) -> torch.Tensor:
    return deterministic_hybrid_action(actor_output)


__all__ = [
    "FROZEN_STAGE1_LOG_STD",
    "Rival2RecurrentActorCritic",
    "Rival2RecurrentPolicyConfig",
    "deterministic_recurrent_action",
]
