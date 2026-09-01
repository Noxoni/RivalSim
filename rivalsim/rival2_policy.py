"""Rival 2.0 actor/critic and exact hybrid action distribution."""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import asdict, dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F

from rivalsim.rival2_contracts import ACTION_NAMES, OBS_DIM, OBS_FIELD_NAMES

LOG_TWO_PI = math.log(2.0 * math.pi)
LOG_TWO = math.log(2.0)
PREVIOUS_ACTION_OBSERVATION_FIELDS = tuple(
    f"previous_action.{action}" for action in ACTION_NAMES
)
PREVIOUS_ACTION_OBSERVATION_INDICES = tuple(
    OBS_FIELD_NAMES.index(field) for field in PREVIOUS_ACTION_OBSERVATION_FIELDS
)


@dataclass(frozen=True, slots=True)
class Rival2PolicyConfig:
    obs_dim: int = OBS_DIM
    hidden_dim: int = 512
    hidden_layers: int = 3
    activation: str = "silu"
    actor_outputs: int = 13
    log_std_min: float = -5.0
    log_std_max: float = 1.0
    dtype: str = "float32"
    autocast: bool = False
    initialization: str = "orthogonal_sqrt2_actor0.01_critic0.01"
    zero_previous_action_inputs: bool = False

    @property
    def content_hash(self) -> str:
        values = asdict(self)
        # Preserve every pre-existing checkpoint's policy hash.  The new field only
        # becomes part of a policy contract when this opt-in lineage enables it.
        if not self.zero_previous_action_inputs:
            values.pop("zero_previous_action_inputs")
        payload = json.dumps(values, sort_keys=True, separators=(",", ":")).encode("ascii")
        return hashlib.sha256(payload).hexdigest().upper()


@dataclass(frozen=True, slots=True)
class HybridDistributionOverride:
    """Campaign-scoped effective hybrid distribution parameters.

    The actor means and button decision boundary remain learned.  Analog log standard
    deviations and the positive button temperature are supplied by one externally frozen
    schedule, ensuring rollout sampling and every likelihood calculation share one exact
    distribution.
    """

    analog_log_std: float
    button_temperature: float

    def __post_init__(self) -> None:
        if not math.isfinite(self.analog_log_std):
            raise ValueError("effective analog log standard deviation must be finite")
        if not math.isfinite(self.button_temperature) or self.button_temperature <= 0.0:
            raise ValueError("button temperature must be finite and positive")


class Rival2ActorCritic(nn.Module):
    """Fixed shared 3x512 SiLU trunk with actor and critic heads."""

    def __init__(self, config: Rival2PolicyConfig | None = None):
        super().__init__()
        self.config = config or Rival2PolicyConfig()
        if self.config.activation != "silu":
            raise ValueError("Rival 2.0 v0.5 freezes SiLU activation")
        layers: list[nn.Module] = []
        input_dim = self.config.obs_dim
        for _ in range(self.config.hidden_layers):
            layers.extend((nn.Linear(input_dim, self.config.hidden_dim), nn.SiLU()))
            input_dim = self.config.hidden_dim
        self.trunk = nn.Sequential(*layers)
        self.actor = nn.Linear(self.config.hidden_dim, self.config.actor_outputs)
        self.critic = nn.Linear(self.config.hidden_dim, 1)
        self.register_buffer(
            "_previous_action_observation_indices",
            torch.tensor(PREVIOUS_ACTION_OBSERVATION_INDICES, dtype=torch.long),
            persistent=False,
        )
        self._initialize()

    def _initialize(self) -> None:
        for module in self.trunk:
            if isinstance(module, nn.Linear):
                nn.init.orthogonal_(module.weight, gain=math.sqrt(2.0))
                nn.init.zeros_(module.bias)
        nn.init.orthogonal_(self.actor.weight, gain=0.01)
        nn.init.zeros_(self.actor.bias)
        nn.init.orthogonal_(self.critic.weight, gain=0.01)
        nn.init.zeros_(self.critic.bias)

    def forward(self, observation: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        if self.config.zero_previous_action_inputs:
            observation = observation.index_fill(
                -1, self._previous_action_observation_indices, 0.0
            )
        hidden = self.trunk(observation)
        return self.actor(hidden), self.critic(hidden).squeeze(-1)

    @property
    def parameter_count(self) -> int:
        return sum(parameter.numel() for parameter in self.parameters())


@dataclass(frozen=True, slots=True)
class HybridSample:
    action: torch.Tensor
    pre_tanh: torch.Tensor
    log_probability: torch.Tensor
    entropy: torch.Tensor


def _distribution_parameters(
    actor_output: torch.Tensor,
    config: Rival2PolicyConfig,
    distribution_override: HybridDistributionOverride | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    if actor_output.shape[-1] != 13:
        raise ValueError("Rival 2.0 actor output must have 13 channels")
    mean = actor_output[..., :5]
    if distribution_override is None:
        log_std = actor_output[..., 5:10].clamp(config.log_std_min, config.log_std_max)
    else:
        log_std = torch.full_like(mean, distribution_override.analog_log_std)
    logits = actor_output[..., 10:13]
    if distribution_override is not None:
        logits = logits / distribution_override.button_temperature
    return mean, log_std, logits


def hybrid_distribution_parameters(
    actor_output: torch.Tensor,
    config: Rival2PolicyConfig | None = None,
    *,
    distribution_override: HybridDistributionOverride | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return the exact mean/log-std/logits used by the hybrid policy."""

    return _distribution_parameters(
        actor_output,
        config or Rival2PolicyConfig(),
        distribution_override,
    )


def _analog_log_probability(
    mean: torch.Tensor,
    log_std: torch.Tensor,
    pre_tanh: torch.Tensor,
) -> torch.Tensor:
    inv_std = torch.exp(-log_std)
    gaussian = -0.5 * (((pre_tanh - mean) * inv_std).square() + 2.0 * log_std + LOG_TWO_PI)
    # Equivalent to log(1 - tanh(u)^2), but stable at large |u|.
    log_jacobian = 2.0 * (LOG_TWO - pre_tanh - F.softplus(-2.0 * pre_tanh))
    return (gaussian - log_jacobian).sum(dim=-1)


def hybrid_log_probability(
    actor_output: torch.Tensor,
    action: torch.Tensor,
    *,
    config: Rival2PolicyConfig | None = None,
    pre_tanh: torch.Tensor | None = None,
    distribution_override: HybridDistributionOverride | None = None,
) -> torch.Tensor:
    config = config or Rival2PolicyConfig()
    mean, log_std, logits = _distribution_parameters(actor_output, config, distribution_override)
    if pre_tanh is None:
        analog = action[..., :5].clamp(-1.0 + 1e-6, 1.0 - 1e-6)
        pre_tanh = torch.atanh(analog)
    analog_log_probability = _analog_log_probability(mean, log_std, pre_tanh)
    buttons = action[..., 5:8]
    button_log_probability = -F.binary_cross_entropy_with_logits(
        logits, buttons, reduction="none"
    ).sum(dim=-1)
    return analog_log_probability + button_log_probability


def hybrid_entropy(
    actor_output: torch.Tensor,
    config: Rival2PolicyConfig | None = None,
    *,
    distribution_override: HybridDistributionOverride | None = None,
) -> torch.Tensor:
    """Finite exploration diagnostic used by PPO (base Gaussian + Bernoulli)."""

    config = config or Rival2PolicyConfig()
    _, log_std, logits = _distribution_parameters(actor_output, config, distribution_override)
    gaussian = (log_std + 0.5 * (1.0 + LOG_TWO_PI)).sum(dim=-1)
    probability = torch.sigmoid(logits)
    bernoulli = (
        -probability * F.logsigmoid(logits) - (1.0 - probability) * F.logsigmoid(-logits)
    ).sum(dim=-1)
    return gaussian + bernoulli


def sample_hybrid_action(
    actor_output: torch.Tensor,
    *,
    generator: torch.Generator,
    config: Rival2PolicyConfig | None = None,
    distribution_override: HybridDistributionOverride | None = None,
) -> HybridSample:
    config = config or Rival2PolicyConfig()
    mean, log_std, logits = _distribution_parameters(actor_output, config, distribution_override)
    epsilon = torch.randn(mean.shape, dtype=mean.dtype, device=mean.device, generator=generator)
    pre_tanh = mean + torch.exp(log_std) * epsilon
    analog = torch.tanh(pre_tanh)
    probability = torch.sigmoid(logits)
    buttons = (
        torch.rand(
            probability.shape,
            dtype=probability.dtype,
            device=probability.device,
            generator=generator,
        )
        < probability
    ).to(probability.dtype)
    action = torch.cat((analog, buttons), dim=-1)
    log_probability = hybrid_log_probability(
        actor_output,
        action,
        config=config,
        pre_tanh=pre_tanh,
        distribution_override=distribution_override,
    )
    return HybridSample(
        action=action,
        pre_tanh=pre_tanh,
        log_probability=log_probability,
        entropy=hybrid_entropy(
            actor_output,
            config,
            distribution_override=distribution_override,
        ),
    )


def deterministic_hybrid_action(
    actor_output: torch.Tensor,
    config: Rival2PolicyConfig | None = None,
) -> torch.Tensor:
    config = config or Rival2PolicyConfig()
    mean, _, logits = _distribution_parameters(actor_output, config)
    analog = torch.tanh(mean)
    buttons = (torch.sigmoid(logits) >= 0.5).to(actor_output.dtype)
    return torch.cat((analog, buttons), dim=-1)


__all__ = [
    "PREVIOUS_ACTION_OBSERVATION_FIELDS",
    "PREVIOUS_ACTION_OBSERVATION_INDICES",
    "HybridDistributionOverride",
    "HybridSample",
    "Rival2ActorCritic",
    "Rival2PolicyConfig",
    "deterministic_hybrid_action",
    "hybrid_distribution_parameters",
    "hybrid_entropy",
    "hybrid_log_probability",
    "sample_hybrid_action",
]
