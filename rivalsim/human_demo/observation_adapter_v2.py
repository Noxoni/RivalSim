"""Read-only masked observation repair for 120 Hz human demonstrations.

The adapter is deliberately external to :class:`Rival2ActorCritic`.  Authoritative
simulator observations take a structural bypass and therefore cannot be changed by
adapter parameters.  Human/gameplay and human/freeplay profiles retain their
committed quality masks; repaired values never promote those classifications.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from enum import StrEnum
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F

from rivalsim.human_demo.bc_observation_bridge import (
    DegradationProfile,
    FieldQuality,
    degradation_quality_mask,
    hybrid_actor_channel_kl,
)
from rivalsim.rival2_contracts import OBS_DIM, OBS_FIELD_NAMES
from rivalsim.rival2_policy import Rival2ActorCritic, Rival2PolicyConfig

OBSERVATION_ADAPTER_VERSION = "RIVAL2_HUMAN_DEMO_OBSERVATION_ADAPTER_V2"
OBSERVATION_ADAPTER_CHECKPOINT_FORMAT = "RIVAL2_HUMAN_DEMO_OBSERVATION_ADAPTER_CHECKPOINT_V2"


class AdapterProfile(StrEnum):
    FULL = "full_authoritative"
    GAMEPLAY = "gameplay"
    FREEPLAY = "freeplay"


@dataclass(frozen=True, slots=True)
class ObservationAdapterConfig:
    observation_dim: int = OBS_DIM
    hidden_dim: int = 256
    hidden_layers: int = 2
    activation: str = "silu"
    profile_features: int = 2
    approximate_residual_limit: float = 1.0
    dtype: str = "float32"
    initialization: str = "orthogonal_sqrt2_zero_output"

    @property
    def content_hash(self) -> str:
        payload = json.dumps(asdict(self), sort_keys=True, separators=(",", ":")).encode("ascii")
        return hashlib.sha256(payload).hexdigest().upper()


def _indices(predicate: Any) -> tuple[int, ...]:
    return tuple(index for index, field in enumerate(OBS_FIELD_NAMES) if predicate(field))


FREEPLAY_NUISANCE_INDICES = _indices(
    lambda field: field.startswith("opponent.") or field.startswith("relative.opponent")
)
BOOST_PAD_INDICES = _indices(lambda field: field.startswith("boost_pad."))


def _bounded_zero_one_indices() -> tuple[int, ...]:
    rows = []
    for index, field in enumerate(OBS_FIELD_NAMES):
        if (
            field.startswith("boost_pad.")
            or field.endswith(
                (
                    ".boost",
                    ".on_ground",
                    ".has_jumped",
                    ".is_jumping",
                    ".has_double_jumped",
                    ".has_flipped",
                    ".is_flipping",
                    ".jump_available",
                    ".dodge_available",
                    ".is_demoed",
                    ".demo_timer_remaining",
                    ".jump_time",
                    ".air_time",
                    ".air_time_since_jump",
                    ".flip_time",
                    ".boosting_time",
                    ".time_since_boosted",
                    ".is_supersonic",
                    ".supersonic_time",
                    ".sticky_ticks",
                )
            )
            or field.startswith("lifecycle.")
            or (
                field.startswith("previous_action.")
                and field.rsplit(".", 1)[-1]
                in {
                    "jump",
                    "boost",
                    "handbrake",
                }
            )
        ):
            rows.append(index)
    return tuple(rows)


ZERO_ONE_INDICES = _bounded_zero_one_indices()


class HumanDemoObservationAdapterV2(nn.Module):
    """Small residual/imputation MLP with immutable quality semantics."""

    def __init__(self, config: ObservationAdapterConfig | None = None):
        super().__init__()
        self.config = config or ObservationAdapterConfig()
        if self.config.observation_dim != OBS_DIM:
            raise ValueError("adapter must preserve the exact 182-field observation order")
        if self.config.activation != "silu":
            raise ValueError("observation adapter V2 freezes SiLU activation")
        input_dim = OBS_DIM * 2 + self.config.profile_features
        layers: list[nn.Module] = []
        for layer_index in range(self.config.hidden_layers):
            layers.extend(
                (
                    nn.Linear(
                        input_dim if layer_index == 0 else self.config.hidden_dim,
                        self.config.hidden_dim,
                    ),
                    nn.SiLU(),
                )
            )
        self.trunk = nn.Sequential(*layers)
        self.output = nn.Linear(self.config.hidden_dim, OBS_DIM)
        self._initialize()

    def _initialize(self) -> None:
        for module in self.trunk:
            if isinstance(module, nn.Linear):
                nn.init.orthogonal_(module.weight, gain=2.0**0.5)
                nn.init.zeros_(module.bias)
        nn.init.zeros_(self.output.weight)
        nn.init.zeros_(self.output.bias)

    @property
    def parameter_count(self) -> int:
        return sum(parameter.numel() for parameter in self.parameters())

    @staticmethod
    def _profile_features(observation: torch.Tensor, profile: AdapterProfile) -> torch.Tensor:
        values = observation.new_zeros((*observation.shape[:-1], 2))
        values[..., 0 if profile is AdapterProfile.GAMEPLAY else 1] = 1.0
        return values

    def forward(
        self,
        observation: torch.Tensor,
        quality: torch.Tensor | None,
        *,
        profile: AdapterProfile | str,
    ) -> torch.Tensor:
        selected = AdapterProfile(profile)
        if observation.shape[-1] != OBS_DIM:
            raise ValueError("observation must end in the 182-field Rival contract")
        if selected is AdapterProfile.FULL:
            # This branch is intentionally parameter-independent.  Returning the input
            # directly gives exact actor/value parity for authoritative simulator use.
            return observation
        if quality is None:
            raise ValueError("degraded profiles require a per-field quality mask")
        if quality.shape[-1] != OBS_DIM:
            raise ValueError("quality must end in 182 fields")
        if quality.device != observation.device:
            raise ValueError("quality and observation must share a device")
        try:
            broadcast_quality = torch.broadcast_to(quality, observation.shape)
        except RuntimeError as exc:
            raise ValueError("quality is not broadcastable to the observation") from exc
        if bool(((broadcast_quality < 0) | (broadcast_quality > 3)).any()):
            raise ValueError("quality contains an unknown classification")
        validate_quality_not_promoted(broadcast_quality, selected)

        normalized_quality = broadcast_quality.to(observation.dtype) / 3.0
        features = torch.cat(
            (
                observation,
                normalized_quality,
                self._profile_features(observation, selected),
            ),
            dim=-1,
        )
        raw = self.output(self.trunk(features))
        exact = broadcast_quality >= int(FieldQuality.EXACT_DERIVED)
        approximate = broadcast_quality == int(FieldQuality.APPROXIMATE)
        unavailable = broadcast_quality == int(FieldQuality.UNAVAILABLE)

        candidate = raw.tanh()
        zero_one = torch.zeros(OBS_DIM, dtype=torch.bool, device=observation.device)
        zero_one[list(ZERO_ONE_INDICES)] = True
        zero_one = torch.broadcast_to(zero_one, observation.shape)
        candidate = torch.where(zero_one, torch.sigmoid(raw), candidate)
        repaired = torch.where(
            approximate,
            observation + raw.tanh() * float(self.config.approximate_residual_limit),
            observation,
        )
        repaired = torch.where(unavailable, candidate, repaired)
        repaired = torch.where(exact, observation, repaired)

        if selected is AdapterProfile.FREEPLAY:
            nuisance = torch.zeros(OBS_DIM, dtype=torch.bool, device=observation.device)
            nuisance[list(FREEPLAY_NUISANCE_INDICES)] = True
            repaired = torch.where(
                torch.broadcast_to(nuisance, observation.shape),
                torch.zeros_like(repaired),
                repaired,
            )
        return repaired


def expected_quality(
    profile: AdapterProfile | str,
    *,
    device: torch.device | str,
) -> torch.Tensor | None:
    selected = AdapterProfile(profile)
    if selected is AdapterProfile.FULL:
        return None
    bridge_profile = (
        DegradationProfile.GAMEPLAY
        if selected is AdapterProfile.GAMEPLAY
        else DegradationProfile.FREEPLAY
    )
    quality = degradation_quality_mask(bridge_profile)
    return torch.from_numpy(quality.copy()).to(device=device)


def validate_quality_not_promoted(
    supplied: torch.Tensor,
    profile: AdapterProfile | str,
) -> None:
    expected = expected_quality(profile, device=supplied.device)
    if expected is None:
        raise ValueError("full authoritative bypass has no degraded quality mask")
    try:
        expected = torch.broadcast_to(expected, supplied.shape)
    except RuntimeError as exc:
        raise ValueError("supplied quality is not broadcastable") from exc
    if bool((supplied > expected).any()):
        raise ValueError("quality mask promotes a committed field classification")


def meaningful_reconstruction_mask(
    profile: AdapterProfile | str,
    quality: torch.Tensor,
) -> torch.Tensor:
    selected = AdapterProfile(profile)
    if selected is AdapterProfile.FULL:
        return torch.zeros_like(quality, dtype=torch.bool)
    mask = quality < int(FieldQuality.EXACT_DERIVED)
    if selected is AdapterProfile.FREEPLAY:
        nuisance = torch.zeros(OBS_DIM, dtype=torch.bool, device=quality.device)
        nuisance[list(FREEPLAY_NUISANCE_INDICES)] = True
        mask = mask & ~torch.broadcast_to(nuisance, mask.shape)
    return mask


@dataclass(frozen=True, slots=True)
class AdapterObjective:
    loss: torch.Tensor
    gameplay_actor_kl: torch.Tensor
    gameplay_reconstruction: torch.Tensor
    freeplay_reconstruction: torch.Tensor
    approximate_residual: torch.Tensor


def adapter_objective(
    adapter: HumanDemoObservationAdapterV2,
    frozen_policy: Rival2ActorCritic,
    full_observation: torch.Tensor,
    gameplay_degraded: torch.Tensor,
    gameplay_quality: torch.Tensor,
    freeplay_degraded: torch.Tensor,
    freeplay_quality: torch.Tensor,
    *,
    policy_config: Rival2PolicyConfig,
    gameplay_actor_weight: float,
    gameplay_reconstruction_weight: float,
    freeplay_reconstruction_weight: float,
    approximate_residual_weight: float,
) -> AdapterObjective:
    """Paired simulator objective; human actions are never inputs or targets."""

    if any(parameter.requires_grad for parameter in frozen_policy.parameters()):
        raise ValueError("Rival policy must be frozen before adapter optimization")
    with torch.no_grad():
        teacher_actor, _teacher_value = frozen_policy(full_observation)
    gameplay_repaired = adapter(
        gameplay_degraded,
        gameplay_quality,
        profile=AdapterProfile.GAMEPLAY,
    )
    freeplay_repaired = adapter(
        freeplay_degraded,
        freeplay_quality,
        profile=AdapterProfile.FREEPLAY,
    )
    gameplay_actor, _gameplay_value = frozen_policy(gameplay_repaired)
    actor_kl = (
        hybrid_actor_channel_kl(
            teacher_actor,
            gameplay_actor,
            policy_config=policy_config,
        )
        .sum(dim=-1)
        .mean()
    )
    gameplay_mask = meaningful_reconstruction_mask(AdapterProfile.GAMEPLAY, gameplay_quality)
    freeplay_mask = meaningful_reconstruction_mask(AdapterProfile.FREEPLAY, freeplay_quality)
    gameplay_reconstruction = F.mse_loss(
        gameplay_repaired.masked_select(gameplay_mask),
        full_observation.masked_select(gameplay_mask),
    )
    freeplay_reconstruction = F.mse_loss(
        freeplay_repaired.masked_select(freeplay_mask),
        full_observation.masked_select(freeplay_mask),
    )
    approximate = gameplay_quality == int(FieldQuality.APPROXIMATE)
    approximate_residual = F.mse_loss(
        gameplay_repaired.masked_select(approximate),
        gameplay_degraded.masked_select(approximate),
    )
    loss = (
        float(gameplay_actor_weight) * actor_kl
        + float(gameplay_reconstruction_weight) * gameplay_reconstruction
        + float(freeplay_reconstruction_weight) * freeplay_reconstruction
        + float(approximate_residual_weight) * approximate_residual
    )
    return AdapterObjective(
        loss=loss,
        gameplay_actor_kl=actor_kl.detach(),
        gameplay_reconstruction=gameplay_reconstruction.detach(),
        freeplay_reconstruction=freeplay_reconstruction.detach(),
        approximate_residual=approximate_residual.detach(),
    )


__all__ = [
    "BOOST_PAD_INDICES",
    "FREEPLAY_NUISANCE_INDICES",
    "OBSERVATION_ADAPTER_CHECKPOINT_FORMAT",
    "OBSERVATION_ADAPTER_VERSION",
    "ZERO_ONE_INDICES",
    "AdapterObjective",
    "AdapterProfile",
    "HumanDemoObservationAdapterV2",
    "ObservationAdapterConfig",
    "adapter_objective",
    "expected_quality",
    "meaningful_reconstruction_mask",
    "validate_quality_not_promoted",
]
