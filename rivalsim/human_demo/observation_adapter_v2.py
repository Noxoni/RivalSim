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

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from rivalsim.human_demo.bc_observation_bridge import (
    DegradationProfile,
    FieldQuality,
    degradation_quality_mask,
    hybrid_actor_channel_kl,
)
from rivalsim.kernels.boost_pad import SOCCAR_PAD_POSITIONS
from rivalsim.rival2_contracts import OBS_DIM, OBS_FIELD_NAMES, ORANGE_PAD_REMAP
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
_FIELD_INDEX = {field: index for index, field in enumerate(OBS_FIELD_NAMES)}


@dataclass(frozen=True, slots=True)
class NativePadOverlayV2:
    """Conservative per-frame pad evidence kept separate from the committed mask."""

    values: np.ndarray
    supported: np.ndarray
    support_quality: np.ndarray
    mapped_physical_indices: tuple[int, ...]
    maximum_xy_error_uu: float


def canonical_pad_index(position: Any, *, maximum_xy_error_uu: float = 4.0) -> tuple[int, float]:
    """Map a native pad position by canonical physical XY, never pointer order."""

    value = np.asarray(position, dtype=np.float64)
    if value.shape != (3,) or not np.isfinite(value).all():
        raise ValueError("native boost-pad position must be a finite xyz vector")
    delta = SOCCAR_PAD_POSITIONS[:, :2].astype(np.float64) - value[:2]
    distances = np.linalg.norm(delta, axis=1)
    order = np.argsort(distances, kind="stable")
    best = int(order[0])
    error = float(distances[best])
    if error > maximum_xy_error_uu:
        raise ValueError(
            f"native boost-pad position has no canonical XY match: {position}, error={error}"
        )
    if len(order) > 1 and distances[order[1]] <= maximum_xy_error_uu:
        raise ValueError("native boost-pad position has an ambiguous canonical XY match")
    return best, error


def native_pad_overlay(frame: dict[str, Any]) -> NativePadOverlayV2:
    """Recover only event-observed pad state; prehistory and unseen pads stay unknown.

    The recorder retains a pad after its first authoritative pickup callback. Its XY
    position identifies the canonical physical pad. Cooldown is the recorder's
    event-timed reconstruction, so both active and cooldown support are conservatively
    labelled approximate in this separate overlay rather than promoted in the committed
    BC quality mask.
    """

    humans = [car for car in frame.get("cars", ()) if car.get("flags", {}).get("is_local_human")]
    if len(humans) != 1:
        raise ValueError("native pad reconstruction requires one unique local human car")
    team = int(humans[0].get("team", -1))
    if team not in (0, 1):
        raise ValueError("native pad reconstruction requires blue or orange team identity")
    physical_to_agent = (
        tuple(range(34))
        if team == 0
        else tuple(ORANGE_PAD_REMAP.index(index) for index in range(34))
    )
    values = np.zeros(OBS_DIM, dtype=np.float32)
    supported = np.zeros(OBS_DIM, dtype=np.bool_)
    support_quality = np.zeros(OBS_DIM, dtype=np.uint8)
    mapped: list[int] = []
    maximum_error = 0.0
    seen: set[int] = set()
    for row in frame.get("boost_pads", ()):
        physical, error = canonical_pad_index(row.get("position"))
        if physical in seen:
            raise ValueError(f"duplicate native row for canonical boost pad {physical}")
        seen.add(physical)
        mapped.append(physical)
        maximum_error = max(maximum_error, error)
        agent_pad = physical_to_agent[physical]
        respawn_delay = float(row.get("respawn_delay", 0.0))
        cooldown_remaining = float(row.get("cooldown_remaining", 0.0))
        if not np.isfinite(respawn_delay) or not np.isfinite(cooldown_remaining):
            raise ValueError("native boost-pad timer is nonfinite")
        if respawn_delay <= 0.0:
            # A pad without an observed authoritative respawn delay remains unknown.
            continue
        cooldown = np.float32(np.clip(cooldown_remaining / respawn_delay, 0.0, 1.0))
        active = np.float32(cooldown <= np.float32(1e-6))
        for suffix, value in (("active", active), ("cooldown", cooldown)):
            index = _FIELD_INDEX[f"boost_pad.{agent_pad}.{suffix}"]
            values[index] = value
            supported[index] = True
            support_quality[index] = int(FieldQuality.APPROXIMATE)
    for value in (values, supported, support_quality):
        value.flags.writeable = False
    return NativePadOverlayV2(
        values=values,
        supported=supported,
        support_quality=support_quality,
        mapped_physical_indices=tuple(sorted(mapped)),
        maximum_xy_error_uu=maximum_error,
    )


def apply_native_pad_overlay(
    repaired: torch.Tensor,
    values: torch.Tensor,
    supported: torch.Tensor,
) -> torch.Tensor:
    """Apply source-supported pad values after learned imputation."""

    if repaired.shape != values.shape or repaired.shape != supported.shape:
        raise ValueError("native pad overlay tensors must match repaired observations")
    if repaired.device != values.device or repaired.device != supported.device:
        raise ValueError("native pad overlay tensors must share a device")
    return torch.where(supported.to(torch.bool), values.to(repaired.dtype), repaired)


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
    "NativePadOverlayV2",
    "ObservationAdapterConfig",
    "adapter_objective",
    "apply_native_pad_overlay",
    "canonical_pad_index",
    "expected_quality",
    "meaningful_reconstruction_mask",
    "native_pad_overlay",
    "validate_quality_not_promoted",
]
