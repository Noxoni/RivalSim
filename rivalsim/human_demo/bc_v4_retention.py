"""Prospective orientation-tail retention infrastructure for Human BC V4."""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F

from rivalsim.human_demo.bc_observation_bridge import hybrid_actor_channel_kl
from rivalsim.human_demo.bc_v3_retention import (
    ROWS_PER_WORLD,
    _group_metrics,
    detailed_retention_guard,
    encoded_rows_for_worlds,
    int64_sha256,
)
from rivalsim.human_demo.bc_v3_retention import (
    gather_encoded_rows as _gather_encoded_rows_v3,
)
from rivalsim.human_demo.missing_feature_distillation import actor_output_statistics
from rivalsim.rival2_contracts import ACTION_NAMES, OBS_FIELD_NAMES
from rivalsim.rival2_policy import Rival2ActorCritic, Rival2PolicyConfig

ORIENTATION_ACTION_NAMES = ("steer", "pitch", "yaw", "roll")
ORIENTATION_ACTION_INDICES = tuple(ACTION_NAMES.index(name) for name in ORIENTATION_ACTION_NAMES)
ORIENTATION_MEAN_INDICES = (1, 2, 3, 4)
ORIENTATION_LOG_STD_INDICES = (6, 7, 8, 9)

SELF_POSITION_Z = OBS_FIELD_NAMES.index("self.position.z")
SELF_LINEAR_VELOCITY_Z = OBS_FIELD_NAMES.index("self.linear_velocity.z")
SELF_FORWARD = tuple(OBS_FIELD_NAMES.index(f"self.forward.{axis}") for axis in "xyz")
SELF_UP = tuple(OBS_FIELD_NAMES.index(f"self.up.{axis}") for axis in "xyz")
SELF_UP_Z = OBS_FIELD_NAMES.index("self.up.z")
SELF_ANGULAR_VELOCITY = tuple(
    OBS_FIELD_NAMES.index(f"self.angular_velocity.{axis}") for axis in "xyz"
)
SELF_ON_GROUND = OBS_FIELD_NAMES.index("self.on_ground")
SELF_IS_FLIPPING = OBS_FIELD_NAMES.index("self.is_flipping")
SELF_FLIP_TIME = OBS_FIELD_NAMES.index("self.flip_time")
SELF_WHEEL_CONTACTS = tuple(
    OBS_FIELD_NAMES.index(f"self.wheel_contact.{wheel}")
    for wheel in ("front_left", "front_right", "back_left", "back_right")
)


def float32_tensor_sha256(value: torch.Tensor) -> str:
    """Hash a float32 tensor with dtype and shape binding in bounded host chunks."""

    tensor = value.detach()
    digest = hashlib.sha256()
    digest.update(b"float32\0")
    digest.update(json.dumps(list(tensor.shape), separators=(",", ":")).encode("ascii"))
    if tensor.ndim == 0:
        digest.update(tensor.to(torch.float32).cpu().contiguous().numpy().tobytes())
    else:
        for start in range(tensor.shape[0]):
            block = tensor[start : start + 1].to(torch.float32).cpu().contiguous().numpy()
            digest.update(block.tobytes(order="C"))
    return digest.hexdigest().upper()


def tensor_sha256(value: torch.Tensor, *, dtype_name: str) -> str:
    """Hash an aligned authority tensor with explicit logical dtype and shape."""

    tensor = value.detach().cpu().contiguous()
    digest = hashlib.sha256()
    digest.update(dtype_name.encode("ascii") + b"\0")
    digest.update(json.dumps(list(tensor.shape), separators=(",", ":")).encode("ascii"))
    digest.update(tensor.numpy().tobytes(order="C"))
    return digest.hexdigest().upper()


def validate_encoded_rows(encoded: torch.Tensor, *, world_count: int) -> None:
    """Validate row identities before any tensor indexing.

    The V3 helper intentionally accepted tensor-like inputs and cast them to int64.
    V4 row identities are frozen authority, so accepting floats, negative indices, or
    foreign-corpus world IDs would be a fail-open provenance bug.
    """

    if not isinstance(encoded, torch.Tensor):
        raise TypeError("encoded rows must be a torch.Tensor")
    if encoded.dtype != torch.int64 or encoded.ndim != 1:
        raise ValueError("encoded rows must be a one-dimensional int64 tensor")
    if world_count <= 0:
        raise ValueError("world_count must be positive")
    if encoded.numel() == 0:
        return
    host = encoded.detach().cpu()
    if int(host.min().item()) < 0:
        raise ValueError("encoded rows cannot be negative")
    if int(host.max().item()) >= world_count * ROWS_PER_WORLD:
        raise ValueError("encoded row belongs outside the bound simulator corpus")


def gather_encoded_rows(observations: torch.Tensor, encoded: torch.Tensor) -> torch.Tensor:
    """Fail-closed V4 wrapper around the established observation gather helper."""

    if observations.ndim != 4:
        raise ValueError("observations must have shape [128, worlds, 2, obs]")
    validate_encoded_rows(encoded, world_count=int(observations.shape[1]))
    return _gather_encoded_rows_v3(observations, encoded)


def gather_aligned_rows(value: torch.Tensor, encoded: torch.Tensor) -> torch.Tensor:
    """Gather `[tick, world, car]` authority metadata using encoded row IDs."""

    if value.ndim != 3 or value.shape[0] != 128 or value.shape[2] != 2:
        raise ValueError("aligned metadata must have shape [128, worlds, 2]")
    validate_encoded_rows(encoded, world_count=int(value.shape[1]))
    selected = encoded.to(device=value.device)
    world = torch.div(selected, ROWS_PER_WORLD, rounding_mode="floor")
    remainder = selected.remainder(ROWS_PER_WORLD)
    tick = torch.div(remainder, 2, rounding_mode="floor")
    car = remainder.remainder(2)
    return value[tick, world, car]


def aligned_role_masks(
    encoded: torch.Tensor,
    opponent_family: torch.Tensor,
    train_mask: torch.Tensor,
) -> dict[str, torch.Tensor]:
    """Classify each tick/perspective from rollout-aligned curriculum metadata."""

    family = gather_aligned_rows(opponent_family, encoded).detach().cpu().to(torch.int64)
    applicable = gather_aligned_rows(train_mask, encoded).detach().cpu().to(torch.bool)
    return {
        "all_perspectives": torch.ones_like(applicable),
        "current_policy_applicable": applicable,
        "counterfactual_opponent": ~applicable,
        "historical_opponent": (family == 1) & ~applicable,
    }


def _orientation_components(
    teacher_actor: torch.Tensor,
    observation: torch.Tensor,
    *,
    policy_config: Rival2PolicyConfig,
) -> dict[str, torch.Tensor]:
    """Return broad teacher/state-only orientation-sensitivity signals."""

    mean = teacher_actor[:, ORIENTATION_MEAN_INDICES]
    log_std = teacher_actor[:, ORIENTATION_LOG_STD_INDICES].clamp(
        policy_config.log_std_min, policy_config.log_std_max
    )
    precision = torch.exp(-2.0 * log_std).max(dim=-1).values
    confidence_normalized_demand = (mean.abs() * torch.exp(-log_std)).max(dim=-1).values
    mean_magnitude = torch.tanh(mean).abs().max(dim=-1).values
    forward = F.normalize(observation[:, SELF_FORWARD], dim=-1, eps=1e-8)
    up = F.normalize(observation[:, SELF_UP], dim=-1, eps=1e-8)
    right = F.normalize(torch.linalg.cross(up, forward, dim=-1), dim=-1, eps=1e-8)
    angular = observation[:, SELF_ANGULAR_VELOCITY]
    angular_speed = torch.stack(
        (
            (angular * forward).sum(dim=-1).abs(),
            (angular * up).sum(dim=-1).abs(),
            (angular * right).sum(dim=-1).abs(),
        ),
        dim=-1,
    ).max(dim=-1).values
    tilt = torch.acos(observation[:, SELF_UP_Z].clamp(-1.0, 1.0)) / math.pi
    return {
        "teacher_orientation_precision": precision,
        "teacher_confidence_normalized_demand": confidence_normalized_demand,
        "teacher_orientation_mean_magnitude": mean_magnitude,
        "state_body_angular_rate": angular_speed,
        "state_tilt_fraction": tilt,
    }


def _quantiles(value: torch.Tensor, levels: tuple[float, ...]) -> dict[str, float]:
    host = value.detach().cpu().to(torch.float64)
    return {
        str(level): float(torch.quantile(host, level).item()) for level in levels
    }


def orientation_score(
    teacher_actor: torch.Tensor,
    observation: torch.Tensor,
    authority: Mapping[str, Any],
    *,
    policy_config: Rival2PolicyConfig,
) -> tuple[torch.Tensor, torch.Tensor, dict[str, torch.Tensor]]:
    """Apply the frozen V4 orientation score and inclusive membership rule."""

    components = _orientation_components(
        teacher_actor, observation, policy_config=policy_config
    )
    ratios = torch.stack(
        tuple(
            components[name] / max(float(authority["core_thresholds"][name]), 1e-12)
            for name in authority["core_feature_order"]
        ),
        dim=-1,
    )
    score = ratios.max(dim=-1).values
    core = score >= 1.0
    wheel_count = (observation[:, SELF_WHEEL_CONTACTS] >= 0.5).sum(dim=-1)
    airborne = (observation[:, SELF_ON_GROUND] < 0.5) & (wheel_count == 0)
    contact = wheel_count > 0
    recent_flip = (observation[:, SELF_IS_FLIPPING] >= 0.5) | (
        observation[:, SELF_FLIP_TIME] > 0.0
    )
    context = authority["context_thresholds"]
    recovery = airborne & (
        (
            (
                observation[:, SELF_POSITION_Z]
                <= float(context["airborne_position_z_q35"])
            )
            & (
                observation[:, SELF_LINEAR_VELOCITY_Z]
                <= float(context["airborne_linear_velocity_z_q35"])
            )
        )
        | recent_flip
    ) & (
        (
            components["state_tilt_fraction"]
            >= float(context["airborne_tilt_q65"])
        )
        | (
            components["state_body_angular_rate"]
            >= float(context["airborne_body_angular_rate_q65"])
        )
    )
    wall_or_ceiling_contact = contact & (
        (
            observation[:, SELF_UP_Z].abs()
            <= float(context["contact_absolute_up_z_q20"])
        )
        | (
            observation[:, SELF_UP_Z]
            <= float(context["contact_up_z_q05"])
        )
    )
    membership = core | recovery | wall_or_ceiling_contact
    components = {
        **components,
        "reason_core": core,
        "reason_recovery_landing": recovery,
        "reason_wall_or_ceiling_contact": wall_or_ceiling_contact,
    }
    return score, membership, components


@dataclass(frozen=True, slots=True)
class V4RetentionPools:
    natural: torch.Tensor
    current_policy_applicable: torch.Tensor
    historical_opponent: torch.Tensor
    low_teacher_variance: torch.Tensor
    orientation_sensitive: torch.Tensor
    mining_candidate_pool: torch.Tensor
    initial_hard_tail_replay: torch.Tensor
    low_variance_threshold_log_std: float
    orientation_authority: dict[str, Any]
    manifest: dict[str, Any]

    def static_by_name(self) -> dict[str, torch.Tensor]:
        return {
            "natural": self.natural,
            "current_policy_applicable": self.current_policy_applicable,
            "historical_opponent": self.historical_opponent,
            "low_teacher_variance": self.low_teacher_variance,
            "orientation_sensitive": self.orientation_sensitive,
        }


def _splitmix64(value: np.ndarray) -> np.ndarray:
    """Version-independent SplitMix64 rank function over uint64 values."""

    item = np.asarray(value, dtype=np.uint64).copy()
    with np.errstate(over="ignore"):
        item = item + np.uint64(0x9E3779B97F4A7C15)
        item = (item ^ (item >> np.uint64(30))) * np.uint64(0xBF58476D1CE4E5B9)
        item = (item ^ (item >> np.uint64(27))) * np.uint64(0x94D049BB133111EB)
        item = item ^ (item >> np.uint64(31))
    return item


def _sample_without_replacement(
    source: np.ndarray,
    count: int,
    *,
    seed: int,
) -> np.ndarray:
    """Select lowest SplitMix64 ranks with encoded-row tie breaking."""

    source = np.asarray(source, dtype=np.int64)
    if source.ndim != 1 or np.any(source < 0):
        raise ValueError("candidate source must be a nonnegative int64 vector")
    source = np.unique(source)
    if count > source.size:
        raise ValueError("candidate-pool quota exceeds source pool")
    rank = _splitmix64(source.astype(np.uint64) ^ np.uint64(seed))
    order = np.lexsort((source, rank))
    return source[order[:count]]


def _build_candidate_pool(
    *,
    natural: torch.Tensor,
    current: torch.Tensor,
    historical: torch.Tensor,
    orientation: torch.Tensor,
    total_rows: int,
    fractions: Mapping[str, float],
    seed: int,
) -> tuple[torch.Tensor, dict[str, Any]]:
    if not math.isclose(sum(float(value) for value in fractions.values()), 1.0):
        raise ValueError("candidate-pool fractions must sum to one")
    quotas = {
        "orientation_sensitive": int(total_rows * float(fractions["orientation_sensitive"])),
        "current_policy_applicable": int(
            total_rows * float(fractions["current_policy_applicable"])
        ),
        "historical_opponent": int(
            total_rows * float(fractions["historical_opponent"])
        ),
    }
    quotas["natural"] = total_rows - sum(quotas.values())
    orientation_np = orientation.numpy()
    current_np = current.numpy()
    historical_np = historical.numpy()
    natural_np = natural.numpy()
    selected_orientation = _sample_without_replacement(
        orientation_np, quotas["orientation_sensitive"], seed=seed ^ 0x01
    )
    current_remaining = np.setdiff1d(current_np, selected_orientation, assume_unique=False)
    selected_current = _sample_without_replacement(
        current_remaining,
        quotas["current_policy_applicable"],
        seed=seed ^ 0x02,
    )
    selected_historical = _sample_without_replacement(
        historical_np, quotas["historical_opponent"], seed=seed ^ 0x03
    )
    already = np.concatenate(
        (selected_orientation, selected_current, selected_historical)
    )
    natural_remaining = np.setdiff1d(natural_np, already, assume_unique=False)
    selected_natural = _sample_without_replacement(
        natural_remaining, quotas["natural"], seed=seed ^ 0x04
    )
    candidate = torch.from_numpy(
        np.concatenate(
            (
                selected_orientation,
                selected_current,
                selected_historical,
                selected_natural,
            )
        ).astype(np.int64, copy=False)
    )
    if candidate.numel() != total_rows or torch.unique(candidate).numel() != total_rows:
        raise RuntimeError("mining candidate pool is not unique and exact-sized")
    return candidate, {
        "rows": total_rows,
        "ordered_int64_sha256": int64_sha256(candidate),
        "unique_rows": int(torch.unique(candidate).numel()),
        "seed": seed,
        "selection_algorithm": (
            "SplitMix64(encoded_row XOR (seed XOR segment_tag)); lowest rank then "
            "encoded row ascending"
        ),
        "segment_order": [
            "orientation_sensitive",
            "current_policy_applicable",
            "historical_opponent",
            "natural",
        ],
        "segment_rows": quotas,
        "fractions": {name: float(value) for name, value in fractions.items()},
        "source": "frozen simulator training rows only",
    }


@torch.no_grad()
def build_v4_retention_pools(
    teacher: Rival2ActorCritic,
    observations: torch.Tensor,
    train_worlds: np.ndarray,
    opponent_family: torch.Tensor,
    train_mask: torch.Tensor,
    *,
    low_variance_quantile: float,
    orientation_core_quantile: float,
    recovery_position_quantile: float,
    recovery_velocity_quantile: float,
    recovery_dynamics_quantile: float,
    contact_absolute_up_quantile: float,
    contact_up_quantile: float,
    candidate_pool_rows: int,
    candidate_pool_fractions: Mapping[str, float],
    candidate_pool_seed: int,
    initial_replay_rows: int,
    policy_config: Rival2PolicyConfig,
    rows_per_batch: int = 65_536,
) -> V4RetentionPools:
    """Build all static V4 strata using only BC-V1 teacher and training states."""

    natural = encoded_rows_for_worlds(train_worlds)
    roles = aligned_role_masks(natural, opponent_family, train_mask)
    teacher_device = next(teacher.parameters()).device
    minimum_log_std: list[torch.Tensor] = []
    raw: dict[str, list[torch.Tensor]] = {}
    context_raw: dict[str, list[torch.Tensor]] = {}
    for start in range(0, natural.numel(), rows_per_batch):
        encoded = natural[start : start + rows_per_batch]
        observation = gather_encoded_rows(observations, encoded)
        if observation.device != teacher_device:
            observation = observation.to(teacher_device)
        actor, _value = teacher(observation)
        log_std = actor[:, 5:10].clamp(
            policy_config.log_std_min, policy_config.log_std_max
        )
        minimum_log_std.append(log_std.min(dim=-1).values.detach().cpu())
        components = _orientation_components(
            actor, observation, policy_config=policy_config
        )
        for name, value in components.items():
            raw.setdefault(name, []).append(value.detach().cpu().to(torch.float32))
        for name, value in {
            "position_z": observation[:, SELF_POSITION_Z],
            "linear_velocity_z": observation[:, SELF_LINEAR_VELOCITY_Z],
            "up_z": observation[:, SELF_UP_Z],
            "on_ground": observation[:, SELF_ON_GROUND],
            "wheel_count": (observation[:, SELF_WHEEL_CONTACTS] >= 0.5).sum(dim=-1),
            "is_flipping": observation[:, SELF_IS_FLIPPING],
            "flip_time": observation[:, SELF_FLIP_TIME],
        }.items():
            context_raw.setdefault(name, []).append(value.detach().cpu().to(torch.float32))
    teacher_minimum = torch.cat(minimum_log_std)
    raw_components = {name: torch.cat(parts) for name, parts in raw.items()}
    state = {name: torch.cat(parts) for name, parts in context_raw.items()}
    low_threshold = float(
        torch.quantile(teacher_minimum.to(torch.float64), low_variance_quantile).item()
    )
    low_mask = teacher_minimum.to(torch.float64) <= low_threshold
    current_mask = roles["current_policy_applicable"]
    core_order = tuple(raw_components)
    core_thresholds = {
        name: float(
            torch.quantile(
                value[current_mask].to(torch.float64), orientation_core_quantile
            ).item()
        )
        for name, value in raw_components.items()
    }
    score = torch.stack(
        tuple(
            raw_components[name] / max(core_thresholds[name], 1e-12)
            for name in core_order
        ),
        dim=-1,
    ).max(dim=-1).values
    core_reason = score >= 1.0
    airborne_mask = current_mask & (state["on_ground"] < 0.5) & (
        state["wheel_count"] == 0
    )
    contact_mask = current_mask & (state["wheel_count"] > 0)
    if not bool(airborne_mask.any()) or not bool(contact_mask.any()):
        raise RuntimeError("orientation context calibration pool is empty")
    context_thresholds = {
        "airborne_position_z_q35": float(
            torch.quantile(
                state["position_z"][airborne_mask].to(torch.float64),
                recovery_position_quantile,
            ).item()
        ),
        "airborne_linear_velocity_z_q35": float(
            torch.quantile(
                state["linear_velocity_z"][airborne_mask].to(torch.float64),
                recovery_velocity_quantile,
            ).item()
        ),
        "airborne_tilt_q65": float(
            torch.quantile(
                raw_components["state_tilt_fraction"][airborne_mask].to(torch.float64),
                recovery_dynamics_quantile,
            ).item()
        ),
        "airborne_body_angular_rate_q65": float(
            torch.quantile(
                raw_components["state_body_angular_rate"][airborne_mask].to(
                    torch.float64
                ),
                recovery_dynamics_quantile,
            ).item()
        ),
        "contact_absolute_up_z_q20": float(
            torch.quantile(
                state["up_z"][contact_mask].abs().to(torch.float64),
                contact_absolute_up_quantile,
            ).item()
        ),
        "contact_up_z_q05": float(
            torch.quantile(
                state["up_z"][contact_mask].to(torch.float64), contact_up_quantile
            ).item()
        ),
    }
    recent_flip = (state["is_flipping"] >= 0.5) | (state["flip_time"] > 0.0)
    recovery_reason = airborne_mask & (
        (
            (
                state["position_z"]
                <= context_thresholds["airborne_position_z_q35"]
            )
            & (
                state["linear_velocity_z"]
                <= context_thresholds["airborne_linear_velocity_z_q35"]
            )
        )
        | recent_flip
    ) & (
        (
            raw_components["state_tilt_fraction"]
            >= context_thresholds["airborne_tilt_q65"]
        )
        | (
            raw_components["state_body_angular_rate"]
            >= context_thresholds["airborne_body_angular_rate_q65"]
        )
    )
    wall_reason = contact_mask & (
        (
            state["up_z"].abs()
            <= context_thresholds["contact_absolute_up_z_q20"]
        )
        | (state["up_z"] <= context_thresholds["contact_up_z_q05"])
    )
    orientation_mask = current_mask & (core_reason | recovery_reason | wall_reason)
    orientation = natural[orientation_mask]
    candidate, candidate_manifest = _build_candidate_pool(
        natural=natural,
        current=natural[current_mask],
        historical=natural[roles["historical_opponent"]],
        orientation=orientation,
        total_rows=candidate_pool_rows,
        fractions=candidate_pool_fractions,
        seed=candidate_pool_seed,
    )
    if initial_replay_rows > candidate_manifest["segment_rows"]["orientation_sensitive"]:
        raise ValueError("initial replay must fit inside candidate orientation segment")
    orientation_segment_rows = candidate_manifest["segment_rows"][
        "orientation_sensitive"
    ]
    orientation_segment = candidate[:orientation_segment_rows]
    natural_np = natural.numpy()
    natural_sort = np.argsort(natural_np, kind="stable")
    natural_sorted = natural_np[natural_sort]
    orientation_positions = natural_sort[
        np.searchsorted(natural_sorted, orientation_segment.numpy())
    ]
    orientation_scores = score[torch.from_numpy(orientation_positions)].numpy()
    initial_order = np.lexsort((orientation_segment.numpy(), -orientation_scores))
    initial_replay = orientation_segment.index_select(
        0, torch.from_numpy(initial_order[:initial_replay_rows].astype(np.int64))
    ).clone()
    orientation_authority = {
        "format": "RIVAL2_HUMAN_BC_V4_ORIENTATION_STRATUM_V1",
        "membership_source": "frozen BC-V1 teacher plus simulator training state only",
        "current_policy_applicable_required": True,
        "core_feature_order": list(core_order),
        "core_feature_quantile": orientation_core_quantile,
        "core_thresholds": core_thresholds,
        "context_quantiles": {
            "recovery_position": recovery_position_quantile,
            "recovery_velocity": recovery_velocity_quantile,
            "recovery_dynamics": recovery_dynamics_quantile,
            "contact_absolute_up_z": contact_absolute_up_quantile,
            "contact_up_z": contact_up_quantile,
        },
        "context_thresholds": context_thresholds,
        "inclusive_rule": (
            "current_policy_applicable AND (max(feature / training_q90) >= 1 "
            "OR recovery_landing_context OR wall_ceiling_contact_context)"
        ),
        "component_training_quantiles": {
            name: _quantiles(
                value[current_mask], (0.5, 0.75, 0.85, 0.9, 0.95, 0.99, 0.999)
            )
            for name, value in raw_components.items()
        },
        "score_training_quantiles": _quantiles(
            score[current_mask], (0.5, 0.75, 0.85, 0.9, 0.95, 0.99, 0.999)
        ),
        "reason_counts": {
            "core": int((current_mask & core_reason).sum().item()),
            "recovery_landing": int(recovery_reason.sum().item()),
            "wall_or_ceiling_contact": int(wall_reason.sum().item()),
            "union": int(orientation_mask.sum().item()),
        },
        "rows": int(orientation.numel()),
        "ordered_int64_sha256": int64_sha256(orientation),
        "student_outputs_used": False,
        "opened_test_rows_used": False,
    }
    pools = {
        "natural": natural,
        "current_policy_applicable": natural[current_mask],
        "historical_opponent": natural[roles["historical_opponent"]],
        "low_teacher_variance": natural[low_mask],
        "orientation_sensitive": orientation,
    }
    manifest = {
        "format": "RIVAL2_HUMAN_BC_V4_RETENTION_STRATA_V1",
        "row_identity": "world_index * 256 + tick * 2 + car_perspective",
        "low_variance_definition": {
            "teacher": "frozen Human BC V1 actor",
            "statistic": "minimum clamped teacher analog log_std across five channels",
            "quantile": low_variance_quantile,
            "threshold_log_std": low_threshold,
            "threshold_std": math.exp(low_threshold),
        },
        "orientation_sensitive_definition": orientation_authority,
        "pools": {
            name: {
                "rows": int(rows.numel()),
                "unique_rows": int(torch.unique(rows).numel()),
                "ordered_int64_sha256": int64_sha256(rows),
            }
            for name, rows in pools.items()
        },
        "mining_candidate_pool": candidate_manifest,
        "initial_hard_tail_replay": {
            "rows": int(initial_replay.numel()),
            "ordered_int64_sha256": int64_sha256(initial_replay),
            "source": (
                "highest frozen teacher/state orientation scores inside the deterministic "
                "orientation candidate segment; score descending then encoded row ascending"
            ),
        },
        "membership_source": "teacher_and_training_state_only_no_student",
    }
    return V4RetentionPools(
        natural=natural,
        current_policy_applicable=pools["current_policy_applicable"],
        historical_opponent=pools["historical_opponent"],
        low_teacher_variance=pools["low_teacher_variance"],
        orientation_sensitive=orientation,
        mining_candidate_pool=candidate,
        initial_hard_tail_replay=initial_replay,
        low_variance_threshold_log_std=low_threshold,
        orientation_authority=orientation_authority,
        manifest=manifest,
    )


def verify_v4_retention_pools(
    pools: V4RetentionPools, expected: Mapping[str, Any]
) -> None:
    checks: dict[str, bool] = {
        "low_variance_threshold": pools.low_variance_threshold_log_std
        == float(expected["low_variance_definition"]["threshold_log_std"]),
        "orientation_authority": pools.orientation_authority
        == expected["orientation_sensitive_definition"],
        "candidate_pool_hash": int64_sha256(pools.mining_candidate_pool)
        == expected["mining_candidate_pool"]["ordered_int64_sha256"],
        "initial_replay_hash": int64_sha256(pools.initial_hard_tail_replay)
        == expected["initial_hard_tail_replay"]["ordered_int64_sha256"],
    }
    for name, rows in pools.static_by_name().items():
        checks[f"{name}.rows"] = int(rows.numel()) == int(
            expected["pools"][name]["rows"]
        )
        checks[f"{name}.hash"] = int64_sha256(rows) == expected["pools"][name][
            "ordered_int64_sha256"
        ]
    checks["complete_manifest"] = pools.manifest == {
        name: expected[name] for name in pools.manifest
    }
    if not all(checks.values()):
        raise RuntimeError(f"V4 retention authority changed: {checks}")


def sample_v4_retention_rows(
    pools: V4RetentionPools,
    hard_tail_replay: HardTailReplayState | torch.Tensor,
    counts: Mapping[str, int],
    *,
    generator: torch.Generator,
) -> tuple[torch.Tensor, dict[str, int]]:
    replay_rows = (
        hard_tail_replay.rows
        if isinstance(hard_tail_replay, HardTailReplayState)
        else hard_tail_replay
    )
    available = {
        **pools.static_by_name(),
        "hard_tail_replay": replay_rows,
    }
    if set(counts) != set(available):
        raise ValueError("V4 retention mixture names changed")
    ordered = (
        "natural",
        "current_policy_applicable",
        "historical_opponent",
        "low_teacher_variance",
        "orientation_sensitive",
        "hard_tail_replay",
    )
    rows: list[torch.Tensor] = []
    realized: dict[str, int] = {}
    for name in ordered:
        count = int(counts[name])
        source = available[name]
        if count <= 0 or source.numel() == 0:
            raise ValueError(f"invalid V4 retention contribution: {name}")
        positions = torch.randint(source.numel(), (count,), generator=generator)
        rows.append(source.index_select(0, positions))
        realized[name] = count
    return torch.cat(rows), realized


@dataclass(frozen=True, slots=True)
class HardTailReplayState:
    """Deterministic bounded replay state saved transactionally with the student."""

    rows: torch.Tensor
    last_seen_generation: torch.Tensor
    scores: torch.Tensor
    provenance: tuple[str, ...]

    def __post_init__(self) -> None:
        count = int(self.rows.numel())
        if self.rows.dtype != torch.int64 or self.rows.ndim != 1:
            raise ValueError("replay rows must be a one-dimensional int64 tensor")
        if (
            self.last_seen_generation.dtype != torch.int64
            or self.last_seen_generation.shape != self.rows.shape
        ):
            raise ValueError("replay generations must align with rows")
        if self.scores.dtype != torch.float64 or self.scores.shape != self.rows.shape:
            raise ValueError("replay scores must be aligned float64 values")
        if len(self.provenance) != count:
            raise ValueError("replay provenance must align with rows")
        if count and torch.unique(self.rows).numel() != count:
            raise ValueError("replay rows must be unique")
        if not bool(torch.isfinite(self.scores).all()):
            raise ValueError("replay scores must be finite")

    def clone(self) -> HardTailReplayState:
        return HardTailReplayState(
            rows=self.rows.clone(),
            last_seen_generation=self.last_seen_generation.clone(),
            scores=self.scores.clone(),
            provenance=tuple(self.provenance),
        )


def initialize_hard_tail_replay(
    rows: torch.Tensor,
    scores: torch.Tensor | None = None,
    *,
    generation: int = 0,
    provenance: str = "teacher_state_bootstrap",
) -> HardTailReplayState:
    """Create the generation-zero orientation bootstrap replay authority."""

    selected = rows.detach().cpu().clone()
    if selected.dtype != torch.int64 or selected.ndim != 1:
        raise ValueError("initial replay rows must be one-dimensional int64")
    if scores is None:
        score = torch.zeros(selected.numel(), dtype=torch.float64)
    else:
        score = scores.detach().cpu().to(torch.float64).clone()
    return HardTailReplayState(
        rows=selected,
        last_seen_generation=torch.full_like(selected, int(generation)),
        scores=score,
        provenance=tuple(provenance for _ in range(selected.numel())),
    )


@dataclass(frozen=True, slots=True)
class HardTailMiningResult:
    replay_state: HardTailReplayState
    telemetry: dict[str, Any]

    @property
    def replay_rows(self) -> torch.Tensor:
        """Compatibility accessor for sampling code."""

        return self.replay_state.rows


@torch.no_grad()
def mine_training_hard_tail(
    teacher: Rival2ActorCritic,
    student: Rival2ActorCritic,
    observations: torch.Tensor,
    candidate_pool: torch.Tensor,
    previous_replay: HardTailReplayState | torch.Tensor,
    *,
    top_k: int,
    max_replay_rows: int,
    replay_lifetime_generations: int,
    policy_config: Rival2PolicyConfig,
    rows_per_batch: int,
    mining_round: int,
) -> HardTailMiningResult:
    """Mine deterministic top-K training negatives and update four-generation replay."""

    if top_k <= 0 or top_k > candidate_pool.numel():
        raise ValueError("invalid hard-tail top-K")
    if max_replay_rows < top_k or replay_lifetime_generations <= 0:
        raise ValueError("invalid bounded replay authority")
    if mining_round <= 0:
        raise ValueError("mining rounds begin at one")
    if isinstance(previous_replay, torch.Tensor):
        previous_state = initialize_hard_tail_replay(previous_replay)
    else:
        previous_state = previous_replay
    model_device = next(student.parameters()).device
    scores: list[torch.Tensor] = []
    channel_sums = torch.zeros(len(ACTION_NAMES), dtype=torch.float64)
    for start in range(0, candidate_pool.numel(), rows_per_batch):
        encoded = candidate_pool[start : start + rows_per_batch]
        observation = gather_encoded_rows(observations, encoded)
        if observation.device != model_device:
            observation = observation.to(model_device)
        teacher_actor, _ = teacher(observation)
        student_actor, _ = student(observation)
        if not bool(
            torch.isfinite(teacher_actor).all() and torch.isfinite(student_actor).all()
        ):
            raise RuntimeError("nonfinite actor output during training-only hard-tail mining")
        channel = hybrid_actor_channel_kl(
            teacher_actor, student_actor, policy_config=policy_config
        )
        channel_host = channel.detach().cpu().to(torch.float64)
        if not bool(torch.isfinite(channel_host).all()):
            raise RuntimeError("nonfinite actor KL during training-only hard-tail mining")
        scores.append(channel_host.sum(dim=-1))
        channel_sums += channel_host.sum(dim=0)
    score = torch.cat(scores)
    encoded_np = candidate_pool.numpy()
    score_np = score.numpy()
    order = np.lexsort((encoded_np, -score_np))
    selected_positions = torch.from_numpy(order[:top_k].astype(np.int64, copy=False))
    selected_rows = candidate_pool.index_select(0, selected_positions).clone()
    selected_scores = score.index_select(0, selected_positions).clone()
    minimum_generation = mining_round - replay_lifetime_generations + 1
    retained_mask = previous_state.last_seen_generation >= minimum_generation
    expired_rows = int((~retained_mask).sum().item())
    combined: dict[int, tuple[int, float, str]] = {
        int(row): (int(generation), float(value), provenance)
        for row, generation, value, provenance in zip(
            previous_state.rows[retained_mask].tolist(),
            previous_state.last_seen_generation[retained_mask].tolist(),
            previous_state.scores[retained_mask].tolist(),
            tuple(
                item
                for item, keep in zip(
                    previous_state.provenance, retained_mask.tolist(), strict=True
                )
                if keep
            ),
            strict=True,
        )
    }
    for row, value in zip(selected_rows.tolist(), selected_scores.tolist(), strict=True):
        combined[int(row)] = (mining_round, float(value), "dynamic_hard_tail")
    union_rows = np.asarray(list(combined), dtype=np.int64)
    union_generations = np.asarray(
        [combined[int(row)][0] for row in union_rows], dtype=np.int64
    )
    union_scores = np.asarray(
        [combined[int(row)][1] for row in union_rows], dtype=np.float64
    )
    union_provenance = [combined[int(row)][2] for row in union_rows]
    replay_order = np.lexsort((union_rows, -union_scores, -union_generations))
    replay_order = replay_order[:max_replay_rows]
    replay = HardTailReplayState(
        rows=torch.from_numpy(union_rows[replay_order].copy()),
        last_seen_generation=torch.from_numpy(union_generations[replay_order].copy()),
        scores=torch.from_numpy(union_scores[replay_order].copy()),
        provenance=tuple(union_provenance[index] for index in replay_order),
    )
    overlap = int(
        np.intersect1d(
            replay.rows.numpy(), previous_state.rows.numpy(), assume_unique=False
        ).size
    )
    telemetry = {
        "format": "RIVAL2_HUMAN_BC_V4_HARD_TAIL_MINING_ROUND_V1",
        "mining_round": mining_round,
        "candidate_rows": int(candidate_pool.numel()),
        "candidate_pool_sha256": int64_sha256(candidate_pool),
        "top_k": top_k,
        "top_fraction": top_k / candidate_pool.numel(),
        "replacement_policy": (
            "refresh top-K; retain current plus prior bounded generations; rank generation "
            "descending, stored KL descending, encoded row ascending"
        ),
        "replay_lifetime_generations": replay_lifetime_generations,
        "maximum_replay_rows": max_replay_rows,
        "replay_rows": int(replay.rows.numel()),
        "replay_sha256": int64_sha256(replay.rows),
        "generation_sha256": int64_sha256(replay.last_seen_generation),
        "overlap_with_previous_rows": overlap,
        "overlap_with_previous_fraction": overlap
        / max(previous_state.rows.numel(), 1),
        "provenance_counts": {
            name: replay.provenance.count(name) for name in sorted(set(replay.provenance))
        },
        "expired_rows": expired_rows,
        "selected_rows_already_present": int(
            np.intersect1d(
                selected_rows.numpy(), previous_state.rows.numpy(), assume_unique=False
            ).size
        ),
        "sample_kl_quantiles": _quantiles(
            score, (0.5, 0.9, 0.95, 0.96875, 0.99, 0.999)
        ),
        "sample_kl_max": float(score.max().item()),
        "selected_minimum_kl": float(score.index_select(0, selected_positions).min().item()),
        "mean_channel_kl": {
            name: float(channel_sums[index].item() / candidate_pool.numel())
            for index, name in enumerate(ACTION_NAMES)
        },
        "validation_or_test_rows_inspected": 0,
    }
    return HardTailMiningResult(replay_state=replay, telemetry=telemetry)


@dataclass(frozen=True, slots=True)
class V4TailRetentionLoss:
    loss: torch.Tensor
    mean_kl: torch.Tensor
    total_mean_barrier: torch.Tensor
    total_cvar_barrier: torch.Tensor
    orientation_cvar_barrier: torch.Tensor
    maximum_sample_kl: torch.Tensor
    maximum_individual_orientation_channel_kl: torch.Tensor
    per_channel_mean_kl: torch.Tensor


def _smooth_squared_barrier(
    value: torch.Tensor, *, threshold: float, temperature: float
) -> torch.Tensor:
    if temperature <= 0.0:
        raise ValueError("barrier temperature must be positive")
    excess = F.softplus((value - threshold) / temperature) * temperature
    return excess.square()


def _upper_tail_mean(value: torch.Tensor, fraction: float) -> torch.Tensor:
    if not 0.0 < fraction <= 1.0:
        raise ValueError("CVaR fraction must be in (0, 1]")
    count = max(1, math.ceil(value.numel() * fraction))
    return torch.topk(value.reshape(-1), count, sorted=False).values.mean()


def v4_tail_aware_actor_retention_loss(
    teacher_actor: torch.Tensor,
    student_actor: torch.Tensor,
    *,
    policy_config: Rival2PolicyConfig,
    mean_kl_coefficient: float,
    total_barrier_threshold: float,
    total_barrier_temperature: float,
    total_barrier_coefficient: float,
    total_cvar_fraction: float,
    total_cvar_coefficient: float,
    orientation_tail_threshold: float,
    orientation_tail_temperature: float,
    orientation_cvar_fraction: float,
    orientation_cvar_coefficient: float,
) -> V4TailRetentionLoss:
    teacher_actor = teacher_actor.detach()
    channel = hybrid_actor_channel_kl(
        teacher_actor, student_actor, policy_config=policy_config
    )
    sample = channel.sum(dim=-1)
    maximum_orientation_channel = channel[:, ORIENTATION_ACTION_INDICES].max(dim=-1).values
    total_barrier_vector = _smooth_squared_barrier(
        sample,
        threshold=total_barrier_threshold,
        temperature=total_barrier_temperature,
    )
    orientation_barrier_vector = _smooth_squared_barrier(
        maximum_orientation_channel,
        threshold=orientation_tail_threshold,
        temperature=orientation_tail_temperature,
    )
    total_mean_barrier = total_barrier_vector.mean()
    total_cvar_barrier = _upper_tail_mean(total_barrier_vector, total_cvar_fraction)
    orientation_cvar_barrier = _upper_tail_mean(
        orientation_barrier_vector, orientation_cvar_fraction
    )
    mean_kl = sample.mean()
    loss = (
        mean_kl_coefficient * mean_kl
        + total_barrier_coefficient * total_mean_barrier
        + total_cvar_coefficient * total_cvar_barrier
        + orientation_cvar_coefficient * orientation_cvar_barrier
    )
    return V4TailRetentionLoss(
        loss=loss,
        mean_kl=mean_kl,
        total_mean_barrier=total_mean_barrier,
        total_cvar_barrier=total_cvar_barrier,
        orientation_cvar_barrier=orientation_cvar_barrier,
        maximum_sample_kl=sample.max(),
        maximum_individual_orientation_channel_kl=maximum_orientation_channel.max(),
        per_channel_mean_kl=channel.mean(dim=0),
    )


def _v4_group_metrics(channel: torch.Tensor) -> dict[str, Any]:
    result = _group_metrics(channel)
    if channel.numel() == 0:
        result["orientation_tail"] = {
            "max_sum_kl": 0.0,
            "counts_above": {str(level): 0 for level in (0.125, 0.25, 0.5, 1.0)},
            "per_channel_counts_above_0.25": {
                name: 0 for name in ORIENTATION_ACTION_NAMES
            },
            "contribution_at_max_sample": {name: 0.0 for name in ACTION_NAMES},
        }
        return result
    value = channel.detach().cpu().to(torch.float64)
    sample = value.sum(dim=-1)
    orientation = value[:, ORIENTATION_ACTION_INDICES].sum(dim=-1)
    max_position = int(sample.argmax().item())
    max_row = value[max_position]
    result["orientation_tail"] = {
        "max_sum_kl": float(orientation.max().item()),
        "counts_above": {
            str(level): int((orientation > level).sum().item())
            for level in (0.125, 0.25, 0.5, 1.0)
        },
        "per_channel_counts_above_0.25": {
            name: int((value[:, index] > 0.25).sum().item())
            for name, index in zip(
                ORIENTATION_ACTION_NAMES, ORIENTATION_ACTION_INDICES, strict=True
            )
        },
        "contribution_at_max_sample": {
            name: float(max_row[index].item())
            for index, name in enumerate(ACTION_NAMES)
        },
    }
    return result


def _actor_distribution_statistics(parts: list[torch.Tensor]) -> dict[str, Any]:
    """Report finite/saturation health without changing the actor contract."""

    return actor_output_statistics(torch.cat(parts))


@torch.no_grad()
def evaluate_v4_retention(
    teacher: Rival2ActorCritic,
    student: Rival2ActorCritic,
    observations: torch.Tensor,
    worlds: np.ndarray,
    opponent_family: torch.Tensor,
    train_mask: torch.Tensor,
    *,
    low_variance_threshold_log_std: float,
    orientation_authority: Mapping[str, Any],
    policy_config: Rival2PolicyConfig,
    worlds_per_batch: int,
) -> dict[str, Any]:
    grouped: dict[str, list[torch.Tensor]] = {
        "all_perspectives": [],
        "current_policy_applicable": [],
        "counterfactual_opponent": [],
        "historical_opponent": [],
        "low_teacher_variance": [],
        "orientation_sensitive": [],
    }
    grouped_actor: dict[str, list[torch.Tensor]] = {
        name: [] for name in grouped
    }
    critic_max = 0.0
    critic_square_sum = 0.0
    critic_count = 0
    actor_finite = True
    value_finite = True
    model_device = next(student.parameters()).device
    for start in range(0, len(worlds), worlds_per_batch):
        selected_worlds = np.asarray(worlds[start : start + worlds_per_batch], dtype=np.int64)
        encoded = encoded_rows_for_worlds(selected_worlds)
        observation = gather_encoded_rows(observations, encoded)
        if observation.device != model_device:
            observation = observation.to(model_device)
        hidden = teacher.trunk(observation)
        teacher_actor = teacher.actor(hidden)
        teacher_value = teacher.critic(hidden).squeeze(-1)
        student_actor, student_value = student(observation)
        channel = hybrid_actor_channel_kl(
            teacher_actor, student_actor, policy_config=policy_config
        )
        role = aligned_role_masks(encoded, opponent_family, train_mask)
        teacher_log_std = teacher_actor[:, 5:10].clamp(
            policy_config.log_std_min, policy_config.log_std_max
        )
        role["low_teacher_variance"] = (
            teacher_log_std.min(dim=-1).values.detach().cpu().to(torch.float64)
            <= low_variance_threshold_log_std
        )
        _score, orientation_member, _components = orientation_score(
            teacher_actor,
            observation,
            orientation_authority,
            policy_config=policy_config,
        )
        role["orientation_sensitive"] = role["current_policy_applicable"] & (
            orientation_member.detach().cpu()
        )
        channel_cpu = channel.detach().cpu()
        student_actor_cpu = student_actor.detach().cpu()
        for name in grouped:
            grouped[name].append(channel_cpu[role[name]])
            grouped_actor[name].append(student_actor_cpu[role[name]])
        drift = (student_value - teacher_value).detach()
        critic_count += int(drift.numel())
        critic_square_sum += float(drift.square().sum().item())
        critic_max = max(critic_max, float(drift.abs().max().item()))
        actor_finite = actor_finite and bool(torch.isfinite(channel).all())
        value_finite = value_finite and bool(
            torch.isfinite(teacher_value).all() and torch.isfinite(student_value).all()
        )
    result = {name: _v4_group_metrics(torch.cat(rows)) for name, rows in grouped.items()}
    for name in grouped:
        result[name]["actor_distribution"] = _actor_distribution_statistics(
            grouped_actor[name]
        )
    result["critic"] = {
        "rmse": math.sqrt(critic_square_sum / critic_count),
        "max_absolute_drift": critic_max,
        "finite": value_finite,
    }
    result["actor_finite"] = actor_finite
    result["actor_distribution"] = result["all_perspectives"]["actor_distribution"]
    return result


def v4_validation_guard(
    metrics: Mapping[str, Any],
    hard_guard: Mapping[str, float | bool],
    selection_margin: Mapping[str, float],
) -> dict[str, Any]:
    contract = detailed_retention_guard(metrics, hard_guard)
    all_rows = metrics["all_perspectives"]
    max_orientation_channel = max(
        float(all_rows["max_channel_kl"][name]) for name in ORIENTATION_ACTION_NAMES
    )
    margin_checks = {
        "selection_max_sample_kl": float(all_rows["max_sample_kl"])
        <= float(selection_margin["maximum_sample_kl"]),
        "selection_max_orientation_channel_kl": max_orientation_channel
        <= float(selection_margin["maximum_individual_orientation_channel_kl"]),
    }
    return {
        "contract": contract,
        "selection_margin_checks": margin_checks,
        "contract_accepted": contract["accepted"],
        "accepted": bool(contract["accepted"] and all(margin_checks.values())),
    }


def v4_combined_validation_eligibility(
    complete_metrics: Mapping[str, Any],
    stress_metrics: Mapping[str, Any],
    hard_guard: Mapping[str, float | bool],
    selection_margin: Mapping[str, float],
) -> dict[str, Any]:
    """Require both prospectively frozen validation authorities to pass."""

    complete = v4_validation_guard(complete_metrics, hard_guard, selection_margin)
    stress = v4_validation_guard(stress_metrics, hard_guard, selection_margin)
    return {
        "complete_validation": complete,
        "stress_validation": stress,
        "accepted": bool(complete["accepted"] and stress["accepted"]),
    }


__all__ = [
    "ORIENTATION_ACTION_INDICES",
    "ORIENTATION_ACTION_NAMES",
    "HardTailMiningResult",
    "HardTailReplayState",
    "V4RetentionPools",
    "V4TailRetentionLoss",
    "aligned_role_masks",
    "build_v4_retention_pools",
    "evaluate_v4_retention",
    "float32_tensor_sha256",
    "gather_aligned_rows",
    "gather_encoded_rows",
    "initialize_hard_tail_replay",
    "mine_training_hard_tail",
    "orientation_score",
    "sample_v4_retention_rows",
    "tensor_sha256",
    "v4_combined_validation_eligibility",
    "v4_tail_aware_actor_retention_loss",
    "v4_validation_guard",
    "validate_encoded_rows",
    "verify_v4_retention_pools",
]
