"""Prospectively frozen tail-safe simulator retention helpers for Human BC V3."""

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
from rivalsim.rival2_contracts import ACTION_NAMES, OBS_DIM
from rivalsim.rival2_policy import Rival2ActorCritic, Rival2PolicyConfig

ROWS_PER_WORLD = 128 * 2
OPPONENT_CURRENT = 0
OPPONENT_HISTORICAL = 1


def int64_sha256(value: torch.Tensor | np.ndarray) -> str:
    """Hash an ordered int64 identity vector with shape and dtype binding."""

    array = np.asarray(
        value.detach().cpu().numpy() if isinstance(value, torch.Tensor) else value,
        dtype=np.int64,
    )
    digest = hashlib.sha256()
    digest.update(b"int64\0")
    digest.update(json.dumps(list(array.shape), separators=(",", ":")).encode("ascii"))
    digest.update(np.ascontiguousarray(array).tobytes(order="C"))
    return digest.hexdigest().upper()


def encoded_rows_for_worlds(worlds: np.ndarray | torch.Tensor) -> torch.Tensor:
    """Return stable world/tick/car row identities in world-major trajectory order."""

    host = torch.as_tensor(np.asarray(worlds, dtype=np.int64)).reshape(-1, 1)
    offsets = torch.arange(ROWS_PER_WORLD, dtype=torch.int64).reshape(1, -1)
    return (host * ROWS_PER_WORLD + offsets).reshape(-1)


def gather_encoded_rows(observations: torch.Tensor, encoded: torch.Tensor) -> torch.Tensor:
    """Gather `[world, tick, car]` identities from `[tick, world, car, obs]`."""

    if observations.ndim != 4 or observations.shape[0] != 128:
        raise ValueError("observations must have shape [128, worlds, 2, obs]")
    if observations.shape[2:] != (2, OBS_DIM):
        raise ValueError("observation contract shape changed")
    selected = encoded.to(device=observations.device, dtype=torch.int64)
    world = torch.div(selected, ROWS_PER_WORLD, rounding_mode="floor")
    remainder = selected.remainder(ROWS_PER_WORLD)
    tick = torch.div(remainder, 2, rounding_mode="floor")
    car = remainder.remainder(2)
    return observations[tick, world, car]


def role_masks_for_encoded(
    encoded: torch.Tensor,
    opponent_family: torch.Tensor,
    rival_side: torch.Tensor,
) -> dict[str, torch.Tensor]:
    """Classify rows using the frozen mixed-opponent current/opponent semantics."""

    host = encoded.detach().cpu().to(torch.int64)
    world = torch.div(host, ROWS_PER_WORLD, rounding_mode="floor")
    car = host.remainder(ROWS_PER_WORLD).remainder(2)
    family = opponent_family.detach().cpu().to(torch.int64).index_select(0, world)
    side = rival_side.detach().cpu().to(torch.int64).index_select(0, world)
    applicable = (family == OPPONENT_CURRENT) | (car == side)
    return {
        "all_perspectives": torch.ones_like(applicable, dtype=torch.bool),
        "current_policy_applicable": applicable,
        "counterfactual_opponent": ~applicable,
        "historical_opponent": (family == OPPONENT_HISTORICAL) & (car != side),
    }


@dataclass(frozen=True, slots=True)
class RetentionPools:
    natural: torch.Tensor
    current_policy_applicable: torch.Tensor
    historical_opponent: torch.Tensor
    low_teacher_variance: torch.Tensor
    low_variance_threshold_log_std: float
    manifest: dict[str, Any]

    def by_name(self) -> dict[str, torch.Tensor]:
        return {
            "natural": self.natural,
            "current_policy_applicable": self.current_policy_applicable,
            "historical_opponent": self.historical_opponent,
            "low_teacher_variance": self.low_teacher_variance,
        }


@torch.no_grad()
def build_retention_pools(
    teacher: Rival2ActorCritic,
    observations: torch.Tensor,
    train_worlds: np.ndarray,
    opponent_family: torch.Tensor,
    rival_side: torch.Tensor,
    *,
    low_variance_quantile: float,
    policy_config: Rival2PolicyConfig,
    rows_per_batch: int = 65_536,
) -> RetentionPools:
    """Build teacher-only V3 sampling strata and their deterministic identities."""

    if not 0.0 < low_variance_quantile < 1.0:
        raise ValueError("low-variance quantile must be in (0, 1)")
    natural = encoded_rows_for_worlds(train_worlds)
    roles = role_masks_for_encoded(natural, opponent_family, rival_side)
    minimum_log_std: list[torch.Tensor] = []
    for start in range(0, natural.numel(), rows_per_batch):
        encoded = natural[start : start + rows_per_batch]
        observation = gather_encoded_rows(observations, encoded)
        actor, _value = teacher(observation)
        log_std = actor[:, 5:10].clamp(policy_config.log_std_min, policy_config.log_std_max)
        minimum_log_std.append(log_std.min(dim=-1).values.detach().cpu())
    teacher_minimum = torch.cat(minimum_log_std)
    threshold = float(
        torch.quantile(teacher_minimum.to(torch.float64), low_variance_quantile).item()
    )
    low_mask = teacher_minimum.to(torch.float64) <= threshold
    pools = {
        "natural": natural,
        "current_policy_applicable": natural[roles["current_policy_applicable"]],
        "historical_opponent": natural[roles["historical_opponent"]],
        "low_teacher_variance": natural[low_mask],
    }
    manifest = {
        "format": "RIVAL2_HUMAN_BC_V3_RETENTION_STRATA_V1",
        "row_identity": "world_index * 256 + tick * 2 + car_perspective",
        "low_variance_definition": {
            "teacher": "frozen Human BC V1 actor on full simulator training observations",
            "statistic": "minimum clamped teacher analog log_std across five channels",
            "quantile": low_variance_quantile,
            "inclusive": True,
            "threshold_log_std": threshold,
            "threshold_std": math.exp(threshold),
            "teacher_minimum_log_std_quantiles": {
                str(q): float(torch.quantile(teacher_minimum.to(torch.float64), q).item())
                for q in (0.001, 0.01, 0.05, 0.1, 0.25, 0.5, 0.9, 0.99)
            },
        },
        "pools": {
            name: {
                "rows": int(rows.numel()),
                "ordered_int64_sha256": int64_sha256(rows),
                "unique_rows": int(torch.unique(rows).numel()),
            }
            for name, rows in pools.items()
        },
        "membership_source": "teacher_only_no_student_outputs",
    }
    return RetentionPools(
        natural=pools["natural"],
        current_policy_applicable=pools["current_policy_applicable"],
        historical_opponent=pools["historical_opponent"],
        low_teacher_variance=pools["low_teacher_variance"],
        low_variance_threshold_log_std=threshold,
        manifest=manifest,
    )


def verify_retention_pools(pools: RetentionPools, expected: Mapping[str, Any]) -> None:
    """Fail closed if regenerated membership differs from prospective authority."""

    checks: dict[str, bool] = {
        "threshold_exact": pools.low_variance_threshold_log_std
        == float(expected["low_variance_definition"]["threshold_log_std"]),
    }
    for name, rows in pools.by_name().items():
        expected_row = expected["pools"][name]
        checks[f"{name}.rows"] = int(rows.numel()) == int(expected_row["rows"])
        checks[f"{name}.hash"] = int64_sha256(rows) == expected_row["ordered_int64_sha256"]
    if not all(checks.values()):
        raise RuntimeError(f"V3 retention strata changed: {checks}")


def sample_retention_rows(
    pools: RetentionPools,
    counts: Mapping[str, int],
    *,
    generator: torch.Generator,
) -> tuple[torch.Tensor, dict[str, int]]:
    """Sample the frozen with-replacement stratum mixture in stable order."""

    rows: list[torch.Tensor] = []
    realized: dict[str, int] = {}
    available = pools.by_name()
    if set(counts) != set(available):
        raise ValueError("retention mixture stratum names changed")
    ordered_names = (
        "natural",
        "current_policy_applicable",
        "historical_opponent",
        "low_teacher_variance",
    )
    for name in ordered_names:
        count = int(counts[name])
        if count <= 0:
            raise ValueError("each V3 retention stratum must contribute positive rows")
        source = available[name]
        positions = torch.randint(source.numel(), (count,), generator=generator)
        rows.append(source.index_select(0, positions))
        realized[name] = count
    return torch.cat(rows), realized


@dataclass(frozen=True, slots=True)
class TailRetentionLoss:
    loss: torch.Tensor
    mean_kl: torch.Tensor
    barrier: torch.Tensor
    maximum_sample_kl: torch.Tensor
    per_channel_mean_kl: torch.Tensor


def tail_aware_actor_retention_loss(
    teacher_actor: torch.Tensor,
    student_actor: torch.Tensor,
    *,
    policy_config: Rival2PolicyConfig,
    mean_kl_coefficient: float,
    barrier_threshold: float,
    barrier_temperature: float,
    barrier_coefficient: float,
) -> TailRetentionLoss:
    """Mean KL plus a smooth squared-softplus barrier on rare sample displacement."""

    if barrier_temperature <= 0.0 or barrier_coefficient < 0.0:
        raise ValueError("invalid tail-barrier configuration")
    channel = hybrid_actor_channel_kl(teacher_actor, student_actor, policy_config=policy_config)
    sample = channel.sum(dim=-1)
    smooth_excess = (
        F.softplus((sample - barrier_threshold) / barrier_temperature) * barrier_temperature
    )
    barrier = smooth_excess.square().mean()
    mean_kl = sample.mean()
    return TailRetentionLoss(
        loss=mean_kl_coefficient * mean_kl + barrier_coefficient * barrier,
        mean_kl=mean_kl,
        barrier=barrier,
        maximum_sample_kl=sample.max(),
        per_channel_mean_kl=channel.mean(dim=0),
    )


def _group_metrics(channel: torch.Tensor) -> dict[str, Any]:
    if channel.numel() == 0:
        return {
            "sample_count": 0,
            "mean_kl": 0.0,
            "max_sample_kl": 0.0,
            "max_individual_channel_kl": 0.0,
            "counts_above": {"0.5": 0, "1.0": 0, "2.0": 0},
            "mean_channel_kl": {name: 0.0 for name in ACTION_NAMES},
            "tail_channel_contribution_above_0.5": {name: 0.0 for name in ACTION_NAMES},
        }
    value = channel.detach().cpu().to(torch.float64)
    sample = value.sum(dim=-1)
    tail = sample > 0.5
    tail_channel = (
        value[tail].sum(dim=0) if bool(tail.any()) else torch.zeros(8, dtype=torch.float64)
    )
    tail_total = float(tail_channel.sum().item())
    return {
        "sample_count": int(sample.numel()),
        "mean_kl": float(sample.mean().item()),
        "max_sample_kl": float(sample.max().item()),
        "max_individual_channel_kl": float(value.max().item()),
        "counts_above": {
            str(threshold): int((sample > threshold).sum().item()) for threshold in (0.5, 1.0, 2.0)
        },
        "mean_channel_kl": {
            name: float(value[:, index].mean().item()) for index, name in enumerate(ACTION_NAMES)
        },
        "max_channel_kl": {
            name: float(value[:, index].max().item()) for index, name in enumerate(ACTION_NAMES)
        },
        "tail_channel_contribution_above_0.5": {
            name: (float(tail_channel[index].item()) / tail_total if tail_total else 0.0)
            for index, name in enumerate(ACTION_NAMES)
        },
    }


@torch.no_grad()
def evaluate_detailed_retention(
    teacher: Rival2ActorCritic,
    student: Rival2ActorCritic,
    observations: torch.Tensor,
    worlds: np.ndarray,
    opponent_family: torch.Tensor,
    rival_side: torch.Tensor,
    *,
    low_variance_threshold_log_std: float,
    policy_config: Rival2PolicyConfig,
    worlds_per_batch: int,
) -> dict[str, Any]:
    """Evaluate complete split metrics for every V3-required retention role."""

    grouped: dict[str, list[torch.Tensor]] = {
        "all_perspectives": [],
        "current_policy_applicable": [],
        "counterfactual_opponent": [],
        "historical_opponent": [],
        "low_teacher_variance": [],
    }
    critic_max = 0.0
    critic_square_sum = 0.0
    critic_count = 0
    actor_finite = True
    value_finite = True
    for start in range(0, len(worlds), worlds_per_batch):
        selected_worlds = np.asarray(worlds[start : start + worlds_per_batch], dtype=np.int64)
        encoded = encoded_rows_for_worlds(selected_worlds)
        observation = gather_encoded_rows(observations, encoded)
        hidden = teacher.trunk(observation)
        teacher_actor = teacher.actor(hidden)
        teacher_value = teacher.critic(hidden).squeeze(-1)
        student_actor, student_value = student(observation)
        channel = hybrid_actor_channel_kl(teacher_actor, student_actor, policy_config=policy_config)
        role = role_masks_for_encoded(encoded, opponent_family, rival_side)
        teacher_log_std = teacher_actor[:, 5:10].clamp(
            policy_config.log_std_min, policy_config.log_std_max
        )
        role["low_teacher_variance"] = (
            teacher_log_std.min(dim=-1).values.detach().cpu().to(torch.float64)
            <= low_variance_threshold_log_std
        )
        channel_cpu = channel.detach().cpu()
        for name in grouped:
            grouped[name].append(channel_cpu[role[name]])
        drift = (student_value - teacher_value).detach()
        critic_count += int(drift.numel())
        critic_square_sum += float(drift.square().sum().item())
        critic_max = max(critic_max, float(drift.abs().max().item()))
        actor_finite = actor_finite and bool(torch.isfinite(channel).all())
        value_finite = value_finite and bool(
            torch.isfinite(teacher_value).all() and torch.isfinite(student_value).all()
        )
    result = {name: _group_metrics(torch.cat(rows)) for name, rows in grouped.items()}
    result["critic"] = {
        "rmse": math.sqrt(critic_square_sum / critic_count),
        "max_absolute_drift": critic_max,
        "finite": value_finite,
    }
    result["actor_finite"] = actor_finite
    return result


def detailed_retention_guard(
    metrics: Mapping[str, Any],
    hard_guard: Mapping[str, float | bool],
) -> dict[str, Any]:
    """Apply the unchanged V1 all-perspective hard boundaries."""

    all_rows = metrics["all_perspectives"]
    checks = {
        "actor_mean_kl": float(all_rows["mean_kl"]) <= float(hard_guard["actor_mean_kl"]),
        "actor_max_sample_kl": float(all_rows["max_sample_kl"])
        <= float(hard_guard["actor_max_sample_kl"]),
        "actor_max_mean_channel_kl": max(all_rows["mean_channel_kl"].values())
        <= float(hard_guard["actor_max_channel_kl"]),
        "critic_rmse": float(metrics["critic"]["rmse"]) <= float(hard_guard["critic_rmse"]),
        "critic_max_absolute_drift": float(metrics["critic"]["max_absolute_drift"])
        <= float(hard_guard["critic_max_absolute_drift"]),
        "finite": bool(metrics["actor_finite"] and metrics["critic"]["finite"]),
    }
    return {"checks": checks, "accepted": all(checks.values())}


__all__ = [
    "ROWS_PER_WORLD",
    "RetentionPools",
    "TailRetentionLoss",
    "build_retention_pools",
    "detailed_retention_guard",
    "encoded_rows_for_worlds",
    "evaluate_detailed_retention",
    "gather_encoded_rows",
    "int64_sha256",
    "role_masks_for_encoded",
    "sample_retention_rows",
    "tail_aware_actor_retention_loss",
    "verify_retention_pools",
]
