"""Deterministic helpers for missing-feature invariance distillation.

The module contains no PPO path and does not read human action targets.  It exposes the
committed BC bridge profiles on torch tensors, whole-world splitting, and exact hybrid actor
retention metrics used by the supervised runner.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch

from rivalsim.human_demo.bc_observation_bridge import (
    DegradationProfile,
    degradation_quality_mask,
    hybrid_actor_channel_kl,
)
from rivalsim.rival2_contracts import ACTION_NAMES, OBS_DIM
from rivalsim.rival2_policy import Rival2PolicyConfig

DISTILLATION_VERSION = "RIVAL2_MISSING_FEATURE_INVARIANCE_DISTILLATION_V1"
DISTILLED_CHECKPOINT_FORMAT = "RIVAL2_120HZ_MISSING_FEATURE_DISTILLED_CHECKPOINT_V1"


def file_sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest().upper()


def canonical_sha256(value: Any) -> str:
    payload = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
    ).encode("ascii")
    return hashlib.sha256(payload).hexdigest().upper()


@dataclass(frozen=True, slots=True)
class WholeWorldSplit:
    train: np.ndarray
    validation: np.ndarray
    test: np.ndarray
    seed: int

    @property
    def manifest(self) -> dict[str, Any]:
        rows = {
            "train": self.train,
            "validation": self.validation,
            "test": self.test,
        }
        return {
            "algorithm": "numpy PCG64 permutation of world indices",
            "seed": self.seed,
            "counts": {name: int(value.size) for name, value in rows.items()},
            "world_indices": {name: value.tolist() for name, value in rows.items()},
            "world_index_sha256": {
                name: hashlib.sha256(value.tobytes(order="C")).hexdigest().upper()
                for name, value in rows.items()
            },
            "split_manifest_sha256": canonical_sha256(
                {name: value.tolist() for name, value in rows.items()}
            ),
            "whole_world_disjoint": True,
        }


def build_whole_world_split(
    *,
    worlds: int,
    train_worlds: int,
    validation_worlds: int,
    test_worlds: int,
    seed: int,
) -> WholeWorldSplit:
    if min(worlds, train_worlds, validation_worlds, test_worlds) <= 0:
        raise ValueError("world split counts must be positive")
    if train_worlds + validation_worlds + test_worlds != worlds:
        raise ValueError("world split counts do not cover the corpus exactly")
    permutation = np.random.default_rng(seed).permutation(worlds).astype(np.int32)
    train_end = train_worlds
    validation_end = train_end + validation_worlds
    split = WholeWorldSplit(
        train=np.ascontiguousarray(permutation[:train_end]),
        validation=np.ascontiguousarray(permutation[train_end:validation_end]),
        test=np.ascontiguousarray(permutation[validation_end:]),
        seed=seed,
    )
    all_worlds = np.concatenate((split.train, split.validation, split.test))
    if np.unique(all_worlds).size != worlds:
        raise RuntimeError("whole-world split overlaps or omits a world")
    for value in (split.train, split.validation, split.test):
        value.flags.writeable = False
    return split


def torch_profile_quality(
    profile: DegradationProfile | str,
    *,
    device: torch.device | str,
) -> torch.Tensor:
    quality = degradation_quality_mask(profile)
    return torch.from_numpy(np.asarray(quality).copy()).to(device=device)


def degrade_observations_torch(
    observation: torch.Tensor,
    quality: torch.Tensor,
) -> torch.Tensor:
    """Apply the committed neutral-placeholder rule without changing available values."""

    if observation.shape[-1] != OBS_DIM:
        raise ValueError("observation must end in the 182-field Rival contract")
    if quality.shape != (OBS_DIM,):
        raise ValueError("quality mask must have shape [182]")
    if quality.device != observation.device:
        raise ValueError("quality mask and observation must share a device")
    return observation.masked_fill(quality == 0, 0.0)


def world_observation_batch(
    observations: torch.Tensor,
    world_indices: np.ndarray | torch.Tensor,
) -> torch.Tensor:
    """Gather complete trajectories and flatten only after whole-world selection."""

    if observations.ndim != 4 or observations.shape[-2:] != (2, OBS_DIM):
        raise ValueError("corpus observations must have shape [T, W, 2, 182]")
    indices = torch.as_tensor(world_indices, dtype=torch.int64, device=observations.device)
    selected = observations.index_select(1, indices)
    return selected.permute(1, 0, 2, 3).reshape(-1, OBS_DIM)


@dataclass(slots=True)
class MetricAccumulator:
    count: int
    channel_kl_sum: torch.Tensor
    sample_kl_sum: float
    sample_kl_max: float
    value_absolute_sum: float
    value_squared_sum: float
    value_absolute_max: float
    actor_finite: bool
    value_finite: bool

    @classmethod
    def create(cls) -> MetricAccumulator:
        return cls(
            count=0,
            channel_kl_sum=torch.zeros(len(ACTION_NAMES), dtype=torch.float64),
            sample_kl_sum=0.0,
            sample_kl_max=0.0,
            value_absolute_sum=0.0,
            value_squared_sum=0.0,
            value_absolute_max=0.0,
            actor_finite=True,
            value_finite=True,
        )

    def update(
        self,
        channel_kl: torch.Tensor,
        teacher_value: torch.Tensor,
        student_value: torch.Tensor,
    ) -> None:
        detached_kl = channel_kl.detach()
        sample = detached_kl.sum(dim=-1)
        drift = (student_value.detach() - teacher_value.detach()).abs()
        batch_count = int(sample.numel())
        self.count += batch_count
        self.channel_kl_sum += detached_kl.sum(dim=0).to("cpu", torch.float64)
        self.sample_kl_sum += float(sample.sum().item())
        self.sample_kl_max = max(self.sample_kl_max, float(sample.max().item()))
        self.value_absolute_sum += float(drift.sum().item())
        self.value_squared_sum += float(drift.square().sum().item())
        self.value_absolute_max = max(
            self.value_absolute_max,
            float(drift.max().item()),
        )
        self.actor_finite = self.actor_finite and bool(torch.isfinite(detached_kl).all())
        self.value_finite = self.value_finite and bool(
            torch.isfinite(teacher_value).all() and torch.isfinite(student_value).all()
        )

    def result(self) -> dict[str, Any]:
        if self.count <= 0:
            raise ValueError("cannot finalize empty metric accumulator")
        channel = self.channel_kl_sum / self.count
        return {
            "sample_count": self.count,
            "actor_mean_kl": self.sample_kl_sum / self.count,
            "actor_max_sample_kl": self.sample_kl_max,
            "actor_channel_kl": {
                name: float(channel[index].item())
                for index, name in enumerate(ACTION_NAMES)
            },
            "value_mae": self.value_absolute_sum / self.count,
            "value_rmse": (self.value_squared_sum / self.count) ** 0.5,
            "value_max_absolute_drift": self.value_absolute_max,
            "actor_finite": self.actor_finite,
            "value_finite": self.value_finite,
        }


def paired_metrics(
    teacher_actor: torch.Tensor,
    student_actor: torch.Tensor,
    teacher_value: torch.Tensor,
    student_value: torch.Tensor,
    *,
    policy_config: Rival2PolicyConfig,
) -> tuple[torch.Tensor, dict[str, Any]]:
    channel = hybrid_actor_channel_kl(
        teacher_actor,
        student_actor,
        policy_config=policy_config,
    )
    accumulator = MetricAccumulator.create()
    accumulator.update(channel, teacher_value, student_value)
    return channel, accumulator.result()


def actor_output_statistics(actor: torch.Tensor) -> dict[str, Any]:
    if actor.ndim != 2 or actor.shape[1] != 13:
        raise ValueError("actor output must have shape [N, 13]")
    value = actor.detach().to(torch.float32)
    means = value[:, :5]
    log_std = value[:, 5:10].clamp(-5.0, 1.0)
    probabilities = torch.sigmoid(value[:, 10:13])
    analog_names = ACTION_NAMES[:5]
    button_names = ACTION_NAMES[5:]
    return {
        "sample_count": int(value.shape[0]),
        "finite": bool(torch.isfinite(value).all()),
        "analog_mean": {
            name: {
                "mean": float(means[:, index].mean().item()),
                "std": float(means[:, index].std(unbiased=False).item()),
                "min": float(means[:, index].min().item()),
                "max": float(means[:, index].max().item()),
                "absolute_ge_5_fraction": float(
                    (means[:, index].abs() >= 5.0).to(torch.float32).mean().item()
                ),
            }
            for index, name in enumerate(analog_names)
        },
        "log_std": {
            name: {
                "mean": float(log_std[:, index].mean().item()),
                "at_min_fraction": float(
                    (log_std[:, index] <= -5.0).to(torch.float32).mean().item()
                ),
                "at_max_fraction": float(
                    (log_std[:, index] >= 1.0).to(torch.float32).mean().item()
                ),
            }
            for index, name in enumerate(analog_names)
        },
        "button_probability": {
            name: {
                "mean": float(probabilities[:, index].mean().item()),
                "std": float(probabilities[:, index].std(unbiased=False).item()),
                "saturation_fraction": float(
                    (
                        (probabilities[:, index] <= 0.001)
                        | (probabilities[:, index] >= 0.999)
                    )
                    .to(torch.float32)
                    .mean()
                    .item()
                ),
            }
            for index, name in enumerate(button_names)
        },
    }


__all__ = [
    "DISTILLATION_VERSION",
    "DISTILLED_CHECKPOINT_FORMAT",
    "MetricAccumulator",
    "WholeWorldSplit",
    "actor_output_statistics",
    "build_whole_world_split",
    "canonical_sha256",
    "degrade_observations_torch",
    "file_sha256",
    "paired_metrics",
    "torch_profile_quality",
    "world_observation_batch",
]
