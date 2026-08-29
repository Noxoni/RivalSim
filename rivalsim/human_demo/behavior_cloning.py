"""Frozen building blocks for Rival's first native-120-Hz human BC stage.

This module contains only supervised objectives, deterministic hierarchical
sampling, and read-only evaluation helpers.  It does not own PPO, rewards,
mechanic detection, or native-recording mutation.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
import torch.nn.functional as F

from rivalsim.human_demo.bc_observation_bridge import hybrid_actor_channel_kl
from rivalsim.rival2_contracts import ACTION_NAMES
from rivalsim.rival2_policy import Rival2PolicyConfig

HUMAN_BC_VERSION = "RIVAL2_HUMAN_BEHAVIOR_CLONING_V1"
HUMAN_BC_CHECKPOINT_FORMAT = "RIVAL2_HUMAN_BEHAVIOR_CLONING_CHECKPOINT_V1"


@dataclass(frozen=True, slots=True)
class HumanBCLoss:
    loss: torch.Tensor
    analog_smooth_l1: torch.Tensor
    button_bce: torch.Tensor
    log_std_retention: torch.Tensor
    analog_per_channel: torch.Tensor
    button_per_channel: torch.Tensor
    log_std_per_channel: torch.Tensor


def human_behavior_cloning_objective(
    student_actor: torch.Tensor,
    teacher_actor: torch.Tensor,
    exact_action: torch.Tensor,
    *,
    smooth_l1_beta: float,
    analog_weight: float,
    button_weight: float,
    log_std_weight: float,
    policy_config: Rival2PolicyConfig | None = None,
) -> HumanBCLoss:
    """Supervise exact actions while retaining teacher exploration scale.

    Analog targets supervise ``tanh(actor_mean)``. Button targets supervise raw
    logits. The bootstrap log standard deviations are retention targets only;
    they are never collapsed to zero or treated as human action labels.
    """

    if student_actor.shape != teacher_actor.shape or student_actor.shape[-1] != 13:
        raise ValueError("student/teacher actor tensors must have matching [N, 13] shape")
    if exact_action.shape != (*student_actor.shape[:-1], 8):
        raise ValueError("exact action tensor must have matching [N, 8] shape")
    if smooth_l1_beta <= 0.0:
        raise ValueError("SmoothL1 beta must be positive")
    config = policy_config or Rival2PolicyConfig()
    analog_prediction = torch.tanh(student_actor[..., :5])
    analog_element = F.smooth_l1_loss(
        analog_prediction,
        exact_action[..., :5],
        reduction="none",
        beta=smooth_l1_beta,
    )
    button_element = F.binary_cross_entropy_with_logits(
        student_actor[..., 10:13], exact_action[..., 5:8], reduction="none"
    )
    teacher_log_std = (
        teacher_actor[..., 5:10].detach().clamp(config.log_std_min, config.log_std_max)
    )
    student_log_std = student_actor[..., 5:10].clamp(config.log_std_min, config.log_std_max)
    log_std_element = (student_log_std - teacher_log_std).square()
    analog_per_channel = analog_element.mean(dim=0)
    button_per_channel = button_element.mean(dim=0)
    log_std_per_channel = log_std_element.mean(dim=0)
    analog = analog_per_channel.mean()
    buttons = button_per_channel.mean()
    log_std = log_std_per_channel.mean()
    loss = analog_weight * analog + button_weight * buttons + log_std_weight * log_std
    return HumanBCLoss(
        loss=loss,
        analog_smooth_l1=analog,
        button_bce=buttons,
        log_std_retention=log_std,
        analog_per_channel=analog_per_channel,
        button_per_channel=button_per_channel,
        log_std_per_channel=log_std_per_channel,
    )


@dataclass(frozen=True, slots=True)
class SimulatorRetentionLoss:
    loss: torch.Tensor
    actor_kl: torch.Tensor
    critic_mse: torch.Tensor
    per_channel_kl: torch.Tensor


def simulator_retention_objective(
    student_actor: torch.Tensor,
    student_value: torch.Tensor,
    teacher_actor: torch.Tensor,
    teacher_value: torch.Tensor,
    *,
    actor_weight: float,
    critic_weight: float,
    policy_config: Rival2PolicyConfig | None = None,
) -> SimulatorRetentionLoss:
    """Retain the frozen teacher actor distribution and detached critic target."""

    if student_actor.shape != teacher_actor.shape or student_actor.shape[-1] != 13:
        raise ValueError("student/teacher actors must have matching [N, 13] shape")
    if student_value.shape != teacher_value.shape:
        raise ValueError("student/teacher values must have matching shape")
    channel = hybrid_actor_channel_kl(
        teacher_actor.detach(), student_actor, policy_config=policy_config
    )
    per_channel = channel.mean(dim=0)
    actor_kl = per_channel.sum()
    critic_mse = F.mse_loss(student_value, teacher_value.detach())
    return SimulatorRetentionLoss(
        loss=actor_weight * actor_kl + critic_weight * critic_mse,
        actor_kl=actor_kl,
        critic_mse=critic_mse,
        per_channel_kl=per_channel,
    )


class MechanicHierarchySampler:
    """Deterministic label -> attempt -> frame sampler with a bounded mixture.

    ``uniform_label_fraction`` is mixed with natural frame sampling.  A runtime
    guard rejects a prospective policy whose expected frame probability exceeds
    ``maximum_oversampling_ratio`` times natural sampling for any frame.
    """

    def __init__(
        self,
        labels: list[str],
        attempts: list[str],
        *,
        uniform_label_fraction: float,
        maximum_oversampling_ratio: float,
        generator: torch.Generator,
    ) -> None:
        if len(labels) != len(attempts) or not labels:
            raise ValueError("labels and attempts must be nonempty and aligned")
        if not 0.0 <= uniform_label_fraction <= 1.0:
            raise ValueError("uniform-label mixture fraction must be in [0, 1]")
        self.generator = generator
        self.uniform_label_fraction = float(uniform_label_fraction)
        self.maximum_oversampling_ratio = float(maximum_oversampling_ratio)
        self.labels = tuple(sorted(set(labels)))
        self.attempt_rows: dict[str, tuple[str, ...]] = {}
        self.frame_rows: dict[str, torch.Tensor] = {}
        for label in self.labels:
            label_attempts = sorted(
                {
                    attempt
                    for current, attempt in zip(labels, attempts, strict=True)
                    if current == label
                }
            )
            self.attempt_rows[label] = tuple(label_attempts)
            for attempt in label_attempts:
                indices = [
                    index
                    for index, (current_label, current_attempt) in enumerate(
                        zip(labels, attempts, strict=True)
                    )
                    if current_label == label and current_attempt == attempt
                ]
                self.frame_rows[attempt] = torch.tensor(indices, dtype=torch.int64)
        self.frame_count = len(labels)
        self.expected_probability = self._expected_frame_probability(labels, attempts)
        natural = 1.0 / self.frame_count
        self.maximum_realized_oversampling_ratio = max(self.expected_probability) / natural
        if self.maximum_realized_oversampling_ratio > self.maximum_oversampling_ratio + 1e-12:
            raise ValueError(
                "mechanic sampling exceeds frozen oversampling cap: "
                f"{self.maximum_realized_oversampling_ratio} > {self.maximum_oversampling_ratio}"
            )

    def _expected_frame_probability(
        self, labels: list[str], attempts: list[str]
    ) -> tuple[float, ...]:
        probability: list[float] = []
        natural = (1.0 - self.uniform_label_fraction) / len(labels)
        for label, attempt in zip(labels, attempts, strict=True):
            balanced = self.uniform_label_fraction / len(self.labels)
            balanced /= len(self.attempt_rows[label])
            balanced /= len(self.frame_rows[attempt])
            probability.append(natural + balanced)
        if abs(sum(probability) - 1.0) > 1e-9:
            raise RuntimeError("mechanic sampling probabilities do not sum to one")
        return tuple(probability)

    def sample(self, count: int) -> torch.Tensor:
        if count <= 0:
            raise ValueError("sample count must be positive")
        result = torch.empty(count, dtype=torch.int64)
        use_uniform = torch.rand(count, generator=self.generator) < self.uniform_label_fraction
        natural_count = int((~use_uniform).sum().item())
        if natural_count:
            result[~use_uniform] = torch.randint(
                self.frame_count, (natural_count,), generator=self.generator
            )
        positions = use_uniform.nonzero(as_tuple=False).flatten()
        for position in positions.tolist():
            label = self.labels[
                int(torch.randint(len(self.labels), (), generator=self.generator).item())
            ]
            attempt_rows = self.attempt_rows[label]
            attempt = attempt_rows[
                int(torch.randint(len(attempt_rows), (), generator=self.generator).item())
            ]
            frames = self.frame_rows[attempt]
            frame = int(torch.randint(len(frames), (), generator=self.generator).item())
            result[position] = frames[frame]
        return result


def action_predictions(actor: torch.Tensor) -> torch.Tensor:
    """Return the eight deterministic supervised predictions in contract order."""

    if actor.ndim != 2 or actor.shape[1] != 13:
        raise ValueError("actor output must have shape [N, 13]")
    return torch.cat((torch.tanh(actor[:, :5]), torch.sigmoid(actor[:, 10:13])), dim=1)


def action_metric_summary(actor: torch.Tensor, target: torch.Tensor) -> dict[str, Any]:
    """Open-loop metrics with every action channel and no thresholded analog loss."""

    if target.ndim != 2 or target.shape[1] != 8 or actor.shape[0] != target.shape[0]:
        raise ValueError("actor/target metric tensors are not aligned")
    predicted = action_predictions(actor)
    difference = predicted - target
    analog_difference = difference[:, :5]
    button_logits = actor[:, 10:13]
    button_target = target[:, 5:8]
    channel = {}
    for index, name in enumerate(ACTION_NAMES):
        row = difference[:, index]
        entry: dict[str, float] = {
            "mae": float(row.abs().mean().item()),
            "rmse": float(row.square().mean().sqrt().item()),
        }
        if index >= 5:
            local = index - 5
            entry["bce"] = float(
                F.binary_cross_entropy_with_logits(
                    button_logits[:, local], button_target[:, local]
                ).item()
            )
            entry["accuracy"] = float(
                ((button_logits[:, local] >= 0.0) == (button_target[:, local] >= 0.5))
                .to(torch.float32)
                .mean()
                .item()
            )
        channel[name] = entry
    return {
        "sample_count": int(target.shape[0]),
        "complete_action_mae": float(difference.abs().mean().item()),
        "complete_action_rmse": float(difference.square().mean().sqrt().item()),
        "analog_mae": float(analog_difference.abs().mean().item()),
        "analog_rmse": float(analog_difference.square().mean().sqrt().item()),
        "button_bce": float(
            F.binary_cross_entropy_with_logits(button_logits, button_target).item()
        ),
        "button_accuracy": float(
            ((button_logits >= 0.0) == (button_target >= 0.5)).to(torch.float32).mean().item()
        ),
        "per_channel": channel,
    }


__all__ = [
    "HUMAN_BC_CHECKPOINT_FORMAT",
    "HUMAN_BC_VERSION",
    "HumanBCLoss",
    "MechanicHierarchySampler",
    "SimulatorRetentionLoss",
    "action_metric_summary",
    "action_predictions",
    "human_behavior_cloning_objective",
    "simulator_retention_objective",
]
