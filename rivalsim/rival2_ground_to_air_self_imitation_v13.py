"""Success-conditioned actor rehearsal for natural aerial continuations.

The V13 stage consumes exact actions sampled by the selected V12 aerial option
only when authoritative physics later records a separated second airborne
contact or a goal inside the six-contact budget.  It changes only the aerial
actor head.  V23, the aerial trunk, the critic, rewards, and simulator physics
remain immutable.
"""

from __future__ import annotations

import copy
from dataclasses import dataclass
from typing import Any

import torch

from rivalsim.human_demo.bc_observation_bridge import hybrid_actor_channel_kl
from rivalsim.human_demo.behavior_cloning import human_behavior_cloning_objective
from rivalsim.rival2_ground_to_air_selfplay_training_v12 import (
    AerialOptionSelfPlayTrainerV12,
)
from rivalsim.rival2_policy import Rival2ActorCritic
from rivalsim.rival2_ppo import Rival2PolicyDisplacementRejected, Rival2RolloutBuffer

GROUND_TO_AIR_SELF_IMITATION_V13_VERSION = (
    "RIVAL2_GROUND_TO_AIR_SELF_IMITATION_V13"
)


def successful_history_mask(
    success_events: torch.Tensor,
    active: torch.Tensor,
    *,
    history_ticks: int,
) -> torch.Tensor:
    """Return the union of active action windows preceding literal successes."""

    if success_events.shape != active.shape or success_events.ndim != 3:
        raise ValueError("success and active masks must align as [T,W,2]")
    if history_ticks <= 0:
        raise ValueError("history_ticks must be positive")
    events = success_events.to(torch.bool)
    selected = torch.zeros_like(events)
    for lag in range(min(history_ticks, events.shape[0])):
        if lag == 0:
            selected |= events
        else:
            selected[:-lag] |= events[lag:]
    return selected & active.to(torch.bool)


@dataclass(frozen=True, slots=True)
class SelfImitationConfig:
    history_ticks: int = 96
    maximum_success_samples: int = 65_536
    maximum_retention_samples: int = 65_536
    smooth_l1_beta: float = 0.1
    analog_weight: float = 1.0
    button_weight: float = 0.5
    log_std_weight: float = 0.05
    teacher_actor_kl_weight: float = 0.02
    maximum_gradient_norm: float = 0.5

    def __post_init__(self) -> None:
        if self.history_ticks <= 0:
            raise ValueError("history_ticks must be positive")
        if self.maximum_success_samples <= 0 or self.maximum_retention_samples <= 0:
            raise ValueError("sample bounds must be positive")
        if self.smooth_l1_beta <= 0.0 or self.maximum_gradient_norm <= 0.0:
            raise ValueError("loss and gradient scales must be positive")
        if min(
            self.analog_weight,
            self.button_weight,
            self.log_std_weight,
            self.teacher_actor_kl_weight,
        ) < 0.0:
            raise ValueError("loss weights cannot be negative")


class AerialSuccessfulSelfImitationV13:
    """Actor-only optimizer around a V12 natural self-play collector."""

    def __init__(
        self,
        collector: AerialOptionSelfPlayTrainerV12,
        *,
        teacher: Rival2ActorCritic,
        learning_rate: float,
        weight_decay: float,
        config: SelfImitationConfig,
        seed: int,
    ) -> None:
        if learning_rate <= 0.0 or weight_decay < 0.0:
            raise ValueError("invalid self-imitation optimizer configuration")
        self.collector = collector
        self.model = collector.model
        self.teacher = teacher.to(collector.device).eval().requires_grad_(False)
        self.config = config
        self.model.trunk.requires_grad_(False)
        self.model.critic.requires_grad_(False)
        self.model.actor.requires_grad_(True)
        self.optimizer = torch.optim.AdamW(
            self.model.actor.parameters(),
            lr=float(learning_rate),
            weight_decay=float(weight_decay),
        )
        self.generator = torch.Generator(device=collector.device).manual_seed(seed)
        self.accepted_blocks = 0
        self.total_success_samples = 0

    def _bounded_indices(self, mask: torch.Tensor, maximum: int) -> torch.Tensor:
        indices = torch.nonzero(mask.reshape(-1), as_tuple=False).flatten()
        if indices.numel() > maximum:
            order = torch.randperm(
                indices.numel(), device=indices.device, generator=self.generator
            )
            indices = indices.index_select(0, order[:maximum])
        return indices

    def update(self, rollout: Rival2RolloutBuffer) -> dict[str, Any]:
        success_events = getattr(rollout, "aerial_success_events", None)
        if success_events is None:
            raise ValueError("rollout does not carry authoritative aerial success events")
        selected = successful_history_mask(
            success_events,
            rollout.train_mask,
            history_ticks=self.config.history_ticks,
        )
        success_indices = self._bounded_indices(
            selected, self.config.maximum_success_samples
        )
        if success_indices.numel() == 0:
            return {
                "accepted": False,
                "reason": "no_successful_trajectory_samples",
                "success_events": int(success_events.sum().item()),
                "success_samples": 0,
            }
        retention_indices = self._bounded_indices(
            rollout.train_mask, self.config.maximum_retention_samples
        )
        flat_observation = rollout.observations.reshape(-1, 182)
        flat_action = rollout.actions.reshape(-1, 8)
        success_observation = flat_observation.index_select(0, success_indices)
        success_action = flat_action.index_select(0, success_indices)
        retention_observation = flat_observation.index_select(0, retention_indices)

        actor_before = copy.deepcopy(self.model.actor.state_dict())
        optimizer_before = copy.deepcopy(self.optimizer.state_dict())
        self.model.train()
        with torch.no_grad():
            teacher_success, _ = self.teacher(success_observation)
            teacher_retention, _ = self.teacher(retention_observation)
        student_success, _ = self.model(success_observation)
        imitation = human_behavior_cloning_objective(
            student_success,
            teacher_success,
            success_action,
            smooth_l1_beta=self.config.smooth_l1_beta,
            analog_weight=self.config.analog_weight,
            button_weight=self.config.button_weight,
            log_std_weight=self.config.log_std_weight,
            policy_config=self.model.config,
        )
        student_retention, _ = self.model(retention_observation)
        retention_channel = hybrid_actor_channel_kl(
            teacher_retention,
            student_retention,
            policy_config=self.model.config,
        )
        retention_per_sample = retention_channel.sum(dim=1)
        retention_kl = retention_per_sample.mean()
        loss = imitation.loss + self.config.teacher_actor_kl_weight * retention_kl
        self.optimizer.zero_grad(set_to_none=True)
        loss.backward()
        gradient = torch.nn.utils.clip_grad_norm_(
            self.model.actor.parameters(), self.config.maximum_gradient_norm
        )
        finite_before_step = bool(torch.isfinite(loss) and torch.isfinite(gradient))
        if not finite_before_step:
            self.model.actor.load_state_dict(actor_before)
            self.optimizer.load_state_dict(optimizer_before)
            self.optimizer.zero_grad(set_to_none=True)
            raise Rival2PolicyDisplacementRejected(
                {"reason": "nonfinite_v13_self_imitation_gradient"}
            )
        self.optimizer.step()
        finite_parameters = all(
            bool(torch.isfinite(parameter).all().item())
            for parameter in self.model.actor.parameters()
        )
        if not finite_parameters:
            self.model.actor.load_state_dict(actor_before)
            self.optimizer.load_state_dict(optimizer_before)
            self.optimizer.zero_grad(set_to_none=True)
            raise Rival2PolicyDisplacementRejected(
                {"reason": "nonfinite_v13_self_imitation_parameter"}
            )
        self.accepted_blocks += 1
        self.total_success_samples += int(success_indices.numel())
        with torch.no_grad():
            post_actor, _ = self.model(retention_observation)
            post_channel = hybrid_actor_channel_kl(
                teacher_retention, post_actor, policy_config=self.model.config
            )
            post_per_sample = post_channel.sum(dim=1)
        return {
            "accepted": True,
            "accepted_block": self.accepted_blocks,
            "success_events": int(success_events.sum().item()),
            "success_samples": int(success_indices.numel()),
            "retention_samples": int(retention_indices.numel()),
            "loss": float(loss.detach()),
            "analog_loss": float(imitation.analog_smooth_l1.detach()),
            "button_loss": float(imitation.button_bce.detach()),
            "log_std_loss": float(imitation.log_std_retention.detach()),
            "pre_step_retention_mean_kl": float(retention_kl.detach()),
            "post_step_retention_mean_kl": float(post_per_sample.mean()),
            "post_step_retention_max_kl": float(post_per_sample.max()),
            "gradient_norm": float(gradient.detach()),
            "finite": True,
        }


__all__ = [
    "GROUND_TO_AIR_SELF_IMITATION_V13_VERSION",
    "AerialSuccessfulSelfImitationV13",
    "SelfImitationConfig",
    "successful_history_mask",
]
