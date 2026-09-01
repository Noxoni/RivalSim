"""Sequence-preserving PPO for the Human Sequence recurrent Rival lineage."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

import torch

from rivalsim.rival2_contracts import ANALOG_ACTION_NAMES, BUTTON_ACTION_NAMES, OBS_DIM
from rivalsim.rival2_policy import (
    HybridDistributionOverride,
    hybrid_distribution_parameters,
    hybrid_entropy,
    hybrid_log_probability,
)
from rivalsim.rival2_ppo import Rival2PPOConfig, compute_gae_gpu
from rivalsim.rival2_recurrent_policy import (
    Rival2RecurrentActorCritic,
    Rival2RecurrentPolicyConfig,
)


class Rival2RecurrentPPOCorruption(RuntimeError):
    """A non-KL numerical failure that requires transactional rollback."""

    def __init__(self, diagnostics: dict[str, Any]):
        self.diagnostics = diagnostics
        super().__init__(
            "recurrent PPO numerical corruption: "
            f"{diagnostics.get('reason', 'unknown')}"
        )


@dataclass(frozen=True, slots=True)
class RecurrentSequenceLayout:
    horizon: int
    world_count: int
    agents_per_world: int
    sequence_count: int
    sequences_per_minibatch: int


class Rival2RecurrentRolloutBuffer:
    """Time-major rollout plus the exact recurrent state at its left boundary."""

    def __init__(
        self,
        horizon: int,
        num_envs: int,
        hidden: torch.Tensor,
        device: torch.device | str,
        *,
        obs_dim: int = OBS_DIM,
    ):
        if horizon <= 0 or num_envs <= 0:
            raise ValueError("horizon and num_envs must be positive")
        if hidden.ndim != 4 or hidden.shape[1:3] != (num_envs, 2):
            raise ValueError("initial hidden must have shape [layers, worlds, 2, hidden]")
        self.horizon = int(horizon)
        self.num_envs = int(num_envs)
        self.device = torch.device(device)
        self.initial_hidden = hidden.detach().clone()
        agent_shape = (self.horizon, self.num_envs, 2)
        self.observations = torch.empty(
            (*agent_shape, obs_dim), dtype=torch.float32, device=self.device
        )
        self.actions = torch.empty((*agent_shape, 8), dtype=torch.float32, device=self.device)
        self.pre_tanh = torch.empty((*agent_shape, 5), dtype=torch.float32, device=self.device)
        self.old_log_probability = torch.empty(agent_shape, dtype=torch.float32, device=self.device)
        self.values = torch.empty(agent_shape, dtype=torch.float32, device=self.device)
        self.rewards = torch.empty(agent_shape, dtype=torch.float32, device=self.device)
        self.terminated = torch.empty(agent_shape, dtype=torch.bool, device=self.device)
        self.truncated = torch.empty(agent_shape, dtype=torch.bool, device=self.device)
        self.next_values = torch.empty(agent_shape, dtype=torch.float32, device=self.device)
        self.train_mask = torch.empty(agent_shape, dtype=torch.bool, device=self.device)
        self.reset_before = torch.empty(agent_shape, dtype=torch.bool, device=self.device)
        self.advantages = torch.empty(agent_shape, dtype=torch.float32, device=self.device)
        self.returns = torch.empty(agent_shape, dtype=torch.float32, device=self.device)
        self.position = 0

    @property
    def logical_bytes(self) -> int:
        tensors = (
            self.initial_hidden,
            self.observations,
            self.actions,
            self.pre_tanh,
            self.old_log_probability,
            self.values,
            self.rewards,
            self.terminated,
            self.truncated,
            self.next_values,
            self.train_mask,
            self.reset_before,
            self.advantages,
            self.returns,
        )
        return sum(tensor.numel() * tensor.element_size() for tensor in tensors)

    def add(
        self,
        *,
        observation: torch.Tensor,
        action: torch.Tensor,
        pre_tanh: torch.Tensor,
        old_log_probability: torch.Tensor,
        value: torch.Tensor,
        reward: torch.Tensor,
        terminated: torch.Tensor,
        truncated: torch.Tensor,
        next_value: torch.Tensor,
        train_mask: torch.Tensor,
        reset_before: torch.Tensor,
    ) -> None:
        if self.position >= self.horizon:
            raise RuntimeError("rollout buffer is full")
        index = self.position
        fields = {
            "observations": observation,
            "actions": action,
            "pre_tanh": pre_tanh,
            "old_log_probability": old_log_probability,
            "values": value,
            "rewards": reward,
            "terminated": terminated,
            "truncated": truncated,
            "next_values": next_value,
            "train_mask": train_mask,
            "reset_before": reset_before,
        }
        for name, value_tensor in fields.items():
            getattr(self, name)[index].copy_(value_tensor)
        self.position += 1

    def compute_gae(self, config: Rival2PPOConfig) -> None:
        if self.position != self.horizon:
            raise RuntimeError("cannot compute GAE before rollout is full")
        advantage, returns = compute_gae_gpu(
            self.rewards,
            self.values,
            self.next_values,
            self.terminated,
            self.truncated,
            gamma=config.gamma,
            gae_lambda=config.gae_lambda,
        )
        self.advantages.copy_(advantage)
        self.returns.copy_(returns)

    def sequence_layout(self, minibatch_size: int) -> RecurrentSequenceLayout:
        if minibatch_size < self.horizon:
            raise ValueError("recurrent minibatch must contain at least one complete sequence")
        return RecurrentSequenceLayout(
            horizon=self.horizon,
            world_count=self.num_envs,
            agents_per_world=2,
            sequence_count=self.num_envs * 2,
            sequences_per_minibatch=max(1, minibatch_size // self.horizon),
        )


def _sequence_major(value: torch.Tensor) -> torch.Tensor:
    """Convert [time, world, agent, ...] into [world-agent, time, ...]."""

    tail = value.shape[3:]
    return value.permute(1, 2, 0, *range(3, value.ndim)).reshape(
        value.shape[1] * value.shape[2], value.shape[0], *tail
    )


def _hidden_sequence_major(hidden: torch.Tensor) -> torch.Tensor:
    return hidden.reshape(hidden.shape[0], hidden.shape[1] * hidden.shape[2], hidden.shape[3])


def _finite_parameters(model: torch.nn.Module) -> bool:
    return all(bool(torch.isfinite(parameter).all().item()) for parameter in model.parameters())


@torch.no_grad()
def _completed_diagnostics(
    model: Rival2RecurrentActorCritic,
    observation: torch.Tensor,
    initial_hidden: torch.Tensor,
    reset_before: torch.Tensor,
    action: torch.Tensor,
    pre_tanh: torch.Tensor,
    old_log_probability: torch.Tensor,
    train_mask: torch.Tensor,
    policy_config: Rival2RecurrentPolicyConfig,
    distribution_override: HybridDistributionOverride,
    sequences_per_minibatch: int,
) -> dict[str, torch.Tensor]:
    device = observation.device
    kl_sum = torch.zeros((), dtype=torch.float64, device=device)
    kl_max = torch.zeros((), dtype=torch.float32, device=device)
    sample_count = torch.zeros((), dtype=torch.float64, device=device)
    mean_sum = torch.zeros(5, dtype=torch.float64, device=device)
    log_std_sum = torch.zeros(5, dtype=torch.float64, device=device)
    button_probability_sum = torch.zeros(3, dtype=torch.float64, device=device)
    value_sum = torch.zeros((), dtype=torch.float64, device=device)
    value_square_sum = torch.zeros((), dtype=torch.float64, device=device)
    value_abs_max = torch.zeros((), dtype=torch.float32, device=device)

    for start in range(0, observation.shape[0], sequences_per_minibatch):
        stop = min(start + sequences_per_minibatch, observation.shape[0])
        actor, value, _ = model(
            observation[start:stop],
            initial_hidden[:, start:stop],
            reset_before=reset_before[start:stop],
        )
        if not bool(torch.isfinite(actor).all().item() and torch.isfinite(value).all().item()):
            raise Rival2RecurrentPPOCorruption({"reason": "nonfinite_completed_output"})
        mask = train_mask[start:stop]
        actor_flat = actor[mask]
        value_flat = value[mask]
        action_flat = action[start:stop][mask]
        pre_tanh_flat = pre_tanh[start:stop][mask]
        old_flat = old_log_probability[start:stop][mask]
        new_log_probability = hybrid_log_probability(
            actor_flat,
            action_flat,
            config=policy_config,
            pre_tanh=pre_tanh_flat,
            distribution_override=distribution_override,
        )
        log_ratio = new_log_probability - old_flat
        ratio = torch.exp(log_ratio)
        sample_kl = (ratio - 1.0) - log_ratio
        mean, log_std, logits = hybrid_distribution_parameters(
            actor_flat,
            policy_config,
            distribution_override=distribution_override,
        )
        count = torch.tensor(float(actor_flat.shape[0]), dtype=torch.float64, device=device)
        sample_count += count
        kl_sum += sample_kl.sum(dtype=torch.float64)
        kl_max = torch.maximum(kl_max, sample_kl.amax())
        mean_sum += mean.sum(dim=0, dtype=torch.float64)
        log_std_sum += log_std.sum(dim=0, dtype=torch.float64)
        button_probability_sum += torch.sigmoid(logits).sum(dim=0, dtype=torch.float64)
        value_sum += value_flat.sum(dtype=torch.float64)
        value_square_sum += value_flat.square().sum(dtype=torch.float64)
        value_abs_max = torch.maximum(value_abs_max, value_flat.abs().amax())

    value_mean = value_sum / sample_count
    value_variance = (value_square_sum / sample_count - value_mean.square()).clamp_min(0.0)
    result: dict[str, torch.Tensor] = {
        "completed_update_mean_kl": (kl_sum / sample_count).to(torch.float32),
        "completed_update_sample_kl_max": kl_max,
        "post_update_value_mean": value_mean.to(torch.float32),
        "post_update_value_std": torch.sqrt(value_variance).to(torch.float32),
        "post_update_value_max_abs": value_abs_max,
    }
    for channel, name in enumerate(ANALOG_ACTION_NAMES):
        result[f"actor_mean_mean_{name}"] = (mean_sum[channel] / sample_count).to(torch.float32)
        result[f"actor_log_std_mean_{name}"] = (
            log_std_sum[channel] / sample_count
        ).to(torch.float32)
    for channel, name in enumerate(BUTTON_ACTION_NAMES):
        result[f"actor_button_probability_{name}"] = (
            button_probability_sum[channel] / sample_count
        ).to(torch.float32)
    return result


def recurrent_ppo_update(
    model: Rival2RecurrentActorCritic,
    optimizer: torch.optim.Optimizer,
    rollout: Rival2RecurrentRolloutBuffer,
    config: Rival2PPOConfig,
    *,
    generator: torch.Generator,
    distribution_override: HybridDistributionOverride,
) -> dict[str, torch.Tensor]:
    """Optimize complete recurrent sequences; KL is measured but never gates updates."""

    rollout.compute_gae(config)
    layout = rollout.sequence_layout(config.minibatch_size)
    observation = _sequence_major(rollout.observations)
    action = _sequence_major(rollout.actions)
    pre_tanh = _sequence_major(rollout.pre_tanh)
    old_log_probability = _sequence_major(rollout.old_log_probability)
    old_value = _sequence_major(rollout.values)
    returns = _sequence_major(rollout.returns)
    raw_advantage = _sequence_major(rollout.advantages)
    train_mask = _sequence_major(rollout.train_mask)
    reset_before = _sequence_major(rollout.reset_before)
    initial_hidden = _hidden_sequence_major(rollout.initial_hidden)
    selected_advantage = raw_advantage[train_mask]
    advantage_mean = selected_advantage.mean()
    advantage_std = selected_advantage.std(unbiased=False).clamp_min(1.0e-8)
    normalized_advantage = (raw_advantage - advantage_mean) / advantage_std

    metric_names = (
        "policy_loss",
        "value_loss",
        "entropy",
        "total_loss",
        "approx_kl",
        "clip_fraction",
        "gradient_norm",
        "post_clip_gradient_norm",
    )
    metrics: dict[str, list[torch.Tensor]] = {name: [] for name in metric_names}
    post_step_kl: list[torch.Tensor] = []
    optimizer_step_index = 0

    for epoch in range(config.epochs):
        permutation = torch.randperm(
            layout.sequence_count, device=rollout.device, generator=generator
        )
        for start in range(0, layout.sequence_count, layout.sequences_per_minibatch):
            sequence_index = permutation[start : start + layout.sequences_per_minibatch]
            batch_observation = observation.index_select(0, sequence_index)
            batch_hidden = initial_hidden.index_select(1, sequence_index)
            batch_reset = reset_before.index_select(0, sequence_index)
            batch_mask = train_mask.index_select(0, sequence_index)
            actor, value, _ = model(
                batch_observation,
                batch_hidden,
                reset_before=batch_reset,
            )
            actor_flat = actor[batch_mask]
            value_flat = value[batch_mask]
            action_flat = action.index_select(0, sequence_index)[batch_mask]
            pre_tanh_flat = pre_tanh.index_select(0, sequence_index)[batch_mask]
            old_log_probability_flat = old_log_probability.index_select(0, sequence_index)[
                batch_mask
            ]
            new_log_probability = hybrid_log_probability(
                actor_flat,
                action_flat,
                config=model.config,
                pre_tanh=pre_tanh_flat,
                distribution_override=distribution_override,
            )
            log_ratio = new_log_probability - old_log_probability_flat
            ratio = torch.exp(log_ratio)
            batch_advantage = normalized_advantage.index_select(0, sequence_index)[batch_mask]
            unclipped = ratio * batch_advantage
            clipped = (
                ratio.clamp(1.0 - config.clip_range, 1.0 + config.clip_range)
                * batch_advantage
            )
            policy_loss = -torch.minimum(unclipped, clipped).mean()
            batch_returns = returns.index_select(0, sequence_index)[batch_mask]
            value_loss = 0.5 * (value_flat - batch_returns).square().mean()
            entropy = hybrid_entropy(
                actor_flat,
                model.config,
                distribution_override=distribution_override,
            ).mean()
            total_loss = (
                policy_loss
                + config.value_loss_coefficient * value_loss
                - config.entropy_coefficient * entropy
            )
            if not bool(torch.isfinite(total_loss).item()):
                raise Rival2RecurrentPPOCorruption(
                    {
                        "reason": "nonfinite_total_loss",
                        "epoch": epoch,
                        "optimizer_step_index": optimizer_step_index,
                        "sequence_minibatch_start": start,
                    }
                )
            optimizer.zero_grad(set_to_none=True)
            total_loss.backward()
            gradient_norm = torch.nn.utils.clip_grad_norm_(
                model.parameters(), config.max_gradient_norm
            )
            gradients = [
                parameter.grad.detach().norm(2)
                for parameter in model.parameters()
                if parameter.grad is not None
            ]
            post_clip_gradient_norm = torch.linalg.vector_norm(torch.stack(gradients), 2)
            if not bool(
                torch.isfinite(gradient_norm).item()
                and torch.isfinite(post_clip_gradient_norm).item()
            ):
                raise Rival2RecurrentPPOCorruption(
                    {
                        "reason": "nonfinite_gradient",
                        "epoch": epoch,
                        "optimizer_step_index": optimizer_step_index,
                    }
                )
            optimizer.step()
            if not _finite_parameters(model):
                raise Rival2RecurrentPPOCorruption(
                    {
                        "reason": "nonfinite_parameter",
                        "epoch": epoch,
                        "optimizer_step_index": optimizer_step_index,
                    }
                )
            with torch.no_grad():
                approx_kl = ((ratio - 1.0) - log_ratio).mean()
                clip_fraction = (
                    (torch.abs(ratio - 1.0) > config.clip_range).to(torch.float32).mean()
                )
                post_actor, _post_value, _ = model(
                    batch_observation,
                    batch_hidden,
                    reset_before=batch_reset,
                )
                post_log_probability = hybrid_log_probability(
                    post_actor[batch_mask],
                    action_flat,
                    config=model.config,
                    pre_tanh=pre_tanh_flat,
                    distribution_override=distribution_override,
                )
                post_log_ratio = post_log_probability - old_log_probability_flat
                post_ratio = torch.exp(post_log_ratio)
                post_step_kl.append(((post_ratio - 1.0) - post_log_ratio).mean())
            for name, value_tensor in (
                ("policy_loss", policy_loss),
                ("value_loss", value_loss),
                ("entropy", entropy),
                ("total_loss", total_loss),
                ("approx_kl", approx_kl),
                ("clip_fraction", clip_fraction),
                ("gradient_norm", gradient_norm),
                ("post_clip_gradient_norm", post_clip_gradient_norm),
            ):
                metrics[name].append(value_tensor.detach())
            optimizer_step_index += 1

    result = {name: torch.stack(values).mean() for name, values in metrics.items()}
    result["optimizer_post_step_approx_kl_mean"] = torch.stack(post_step_kl).mean()
    result["optimizer_post_step_approx_kl_max"] = torch.stack(post_step_kl).amax()
    result["optimizer_steps"] = torch.tensor(
        optimizer_step_index, dtype=torch.int64, device=rollout.device
    )
    result["old_value_mean"] = old_value[train_mask].mean()
    result["return_mean"] = returns[train_mask].mean()
    result["return_std"] = returns[train_mask].std(unbiased=False)
    result["advantage_before_normalization_mean"] = selected_advantage.mean()
    result["advantage_before_normalization_std"] = selected_advantage.std(unbiased=False)
    result.update(
        _completed_diagnostics(
            model,
            observation,
            initial_hidden,
            reset_before,
            action,
            pre_tanh,
            old_log_probability,
            train_mask,
            model.config,
            distribution_override,
            layout.sequences_per_minibatch,
        )
    )
    result["approx_kl"] = result["completed_update_mean_kl"]
    for channel, name in enumerate(ANALOG_ACTION_NAMES):
        result[f"emitted_action_saturation_fraction_{name}"] = (
            (action[..., channel][train_mask].abs() > 0.95).to(torch.float32).mean()
        )
    if any(
        not math.isfinite(float(value.item()))
        for name, value in result.items()
        if name not in {
            "approx_kl",
            "completed_update_mean_kl",
            "completed_update_sample_kl_max",
            "optimizer_post_step_approx_kl_mean",
            "optimizer_post_step_approx_kl_max",
        }
    ):
        raise Rival2RecurrentPPOCorruption({"reason": "nonfinite_non_kl_telemetry"})
    return result


__all__ = [
    "RecurrentSequenceLayout",
    "Rival2RecurrentPPOCorruption",
    "Rival2RecurrentRolloutBuffer",
    "recurrent_ppo_update",
]
