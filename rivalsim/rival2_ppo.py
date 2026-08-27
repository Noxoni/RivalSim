"""GPU-resident rollout storage, GAE, and PPO for Rival 2.0."""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import asdict, dataclass
from typing import Any

import torch

from rivalsim.rival2_contracts import (
    ANALOG_ACTION_NAMES,
    BUTTON_ACTION_NAMES,
    OBS_DIM,
)
from rivalsim.rival2_policy import (
    Rival2ActorCritic,
    Rival2PolicyConfig,
    hybrid_entropy,
    hybrid_log_probability,
)


@dataclass(frozen=True, slots=True)
class Rival2PPOConfig:
    gamma: float = 0.995
    gae_lambda: float = 0.95
    clip_range: float = 0.20
    value_loss_coefficient: float = 0.50
    entropy_coefficient: float = 0.01
    max_gradient_norm: float = 0.50
    learning_rate: float = 3e-4
    epochs: int = 2
    rollout_horizon: int = 32
    minibatch_size: int = 65536

    @property
    def content_hash(self) -> str:
        payload = json.dumps(asdict(self), sort_keys=True, separators=(",", ":")).encode("ascii")
        return hashlib.sha256(payload).hexdigest().upper()


@dataclass(frozen=True, slots=True)
class Rival2KLGuardConfig:
    """Transactional PPO corruption guard; not part of the frozen PPO identity."""

    minibatch_kl_limit: float = 0.10
    completed_update_mean_kl_limit: float = 0.05

    def __post_init__(self) -> None:
        if not math.isfinite(self.minibatch_kl_limit) or self.minibatch_kl_limit <= 0.0:
            raise ValueError("minibatch KL limit must be finite and positive")
        if (
            not math.isfinite(self.completed_update_mean_kl_limit)
            or self.completed_update_mean_kl_limit <= 0.0
        ):
            raise ValueError("completed-update KL limit must be finite and positive")


class Rival2PolicyDisplacementRejected(RuntimeError):
    """Raised only after a guarded PPO update violates its hard safety boundary."""

    def __init__(self, diagnostics: dict[str, Any]):
        self.diagnostics = diagnostics
        super().__init__(
            "Rival 2.0 PPO update rejected: "
            f"{diagnostics.get('reason', 'policy displacement guard')}"
        )


class Rival2RolloutBuffer:
    """Time-major bounded CUDA storage with no host packing."""

    def __init__(
        self,
        horizon: int,
        num_envs: int,
        device: torch.device | str,
        *,
        obs_dim: int = OBS_DIM,
    ):
        if horizon <= 0 or num_envs <= 0:
            raise ValueError("horizon and num_envs must be positive")
        self.horizon = horizon
        self.num_envs = num_envs
        self.device = torch.device(device)
        agent_shape = (horizon, num_envs, 2)
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
        self.policy_version = torch.empty(agent_shape, dtype=torch.int64, device=self.device)
        self.opponent_version = torch.empty(agent_shape, dtype=torch.int64, device=self.device)
        self.train_mask = torch.empty(agent_shape, dtype=torch.bool, device=self.device)
        self.advantages = torch.empty(agent_shape, dtype=torch.float32, device=self.device)
        self.returns = torch.empty(agent_shape, dtype=torch.float32, device=self.device)
        self.position = 0

    @property
    def logical_bytes(self) -> int:
        tensors = (
            self.observations,
            self.actions,
            self.pre_tanh,
            self.old_log_probability,
            self.values,
            self.rewards,
            self.terminated,
            self.truncated,
            self.next_values,
            self.policy_version,
            self.opponent_version,
            self.train_mask,
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
        policy_version: torch.Tensor,
        opponent_version: torch.Tensor,
        train_mask: torch.Tensor,
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
            "policy_version": policy_version,
            "opponent_version": opponent_version,
            "train_mask": train_mask,
        }
        for name, value_tensor in fields.items():
            getattr(self, name)[index].copy_(value_tensor)
        self.position += 1

    def compute_gae(self, config: Rival2PPOConfig) -> tuple[torch.Tensor, torch.Tensor]:
        if self.position != self.horizon:
            raise RuntimeError("GAE requires a complete rollout")
        advantages, returns = compute_gae_gpu(
            self.rewards,
            self.values,
            self.next_values,
            self.terminated,
            self.truncated,
            gamma=config.gamma,
            gae_lambda=config.gae_lambda,
        )
        self.advantages.copy_(advantages)
        self.returns.copy_(returns)
        return self.advantages, self.returns


def compute_gae_gpu(
    rewards: torch.Tensor,
    values: torch.Tensor,
    next_values: torch.Tensor,
    terminated: torch.Tensor,
    truncated: torch.Tensor,
    *,
    gamma: float,
    gae_lambda: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """GAE with terminal zero-bootstrap and truncation final-state bootstrap."""

    if not (
        rewards.shape == values.shape == next_values.shape == terminated.shape == truncated.shape
    ):
        raise ValueError("all GAE tensors must have the same shape")
    advantages = torch.empty_like(rewards)
    carry = torch.zeros_like(rewards[0])
    for time in range(rewards.shape[0] - 1, -1, -1):
        terminal_nonbootstrap = (~terminated[time]).to(rewards.dtype)
        episode_continuation = (~(terminated[time] | truncated[time])).to(rewards.dtype)
        delta = rewards[time] + gamma * next_values[time] * terminal_nonbootstrap - values[time]
        carry = delta + gamma * gae_lambda * episode_continuation * carry
        advantages[time] = carry
    return advantages, advantages + values


@torch.no_grad()
def _completed_update_diagnostics(
    model: Rival2ActorCritic,
    observation: torch.Tensor,
    action: torch.Tensor,
    pre_tanh: torch.Tensor,
    old_log_probability: torch.Tensor,
    policy_config: Rival2PolicyConfig,
    chunk_size: int,
) -> dict[str, torch.Tensor]:
    """Measure the final policy once against every trainable rollout sample."""

    device = observation.device
    count = observation.shape[0]
    scalar_count = torch.tensor(float(count), dtype=torch.float64, device=device)
    mean_sum = torch.zeros(5, dtype=torch.float64, device=device)
    mean_abs_sum = torch.zeros(5, dtype=torch.float64, device=device)
    mean_abs_max = torch.zeros(5, dtype=torch.float32, device=device)
    log_std_sum = torch.zeros(5, dtype=torch.float64, device=device)
    button_probability_sum = torch.zeros(3, dtype=torch.float64, device=device)
    value_sum = torch.zeros((), dtype=torch.float64, device=device)
    value_square_sum = torch.zeros((), dtype=torch.float64, device=device)
    value_abs_max = torch.zeros((), dtype=torch.float32, device=device)
    kl_sum = torch.zeros((), dtype=torch.float64, device=device)
    kl_max = torch.zeros((), dtype=torch.float32, device=device)

    for start in range(0, count, chunk_size):
        stop = min(start + chunk_size, count)
        actor_output, value = model(observation[start:stop])
        mean = actor_output[..., :5]
        log_std = actor_output[..., 5:10].clamp(
            policy_config.log_std_min, policy_config.log_std_max
        )
        probability = torch.sigmoid(actor_output[..., 10:13])
        new_log_probability = hybrid_log_probability(
            actor_output,
            action[start:stop],
            config=policy_config,
            pre_tanh=pre_tanh[start:stop],
        )
        log_ratio = new_log_probability - old_log_probability[start:stop]
        ratio = torch.exp(log_ratio)
        sample_kl = (ratio - 1.0) - log_ratio
        mean_sum += mean.sum(dim=0, dtype=torch.float64)
        mean_abs_sum += mean.abs().sum(dim=0, dtype=torch.float64)
        mean_abs_max = torch.maximum(mean_abs_max, mean.abs().amax(dim=0))
        log_std_sum += log_std.sum(dim=0, dtype=torch.float64)
        button_probability_sum += probability.sum(dim=0, dtype=torch.float64)
        value_sum += value.sum(dtype=torch.float64)
        value_square_sum += value.square().sum(dtype=torch.float64)
        value_abs_max = torch.maximum(value_abs_max, value.abs().amax())
        kl_sum += sample_kl.sum(dtype=torch.float64)
        kl_max = torch.maximum(kl_max, sample_kl.amax())

    value_mean = value_sum / scalar_count
    value_variance = (value_square_sum / scalar_count - value_mean.square()).clamp_min(0.0)
    result: dict[str, torch.Tensor] = {
        "completed_update_mean_kl": (kl_sum / scalar_count).to(torch.float32),
        "completed_update_sample_kl_max": kl_max,
        "post_update_value_mean": value_mean.to(torch.float32),
        "post_update_value_std": torch.sqrt(value_variance).to(torch.float32),
        "post_update_value_max_abs": value_abs_max,
    }
    for channel, name in enumerate(ANALOG_ACTION_NAMES):
        result[f"actor_mean_mean_{name}"] = (mean_sum[channel] / scalar_count).to(torch.float32)
        result[f"actor_mean_abs_mean_{name}"] = (mean_abs_sum[channel] / scalar_count).to(
            torch.float32
        )
        result[f"actor_mean_abs_max_{name}"] = mean_abs_max[channel]
        result[f"actor_log_std_mean_{name}"] = (log_std_sum[channel] / scalar_count).to(
            torch.float32
        )
    for channel, name in enumerate(BUTTON_ACTION_NAMES):
        result[f"actor_button_probability_{name}"] = (
            button_probability_sum[channel] / scalar_count
        ).to(torch.float32)
    return result


def ppo_update(
    model: Rival2ActorCritic,
    optimizer: torch.optim.Optimizer,
    rollout: Rival2RolloutBuffer,
    config: Rival2PPOConfig,
    *,
    generator: torch.Generator,
    policy_config: Rival2PolicyConfig | None = None,
    gae_ready: bool = False,
    kl_guard: Rival2KLGuardConfig | None = None,
) -> dict[str, torch.Tensor]:
    """Run all PPO shuffling, gathers, losses, and optimizer work on CUDA."""

    policy_config = policy_config or model.config
    if not gae_ready:
        rollout.compute_gae(config)
    flat_mask = rollout.train_mask.reshape(-1)
    indices = torch.nonzero(flat_mask, as_tuple=False).squeeze(-1)
    sample_count = indices.shape[0]
    if sample_count == 0:
        raise RuntimeError("PPO rollout contains no trainable samples")

    observation = rollout.observations.reshape(-1, OBS_DIM).index_select(0, indices)
    action = rollout.actions.reshape(-1, 8).index_select(0, indices)
    pre_tanh = rollout.pre_tanh.reshape(-1, 5).index_select(0, indices)
    old_log_probability = rollout.old_log_probability.reshape(-1).index_select(0, indices)
    old_value = rollout.values.reshape(-1).index_select(0, indices)
    returns = rollout.returns.reshape(-1).index_select(0, indices)
    raw_advantage = rollout.advantages.reshape(-1).index_select(0, indices)
    advantage = (raw_advantage - raw_advantage.mean()) / raw_advantage.std(
        unbiased=False
    ).clamp_min(1e-8)

    metrics: dict[str, list[torch.Tensor]] = {
        "policy_loss": [],
        "value_loss": [],
        "entropy": [],
        "total_loss": [],
        "approx_kl": [],
        "clip_fraction": [],
        "gradient_norm": [],
        "post_clip_gradient_norm": [],
    }
    guarded_post_step_kl: list[torch.Tensor] = []
    optimizer_step_index = 0
    for epoch in range(config.epochs):
        permutation = torch.randperm(sample_count, device=rollout.device, generator=generator)
        for start in range(0, sample_count, config.minibatch_size):
            batch = permutation[start : start + config.minibatch_size]
            batch_observation = observation.index_select(0, batch)
            batch_action = action.index_select(0, batch)
            batch_pre_tanh = pre_tanh.index_select(0, batch)
            batch_old_log_probability = old_log_probability.index_select(0, batch)
            actor_output, value = model(batch_observation)
            new_log_probability = hybrid_log_probability(
                actor_output,
                batch_action,
                config=policy_config,
                pre_tanh=batch_pre_tanh,
            )
            log_ratio = new_log_probability - batch_old_log_probability
            ratio = torch.exp(log_ratio)
            batch_advantage = advantage.index_select(0, batch)
            unclipped = ratio * batch_advantage
            clipped = (
                ratio.clamp(1.0 - config.clip_range, 1.0 + config.clip_range) * batch_advantage
            )
            policy_loss = -torch.minimum(unclipped, clipped).mean()
            value_loss = 0.5 * (value - returns.index_select(0, batch)).square().mean()
            entropy = hybrid_entropy(actor_output, policy_config).mean()
            total_loss = (
                policy_loss
                + config.value_loss_coefficient * value_loss
                - config.entropy_coefficient * entropy
            )
            if kl_guard is not None and not bool(torch.isfinite(total_loss).item()):
                raise Rival2PolicyDisplacementRejected(
                    {
                        "reason": "nonfinite_total_loss",
                        "epoch": epoch,
                        "optimizer_step_index": optimizer_step_index,
                        "minibatch_start": start,
                        "minibatch_samples": int(batch.shape[0]),
                        "minibatch_kl_limit": kl_guard.minibatch_kl_limit,
                        "completed_update_mean_kl_limit": (kl_guard.completed_update_mean_kl_limit),
                    }
                )
            optimizer.zero_grad(set_to_none=True)
            total_loss.backward()
            gradient_norm = torch.nn.utils.clip_grad_norm_(
                model.parameters(), config.max_gradient_norm
            )
            post_clip_gradient_norm = torch.linalg.vector_norm(
                torch.stack(
                    [
                        parameter.grad.detach().norm(2)
                        for parameter in model.parameters()
                        if parameter.grad is not None
                    ]
                ),
                2,
            )
            if kl_guard is not None and not bool(
                torch.isfinite(gradient_norm) & torch.isfinite(post_clip_gradient_norm)
            ):
                raise Rival2PolicyDisplacementRejected(
                    {
                        "reason": "nonfinite_gradient",
                        "epoch": epoch,
                        "optimizer_step_index": optimizer_step_index,
                        "minibatch_start": start,
                        "minibatch_samples": int(batch.shape[0]),
                        "raw_gradient_norm": float(gradient_norm.item()),
                        "post_clip_gradient_norm": float(post_clip_gradient_norm.item()),
                        "minibatch_kl_limit": kl_guard.minibatch_kl_limit,
                        "completed_update_mean_kl_limit": (kl_guard.completed_update_mean_kl_limit),
                    }
                )
            optimizer.step()

            with torch.no_grad():
                approx_kl = ((ratio - 1.0) - log_ratio).mean()
                clip_fraction = (
                    (torch.abs(ratio - 1.0) > config.clip_range).to(torch.float32).mean()
                )
                if kl_guard is not None:
                    post_actor_output, _post_value = model(batch_observation)
                    post_log_probability = hybrid_log_probability(
                        post_actor_output,
                        batch_action,
                        config=policy_config,
                        pre_tanh=batch_pre_tanh,
                    )
                    post_log_ratio = post_log_probability - batch_old_log_probability
                    post_ratio = torch.exp(post_log_ratio)
                    post_step_kl = ((post_ratio - 1.0) - post_log_ratio).mean()
                    guarded_post_step_kl.append(post_step_kl.detach())
                    post_step_kl_value = float(post_step_kl.item())
                    if (
                        not math.isfinite(post_step_kl_value)
                        or post_step_kl_value > kl_guard.minibatch_kl_limit
                    ):
                        raise Rival2PolicyDisplacementRejected(
                            {
                                "reason": "minibatch_kl_limit_exceeded",
                                "epoch": epoch,
                                "optimizer_step_index": optimizer_step_index,
                                "minibatch_start": start,
                                "minibatch_samples": int(batch.shape[0]),
                                "pre_step_approx_kl": float(approx_kl.item()),
                                "post_step_approx_kl": post_step_kl_value,
                                "policy_loss": float(policy_loss.item()),
                                "value_loss": float(value_loss.item()),
                                "entropy_diagnostic": float(entropy.item()),
                                "total_loss": float(total_loss.item()),
                                "raw_gradient_norm": float(gradient_norm.item()),
                                "post_clip_gradient_norm": float(post_clip_gradient_norm.item()),
                                "minibatch_kl_limit": kl_guard.minibatch_kl_limit,
                                "completed_update_mean_kl_limit": (
                                    kl_guard.completed_update_mean_kl_limit
                                ),
                            }
                        )
            metrics["policy_loss"].append(policy_loss.detach())
            metrics["value_loss"].append(value_loss.detach())
            metrics["entropy"].append(entropy.detach())
            metrics["total_loss"].append(total_loss.detach())
            metrics["approx_kl"].append(approx_kl.detach())
            metrics["clip_fraction"].append(clip_fraction.detach())
            metrics["gradient_norm"].append(gradient_norm.detach())
            metrics["post_clip_gradient_norm"].append(post_clip_gradient_norm.detach())
            optimizer_step_index += 1

    result = {name: torch.stack(values).mean() for name, values in metrics.items()}
    result["old_value_mean"] = old_value.mean()
    if kl_guard is not None:
        result["optimizer_pre_step_approx_kl_mean"] = result["approx_kl"]
        result["optimizer_post_step_approx_kl_mean"] = torch.stack(guarded_post_step_kl).mean()
        result["optimizer_post_step_approx_kl_max"] = torch.stack(guarded_post_step_kl).amax()
        completed = _completed_update_diagnostics(
            model,
            observation,
            action,
            pre_tanh,
            old_log_probability,
            policy_config,
            config.minibatch_size,
        )
        result.update(completed)
        result["approx_kl"] = completed["completed_update_mean_kl"]
        result["predicted_value_mean"] = old_value.mean()
        result["predicted_value_std"] = old_value.std(unbiased=False)
        result["predicted_value_max_abs"] = old_value.abs().amax()
        result["return_mean"] = returns.mean()
        result["return_std"] = returns.std(unbiased=False)
        result["return_max_abs"] = returns.abs().amax()
        result["advantage_before_normalization_mean"] = raw_advantage.mean()
        result["advantage_before_normalization_std"] = raw_advantage.std(unbiased=False)
        result["advantage_before_normalization_max_abs"] = raw_advantage.abs().amax()
        for channel, name in enumerate(ANALOG_ACTION_NAMES):
            result[f"emitted_action_saturation_fraction_{name}"] = (
                (action[:, channel].abs() > 0.95).to(torch.float32).mean()
            )
        completed_kl = float(result["completed_update_mean_kl"].item())
        if (
            not math.isfinite(completed_kl)
            or completed_kl > kl_guard.completed_update_mean_kl_limit
        ):
            raise Rival2PolicyDisplacementRejected(
                {
                    "reason": "completed_update_mean_kl_limit_exceeded",
                    "optimizer_steps_completed": optimizer_step_index,
                    "completed_update_mean_kl": completed_kl,
                    "completed_update_sample_kl_max": float(
                        result["completed_update_sample_kl_max"].item()
                    ),
                    "optimizer_post_step_approx_kl_mean": float(
                        result["optimizer_post_step_approx_kl_mean"].item()
                    ),
                    "optimizer_post_step_approx_kl_max": float(
                        result["optimizer_post_step_approx_kl_max"].item()
                    ),
                    "policy_loss": float(result["policy_loss"].item()),
                    "value_loss": float(result["value_loss"].item()),
                    "raw_gradient_norm": float(result["gradient_norm"].item()),
                    "post_clip_gradient_norm": float(result["post_clip_gradient_norm"].item()),
                    "minibatch_kl_limit": kl_guard.minibatch_kl_limit,
                    "completed_update_mean_kl_limit": (kl_guard.completed_update_mean_kl_limit),
                }
            )
    return result


@torch.no_grad()
def evaluate_clipped_policy_objective(
    model: Rival2ActorCritic,
    rollout: Rival2RolloutBuffer,
    config: Rival2PPOConfig,
    *,
    policy_config: Rival2PolicyConfig | None = None,
) -> dict[str, torch.Tensor]:
    """Evaluate one frozen rollout without gradients or optimizer mutation."""

    policy_config = policy_config or model.config
    rollout.compute_gae(config)
    indices = torch.nonzero(rollout.train_mask.reshape(-1), as_tuple=False).squeeze(-1)
    observation = rollout.observations.reshape(-1, OBS_DIM).index_select(0, indices)
    action = rollout.actions.reshape(-1, 8).index_select(0, indices)
    pre_tanh = rollout.pre_tanh.reshape(-1, 5).index_select(0, indices)
    old_log_probability = rollout.old_log_probability.reshape(-1).index_select(0, indices)
    advantage = rollout.advantages.reshape(-1).index_select(0, indices)
    advantage = (advantage - advantage.mean()) / advantage.std(unbiased=False).clamp_min(1e-8)
    actor_output, value = model(observation)
    log_ratio = (
        hybrid_log_probability(
            actor_output,
            action,
            config=policy_config,
            pre_tanh=pre_tanh,
        )
        - old_log_probability
    )
    ratio = torch.exp(log_ratio)
    objective_samples = torch.minimum(
        ratio * advantage,
        ratio.clamp(1.0 - config.clip_range, 1.0 + config.clip_range) * advantage,
    )
    objective = objective_samples.mean()
    change = objective_samples - advantage
    return {
        "clipped_policy_objective": objective,
        "change_from_behavior": change.mean(),
        "change_standard_error": change.std(unbiased=False)
        / torch.sqrt(torch.full((), change.numel(), dtype=torch.float32, device=change.device)),
        "mean_log_ratio": log_ratio.mean(),
        "approx_kl": ((ratio - 1.0) - log_ratio).mean(),
        "clip_fraction": ((torch.abs(ratio - 1.0) > config.clip_range).to(torch.float32).mean()),
        "value_mse": 0.5
        * (value - rollout.returns.reshape(-1).index_select(0, indices)).square().mean(),
    }


__all__ = [
    "Rival2KLGuardConfig",
    "Rival2PPOConfig",
    "Rival2PolicyDisplacementRejected",
    "Rival2RolloutBuffer",
    "compute_gae_gpu",
    "evaluate_clipped_policy_objective",
    "ppo_update",
]
