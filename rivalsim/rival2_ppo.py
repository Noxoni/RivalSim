"""GPU-resident rollout storage, GAE, and PPO for Rival 2.0."""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass

import torch

from rivalsim.rival2_contracts import OBS_DIM
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


def ppo_update(
    model: Rival2ActorCritic,
    optimizer: torch.optim.Optimizer,
    rollout: Rival2RolloutBuffer,
    config: Rival2PPOConfig,
    *,
    generator: torch.Generator,
    policy_config: Rival2PolicyConfig | None = None,
    gae_ready: bool = False,
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
    advantage = rollout.advantages.reshape(-1).index_select(0, indices)
    advantage = (advantage - advantage.mean()) / advantage.std(unbiased=False).clamp_min(1e-8)

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
    for _ in range(config.epochs):
        permutation = torch.randperm(sample_count, device=rollout.device, generator=generator)
        for start in range(0, sample_count, config.minibatch_size):
            batch = permutation[start : start + config.minibatch_size]
            actor_output, value = model(observation.index_select(0, batch))
            new_log_probability = hybrid_log_probability(
                actor_output,
                action.index_select(0, batch),
                config=policy_config,
                pre_tanh=pre_tanh.index_select(0, batch),
            )
            log_ratio = new_log_probability - old_log_probability.index_select(0, batch)
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
            optimizer.step()

            with torch.no_grad():
                approx_kl = ((ratio - 1.0) - log_ratio).mean()
                clip_fraction = (
                    (torch.abs(ratio - 1.0) > config.clip_range).to(torch.float32).mean()
                )
            metrics["policy_loss"].append(policy_loss.detach())
            metrics["value_loss"].append(value_loss.detach())
            metrics["entropy"].append(entropy.detach())
            metrics["total_loss"].append(total_loss.detach())
            metrics["approx_kl"].append(approx_kl.detach())
            metrics["clip_fraction"].append(clip_fraction.detach())
            metrics["gradient_norm"].append(gradient_norm.detach())
            metrics["post_clip_gradient_norm"].append(post_clip_gradient_norm.detach())

    result = {name: torch.stack(values).mean() for name, values in metrics.items()}
    result["old_value_mean"] = old_value.mean()
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
    "Rival2PPOConfig",
    "Rival2RolloutBuffer",
    "compute_gae_gpu",
    "evaluate_clipped_policy_objective",
    "ppo_update",
]
