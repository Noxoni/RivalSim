"""End-to-end GPU-native Rival 2.0 self-play trainer."""

from __future__ import annotations

import copy
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import torch

from rivalsim.rival2_contracts import contract_hashes_for_reward
from rivalsim.rival2_env import Rival2Env
from rivalsim.rival2_policy import (
    Rival2ActorCritic,
    Rival2PolicyConfig,
    deterministic_hybrid_action,
    sample_hybrid_action,
)
from rivalsim.rival2_ppo import (
    Rival2PPOConfig,
    Rival2RolloutBuffer,
    ppo_update,
)


@dataclass(frozen=True, slots=True)
class Rival2SelfPlayConfig:
    historical_chance: float = 0.20
    historical_pool_bound: int = 16

    def __post_init__(self) -> None:
        if not 0.0 <= self.historical_chance <= 1.0:
            raise ValueError("historical_chance must be in [0,1]")
        if self.historical_pool_bound <= 0:
            raise ValueError("historical_pool_bound must be positive")


class HistoricalPolicyPool:
    """Bounded frozen GPU policy pool; inference is grouped by resident version."""

    def __init__(
        self,
        policy_config: Rival2PolicyConfig,
        device: torch.device,
        bound: int = 16,
    ):
        self.policy_config = policy_config
        self.device = device
        self.bound = bound
        self.versions: list[int] = []
        self.policies: list[Rival2ActorCritic] = []
        self.version_tensor = torch.empty(0, dtype=torch.int64, device=device)

    def _refresh_version_tensor(self) -> None:
        self.version_tensor = torch.as_tensor(self.versions, dtype=torch.int64, device=self.device)

    def add(self, model: Rival2ActorCritic, version: int) -> int | None:
        frozen = copy.deepcopy(model).to(self.device).eval()
        frozen.requires_grad_(False)
        self.versions.append(int(version))
        self.policies.append(frozen)
        evicted_version: int | None = None
        if len(self.versions) > self.bound:
            evicted_version = self.versions.pop(0)
            self.policies.pop(0)
        self._refresh_version_tensor()
        return evicted_version

    def clear(self) -> None:
        self.versions.clear()
        self.policies.clear()
        self._refresh_version_tensor()

    def checkpoint_state(self) -> list[dict[str, Any]]:
        return [
            {"version": version, "model": policy.state_dict()}
            for version, policy in zip(self.versions, self.policies, strict=True)
        ]

    def load_checkpoint_state(self, entries: list[dict[str, Any]]) -> None:
        self.clear()
        for entry in entries:
            policy = Rival2ActorCritic(self.policy_config).to(self.device)
            policy.load_state_dict(entry["model"])
            policy.eval().requires_grad_(False)
            self.versions.append(int(entry["version"]))
            self.policies.append(policy)
        self._refresh_version_tensor()


class Rival2Trainer:
    """Own the current policy, resident opponent state, rollouts, and PPO."""

    def __init__(
        self,
        env: Rival2Env,
        *,
        policy_config: Rival2PolicyConfig | None = None,
        ppo_config: Rival2PPOConfig | None = None,
        self_play_config: Rival2SelfPlayConfig | None = None,
        seed: int = 20260825,
    ):
        self.env = env
        self.device = env.device
        self.policy_config = policy_config or Rival2PolicyConfig()
        self.ppo_config = ppo_config or Rival2PPOConfig()
        self.self_play_config = self_play_config or Rival2SelfPlayConfig()
        self.model = Rival2ActorCritic(self.policy_config).to(self.device)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.ppo_config.learning_rate)
        self.policy_generator = torch.Generator(device=self.device).manual_seed(seed)
        self.opponent_generator = torch.Generator(device=self.device).manual_seed(
            seed ^ 0x5A17A5EED
        )
        self.opponent_pool = HistoricalPolicyPool(
            self.policy_config,
            self.device,
            self.self_play_config.historical_pool_bound,
        )
        self.opponent_assignment = torch.full(
            (env.num_envs,), -1, dtype=torch.int64, device=self.device
        )
        self.policy_version = 0
        self.iteration = 0
        self.total_agent_samples = 0
        self.curriculum_transition: dict[str, Any] | None = None

    def add_historical_snapshot(self) -> None:
        evicted_version = self.opponent_pool.add(self.model, self.policy_version)
        if evicted_version is not None:
            self.opponent_assignment.masked_fill_(
                self.opponent_assignment == evicted_version,
                -1,
            )

    def assign_opponents_at_reset(self, reset_mask: torch.Tensor) -> None:
        """Change opponent version only for reset worlds, entirely on CUDA."""

        if not self.opponent_pool.versions:
            self.opponent_assignment.masked_fill_(reset_mask, -1)
            return
        historical = (
            torch.rand(
                (self.env.num_envs,),
                device=self.device,
                generator=self.opponent_generator,
            )
            < self.self_play_config.historical_chance
        )
        pool_index = torch.randint(
            len(self.opponent_pool.versions),
            (self.env.num_envs,),
            device=self.device,
            generator=self.opponent_generator,
        )
        selected = torch.where(
            historical,
            self.opponent_pool.version_tensor.index_select(0, pool_index),
            -1,
        )
        self.opponent_assignment.copy_(torch.where(reset_mask, selected, self.opponent_assignment))

    def _policy_outputs(
        self, observation: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        worlds = observation.shape[0]
        current_actor, current_value = self.model(observation.reshape(-1, observation.shape[-1]))
        actor = current_actor.reshape(worlds, 2, 13)
        value = current_value.reshape(worlds, 2)
        acting_version = torch.full(
            (worlds, 2), self.policy_version, dtype=torch.int64, device=self.device
        )
        train_mask = torch.ones((worlds, 2), dtype=torch.bool, device=self.device)
        train_mask[:, 1] = self.opponent_assignment < 0
        acting_version[:, 1] = torch.where(
            self.opponent_assignment >= 0,
            self.opponent_assignment,
            acting_version[:, 1],
        )
        for version, policy in zip(
            self.opponent_pool.versions, self.opponent_pool.policies, strict=True
        ):
            indices = torch.nonzero(self.opponent_assignment == version, as_tuple=False).squeeze(-1)
            historical_actor, historical_value = policy(observation[:, 1].index_select(0, indices))
            actor[:, 1].index_copy_(0, indices, historical_actor)
            value[:, 1].index_copy_(0, indices, historical_value)
        return actor, value, acting_version, train_mask

    def collect_rollout(self) -> Rival2RolloutBuffer:
        config = self.ppo_config
        rollout = Rival2RolloutBuffer(config.rollout_horizon, self.env.num_envs, self.device)
        observation = self.env.observation
        self.model.eval()
        for _ in range(config.rollout_horizon):
            with torch.no_grad():
                actor, value, acting_version, train_mask = self._policy_outputs(observation)
                sample = sample_hybrid_action(
                    actor, generator=self.policy_generator, config=self.policy_config
                )
                transition = self.env.step(sample.action)
                _, transition_value = self.model(
                    transition.transition_observation.reshape(-1, self.policy_config.obs_dim)
                )
                next_value = transition_value.reshape(self.env.num_envs, 2)
                terminated = transition.terminated[:, None].expand(-1, 2)
                truncated = transition.truncated[:, None].expand(-1, 2)
                opponent_version = torch.empty_like(acting_version)
                opponent_version[:, 0] = acting_version[:, 1]
                opponent_version[:, 1] = acting_version[:, 0]
                rollout.add(
                    observation=observation,
                    action=transition.emitted_action,
                    pre_tanh=sample.pre_tanh,
                    old_log_probability=sample.log_probability,
                    value=value,
                    reward=transition.reward,
                    terminated=terminated,
                    truncated=truncated,
                    next_value=next_value,
                    policy_version=acting_version,
                    opponent_version=opponent_version,
                    train_mask=train_mask,
                )
                self.assign_opponents_at_reset(transition.reset_mask)
                observation = transition.observation
        self.env.observation = observation
        self.total_agent_samples += config.rollout_horizon * self.env.num_envs * 2
        return rollout

    def update(self, rollout: Rival2RolloutBuffer) -> dict[str, torch.Tensor]:
        self.model.train()
        metrics = ppo_update(
            self.model,
            self.optimizer,
            rollout,
            self.ppo_config,
            generator=self.policy_generator,
            policy_config=self.policy_config,
        )
        self.policy_version += 1
        self.iteration += 1
        return metrics

    def train_iteration(self) -> tuple[Rival2RolloutBuffer, dict[str, torch.Tensor]]:
        rollout = self.collect_rollout()
        return rollout, self.update(rollout)

    @torch.no_grad()
    def deterministic_action_value(
        self, observation: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        self.model.eval()
        actor, value = self.model(observation)
        return deterministic_hybrid_action(actor, self.policy_config), value

    def checkpoint_payload(self) -> dict[str, Any]:
        payload = {
            "format": "RIVAL2_CHECKPOINT_V1",
            "model": self.model.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "policy_config": asdict(self.policy_config),
            "ppo_config": asdict(self.ppo_config),
            "self_play_config": asdict(self.self_play_config),
            "contract_hashes": dict(self.env.contract_hashes),
            "reward_version": self.env.reward_version,
            "episode_version": self.env.episode_version,
            "policy_config_hash": self.policy_config.content_hash,
            "ppo_config_hash": self.ppo_config.content_hash,
            "policy_version": self.policy_version,
            "iteration": self.iteration,
            "total_agent_samples": self.total_agent_samples,
            "torch_cpu_rng_state": torch.get_rng_state(),
            "torch_cuda_rng_state": torch.cuda.get_rng_state(self.device),
            "policy_generator_state": self.policy_generator.get_state(),
            "opponent_generator_state": self.opponent_generator.get_state(),
            "opponent_assignment": self.opponent_assignment,
            "historical_opponents": self.opponent_pool.checkpoint_state(),
        }
        if self.curriculum_transition is not None:
            payload["curriculum_transition"] = copy.deepcopy(self.curriculum_transition)
        return payload

    def save_checkpoint(self, path: str | Path) -> None:
        torch.save(self.checkpoint_payload(), Path(path))

    def load_checkpoint(self, path: str | Path) -> None:
        payload = torch.load(Path(path), map_location=self.device, weights_only=False)
        if payload.get("format") != "RIVAL2_CHECKPOINT_V1":
            raise ValueError("unsupported Rival 2.0 checkpoint format")
        if payload["contract_hashes"] != self.env.contract_hashes:
            raise ValueError("Rival 2.0 checkpoint contract hashes are incompatible")
        if payload.get("reward_version", self.env.reward_version) != self.env.reward_version:
            raise ValueError("Rival 2.0 checkpoint reward version is incompatible")
        if payload.get("episode_version", "RIVAL2_EPISODE_V1") != self.env.episode_version:
            raise ValueError("Rival 2.0 checkpoint episode version is incompatible")
        if payload["policy_config_hash"] != self.policy_config.content_hash:
            raise ValueError("Rival 2.0 checkpoint policy configuration is incompatible")
        if payload["ppo_config_hash"] != self.ppo_config.content_hash:
            raise ValueError("Rival 2.0 checkpoint PPO configuration is incompatible")
        self.model.load_state_dict(payload["model"])
        self.optimizer.load_state_dict(payload["optimizer"])
        self.policy_version = int(payload["policy_version"])
        self.iteration = int(payload["iteration"])
        self.total_agent_samples = int(payload["total_agent_samples"])
        self.self_play_config = Rival2SelfPlayConfig(**payload["self_play_config"])
        self.opponent_pool.bound = self.self_play_config.historical_pool_bound
        torch.set_rng_state(payload["torch_cpu_rng_state"].cpu())
        torch.cuda.set_rng_state(payload["torch_cuda_rng_state"].cpu(), self.device)
        self.policy_generator.set_state(payload["policy_generator_state"].cpu())
        self.opponent_generator.set_state(payload["opponent_generator_state"].cpu())
        self.opponent_assignment.copy_(payload["opponent_assignment"])
        self.opponent_pool.load_checkpoint_state(payload["historical_opponents"])
        self.curriculum_transition = copy.deepcopy(payload.get("curriculum_transition"))

    def transition_reward_curriculum(
        self,
        *,
        source_reward_version: str,
        destination_reward_version: str,
        transition_record: dict[str, Any],
    ) -> dict[str, Any]:
        """Apply one explicit reward-only curriculum transition in place.

        Ordinary checkpoint loading remains contract-strict. This separate path is for an
        authorized update-boundary curriculum migration where model, optimizer, RNG, counters,
        opponent assignments, historical policies, and live world state must remain untouched.
        """

        if self.curriculum_transition is not None:
            raise ValueError("a Rival 2.0 curriculum transition is already recorded")
        if source_reward_version == destination_reward_version:
            raise ValueError("curriculum reward source and destination must differ")
        source_contracts = contract_hashes_for_reward(
            source_reward_version, self.env.episode_version
        )
        destination_contracts = contract_hashes_for_reward(
            destination_reward_version, self.env.episode_version
        )
        if self.env.reward_version != source_reward_version:
            raise ValueError("active Rival 2.0 reward does not match the transition source")
        if self.env.contract_hashes != source_contracts:
            raise ValueError("active Rival 2.0 contracts do not match the transition source")

        source_shared = {
            name: digest
            for name, digest in source_contracts.items()
            if name != source_reward_version
        }
        destination_shared = {
            name: digest
            for name, digest in destination_contracts.items()
            if name != destination_reward_version
        }
        if source_shared != destination_shared:
            raise ValueError("curriculum transition would change a non-reward contract")
        if not transition_record:
            raise ValueError("curriculum transition requires an explicit authority record")

        recorded = {
            **copy.deepcopy(transition_record),
            "source_reward_version": source_reward_version,
            "source_contract_hashes": source_contracts,
            "destination_reward_version": destination_reward_version,
            "destination_contract_hashes": destination_contracts,
            "transition_iteration": self.iteration,
            "transition_policy_version": self.policy_version,
            "transition_agent_decision_samples": self.total_agent_samples,
            "changed_semantics": ["reward_contract"],
            "preserved_runtime_state": [
                "model",
                "optimizer",
                "torch_cpu_rng",
                "torch_cuda_rng",
                "policy_generator_rng",
                "opponent_generator_rng",
                "policy_and_sample_counters",
                "opponent_assignment",
                "historical_policy_pool",
                "live_world_state",
            ],
        }
        set_reward_version = getattr(self.env, "set_reward_version", None)
        if set_reward_version is None:
            self.env.reward_version = destination_reward_version
            self.env.contract_hashes = destination_contracts
        else:
            set_reward_version(destination_reward_version)
            if self.env.contract_hashes != destination_contracts:
                raise RuntimeError("reward transition installed unexpected contracts")
        self.curriculum_transition = recorded
        return copy.deepcopy(recorded)


__all__ = [
    "HistoricalPolicyPool",
    "Rival2SelfPlayConfig",
    "Rival2Trainer",
]
