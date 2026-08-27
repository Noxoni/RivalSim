"""Bounded Gameplay V2 mixed-opponent training runtime."""

from __future__ import annotations

import copy
import hashlib
from dataclasses import asdict, dataclass
from typing import Any

import numpy as np
import torch

from rivalsim.rival2_env import Rival2Env
from rivalsim.rival2_mixed_ppo import (
    Rival2MixedPPOSafetyConfig,
    build_retention_observation_corpus,
    make_empty_mixed_optimizer,
    migrate_adam_to_mixed_groups,
    mixed_optimizer_learning_rates,
    ppo_update_mixed_curriculum,
    retention_observation_sha256,
)
from rivalsim.rival2_policy import sample_hybrid_action
from rivalsim.rival2_ppo import (
    Rival2KLGuardConfig,
    Rival2PolicyDisplacementRejected,
    Rival2RolloutBuffer,
)
from rivalsim.rival2_training import Rival2Trainer
from third_party.nexto.adapter import NextoPolicyAdapter, NextoStateTensors
from third_party.wisp75b.adapter import WispPolicyAdapter, WispStateTensors

OPPONENT_CURRENT = 0
OPPONENT_HISTORICAL = 1
OPPONENT_NEXTO = 2
OPPONENT_WISP = 3
OPPONENT_NAMES = ("current", "historical", "nexto", "wisp")
NEXTO_ACTING_VERSION = -2
WISP_ACTING_VERSION = -3


@dataclass(frozen=True, slots=True)
class Rival2OpponentCurriculumConfig:
    nexto_probability: float = 0.35
    wisp_probability: float = 0.35
    current_probability: float = 0.20
    historical_probability: float = 0.10
    seed: int = 2026082703

    def __post_init__(self) -> None:
        probabilities = (
            self.nexto_probability,
            self.wisp_probability,
            self.current_probability,
            self.historical_probability,
        )
        if any(value < 0.0 for value in probabilities):
            raise ValueError("opponent probabilities must be non-negative")
        if abs(sum(probabilities) - 1.0) > 1.0e-12:
            raise ValueError("opponent probabilities must sum to one")


class Rival2OpponentCurriculumTrainer(Rival2Trainer):
    """Rival trainer with episode-fixed frozen/current opponent families."""

    def __init__(
        self,
        env: Rival2Env,
        *,
        opponent_curriculum: Rival2OpponentCurriculumConfig | None = None,
        **kwargs: Any,
    ):
        super().__init__(env, **kwargs)
        self.opponent_curriculum = opponent_curriculum or Rival2OpponentCurriculumConfig()
        self.curriculum_generator = torch.Generator(device=self.device).manual_seed(
            self.opponent_curriculum.seed
        )
        self.opponent_family = torch.full(
            (env.num_envs,), -1, dtype=torch.int64, device=self.device
        )
        self.rival_side = torch.zeros(env.num_envs, dtype=torch.int64, device=self.device)
        self.realized_family_assignments = torch.zeros(4, dtype=torch.int64, device=self.device)
        self.nexto = NextoPolicyAdapter(env.num_envs, device=self.device)
        self.wisp = WispPolicyAdapter(env.num_envs, device=self.device)
        self.nexto_state = NextoStateTensors.from_bridge(env.bridge)
        self.wisp_state = WispStateTensors.from_bridge(env.bridge)
        self._world_rows = torch.arange(env.num_envs, device=self.device)
        self.last_rollout_curriculum_metrics: dict[str, Any] | None = None
        self.mixed_ppo_safety: Rival2MixedPPOSafetyConfig | None = None
        self.optimizer_migration_proof: dict[str, Any] | None = None
        self.retention_observations: torch.Tensor | None = None
        self.retention_corpus_summary: dict[str, Any] | None = None
        self.last_adaptive_ppo_diagnostics: dict[str, Any] | None = None

    def initialize_curriculum_assignments(self) -> None:
        if bool((self.opponent_family >= 0).any()):
            raise ValueError("opponent curriculum assignments are already initialized")
        source_assignment = self.opponent_assignment.detach().cpu()
        self.assign_opponents_at_reset(
            torch.ones(self.env.num_envs, dtype=torch.bool, device=self.device)
        )
        if self.curriculum_transition is not None:
            generator_state = self.curriculum_generator.get_state().cpu().numpy().tobytes()
            self.curriculum_transition["opponent_curriculum_initialization"] = {
                "config": asdict(self.opponent_curriculum),
                "source_assignment_current_count": int((source_assignment < 0).sum()),
                "source_assignment_historical_count": int((source_assignment >= 0).sum()),
                "source_assignment_replaced_at_fresh_episode_boundary": True,
                "dedicated_generator_state_sha256_after_initialization": hashlib.sha256(
                    generator_state
                )
                .hexdigest()
                .upper(),
            }

    def assign_opponents_at_reset(self, reset_mask: torch.Tensor) -> None:
        """Prospectively sample family and side only at episode boundaries."""

        if (
            reset_mask.shape != (self.env.num_envs,)
            or reset_mask.dtype != torch.bool
            or reset_mask.device != self.device
        ):
            raise ValueError("opponent reset mask shape/device mismatch")
        if not bool(reset_mask.any()):
            return
        old_family = self.opponent_family.clone()
        draw = torch.rand(
            self.env.num_envs,
            device=self.device,
            generator=self.curriculum_generator,
        )
        nexto_end = self.opponent_curriculum.nexto_probability
        wisp_end = nexto_end + self.opponent_curriculum.wisp_probability
        current_end = wisp_end + self.opponent_curriculum.current_probability
        sampled_family = torch.where(
            draw < nexto_end,
            OPPONENT_NEXTO,
            torch.where(
                draw < wisp_end,
                OPPONENT_WISP,
                torch.where(draw < current_end, OPPONENT_CURRENT, OPPONENT_HISTORICAL),
            ),
        ).to(torch.int64)
        sampled_side = torch.randint(
            2,
            (self.env.num_envs,),
            device=self.device,
            generator=self.curriculum_generator,
        )
        self.opponent_family.copy_(torch.where(reset_mask, sampled_family, self.opponent_family))
        self.rival_side.copy_(torch.where(reset_mask, sampled_side, self.rival_side))

        if not self.opponent_pool.versions:
            self.opponent_family.copy_(
                torch.where(
                    reset_mask & (self.opponent_family == OPPONENT_HISTORICAL),
                    torch.full_like(self.opponent_family, OPPONENT_CURRENT),
                    self.opponent_family,
                )
            )
            self.opponent_assignment.masked_fill_(reset_mask, -1)
        else:
            pool_index = torch.randint(
                len(self.opponent_pool.versions),
                (self.env.num_envs,),
                device=self.device,
                generator=self.curriculum_generator,
            )
            version = self.opponent_pool.version_tensor.index_select(0, pool_index)
            historical = reset_mask & (self.opponent_family == OPPONENT_HISTORICAL)
            selected = torch.where(
                historical,
                version,
                torch.full_like(version, -1),
            )
            self.opponent_assignment.copy_(
                torch.where(reset_mask, selected, self.opponent_assignment)
            )

        opponent_side = 1 - self.rival_side
        self.nexto.set_player_index(opponent_side)
        self.wisp.set_player_index(opponent_side)
        nexto_reset = reset_mask & (self.opponent_family == OPPONENT_NEXTO)
        new_nexto = nexto_reset & (old_family != OPPONENT_NEXTO)
        if bool(new_nexto.any()):
            self.nexto.activate(new_nexto)
        self.nexto.notify_kickoff(nexto_reset)
        # A RivalSim short-episode reset is a fresh Wisp episode.  The pinned
        # source requires all Wisp observation/action-delay history to reset at
        # that boundary even when Wisp happens to be sampled again.
        wisp_reset = reset_mask & (self.opponent_family == OPPONENT_WISP)
        if bool(wisp_reset.any()):
            self.wisp.activate(wisp_reset)

        counts = torch.bincount(self.opponent_family[reset_mask], minlength=4).to(torch.int64)
        self.realized_family_assignments.add_(counts)

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
        train_mask = torch.zeros((worlds, 2), dtype=torch.bool, device=self.device)
        current = self.opponent_family == OPPONENT_CURRENT
        train_mask[current] = True
        non_current = ~current
        train_mask[self._world_rows[non_current], self.rival_side[non_current]] = True
        opponent_side = 1 - self.rival_side
        acting_version[
            self._world_rows[self.opponent_family == OPPONENT_NEXTO],
            opponent_side[self.opponent_family == OPPONENT_NEXTO],
        ] = NEXTO_ACTING_VERSION
        acting_version[
            self._world_rows[self.opponent_family == OPPONENT_WISP],
            opponent_side[self.opponent_family == OPPONENT_WISP],
        ] = WISP_ACTING_VERSION

        for version, policy in zip(
            self.opponent_pool.versions, self.opponent_pool.policies, strict=True
        ):
            selected = (self.opponent_family == OPPONENT_HISTORICAL) & (
                self.opponent_assignment == version
            )
            indices = torch.nonzero(selected, as_tuple=False).squeeze(-1)
            if indices.numel() == 0:
                continue
            sides = opponent_side.index_select(0, indices)
            historical_actor, historical_value = policy(observation[indices, sides])
            actor[indices, sides] = historical_actor
            value[indices, sides] = historical_value
            acting_version[indices, sides] = version
        return actor, value, acting_version, train_mask

    def _step_with_frozen_opponents(
        self,
        base_action: torch.Tensor,
    ):
        nexto_mask = self.opponent_family == OPPONENT_NEXTO
        wisp_mask = self.opponent_family == OPPONENT_WISP
        opponent_side = 1 - self.rival_side

        def tick_action(_tick: int) -> torch.Tensor:
            actions = base_action.clone()
            ball = self.nexto_state.ball_pos
            kickoff = (ball[:, 0] == 0.0) & (ball[:, 1] == 0.0)
            if bool(nexto_mask.any()):
                nexto_action, _indices = self.nexto.tick_action(
                    self.nexto_state,
                    kickoff,
                    active_mask=nexto_mask,
                )
                rows = self._world_rows[nexto_mask]
                actions[rows, opponent_side[nexto_mask]] = nexto_action[nexto_mask]
            if bool(wisp_mask.any()):
                wisp_action, _indices = self.wisp.tick_action(
                    self.wisp_state,
                    active_mask=wisp_mask,
                )
                rows = self._world_rows[wisp_mask]
                actions[rows, opponent_side[wisp_mask]] = wisp_action[wisp_mask]
            return actions

        return self.env.step_with_tick_actions(base_action, tick_action)

    def collect_rollout(self, active_world_mask: torch.Tensor | None = None) -> Rival2RolloutBuffer:
        config = self.ppo_config
        if active_world_mask is not None and (
            active_world_mask.shape != (self.env.num_envs,)
            or active_world_mask.dtype != torch.bool
            or active_world_mask.device != self.device
        ):
            raise ValueError("active world mask shape/dtype/device mismatch")
        if bool((self.opponent_family < 0).any()):
            raise RuntimeError("opponent curriculum assignments were not initialized")
        rollout = Rival2RolloutBuffer(
            config.rollout_horizon,
            self.env.num_envs,
            self.device,
            store_opponent_family=True,
        )
        observation = self.env.observation
        active_agent_samples = torch.zeros((), dtype=torch.int64, device=self.device)
        family_world_decisions = torch.zeros(4, dtype=torch.int64, device=self.device)
        family_trainable_samples = torch.zeros(4, dtype=torch.int64, device=self.device)
        family_terminated = torch.zeros(4, dtype=torch.int64, device=self.device)
        family_truncated = torch.zeros(4, dtype=torch.int64, device=self.device)
        family_rival_wins = torch.zeros(4, dtype=torch.int64, device=self.device)
        family_opponent_wins = torch.zeros(4, dtype=torch.int64, device=self.device)
        family_trainable_reward = torch.zeros(4, dtype=torch.float64, device=self.device)
        self.model.eval()
        for _ in range(config.rollout_horizon):
            with torch.no_grad():
                actor, value, acting_version, train_mask = self._policy_outputs(observation)
                sample = sample_hybrid_action(
                    actor, generator=self.policy_generator, config=self.policy_config
                )
                action = sample.action
                if active_world_mask is not None:
                    train_mask = train_mask & active_world_mask[:, None]
                    action = torch.where(
                        active_world_mask[:, None, None], action, torch.zeros_like(action)
                    )
                active_agent_samples += train_mask.sum()
                transition = self._step_with_frozen_opponents(action)
                active_family = self.opponent_family
                family_world_decisions += torch.bincount(active_family, minlength=4)
                family_trainable_samples.scatter_add_(
                    0,
                    active_family,
                    train_mask.sum(dim=1).to(torch.int64),
                )
                family_terminated += torch.bincount(
                    active_family[transition.terminated], minlength=4
                )
                family_truncated += torch.bincount(active_family[transition.truncated], minlength=4)
                trainable_reward_per_world = torch.where(
                    train_mask, transition.reward, torch.zeros_like(transition.reward)
                ).sum(dim=1)
                family_trainable_reward += torch.bincount(
                    active_family,
                    weights=trainable_reward_per_world.to(torch.float64),
                    minlength=4,
                )
                terminal_rows = transition.terminated
                scoring_team = self.env.bridge.views["rival2.scoring_team_latched"].to(torch.int64)
                rival_won = terminal_rows & (scoring_team == self.rival_side)
                opponent_won = terminal_rows & (scoring_team == (1 - self.rival_side))
                family_rival_wins += torch.bincount(active_family[rival_won], minlength=4)
                family_opponent_wins += torch.bincount(active_family[opponent_won], minlength=4)
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
                    opponent_family=active_family[:, None].expand(-1, 2),
                )
                self.assign_opponents_at_reset(transition.reset_mask)
                if active_world_mask is not None:
                    active_world_mask.logical_and_(~transition.terminated)
                observation = transition.observation
        self.env.observation = observation
        self.total_agent_samples += int(active_agent_samples.item())
        self.last_rollout_curriculum_metrics = {
            name: {OPPONENT_NAMES[index]: int(values[index].item()) for index in range(4)}
            for name, values in (
                ("world_decisions", family_world_decisions),
                ("trainable_agent_samples", family_trainable_samples),
                ("terminated_episodes", family_terminated),
                ("truncated_episodes", family_truncated),
                ("rival_wins", family_rival_wins),
                ("opponent_wins", family_opponent_wins),
            )
        }
        self.last_rollout_curriculum_metrics["trainable_reward_sum"] = {
            OPPONENT_NAMES[index]: float(family_trainable_reward[index].item())
            for index in range(4)
        }
        return rollout

    def enable_safe_mixed_ppo(
        self,
        config: Rival2MixedPPOSafetyConfig | None = None,
    ) -> dict[str, Any]:
        """Install the curriculum-only split optimizer with exact Adam migration."""

        if self.mixed_ppo_safety is not None:
            raise ValueError("safe mixed PPO is already enabled")
        selected = config or Rival2MixedPPOSafetyConfig()
        self.optimizer, proof = migrate_adam_to_mixed_groups(self.model, self.optimizer, selected)
        self.mixed_ppo_safety = selected
        self.optimizer_migration_proof = proof
        if self.curriculum_transition is not None:
            self.curriculum_transition["mixed_ppo_safety_transition"] = {
                "config": asdict(selected),
                "config_hash": selected.content_hash,
                "optimizer_migration": copy.deepcopy(proof),
                "legacy_ppo_path_changed": False,
                "model_architecture_changed": False,
            }
        return copy.deepcopy(proof)

    def initialize_retention_corpus_from_rollout(
        self,
        rollout: Rival2RolloutBuffer,
        *,
        source_identity: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Freeze the source-policy safety corpus before the first mixed update."""

        if self.mixed_ppo_safety is None:
            raise RuntimeError("safe mixed PPO must be enabled before retention setup")
        if self.retention_observations is not None:
            raise ValueError("retention corpus is already initialized")
        observations, summary = build_retention_observation_corpus(
            rollout,
            corpus_size=self.mixed_ppo_safety.retention_corpus_size,
        )
        summary.update(
            {
                "source_iteration": self.iteration,
                "source_policy_version": self.policy_version,
                "source_agent_decision_samples_after_rollout": self.total_agent_samples,
                "source_identity": copy.deepcopy(source_identity or {}),
            }
        )
        self.install_retention_corpus(observations, summary)
        return copy.deepcopy(summary)

    def install_retention_corpus(
        self,
        observations: torch.Tensor,
        summary: dict[str, Any],
    ) -> None:
        """Install and verify the immutable source-policy retention observations."""

        if self.mixed_ppo_safety is None:
            raise RuntimeError("safe mixed PPO must be enabled before retention setup")
        if self.retention_observations is not None:
            raise ValueError("retention corpus is already initialized")
        installed = observations.to(self.device, dtype=torch.float32).detach().clone()
        expected_shape = (self.mixed_ppo_safety.retention_corpus_size, self.policy_config.obs_dim)
        checks = {
            "shape_exact": installed.shape == expected_shape,
            "finite": bool(torch.isfinite(installed).all().item()),
            "sha256_exact": retention_observation_sha256(installed) == summary.get("sha256"),
            "source_policy_identity_present": bool(summary.get("source_identity")),
        }
        if not all(checks.values()):
            raise RuntimeError(f"retention corpus installation failed closed: {checks}")
        self.retention_observations = installed
        self.retention_corpus_summary = copy.deepcopy(summary)
        if self.curriculum_transition is not None:
            self.curriculum_transition["retention_corpus"] = copy.deepcopy(summary)

    def update(
        self,
        rollout: Rival2RolloutBuffer,
        *,
        kl_guard: Rival2KLGuardConfig | None = None,
    ) -> dict[str, torch.Tensor]:
        if self.mixed_ppo_safety is None:
            return super().update(rollout, kl_guard=kl_guard)
        if kl_guard is None:
            raise ValueError("safe mixed PPO requires the unchanged hard KL guard")
        if self.retention_observations is None:
            raise RuntimeError("safe mixed PPO retention corpus is not initialized")

        self.model.train()
        rollback = {
            "model": copy.deepcopy(self.model.state_dict()),
            "optimizer": copy.deepcopy(self.optimizer.state_dict()),
            "gradients": [
                None if parameter.grad is None else parameter.grad.detach().clone()
                for parameter in self.model.parameters()
            ],
            "torch_cpu_rng_state": torch.get_rng_state().clone(),
            "torch_cuda_rng_state": torch.cuda.get_rng_state(self.device).clone(),
            "policy_generator_state": self.policy_generator.get_state().clone(),
            "opponent_generator_state": self.opponent_generator.get_state().clone(),
            "curriculum_generator_state": self.curriculum_generator.get_state().clone(),
            "model_training": self.model.training,
        }
        try:
            metrics, diagnostics = ppo_update_mixed_curriculum(
                self.model,
                self.optimizer,
                rollout,
                self.ppo_config,
                self.mixed_ppo_safety,
                retention_observations=self.retention_observations,
                family_names=OPPONENT_NAMES,
                generator=self.policy_generator,
                policy_config=self.policy_config,
                kl_guard=kl_guard,
            )
        except Rival2PolicyDisplacementRejected as error:
            self.model.load_state_dict(rollback["model"])
            self.optimizer.load_state_dict(rollback["optimizer"])
            for parameter, gradient in zip(
                self.model.parameters(), rollback["gradients"], strict=True
            ):
                parameter.grad = None if gradient is None else gradient.clone()
            self.policy_generator.set_state(rollback["policy_generator_state"])
            self.opponent_generator.set_state(rollback["opponent_generator_state"])
            self.curriculum_generator.set_state(rollback["curriculum_generator_state"])
            torch.set_rng_state(rollback["torch_cpu_rng_state"])
            torch.cuda.set_rng_state(rollback["torch_cuda_rng_state"], self.device)
            self.model.train(bool(rollback["model_training"]))
            error.diagnostics.update(
                {
                    "rejected_iteration": self.iteration + 1,
                    "restored_iteration": self.iteration,
                    "restored_policy_version": self.policy_version,
                    "pre_update_agent_decision_samples": self.total_agent_samples,
                    "transactional_rollback_completed": True,
                    "mixed_ppo_safety_config": asdict(self.mixed_ppo_safety),
                }
            )
            raise
        self.last_adaptive_ppo_diagnostics = diagnostics
        self.policy_version += 1
        self.iteration += 1
        return metrics

    def checkpoint_payload(self) -> dict[str, Any]:
        payload = super().checkpoint_payload()
        payload["opponent_curriculum"] = {
            "config": asdict(self.opponent_curriculum),
            "generator_state": self.curriculum_generator.get_state(),
            "family": self.opponent_family,
            "rival_side": self.rival_side,
            "realized_family_assignments": self.realized_family_assignments,
            "adaptive_ppo": (
                None
                if self.mixed_ppo_safety is None
                else {
                    "config": asdict(self.mixed_ppo_safety),
                    "config_hash": self.mixed_ppo_safety.content_hash,
                    "optimizer_migration_proof": copy.deepcopy(self.optimizer_migration_proof),
                    "retention_observations": self.retention_observations,
                    "retention_corpus_summary": copy.deepcopy(self.retention_corpus_summary),
                    "optimizer_learning_rates": mixed_optimizer_learning_rates(self.optimizer),
                }
            ),
            "nexto": {
                "player_index": self.nexto.player_index,
                "previous_action": self.nexto.previous_action,
                "neural_counter": self.nexto.neural_counter,
                "kickoff_index": self.nexto.kickoff_index,
                "cadence_tick": self.nexto._cadence_tick,
            },
            "wisp": {
                "player_index": self.wisp.player_index,
                "old_action": self.wisp.old_action,
                "new_action": self.wisp.new_action,
                "previous_action": self.wisp.previous_action,
                "ticks": self.wisp.ticks,
                "update_flag": self.wisp.update_flag,
                "eta_cache": self.wisp.eta_cache.copy(),
                "observation_generator_state": self.wisp.observation_generator.get_state(),
                "opponent_slot": self.wisp.opponent_slot,
            },
        }
        return payload

    def _restore_checkpoint_state(self, payload: dict[str, Any]) -> None:
        optimizer_groups = payload["optimizer"]["param_groups"]
        curriculum = payload.get("opponent_curriculum")
        adaptive = None if curriculum is None else curriculum.get("adaptive_ppo")
        if len(optimizer_groups) == 2:
            if adaptive is None:
                raise ValueError("split optimizer checkpoint lacks adaptive PPO state")
            restored_config = Rival2MixedPPOSafetyConfig(**adaptive["config"])
            if restored_config.content_hash != adaptive["config_hash"]:
                raise ValueError("adaptive PPO checkpoint configuration hash mismatch")
            group_by_name = {group.get("name"): group for group in optimizer_groups}
            if set(group_by_name) != {"policy", "critic"}:
                raise ValueError("split optimizer checkpoint group names are invalid")
            self.optimizer = make_empty_mixed_optimizer(
                self.model,
                policy_learning_rate=float(group_by_name["policy"]["lr"]),
                critic_learning_rate=float(group_by_name["critic"]["lr"]),
            )
        elif len(optimizer_groups) != 1:
            raise ValueError("unsupported optimizer parameter-group count")
        super()._restore_checkpoint_state(payload)
        self.mixed_ppo_safety = None
        self.optimizer_migration_proof = None
        self.retention_observations = None
        self.retention_corpus_summary = None
        self.last_adaptive_ppo_diagnostics = None
        if curriculum is None:
            return
        if curriculum["config"] != asdict(self.opponent_curriculum):
            raise ValueError("opponent curriculum checkpoint configuration mismatch")
        self.curriculum_generator.set_state(curriculum["generator_state"].cpu())
        self.opponent_family.copy_(curriculum["family"])
        self.rival_side.copy_(curriculum["rival_side"])
        self.realized_family_assignments.copy_(curriculum["realized_family_assignments"])
        nexto = curriculum["nexto"]
        for name in ("player_index", "previous_action", "neural_counter", "kickoff_index"):
            getattr(self.nexto, name).copy_(nexto[name])
        self.nexto._cadence_tick = int(nexto["cadence_tick"])
        wisp = curriculum["wisp"]
        for name in (
            "player_index",
            "old_action",
            "new_action",
            "previous_action",
            "ticks",
            "update_flag",
            "opponent_slot",
        ):
            getattr(self.wisp, name).copy_(wisp[name])
        self.wisp.eta_cache[:] = np.asarray(wisp["eta_cache"], dtype=np.float64)
        self.wisp.observation_generator.set_state(wisp["observation_generator_state"].cpu())
        if adaptive is not None:
            self.mixed_ppo_safety = Rival2MixedPPOSafetyConfig(**adaptive["config"])
            self.optimizer_migration_proof = copy.deepcopy(adaptive["optimizer_migration_proof"])
            self.retention_observations = adaptive["retention_observations"].to(self.device)
            self.retention_corpus_summary = copy.deepcopy(adaptive["retention_corpus_summary"])


__all__ = [
    "NEXTO_ACTING_VERSION",
    "OPPONENT_CURRENT",
    "OPPONENT_HISTORICAL",
    "OPPONENT_NAMES",
    "OPPONENT_NEXTO",
    "OPPONENT_WISP",
    "WISP_ACTING_VERSION",
    "Rival2OpponentCurriculumConfig",
    "Rival2OpponentCurriculumTrainer",
]
