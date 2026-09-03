"""Sparse-option PPO for the protected aerial scorer in natural self-play."""

from __future__ import annotations

import copy
from dataclasses import asdict
from pathlib import Path
from typing import Any

import torch

from rivalsim.rival2_contracts import CAR_LINEAR_SPEED_SCALE, OBS_FIELD_NAMES
from rivalsim.rival2_env import Rival2Env
from rivalsim.rival2_ground_to_air_selfplay_v12 import (
    AerialOptionRouterConfig,
    AerialOptionSelfPlayRouter,
    AerialSelfPlayRewardConfig,
)
from rivalsim.rival2_policy import (
    HybridDistributionOverride,
    Rival2ActorCritic,
    deterministic_hybrid_action,
    sample_hybrid_action,
)
from rivalsim.rival2_ppo import (
    Rival2KLGuardConfig,
    Rival2PolicyDisplacementRejected,
    Rival2PPOConfig,
    Rival2RolloutBuffer,
    ppo_update,
)


class AerialOptionSelfPlayTrainerV12:
    """Train only the direct aerial option while V23 controls ordinary play.

    Both physical sides use the same current aerial option, so every option
    action is current-policy self-play.  The V23 side-specialized policies are
    immutable.  The aerial option trunk is also frozen: PPO updates its actor
    and critic heads, and value loss therefore cannot alter actor features.
    """

    def __init__(
        self,
        env: Rival2Env,
        *,
        blue_base: Rival2ActorCritic,
        orange_base: Rival2ActorCritic,
        option: Rival2ActorCritic,
        ppo_config: Rival2PPOConfig,
        router_config: AerialOptionRouterConfig,
        reward_config: AerialSelfPlayRewardConfig,
        exploration: HybridDistributionOverride,
        seed: int,
        actor_learning_rate: float,
        critic_learning_rate: float,
    ) -> None:
        if env.policy_hz != 120 or env.physics_hz != 120:
            raise ValueError("V12 aerial self-play requires native 120 Hz cadence")
        configs = (blue_base.config, orange_base.config, option.config)
        if any(asdict(config) != asdict(configs[0]) for config in configs[1:]):
            raise ValueError("hybrid policy architectures differ")
        self.env = env
        self.device = env.device
        self.policy_config = option.config
        self.ppo_config = ppo_config
        self.blue_base = blue_base.to(self.device).eval().requires_grad_(False)
        self.orange_base = orange_base.to(self.device).eval().requires_grad_(False)
        self.model = option.to(self.device)
        self.model.trunk.requires_grad_(False)
        self.model.actor.requires_grad_(True)
        self.model.critic.requires_grad_(True)
        self.optimizer = torch.optim.Adam(
            [
                {
                    "name": "aerial_actor",
                    "params": self.model.actor.parameters(),
                    "lr": float(actor_learning_rate),
                },
                {
                    "name": "aerial_critic",
                    "params": self.model.critic.parameters(),
                    "lr": float(critic_learning_rate),
                },
            ]
        )
        self.exploration = exploration
        self.policy_generator = torch.Generator(device=self.device).manual_seed(seed)
        self.router = AerialOptionSelfPlayRouter(
            env.num_envs * 2,
            device=self.device,
            router_config=router_config,
            reward_config=reward_config,
        )
        self.iteration = 0
        self.policy_version = 0
        self.total_option_samples = 0
        self.total_physics_ticks = 0
        self.last_rollout_metrics: dict[str, Any] | None = None

    @property
    def router_config(self) -> AerialOptionRouterConfig:
        return self.router.config

    @property
    def reward_config(self) -> AerialSelfPlayRewardConfig:
        return self.router.reward_config

    def _base_action(self, observation: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            blue_actor, _ = self.blue_base(observation[:, 0])
            orange_actor, _ = self.orange_base(observation[:, 1])
            return torch.stack(
                (
                    deterministic_hybrid_action(blue_actor, self.policy_config),
                    deterministic_hybrid_action(orange_actor, self.policy_config),
                ),
                dim=1,
            )

    @staticmethod
    def _counter_snapshot(router: AerialOptionSelfPlayRouter) -> dict[str, int]:
        return {name: int(value.item()) for name, value in router.counters.items()}

    def collect_rollout(self) -> Rival2RolloutBuffer:
        horizon = self.ppo_config.rollout_horizon
        worlds = self.env.num_envs
        rollout = Rival2RolloutBuffer(horizon, worlds, self.device)
        observation = self.env.observation
        router_before = self._counter_snapshot(self.router)
        reward_before = float(self.router.reward_sum.item())
        counter_names = (
            "legitimate_touch_total",
            "flip_touch_total",
            "bad_flip_total",
        )
        gameplay_before = {
            name: self.env.bridge.views[f"gameplay_120.{name}"].clone()
            for name in counter_names
        }
        no_touch_index = OBS_FIELD_NAMES.index("lifecycle.no_touch_age")
        speed_start = OBS_FIELD_NAMES.index("self.linear_velocity.x")
        option_samples = torch.zeros((), dtype=torch.int64, device=self.device)
        goal_count = torch.zeros(2, dtype=torch.int64, device=self.device)
        no_touch = torch.zeros((), dtype=torch.int64, device=self.device)
        speed_sum = torch.zeros((), dtype=torch.float64, device=self.device)
        saturation = torch.zeros((), dtype=torch.int64, device=self.device)
        action_samples = torch.zeros((), dtype=torch.int64, device=self.device)
        self.model.eval()
        for _tick in range(horizon):
            flat_observation = observation.reshape(-1, self.policy_config.obs_dim)
            kickoff = (
                self.env.bridge.views["rival2.kickoff_indicator"] != 0
            )[:, None].expand(-1, 2).reshape(-1)
            done = torch.zeros_like(kickoff)
            selection = self.router.select(
                flat_observation,
                kickoff_active=kickoff,
                match_done=done,
            )
            active = selection.active.reshape(worlds, 2)
            with torch.no_grad():
                actor, flat_value = self.model(flat_observation)
                sample = sample_hybrid_action(
                    actor,
                    generator=self.policy_generator,
                    config=self.policy_config,
                    distribution_override=self.exploration,
                )
                base_action = self._base_action(observation)
                option_action = sample.action.reshape(worlds, 2, 8)
                action = torch.where(active[..., None], option_action, base_action)
                value = flat_value.reshape(worlds, 2)
            transition = self.env.step(action)
            scoring_team = self.env.bridge.views[
                "rival2.scoring_team_latched"
            ].to(torch.int64)
            side = torch.arange(2, device=self.device)[None, :]
            goal_for = (
                transition.terminated[:, None]
                & (scoring_team[:, None] == side)
            )
            outcome = self.router.observe(
                flat_observation,
                transition.transition_observation.reshape(-1, 182),
                active_before=selection.active,
                goal_for_lane=goal_for.reshape(-1),
            )
            reward = transition.reward + outcome.supplemental_reward.reshape(worlds, 2)
            with torch.no_grad():
                _, next_flat_value = self.model(
                    transition.transition_observation.reshape(-1, 182)
                )
            next_value = next_flat_value.reshape(worlds, 2)
            terminated = transition.terminated[:, None].expand(-1, 2)
            truncated = transition.truncated[:, None].expand(-1, 2)
            version = torch.full(
                (worlds, 2),
                self.policy_version,
                dtype=torch.int64,
                device=self.device,
            )
            rollout.add(
                observation=observation,
                action=transition.emitted_action,
                pre_tanh=sample.pre_tanh.reshape(worlds, 2, 5),
                old_log_probability=sample.log_probability.reshape(worlds, 2),
                value=value,
                reward=reward,
                terminated=terminated,
                truncated=truncated,
                next_value=next_value,
                policy_version=version,
                opponent_version=version,
                train_mask=active,
            )
            option_samples += active.sum()
            action_samples += active.sum()
            saturation += (
                (transition.emitted_action[..., :5].abs() > 0.95)
                & active[..., None]
            ).sum()
            normalized_speed = torch.linalg.vector_norm(
                observation[..., speed_start : speed_start + 3], dim=-1
            )
            speed_sum += torch.where(
                active, normalized_speed, torch.zeros_like(normalized_speed)
            ).sum(dtype=torch.float64)
            for team in (0, 1):
                goal_count[team] += (
                    transition.terminated & (scoring_team == team)
                ).sum()
            no_touch += (
                transition.truncated
                & (
                    transition.transition_observation[:, 0, no_touch_index]
                    >= 1.0 - 1.0e-6
                )
            ).sum()
            observation = transition.observation
        self.env.observation = observation
        count = int(option_samples.item())
        self.total_option_samples += count
        self.total_physics_ticks += horizon * worlds
        router_after = self._counter_snapshot(self.router)
        router_delta = {
            name: router_after[name] - router_before[name] for name in router_after
        }
        gameplay_delta = {
            name: int(
                (
                    self.env.bridge.views[f"gameplay_120.{name}"]
                    - gameplay_before[name]
                ).sum(dtype=torch.int64).item()
            )
            for name in counter_names
        }
        player_minutes = horizon * worlds * 2 / (120.0 * 60.0)
        self.last_rollout_metrics = {
            "option_samples": count,
            "option_fraction": count / (horizon * worlds * 2),
            "router": router_delta,
            "route_activations_cumulative": self.router.telemetry()[
                "route_activations"
            ],
            "supplemental_reward_sum": float(self.router.reward_sum.item())
            - reward_before,
            "goals_by_team": goal_count.detach().cpu().tolist(),
            "no_touch_truncations": int(no_touch.item()),
            "touches_per_player_minute": gameplay_delta["legitimate_touch_total"]
            / player_minutes,
            "unnecessary_flip_contacts_per_player_minute": gameplay_delta[
                "bad_flip_total"
            ]
            / player_minutes,
            "unnecessary_fraction_of_flip_contacts": gameplay_delta["bad_flip_total"]
            / max(gameplay_delta["flip_touch_total"], 1),
            "mean_option_speed_uu_per_second": float(speed_sum.item())
            / max(count, 1)
            * CAR_LINEAR_SPEED_SCALE,
            "option_analog_saturation_fraction": int(saturation.item())
            / max(int(action_samples.item()) * 5, 1),
        }
        return rollout

    def update(self, rollout: Rival2RolloutBuffer) -> dict[str, torch.Tensor]:
        """Perform telemetry-only-KL PPO with transactional nonfinite recovery."""

        model_before = copy.deepcopy(self.model.state_dict())
        optimizer_before = copy.deepcopy(self.optimizer.state_dict())
        generator_before = self.policy_generator.get_state().clone()
        self.model.train()
        try:
            metrics = ppo_update(
                self.model,
                self.optimizer,
                rollout,
                self.ppo_config,
                generator=self.policy_generator,
                policy_config=self.policy_config,
                kl_guard=Rival2KLGuardConfig(
                    reject_minibatch_kl=False,
                    reject_completed_update_kl=False,
                ),
                distribution_override=self.exploration,
            )
        except Rival2PolicyDisplacementRejected:
            self.model.load_state_dict(model_before)
            self.optimizer.load_state_dict(optimizer_before)
            self.policy_generator.set_state(generator_before)
            raise
        if not all(
            bool(torch.isfinite(parameter).all().item())
            for parameter in self.model.parameters()
        ):
            self.model.load_state_dict(model_before)
            self.optimizer.load_state_dict(optimizer_before)
            self.policy_generator.set_state(generator_before)
            raise Rival2PolicyDisplacementRejected(
                {"reason": "nonfinite_parameter_after_v12_update"}
            )
        self.iteration += 1
        self.policy_version += 1
        return metrics

    def checkpoint_payload(self, provenance: dict[str, Any]) -> dict[str, Any]:
        return {
            "format": "RIVAL2_GROUND_TO_AIR_SELFPLAY_V12_CHECKPOINT",
            "model": self.model.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "policy_config": asdict(self.policy_config),
            "ppo_config": asdict(self.ppo_config),
            "router_config": asdict(self.router_config),
            "aerial_reward_config": asdict(self.reward_config),
            "exploration": asdict(self.exploration),
            "iteration": self.iteration,
            "policy_version": self.policy_version,
            "total_option_samples": self.total_option_samples,
            "total_physics_ticks": self.total_physics_ticks,
            "policy_generator_state": self.policy_generator.get_state(),
            "router_telemetry": self.router.telemetry(),
            "trunk_frozen": True,
            "base_policies_frozen": True,
            "critic_value_loss_isolated_from_trunk": True,
            "kl_policy": "telemetry_only_no_KL_rejection_or_KL_rollback",
            "provenance": copy.deepcopy(provenance),
        }

    def save_checkpoint(self, path: str | Path, provenance: dict[str, Any]) -> None:
        torch.save(self.checkpoint_payload(provenance), Path(path))


__all__ = ["AerialOptionSelfPlayTrainerV12"]
