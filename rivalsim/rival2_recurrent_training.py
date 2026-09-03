"""Pure-current recurrent self-play trainer for Human Sequence PPO."""

from __future__ import annotations

import copy
from dataclasses import asdict
from pathlib import Path
from typing import Any

import torch

from rivalsim.rival2_contracts import (
    CAR_LINEAR_SPEED_SCALE,
    OBS_FIELD_NAMES,
)
from rivalsim.rival2_env import Rival2Env
from rivalsim.rival2_exploration import FreshHumanSeedExploration
from rivalsim.rival2_policy import sample_hybrid_action
from rivalsim.rival2_ppo import Rival2PPOConfig
from rivalsim.rival2_recurrent_policy import (
    Rival2RecurrentActorCritic,
    Rival2RecurrentPolicyConfig,
)
from rivalsim.rival2_recurrent_ppo import (
    Rival2RecurrentPPOCorruption,
    Rival2RecurrentRolloutBuffer,
    recurrent_ppo_update,
)

CHECKPOINT_FORMAT = "RIVAL2_HUMAN_SEQUENCE_RECURRENT_PPO_V1_CHECKPOINT"
_TOUCH_EVENT_INDEX = OBS_FIELD_NAMES.index("lifecycle.self_touch_event")
_NO_TOUCH_AGE_INDEX = OBS_FIELD_NAMES.index("lifecycle.no_touch_age")
_SELF_VELOCITY_START = OBS_FIELD_NAMES.index("self.linear_velocity.x")
_SELF_SUPERSONIC_INDEX = OBS_FIELD_NAMES.index("self.is_supersonic")
_BALL_VELOCITY_Y_INDEX = OBS_FIELD_NAMES.index("ball.linear_velocity.y")


class Rival2RecurrentTrainer:
    """Current-v-current recurrent PPO with continuous per-agent hidden state."""

    def __init__(
        self,
        env: Rival2Env,
        *,
        policy_config: Rival2RecurrentPolicyConfig,
        ppo_config: Rival2PPOConfig,
        phase: str,
        source_identity: dict[str, Any],
        seed: int,
        model: torch.nn.Module | None = None,
        checkpoint_format: str = CHECKPOINT_FORMAT,
        lineage: str = "Human Sequence Seed v1 recurrent PPO",
    ):
        if phase not in {
            "phase_a_acquisition",
            "phase_b_gameplay_120_v2",
            "unified_ground_selfplay_v1",
            "unified_ground_acquisition_v2",
            "unified_ground_gameplay_v2",
        }:
            raise ValueError(f"unsupported recurrent PPO phase: {phase}")
        self.env = env
        self.device = env.device
        self.policy_config = policy_config
        self.ppo_config = ppo_config
        self.phase = phase
        self.source_identity = copy.deepcopy(source_identity)
        self.checkpoint_format = checkpoint_format
        self.lineage = lineage
        self.model = (
            Rival2RecurrentActorCritic(policy_config) if model is None else model
        ).to(self.device)
        if getattr(self.model, "config", None) != policy_config:
            raise ValueError("recurrent PPO model/config mismatch")
        self.optimizer = torch.optim.Adam(
            self.model.parameters(), lr=self.ppo_config.learning_rate
        )
        self.policy_generator = torch.Generator(device=self.device).manual_seed(int(seed))
        self.shuffle_generator = torch.Generator(device=self.device).manual_seed(
            int(seed) ^ 0x51E0A11
        )
        self.hidden = self.model.initial_hidden(
            env.num_envs * 2, device=self.device
        ).reshape(
            policy_config.recurrent_layers,
            env.num_envs,
            2,
            policy_config.hidden_dim,
        )
        self.reset_before = torch.ones(
            (env.num_envs, 2), dtype=torch.bool, device=self.device
        )
        self.episode_has_touch = torch.zeros(
            (env.num_envs, 2), dtype=torch.bool, device=self.device
        )
        self.exploration: FreshHumanSeedExploration | None = None
        self.accepted_updates_total = 0
        self.phase_accepted_updates = 0
        self.policy_version = 0
        self.total_agent_samples = 0
        self.physical_physics_ticks_experienced = 0
        self.last_rollout_metrics: dict[str, Any] = {}
        self.phase_transition: dict[str, Any] | None = None
        self.resume_count = 0

    def set_exploration(self, value: FreshHumanSeedExploration) -> None:
        self.exploration = value

    def replace_optimizer(self) -> None:
        self.optimizer = torch.optim.Adam(
            self.model.parameters(), lr=self.ppo_config.learning_rate
        )

    def _flat_hidden(self) -> torch.Tensor:
        return self.hidden.reshape(
            self.policy_config.recurrent_layers,
            self.env.num_envs * 2,
            self.policy_config.hidden_dim,
        )

    def _set_flat_hidden(self, hidden: torch.Tensor) -> None:
        self.hidden = hidden.reshape(
            self.policy_config.recurrent_layers,
            self.env.num_envs,
            2,
            self.policy_config.hidden_dim,
        )

    def _gameplay_counter_snapshot(self) -> dict[str, torch.Tensor] | None:
        if self.env.world.gameplay_120 is None:
            return None
        names = (
            "legitimate_touch_total",
            "flip_touch_total",
            "bad_flip_total",
            "contest_exempt_total",
            "power_exempt_total",
            "retained_control_exempt_total",
            "control_score_sum_total",
            "control_score_tick_total",
            "control_score_positive_total",
            "control_score_ge_025_total",
            "control_score_ge_05_total",
            "control_reward_sum_total",
        )
        return {
            name: self.env.bridge.views[f"gameplay_120.{name}"].clone()
            for name in names
            if f"gameplay_120.{name}" in self.env.bridge.views
        }

    def collect_rollout(self) -> Rival2RecurrentRolloutBuffer:
        if self.exploration is None:
            raise RuntimeError("exploration must be frozen before rollout collection")
        config = self.ppo_config
        rollout = Rival2RecurrentRolloutBuffer(
            config.rollout_horizon,
            self.env.num_envs,
            self.hidden,
            self.device,
            obs_dim=self.policy_config.obs_dim,
        )
        observation = self.env.observation
        gameplay_before = self._gameplay_counter_snapshot()
        touch_events = torch.zeros((), dtype=torch.int64, device=self.device)
        first_touch_events = torch.zeros((), dtype=torch.int64, device=self.device)
        goal_events = torch.zeros((), dtype=torch.int64, device=self.device)
        goalward_touch_events = torch.zeros((), dtype=torch.int64, device=self.device)
        no_touch_truncations = torch.zeros((), dtype=torch.int64, device=self.device)
        hard_limit_truncations = torch.zeros((), dtype=torch.int64, device=self.device)
        reward_sum = torch.zeros((), dtype=torch.float64, device=self.device)
        reward_abs_sum = torch.zeros((), dtype=torch.float64, device=self.device)
        speed_sum = torch.zeros((), dtype=torch.float64, device=self.device)
        supersonic_ticks = torch.zeros((), dtype=torch.int64, device=self.device)
        saturation_count = torch.zeros((), dtype=torch.int64, device=self.device)
        action_samples = config.rollout_horizon * self.env.num_envs * 2

        self.model.eval()
        for _ in range(config.rollout_horizon):
            step_reset = self.reset_before
            with torch.no_grad():
                actor, value_flat, hidden_after_observation = self.model(
                    observation.reshape(-1, self.policy_config.obs_dim),
                    self._flat_hidden(),
                    reset_before=step_reset.reshape(-1),
                )
                sample = sample_hybrid_action(
                    actor,
                    generator=self.policy_generator,
                    config=self.policy_config,
                    distribution_override=self.exploration.distribution_override,
                )
                sampled_action = sample.action.reshape(self.env.num_envs, 2, 8)
                transition = self.env.step(sampled_action)
                _next_actor, next_value_flat, _terminal_hidden = self.model(
                    transition.transition_observation.reshape(
                        -1, self.policy_config.obs_dim
                    ),
                    hidden_after_observation,
                )
                value = value_flat.reshape(self.env.num_envs, 2)
                next_value = next_value_flat.reshape(self.env.num_envs, 2)
                terminated = transition.terminated[:, None].expand(-1, 2)
                truncated = transition.truncated[:, None].expand(-1, 2)
                train_mask = torch.ones_like(terminated)
                rollout.add(
                    observation=observation,
                    action=transition.emitted_action,
                    pre_tanh=sample.pre_tanh.reshape(self.env.num_envs, 2, 5),
                    old_log_probability=sample.log_probability.reshape(
                        self.env.num_envs, 2
                    ),
                    value=value,
                    reward=transition.reward,
                    terminated=terminated,
                    truncated=truncated,
                    next_value=next_value,
                    train_mask=train_mask,
                    reset_before=step_reset,
                )

                touch = (
                    transition.transition_observation[..., _TOUCH_EVENT_INDEX] > 0.5
                )
                first_touch = touch & ~self.episode_has_touch
                self.episode_has_touch.logical_or_(touch)
                touch_events += touch.sum()
                goalward_touch_events += (
                    touch
                    & (
                        transition.transition_observation[
                            ..., _BALL_VELOCITY_Y_INDEX
                        ]
                        > 0.0
                    )
                ).sum()
                first_touch_events += first_touch.sum()
                goal_events += transition.terminated.sum()
                no_touch_terminal = transition.truncated & (
                    transition.transition_observation[:, 0, _NO_TOUCH_AGE_INDEX]
                    >= 1.0 - 1.0e-6
                )
                no_touch_truncations += no_touch_terminal.sum()
                hard_limit_truncations += (
                    transition.truncated & ~no_touch_terminal
                ).sum()
                reward_sum += transition.reward.sum(dtype=torch.float64)
                reward_abs_sum += transition.reward.abs().sum(dtype=torch.float64)
                speed = torch.linalg.vector_norm(
                    observation[..., _SELF_VELOCITY_START : _SELF_VELOCITY_START + 3],
                    dim=-1,
                )
                speed_sum += speed.sum(dtype=torch.float64)
                supersonic_ticks += (
                    observation[..., _SELF_SUPERSONIC_INDEX] > 0.5
                ).sum()
                saturation_count += (
                    transition.emitted_action[..., :5].abs() > 0.95
                ).sum()

                reset_agent = transition.reset_mask[:, None].expand(-1, 2)
                next_hidden = hidden_after_observation.reshape(
                    self.policy_config.recurrent_layers,
                    self.env.num_envs,
                    2,
                    self.policy_config.hidden_dim,
                )
                next_hidden = next_hidden.masked_fill(
                    reset_agent.view(1, self.env.num_envs, 2, 1), 0.0
                )
                self.hidden = next_hidden
                self.reset_before = reset_agent.clone()
                self.episode_has_touch.masked_fill_(reset_agent, False)
                observation = transition.observation

        self.env.observation = observation
        self.total_agent_samples += action_samples
        self.physical_physics_ticks_experienced += (
            config.rollout_horizon
            * self.env.num_envs
            * self.env.physics_ticks_per_decision
        )
        physical_player_minutes = action_samples / (120.0 * 60.0)
        gameplay_delta: dict[str, int | float] = {}
        gameplay_after = self._gameplay_counter_snapshot()
        if gameplay_before is not None and gameplay_after is not None:
            for name in gameplay_before:
                delta = gameplay_after[name] - gameplay_before[name]
                if delta.dtype.is_floating_point:
                    gameplay_delta[name] = float(delta.sum(dtype=torch.float64).item())
                else:
                    gameplay_delta[name] = int(delta.sum(dtype=torch.int64).item())
        control_ticks = int(gameplay_delta.get("control_score_tick_total", 0))
        control_summary = {
            "mean": (
                float(gameplay_delta.get("control_score_sum_total", 0.0))
                / control_ticks
                if control_ticks
                else 0.0
            ),
            "positive_fraction": (
                int(gameplay_delta.get("control_score_positive_total", 0))
                / control_ticks
                if control_ticks
                else 0.0
            ),
            "ge_025_fraction": (
                int(gameplay_delta.get("control_score_ge_025_total", 0))
                / control_ticks
                if control_ticks
                else 0.0
            ),
            "ge_05_fraction": (
                int(gameplay_delta.get("control_score_ge_05_total", 0))
                / control_ticks
                if control_ticks
                else 0.0
            ),
            "reward_sum": float(gameplay_delta.get("control_reward_sum_total", 0.0)),
        }
        self.last_rollout_metrics = {
            "reward_version": self.env.reward_version,
            "policy_hz": self.env.policy_hz,
            "trainable_agent_samples": action_samples,
            "physical_player_minutes": physical_player_minutes,
            "touch_events": int(touch_events.item()),
            "touches_per_minute": float(touch_events.item()) / physical_player_minutes,
            "goalward_touch_events": int(goalward_touch_events.item()),
            "goalward_touch_fraction": (
                int(goalward_touch_events.item()) / max(1, int(touch_events.item()))
            ),
            "first_touch_events_per_player_episode": int(first_touch_events.item()),
            "goal_events": int(goal_events.item()),
            "no_touch_truncations": int(no_touch_truncations.item()),
            "hard_limit_truncations": int(hard_limit_truncations.item()),
            "reward_signed_sum": float(reward_sum.item()),
            "reward_absolute_sum": float(reward_abs_sum.item()),
            "mean_movement_speed_uu_per_second": (
                float(speed_sum.item()) / action_samples * CAR_LINEAR_SPEED_SCALE
            ),
            "supersonic_occupancy_fraction": (
                int(supersonic_ticks.item()) / action_samples
            ),
            "analog_action_saturation_fraction": (
                int(saturation_count.item()) / (action_samples * 5)
            ),
            "gameplay_120_counter_delta": gameplay_delta,
            "control_score": control_summary,
            "named_mechanics_hot_path_absent": self.env.world.gameplay_v3 is None,
            "recurrent_hidden_continuous_within_episode": True,
            "recurrent_reset_only_on_native_reset": True,
        }
        return rollout

    def update(self, rollout: Rival2RecurrentRolloutBuffer) -> dict[str, torch.Tensor]:
        if self.exploration is None:
            raise RuntimeError("exploration must be frozen before PPO update")
        rollback = {
            "model": copy.deepcopy(self.model.state_dict()),
            "optimizer": copy.deepcopy(self.optimizer.state_dict()),
            "shuffle_generator": self.shuffle_generator.get_state().clone(),
            "torch_cpu_rng": torch.get_rng_state().clone(),
            "torch_cuda_rng": torch.cuda.get_rng_state(self.device).clone(),
        }
        self.model.train()
        try:
            metrics = recurrent_ppo_update(
                self.model,
                self.optimizer,
                rollout,
                self.ppo_config,
                generator=self.shuffle_generator,
                distribution_override=self.exploration.distribution_override,
            )
        except Rival2RecurrentPPOCorruption as error:
            self.model.load_state_dict(rollback["model"])
            self.optimizer.load_state_dict(rollback["optimizer"])
            self.shuffle_generator.set_state(rollback["shuffle_generator"])
            torch.set_rng_state(rollback["torch_cpu_rng"])
            torch.cuda.set_rng_state(rollback["torch_cuda_rng"], self.device)
            error.diagnostics.update(
                {
                    "accepted_updates_total": self.accepted_updates_total,
                    "phase": self.phase,
                    "phase_accepted_updates": self.phase_accepted_updates,
                    "transactional_rollback_completed": True,
                    "kl_caused_rejection": False,
                }
            )
            raise
        self.accepted_updates_total += 1
        self.phase_accepted_updates += 1
        self.policy_version += 1
        return metrics

    def checkpoint_payload(self, *, include_optimizer: bool = True) -> dict[str, Any]:
        return {
            "format": self.checkpoint_format,
            "lineage": self.lineage,
            "source": copy.deepcopy(self.source_identity),
            "phase": self.phase,
            "model": self.model.state_dict(),
            "optimizer": self.optimizer.state_dict() if include_optimizer else None,
            "optimizer_included": include_optimizer,
            "optimizer_group_lrs": {
                str(group.get("name", f"group_{index}")): float(group["lr"])
                for index, group in enumerate(self.optimizer.param_groups)
            },
            "policy_config": asdict(self.policy_config),
            "policy_config_sha256": self.policy_config.content_hash,
            "ppo_config": asdict(self.ppo_config),
            "ppo_config_sha256": self.ppo_config.content_hash,
            "reward_version": self.env.reward_version,
            "episode_version": self.env.episode_version,
            "observation_version": self.env.observation_version,
            "action_version": self.env.action_version,
            "contract_hashes": dict(self.env.contract_hashes),
            "physics_hz": self.env.physics_hz,
            "policy_hz": self.env.policy_hz,
            "accepted_updates_total": self.accepted_updates_total,
            "phase_accepted_updates": self.phase_accepted_updates,
            "policy_version": self.policy_version,
            "total_agent_samples": self.total_agent_samples,
            "physical_physics_ticks_experienced": self.physical_physics_ticks_experienced,
            "exploration": None if self.exploration is None else self.exploration.as_dict(),
            "policy_generator_state": self.policy_generator.get_state(),
            "shuffle_generator_state": self.shuffle_generator.get_state(),
            "torch_cpu_rng_state": torch.get_rng_state(),
            "torch_cuda_rng_state": torch.cuda.get_rng_state(self.device),
            "phase_transition": copy.deepcopy(self.phase_transition),
            "opponents": {
                "current_policy_probability": 1.0,
                "both_sides_trainable": True,
                "historical": False,
                "nexto": False,
                "wisp": False,
            },
            "recurrent_state": {
                "continuous_during_live_run": True,
                "reset_only_at_native_episode_boundary": True,
                "checkpoint_resume_starts_fresh_simulator_kickoff": True,
                "hidden_serialized": False,
            },
            "kl_policy": {
                "telemetry_only": True,
                "minibatch_rejection": False,
                "completed_update_rejection": False,
                "kl_retry": False,
                "kl_rollback": False,
                "nonfinite_transactional_rollback": True,
            },
        }

    def save_checkpoint(self, path: str | Path, *, include_optimizer: bool = True) -> None:
        destination = Path(path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        torch.save(self.checkpoint_payload(include_optimizer=include_optimizer), destination)

    def load_checkpoint(self, path: str | Path) -> dict[str, Any]:
        payload = torch.load(Path(path), map_location=self.device, weights_only=False)
        if payload.get("format") != self.checkpoint_format:
            raise ValueError("unsupported recurrent PPO checkpoint")
        if payload.get("policy_config_sha256") != self.policy_config.content_hash:
            raise ValueError("recurrent PPO policy contract mismatch")
        if payload.get("ppo_config_sha256") != self.ppo_config.content_hash:
            raise ValueError("recurrent PPO configuration mismatch")
        if payload.get("phase") != self.phase:
            raise ValueError("recurrent PPO phase mismatch")
        if payload.get("reward_version") != self.env.reward_version:
            raise ValueError("recurrent PPO reward mismatch")
        self.model.load_state_dict(payload["model"], strict=True)
        if payload.get("optimizer") is None:
            raise ValueError("inference-only snapshot is not resumable")
        self.optimizer.load_state_dict(payload["optimizer"])
        self.accepted_updates_total = int(payload["accepted_updates_total"])
        self.phase_accepted_updates = int(payload["phase_accepted_updates"])
        self.policy_version = int(payload["policy_version"])
        self.total_agent_samples = int(payload["total_agent_samples"])
        self.physical_physics_ticks_experienced = int(
            payload["physical_physics_ticks_experienced"]
        )
        self.policy_generator.set_state(payload["policy_generator_state"].cpu())
        self.shuffle_generator.set_state(payload["shuffle_generator_state"].cpu())
        self.phase_transition = copy.deepcopy(payload.get("phase_transition"))
        torch.set_rng_state(payload["torch_cpu_rng_state"].cpu())
        torch.cuda.set_rng_state(payload["torch_cuda_rng_state"].cpu(), self.device)
        self.hidden.zero_()
        self.reset_before.fill_(True)
        self.episode_has_touch.zero_()
        self.resume_count += 1
        return payload


__all__ = [
    "CHECKPOINT_FORMAT",
    "Rival2RecurrentTrainer",
]
