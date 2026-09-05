"""Recurrent mixed-opponent PPO for the SSL Foundation reward lane."""

from __future__ import annotations

import copy
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import torch
import warp as wp

from rivalsim.rival2_contracts import CAR_LINEAR_SPEED_SCALE, OBS_FIELD_NAMES
from rivalsim.rival2_policy import sample_hybrid_action
from rivalsim.rival2_recurrent_ppo import Rival2RecurrentRolloutBuffer
from rivalsim.rival2_recurrent_training import Rival2RecurrentTrainer
from rivalsim.rival2_unified_policy import Rival2UnifiedActorCritic, deterministic_unified_action
from third_party.nexto.adapter import NextoPolicyAdapter, NextoStateTensors

OPPONENT_CURRENT = 0
OPPONENT_NEXTO = 1
OPPONENT_FROZEN_V5 = 2
OPPONENT_NAMES = ("current", "nexto", "frozen_unified_v5")

_TOUCH_INDEX = OBS_FIELD_NAMES.index("lifecycle.self_touch_event")
_NO_TOUCH_INDEX = OBS_FIELD_NAMES.index("lifecycle.no_touch_age")
_BALL_VELOCITY_Y_INDEX = OBS_FIELD_NAMES.index("ball.linear_velocity.y")
_SELF_VELOCITY_START = OBS_FIELD_NAMES.index("self.linear_velocity.x")


@dataclass(frozen=True, slots=True)
class SslFoundationOpponentConfig:
    current_probability: float = 0.40
    nexto_probability: float = 0.30
    frozen_v5_probability: float = 0.30
    seed: int = 2026090302

    def __post_init__(self) -> None:
        values = (
            self.current_probability,
            self.nexto_probability,
            self.frozen_v5_probability,
        )
        if any(value < 0.0 for value in values) or abs(sum(values) - 1.0) > 1.0e-12:
            raise ValueError("SSL Foundation opponent probabilities must sum to one")


class Rival2SslFoundationTrainer(Rival2RecurrentTrainer):
    """Current recurrent policy versus current, Nexto, and frozen Unified V5."""

    def __init__(
        self,
        *args: Any,
        frozen_v5_model: Rival2UnifiedActorCritic,
        opponent_config: SslFoundationOpponentConfig,
        scenario_family: torch.Tensor,
        **kwargs: Any,
    ) -> None:
        super().__init__(*args, **kwargs)
        if scenario_family.shape != (self.env.num_envs,):
            raise ValueError("scenario family must provide one entry per world")
        self.opponent_config = opponent_config
        self.opponent_generator = torch.Generator(device=self.device).manual_seed(
            opponent_config.seed
        )
        self.opponent_family = torch.full(
            (self.env.num_envs,), -1, dtype=torch.int64, device=self.device
        )
        self.rival_side = torch.zeros(self.env.num_envs, dtype=torch.int64, device=self.device)
        self.world_rows = torch.arange(self.env.num_envs, device=self.device)
        self.realized_family_assignments = torch.zeros(
            len(OPPONENT_NAMES), dtype=torch.int64, device=self.device
        )
        self.scenario_family = scenario_family.to(device=self.device, dtype=torch.int64).clone()
        self.frozen_v5 = frozen_v5_model.to(self.device).eval()
        self.frozen_v5.requires_grad_(False)
        self.frozen_hidden = self.frozen_v5.initial_hidden(
            self.env.num_envs * 2, device=self.device
        ).reshape(
            self.policy_config.recurrent_layers,
            self.env.num_envs,
            2,
            self.policy_config.hidden_dim,
        )
        self.nexto = NextoPolicyAdapter(self.env.num_envs, device=self.device)
        self.nexto_state = NextoStateTensors.from_bridge(self.env.bridge)
        self.last_rollout_curriculum_metrics: dict[str, Any] = {}
        self.assign_opponents_at_reset(
            torch.ones(self.env.num_envs, dtype=torch.bool, device=self.device)
        )

    def assign_opponents_at_reset(self, reset_mask: torch.Tensor) -> None:
        if reset_mask.shape != (self.env.num_envs,) or reset_mask.device != self.device:
            raise ValueError("opponent reset mask mismatch")
        if not bool(reset_mask.any()):
            return
        draw = torch.rand(
            self.env.num_envs,
            generator=self.opponent_generator,
            device=self.device,
        )
        current_end = self.opponent_config.current_probability
        nexto_end = current_end + self.opponent_config.nexto_probability
        sampled = torch.where(
            draw < current_end,
            OPPONENT_CURRENT,
            torch.where(draw < nexto_end, OPPONENT_NEXTO, OPPONENT_FROZEN_V5),
        ).to(torch.int64)
        side = torch.randint(
            2,
            (self.env.num_envs,),
            generator=self.opponent_generator,
            device=self.device,
        )
        self.opponent_family.copy_(torch.where(reset_mask, sampled, self.opponent_family))
        self.rival_side.copy_(torch.where(reset_mask, side, self.rival_side))
        counts = torch.bincount(self.opponent_family[reset_mask], minlength=len(OPPONENT_NAMES))
        self.realized_family_assignments.add_(counts)
        opponent_side = 1 - self.rival_side
        self.nexto.set_player_index(opponent_side)
        nexto_reset = reset_mask & (self.opponent_family == OPPONENT_NEXTO)
        if bool(nexto_reset.any()):
            self.nexto.activate(nexto_reset)

    def _flat_frozen_hidden(self) -> torch.Tensor:
        return self.frozen_hidden.reshape(
            self.policy_config.recurrent_layers,
            self.env.num_envs * 2,
            self.policy_config.hidden_dim,
        )

    def active_frozen_forward(self, observation, step_reset):
        """Advance only assigned frozen opponents. Reassignments occur after world resets.

        Unused hidden rows are intentionally not advanced; the existing reset path
        zeroes both sides before any new assignment can become active.
        Current-policy sampling still consumes exactly the original full RNG draw.
        """
        rows = self.world_rows[self.opponent_family == OPPONENT_FROZEN_V5]
        flat_index = rows * 2 + 1 - self.rival_side.index_select(0, rows)
        hidden = self._flat_frozen_hidden()
        actors = observation.new_zeros((self.env.num_envs * 2, 13))
        next_hidden = hidden.clone()
        if flat_index.numel():
            selected, selected_hidden = self.frozen_v5.forward_actor(
                observation.reshape(-1, self.policy_config.obs_dim).index_select(0, flat_index),
                hidden.index_select(1, flat_index),
                reset_before=step_reset.reshape(-1).index_select(0, flat_index),
            )
            actors.index_copy_(0, flat_index, selected)
            next_hidden.index_copy_(1, flat_index, selected_hidden)
        return actors, next_hidden

    @torch.no_grad()
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
            store_opponent_family=True,
        )
        observation = self.env.observation
        family_decisions = torch.zeros(len(OPPONENT_NAMES), dtype=torch.int64, device=self.device)
        family_trainable = torch.zeros_like(family_decisions)
        family_goals_for = torch.zeros_like(family_decisions)
        family_goals_against = torch.zeros_like(family_decisions)
        family_reward = torch.zeros(len(OPPONENT_NAMES), dtype=torch.float64, device=self.device)
        component_sum = {
            name: torch.zeros((), dtype=torch.float64, device=self.device)
            for name in (
                "field",
                "access",
                "control",
                "defense",
                "alignment",
                "boost",
                "total",
            )
        }
        touch_events = torch.zeros((), dtype=torch.int64, device=self.device)
        goalward_touches = torch.zeros((), dtype=torch.int64, device=self.device)
        no_touch_truncations = torch.zeros((), dtype=torch.int64, device=self.device)
        speed_sum = torch.zeros((), dtype=torch.float64, device=self.device)
        analog_saturation = torch.zeros((), dtype=torch.int64, device=self.device)
        trainable_samples = torch.zeros((), dtype=torch.int64, device=self.device)

        self.model.eval()
        for _ in range(config.rollout_horizon):
            step_reset = self.reset_before
            actor_flat, value_flat, hidden_after_flat = self.model(
                observation.reshape(-1, self.policy_config.obs_dim),
                self._flat_hidden(),
                reset_before=step_reset.reshape(-1),
            )
            if getattr(self, "optimize_execution", False):
                frozen_actor_flat, frozen_hidden_after_flat = self.active_frozen_forward(
                    observation, step_reset
                )
            else:
                frozen_actor_flat, _frozen_value, frozen_hidden_after_flat = self.frozen_v5(
                    observation.reshape(-1, self.policy_config.obs_dim),
                    self._flat_frozen_hidden(),
                    reset_before=step_reset.reshape(-1),
                )
            value = value_flat.reshape(self.env.num_envs, 2)
            frozen_actor = frozen_actor_flat.reshape(self.env.num_envs, 2, 13)
            sample = sample_hybrid_action(
                actor_flat,
                generator=self.policy_generator,
                config=self.policy_config,
                distribution_override=self.exploration.distribution_override,
            )
            action = sample.action.reshape(self.env.num_envs, 2, 8)
            opponent_side = 1 - self.rival_side
            frozen_rows = self.opponent_family == OPPONENT_FROZEN_V5
            if bool(frozen_rows.any()):
                rows = self.world_rows[frozen_rows]
                sides = opponent_side[frozen_rows]
                action[rows, sides] = deterministic_unified_action(frozen_actor[rows, sides])
            current = self.opponent_family == OPPONENT_CURRENT
            train_mask = torch.zeros((self.env.num_envs, 2), dtype=torch.bool, device=self.device)
            train_mask[current] = True
            noncurrent = ~current
            train_mask[self.world_rows[noncurrent], self.rival_side[noncurrent]] = True

            nexto_mask = self.opponent_family == OPPONENT_NEXTO

            def tick_action(
                _tick: int,
                *,
                base_action: torch.Tensor = action,
                active_nexto: torch.Tensor = nexto_mask,
                active_opponent_side: torch.Tensor = opponent_side,
            ) -> torch.Tensor:
                tick = base_action.clone()
                if bool(active_nexto.any()):
                    ball = self.nexto_state.ball_pos
                    kickoff = (ball[:, 0] == 0.0) & (ball[:, 1] == 0.0)
                    nexto_action, _ = self.nexto.tick_action(
                        self.nexto_state, kickoff, active_mask=active_nexto
                    )
                    rows = self.world_rows[active_nexto]
                    tick[rows, active_opponent_side[active_nexto]] = nexto_action[active_nexto]
                return tick

            transition = self.env.step_with_tick_actions(action, tick_action)
            if getattr(self, "optimize_execution", False) and getattr(
                self.model, "critic_is_independent", False
            ):
                next_value_flat = self.model.isolated_value(
                    transition.transition_observation.reshape(-1, self.policy_config.obs_dim)
                )
            else:
                _next_actor, next_value_flat, _ = self.model(
                    transition.transition_observation.reshape(-1, self.policy_config.obs_dim),
                    hidden_after_flat,
                )
            next_value = next_value_flat.reshape(self.env.num_envs, 2)
            terminated = transition.terminated[:, None].expand(-1, 2)
            truncated = transition.truncated[:, None].expand(-1, 2)
            rollout.add(
                observation=observation,
                action=transition.emitted_action,
                pre_tanh=sample.pre_tanh.reshape(self.env.num_envs, 2, 5),
                old_log_probability=sample.log_probability.reshape(self.env.num_envs, 2),
                value=value,
                reward=transition.reward,
                terminated=terminated,
                truncated=truncated,
                next_value=next_value,
                train_mask=train_mask,
                reset_before=step_reset,
                opponent_family=self.opponent_family[:, None].expand(-1, 2),
            )

            family_decisions += torch.bincount(self.opponent_family, minlength=len(OPPONENT_NAMES))
            family_trainable.scatter_add_(
                0, self.opponent_family, train_mask.sum(dim=1).to(torch.int64)
            )
            selected_reward = torch.where(
                train_mask, transition.reward, torch.zeros_like(transition.reward)
            ).sum(dim=1)
            family_reward += torch.bincount(
                self.opponent_family,
                weights=selected_reward.to(torch.float64),
                minlength=len(OPPONENT_NAMES),
            )
            scoring_team = self.env.bridge.views["rival2.scoring_team_latched"].to(torch.int64)
            goal_rows = transition.terminated
            rival_won = goal_rows & (scoring_team == self.rival_side)
            rival_lost = goal_rows & (scoring_team == opponent_side)
            family_goals_for += torch.bincount(
                self.opponent_family[rival_won], minlength=len(OPPONENT_NAMES)
            )
            family_goals_against += torch.bincount(
                self.opponent_family[rival_lost], minlength=len(OPPONENT_NAMES)
            )
            touch = (transition.transition_observation[..., _TOUCH_INDEX] > 0.5) & train_mask
            touch_events += touch.sum()
            goalward_touches += (
                touch & (transition.transition_observation[..., _BALL_VELOCITY_Y_INDEX] > 0.0)
            ).sum()
            no_touch_truncations += (
                transition.truncated
                & (transition.transition_observation[:, 0, _NO_TOUCH_INDEX] >= 1.0 - 1.0e-6)
            ).sum()
            normalized_speed = torch.linalg.vector_norm(
                observation[
                    ...,
                    _SELF_VELOCITY_START : _SELF_VELOCITY_START + 3,
                ],
                dim=-1,
            )
            speed_sum += torch.where(
                train_mask, normalized_speed, torch.zeros_like(normalized_speed)
            ).sum(dtype=torch.float64)
            analog_saturation += (
                (transition.emitted_action[..., :5].abs() > 0.95) & train_mask.unsqueeze(-1)
            ).sum()
            trainable_samples += train_mask.sum()
            if self.env.last_ssl_foundation_components is None:
                raise RuntimeError("SSL Foundation reward components unavailable")
            for name, component in self.env.last_ssl_foundation_components.items():
                component_sum[name] += torch.where(
                    train_mask, component, torch.zeros_like(component)
                ).sum(dtype=torch.float64)

            reset_agent = transition.reset_mask[:, None].expand(-1, 2)
            next_hidden = hidden_after_flat.reshape_as(self.hidden).masked_fill(
                reset_agent.view(1, self.env.num_envs, 2, 1), 0.0
            )
            next_frozen_hidden = frozen_hidden_after_flat.reshape_as(
                self.frozen_hidden
            ).masked_fill(reset_agent.view(1, self.env.num_envs, 2, 1), 0.0)
            self.hidden = next_hidden
            self.frozen_hidden = next_frozen_hidden
            self.reset_before = reset_agent.clone()
            self.assign_opponents_at_reset(transition.reset_mask)
            if self.env.world.ssl_foundation_reset is not None:
                self.scenario_family.copy_(
                    wp.to_torch(self.env.world.ssl_foundation_reset.current_family).to(torch.int64)
                )
            observation = transition.observation

        self.env.observation = observation
        sample_count = int(trainable_samples.item())
        self.total_agent_samples += sample_count
        self.physical_physics_ticks_experienced += (
            config.rollout_horizon * self.env.num_envs * self.env.physics_ticks_per_decision
        )
        player_minutes = sample_count / (120.0 * 60.0)
        touches = int(touch_events.item())
        self.last_rollout_curriculum_metrics = {
            "opponent_world_decisions": {
                name: int(family_decisions[index].item())
                for index, name in enumerate(OPPONENT_NAMES)
            },
            "opponent_trainable_samples": {
                name: int(family_trainable[index].item())
                for index, name in enumerate(OPPONENT_NAMES)
            },
            "goals_for": {
                name: int(family_goals_for[index].item())
                for index, name in enumerate(OPPONENT_NAMES)
            },
            "goals_against": {
                name: int(family_goals_against[index].item())
                for index, name in enumerate(OPPONENT_NAMES)
            },
            "reward_sum": {
                name: float(family_reward[index].item())
                for index, name in enumerate(OPPONENT_NAMES)
            },
        }
        self.last_rollout_metrics = {
            "reward_version": self.env.reward_version,
            "trainable_agent_samples": sample_count,
            "physical_player_minutes": player_minutes,
            "touch_events": touches,
            "touches_per_minute": touches / max(player_minutes, 1.0e-12),
            "goalward_touch_fraction": int(goalward_touches.item()) / max(1, touches),
            "no_touch_truncations": int(no_touch_truncations.item()),
            "mean_movement_speed_uu_per_second": (
                float(speed_sum.item()) / max(1, sample_count) * CAR_LINEAR_SPEED_SCALE
            ),
            "analog_action_saturation_fraction": (
                int(analog_saturation.item()) / max(1, sample_count * 5)
            ),
            "potential_component_signed_sum": {
                name: float(value.item()) for name, value in component_sum.items()
            },
            "direct_non_goal_reward_terms": 0,
            "named_mechanics_hot_path_absent": (
                self.env.world.gameplay_v3 is None and self.env.world.gameplay_120 is None
            ),
            "curriculum": copy.deepcopy(self.last_rollout_curriculum_metrics),
        }
        return rollout

    def checkpoint_payload(self, *, include_optimizer: bool = True) -> dict[str, Any]:
        payload = super().checkpoint_payload(include_optimizer=include_optimizer)
        payload["opponents"] = {
            "config": asdict(self.opponent_config),
            "names": list(OPPONENT_NAMES),
            "frozen_v5_deterministic": True,
            "nexto_deterministic": True,
            "current_selfplay_both_sides_trainable": True,
            "noncurrent_opponents_inference_only": True,
        }
        payload["opponent_curriculum_state"] = {
            "generator_state": self.opponent_generator.get_state(),
            "realized_family_assignments": self.realized_family_assignments,
        }
        payload["reset_curriculum"] = {
            "scenario_family": self.scenario_family,
            "summary": copy.deepcopy(self.env.world.ssl_foundation_reset.summary),
            "scenario_id_in_observation": False,
        }
        return payload

    def load_checkpoint(self, path: str | Path) -> dict[str, Any]:
        payload = super().load_checkpoint(path)
        state = payload["opponent_curriculum_state"]
        self.opponent_generator.set_state(state["generator_state"].cpu())
        self.realized_family_assignments.copy_(state["realized_family_assignments"].to(self.device))
        self.opponent_family.fill_(-1)
        self.frozen_hidden.zero_()
        self.assign_opponents_at_reset(
            torch.ones(self.env.num_envs, dtype=torch.bool, device=self.device)
        )
        return payload


__all__ = [
    "OPPONENT_CURRENT",
    "OPPONENT_FROZEN_V5",
    "OPPONENT_NAMES",
    "OPPONENT_NEXTO",
    "Rival2SslFoundationTrainer",
    "SslFoundationOpponentConfig",
]
