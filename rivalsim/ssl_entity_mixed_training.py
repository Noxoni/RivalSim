"""Same entity policy and PPO objective, with staged inference-only Nexto.

Separate from the immutable 100-update pilot. Training masks and family-local
advantage normalization exclude opponent actions/returns from optimization.
No rewards, physics, observation fields, or actor architecture are changed.
"""

from __future__ import annotations

import copy

import torch

from rivalsim.fresh_ground_30hz import SEED, authority, ppo_config
from rivalsim.rival2_recurrent_ppo import (
    Rival2RecurrentPPOCorruption,
    Rival2RecurrentRolloutBuffer,
    _sequence_major,
)
from rivalsim.ssl_entity_training import finite_model_and_optimizer, joint_sequence_loss
from rivalsim.ssl_joint_control_policy import categorical_statistics
from third_party.nexto.adapter import NextoPolicyAdapter, NextoStateTensors


def learner_mask(is_nexto, side):
    mask = (~is_nexto)[:, None].expand(-1, 2).clone()
    mask[torch.arange(len(side), device=side.device), side] = True
    return mask


def routine_acquisition(result):
    acquisition, finishing = result["acquisition_selfplay"], result["finishing_selfplay"]
    median = acquisition["median_first_touch_seconds_if_touched"]
    return (
        acquisition["focal_touch_fraction"] >= 0.60
        and median is not None
        and median <= 5
        and finishing["goals_for"] >= 1
    )


def family_normalize(advantage, mask, family):
    if not bool(mask.any()):
        raise ValueError("No current-policy samples")
    identities = torch.unique(family[mask])
    if not bool(((identities == 0) | (identities == 1)).all()):
        raise ValueError("Only current self-play and Nexto families are authorized")
    normalized = torch.zeros_like(advantage)
    for identity in identities.tolist():
        selected = mask & (family == identity)
        values = advantage[selected]
        normalized[selected] = (values - values.mean()) / values.std(unbiased=False).clamp_min(1e-8)
    return normalized


class MixedJointRollout(Rival2RecurrentRolloutBuffer):
    def __init__(self, horizon, worlds, hidden, device):
        super().__init__(horizon, worlds, hidden, device, store_opponent_family=True)
        self.action_indices = torch.empty((horizon, worlds, 2), dtype=torch.int64, device=device)

    @property
    def logical_bytes(self):
        return (
            super().logical_bytes + self.action_indices.numel() * self.action_indices.element_size()
        )


class MixedEntityRolloutCollector:
    def __init__(self, env, model, seed=SEED, *, config=None, reward_component_names=None):
        self.env, self.model = env, model.to(env.device)
        self.config = config if config is not None else ppo_config()
        self.reward_component_names = reward_component_names
        self.generator = torch.Generator(device=env.device).manual_seed(seed)
        initial_hidden = model.initial_hidden(env.num_envs * 2)
        self.hidden = initial_hidden.reshape(initial_hidden.shape[0], env.num_envs, 2, -1)
        self.reset_before = torch.ones((env.num_envs, 2), dtype=torch.bool, device=env.device)
        self.episode_has_touch = torch.zeros_like(self.reset_before)
        self.last_metrics = {}
        self.nexto_probability = 0.0
        self.competence_streak = 0
        self.opponent_generator = torch.Generator(device=env.device).manual_seed(seed + 84)
        self.is_nexto = torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)
        self.side = torch.zeros(env.num_envs, dtype=torch.long, device=env.device)
        self.rows = torch.arange(env.num_envs, device=env.device)
        self.nexto = None
        self.nexto_state = None
        self.assign(torch.ones_like(self.is_nexto))

    def assign(self, reset):
        # Family and learner side change only with fresh physical episodes.
        draw = torch.rand(
            self.env.num_envs, device=self.env.device, generator=self.opponent_generator
        )
        side = torch.randint(
            2, (self.env.num_envs,), device=self.env.device, generator=self.opponent_generator
        )
        self.is_nexto.copy_(torch.where(reset, draw < self.nexto_probability, self.is_nexto))
        self.side.copy_(torch.where(reset, side, self.side))
        if self.nexto_probability and self.nexto is None:
            self.nexto = NextoPolicyAdapter(self.env.num_envs, device=self.env.device)
            for parameter in self.nexto.actor.parameters():
                parameter.requires_grad_(False)
            self.nexto_state = NextoStateTensors.from_bridge(self.env.bridge)
        if self.nexto is not None:
            self.nexto.set_player_index(1 - self.side)
            self.nexto.activate(reset & self.is_nexto)

    def accept_evaluation(self, result):
        self.competence_streak = self.competence_streak + 1 if routine_acquisition(result) else 0
        if self.competence_streak >= 2:
            self.nexto_probability = 0.20
        return self.nexto_probability

    def training_mask(self):
        return learner_mask(self.is_nexto, self.side)

    def opponent_checkpoint_state(self):
        cache = None
        if self.nexto is not None:
            cache = {
                name: getattr(self.nexto, name).clone()
                for name in ("previous_action", "neural_counter", "kickoff_index", "player_index")
            }
        return dict(
            nexto_probability=self.nexto_probability,
            competence_streak=self.competence_streak,
            generator_state=self.opponent_generator.get_state(),
            is_nexto=self.is_nexto.clone(),
            learner_side=self.side.clone(),
            nexto_cache=cache,
            resume_semantics="Fresh physical episodes and hidden; cache/assignments "
            "are provenance only, redraw using saved opponent RNG.",
        )

    @torch.no_grad()
    def collect(self):
        env, model = self.env, self.model
        n, horizon = env.num_envs, self.config.rollout_horizon
        buffer = MixedJointRollout(horizon, n, self.hidden, env.device)
        model.eval()
        obs = env.observation
        totals = {
            key: torch.zeros((), dtype=torch.float64, device=env.device)
            for key in (
                "touches",
                "goalward_touches",
                "goals",
                "concedes",
                "resets",
                "no_touch",
                "time_limit",
                "speed",
                "first_touches",
                "first_touch_age",
                "ended_player_episodes",
                "episodes_with_touch",
                "entropy",
                "jump",
                "boost",
                "handbrake",
                "samples",
                "nexto_samples",
                "physical_goals",
            )
        }
        components = {
            key: torch.zeros_like(totals["touches"])
            for key in (self.reward_component_names if self.reward_component_names is not None
                        else (*authority()["weights"], "terminal_goal", "total"))
        }
        counts = torch.zeros(90, dtype=torch.int64, device=env.device)
        for tick in range(horizon):
            logits, value, hidden = model(
                obs.reshape(-1, 182),
                self.hidden.reshape(self.hidden.shape[0], -1, model.config.context_hidden_dim),
                reset_before=self.reset_before.reshape(-1),
            )
            if not bool(
                torch.isfinite(logits).all()
                & torch.isfinite(value).all()
                & torch.isfinite(hidden).all()
            ):
                raise Rival2RecurrentPPOCorruption({"reason": "nonfinite_entity_rollout_policy"})
            index, action, logp = model.sample(logits, self.generator)
            _, entropy = categorical_statistics(logits, index)
            action = action.reshape(n, 2, 8)
            train_mask = self.training_mask()
            active = self.is_nexto.clone()

            def tick_provider(_tick, action=action, active=active):
                applied = action.clone()
                ball = self.nexto_state.ball_pos
                kickoff = (ball[:, 0] == 0) & (ball[:, 1] == 0)
                controls, _ = self.nexto.tick_action(self.nexto_state, kickoff, active_mask=active)
                selected = self.rows[active]
                applied[selected, 1 - self.side[active]] = controls[active]
                return applied

            transition = (
                env.step_with_tick_actions(action, tick_provider)
                if bool(active.any())
                else env.step(action)
            )
            native = env.last_native
            successor = transition.transition_observation.reshape(-1, 182)
            if hasattr(model, "bootstrap_value"):
                # Peek with history AFTER the current observation. Do not carry
                # the peek state: the next observation is consumed on the next
                # decision, exactly once. Timeout uses this pre-reset state.
                next_value = model.bootstrap_value(successor, hidden).reshape(n, 2)
            else:
                next_value = model.isolated_value(successor).reshape(n, 2)
            reset = transition.reset_mask[:, None].expand(-1, 2)
            buffer.action_indices[tick].copy_(index.reshape(n, 2))
            buffer.add(
                observation=obs,
                action=action,
                pre_tanh=torch.zeros((n, 2, 5), device=env.device),
                old_log_probability=logp.reshape(n, 2),
                value=value.reshape(n, 2),
                reward=transition.reward,
                terminated=transition.terminated[:, None].expand(-1, 2),
                truncated=transition.truncated[:, None].expand(-1, 2),
                next_value=next_value,
                train_mask=train_mask,
                opponent_family=active.long()[:, None].expand(-1, 2),
                reset_before=self.reset_before,
            )
            counts += torch.bincount(index[train_mask.flatten()], minlength=90)
            totals["samples"] += train_mask.sum()
            totals["nexto_samples"] += (train_mask & active[:, None]).sum()
            touches = native["touch_count"] * train_mask
            first = (touches > 0) & ~self.episode_has_touch
            self.episode_has_touch |= touches > 0
            totals["touches"] += touches.sum()
            totals["goalward_touches"] += (
                touches * (transition.transition_observation[..., 4] > 0)
            ).sum()
            totals["first_touches"] += first.sum()
            age = native["episode_ticks"][:, None] - 4 + native["first_touch_tick"] + 1
            totals["first_touch_age"] += age.masked_fill(~first, 0).sum() / 120
            ended = reset & train_mask
            totals["ended_player_episodes"] += ended.sum()
            totals["episodes_with_touch"] += (ended & self.episode_has_touch).sum()
            ended_by_goal = transition.terminated[:, None] & train_mask
            winner = native["scoring_team"][:, None] == torch.arange(2, device=env.device)
            totals["goals"] += (ended_by_goal & winner).sum()
            totals["concedes"] += (ended_by_goal & ~winner).sum()
            totals["physical_goals"] += transition.terminated.sum()
            no_touch = transition.truncated & (native["no_touch_ticks"] >= 1800)
            totals["no_touch"] += no_touch.sum()
            totals["time_limit"] += (transition.truncated & ~no_touch).sum()
            totals["resets"] += transition.reset_mask.sum()
            totals["speed"] += obs[..., 12:15].norm(dim=-1).masked_fill(~train_mask, 0).sum() * 2300
            totals["entropy"] += entropy[train_mask.flatten()].sum()
            for j, key in enumerate(("jump", "boost", "handbrake"), 5):
                totals[key] += action[..., j].masked_fill(~train_mask, 0).sum()
            for key, component in env.last_components.items():
                components[key] += component.masked_fill(~train_mask, 0).sum(dtype=torch.float64)
            self.hidden = hidden.reshape_as(self.hidden).masked_fill(reset[None, ..., None], 0)
            self.reset_before = reset.clone()
            self.episode_has_touch.masked_fill_(reset, False)
            self.assign(transition.reset_mask)
            obs = transition.observation
        for name in (
            "observations",
            "actions",
            "old_log_probability",
            "values",
            "rewards",
            "next_values",
        ):
            if not bool(torch.isfinite(getattr(buffer, name)).all()):
                raise Rival2RecurrentPPOCorruption({"reason": "nonfinite_entity_rollout_" + name})
        samples = int(totals["samples"])
        raw = {key: float(v) for key, v in totals.items()}
        self.last_metrics = dict(
            raw,
            trainable_agent_samples=samples,
            physical_physics_ticks=horizon * n * 4,
            touches_per_minute=raw["touches"] / (samples / 1800),
            movement_speed=raw["speed"] / samples,
            mean_entropy=raw["entropy"] / samples,
            goalward_touch_fraction=raw["goalward_touches"] / max(raw["touches"], 1),
            mean_first_touch_seconds_if_touched=raw["first_touch_age"]
            / max(raw["first_touches"], 1),
            ended_player_episode_touch_fraction=raw["episodes_with_touch"]
            / max(raw["ended_player_episodes"], 1),
            action_index_counts=counts.cpu().tolist(),
            potential_reward_components={k: float(v) for k, v in components.items()},
            current_selfplay_only=bool(raw["nexto_samples"] == 0),
            opponent_nexto_probability=self.nexto_probability,
            nexto_training_sample_count=int(raw["nexto_samples"]),
            learner_only_action_index_counts=True,
            unused_opponent_buffer_actions="Dummy current-policy proposals, not PPO targets. "
            "Nexto controls emitted by per-physics-tick provider.",
            world_reset_counts_include_all_opponent_families=True,
            kl_telemetry_only=True,
        )
        return buffer


def mixed_sequence_data(rollout, config):
    rollout.compute_gae(config)
    data = {
        key: _sequence_major(getattr(rollout, key))
        for key in (
            "observations",
            "action_indices",
            "old_log_probability",
            "advantages",
            "returns",
            "values",
            "reset_before",
            "train_mask",
        )
    }
    data["initial_hidden"] = rollout.initial_hidden.reshape(
        rollout.initial_hidden.shape[0], -1, rollout.initial_hidden.shape[-1])
    family = _sequence_major(rollout.opponent_family)
    data["normalized_advantage"] = family_normalize(data["advantages"], data["train_mask"], family)
    data["opponent_family"] = family
    return data


def mixed_joint_ppo_update(model, optimizer, rollout, config, generator):
    """Complete-sequence PPO with full update rollback for corruption, not KL.

    Categorical endpoint actions are intentional; hybrid saturation thresholds
    must not be misapplied to this parser. Entropy/action occupancy are reported.
    """
    old_model = {n: v.detach().clone() for n, v in model.state_dict().items()}
    old_optimizer = copy.deepcopy(optimizer.state_dict())
    rng = generator.get_state().clone()
    try:
        data = mixed_sequence_data(rollout, config)
        if not all(
            bool(torch.isfinite(data[k]).all())
            for k in ("returns", "advantages", "normalized_advantage")
        ):
            raise Rival2RecurrentPPOCorruption({"reason": "nonfinite_entity_gae"})
        eligible = data["train_mask"].any(dim=1).nonzero().flatten()
        if eligible.numel() == 0:
            raise ValueError("No current-policy training sequences")
        sequences = eligible.numel()
        size = config.minibatch_size // rollout.horizon
        metrics = []
        model.train()
        for _epoch in range(config.epochs):
            order = eligible[torch.randperm(sequences, device=rollout.device, generator=generator)]
            for start in range(0, sequences, size):
                index = order[start : start + size]
                optimizer.zero_grad(set_to_none=True)
                total, report = joint_sequence_loss(model, data, index, config)
                if not bool(torch.isfinite(total)):
                    raise Rival2RecurrentPPOCorruption({"reason": "nonfinite_entity_loss"})
                total.backward()
                try:
                    norm = torch.nn.utils.clip_grad_norm_(
                        model.parameters(), config.max_gradient_norm, error_if_nonfinite=True
                    )
                except RuntimeError as exc:
                    if "non-finite" in str(exc):
                        raise Rival2RecurrentPPOCorruption(
                            {"reason": "nonfinite_entity_gradient"}
                        ) from exc
                    raise
                optimizer.step()
                if not finite_model_and_optimizer(model, optimizer):
                    raise Rival2RecurrentPPOCorruption(
                        {"reason": "nonfinite_entity_parameter_or_adam"}
                    )
                report.update(total_loss=total.detach(), gradient_norm=norm.detach())
                metrics.append(report)
        with torch.no_grad():
            kl_sum = torch.zeros((), dtype=torch.float64, device=rollout.device)
            kl_max = torch.zeros((), device=rollout.device)
            count = 0
            for start in range(0, sequences, size):
                index = eligible[start : min(start + size, sequences)]
                logits, _ = model.forward_actor(
                    data["observations"][index],
                    data["initial_hidden"][:, index],
                    reset_before=data["reset_before"][index],
                )
                if not bool(torch.isfinite(logits).all()):
                    raise Rival2RecurrentPPOCorruption(
                        {"reason": "nonfinite_entity_completed_output"}
                    )
                logp, _ = categorical_statistics(logits, data["action_indices"][index])
                mask = data["train_mask"][index]
                lr = (logp - data["old_log_probability"][index])[mask]
                kl = lr.exp() - 1 - lr
                kl_sum += kl.sum(dtype=torch.float64)
                kl_max = torch.maximum(kl_max, kl.max())
                count += kl.numel()
        result = {key: float(torch.stack([m[key] for m in metrics]).mean()) for key in metrics[0]}
        result.update(
            optimizer_steps=len(metrics),
            completed_update_mean_kl=float(kl_sum / count),
            completed_update_sample_kl_max=float(kl_max),
            kl_rejections=0,
        )
        return result
    except Exception:
        model.load_state_dict(old_model, strict=True)
        optimizer.load_state_dict(old_optimizer)
        generator.set_state(rng)
        optimizer.zero_grad(set_to_none=True)
        raise
