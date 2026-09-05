"""Categorical rollout/sequence loss for the entity candidate, not a hybrid shim.

No campaign is started on import. The unchanged native 30Hz environment owns
reward/goal/reset/truncation semantics. Current-v-current only in this version.
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
from rivalsim.ssl_joint_control_policy import categorical_statistics


class JointRollout(Rival2RecurrentRolloutBuffer):
    def __init__(self, horizon, worlds, hidden, device):
        super().__init__(horizon, worlds, hidden, device)
        self.action_indices = torch.empty((horizon, worlds, 2), dtype=torch.int64, device=device)

    @property
    def logical_bytes(self):
        return (
            super().logical_bytes + self.action_indices.numel() * self.action_indices.element_size()
        )


class EntityRolloutCollector:
    def __init__(self, env, model, seed=SEED):
        self.env, self.model = env, model.to(env.device)
        self.config = ppo_config()
        self.generator = torch.Generator(device=env.device).manual_seed(seed)
        self.hidden = model.initial_hidden(env.num_envs * 2).reshape(1, env.num_envs, 2, -1)
        self.reset_before = torch.ones((env.num_envs, 2), dtype=torch.bool, device=env.device)
        self.episode_has_touch = torch.zeros_like(self.reset_before)
        self.last_metrics = {}

    @torch.no_grad()
    def collect(self):
        env, model = self.env, self.model
        n, horizon = env.num_envs, self.config.rollout_horizon
        buffer = JointRollout(horizon, n, self.hidden, env.device)
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
            )
        }
        components = {
            key: torch.zeros_like(totals["touches"])
            for key in (*authority()["weights"], "terminal_goal", "total")
        }
        counts = torch.zeros(90, dtype=torch.int64, device=env.device)
        for tick in range(horizon):
            logits, value, hidden = model(
                obs.reshape(-1, 182),
                self.hidden.reshape(1, -1, model.config.context_hidden_dim),
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
            transition = env.step(action)
            native = env.last_native
            next_value = model.isolated_value(
                transition.transition_observation.reshape(-1, 182)
            ).reshape(n, 2)
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
                train_mask=torch.ones_like(reset),
                reset_before=self.reset_before,
            )
            counts += torch.bincount(index, minlength=90)
            touches = native["touch_count"]
            first = (touches > 0) & ~self.episode_has_touch
            self.episode_has_touch |= touches > 0
            totals["touches"] += touches.sum()
            totals["goalward_touches"] += (
                touches * (transition.transition_observation[..., 4] > 0)
            ).sum()
            totals["first_touches"] += first.sum()
            age = native["episode_ticks"][:, None] - 4 + native["first_touch_tick"] + 1
            totals["first_touch_age"] += age.masked_fill(~first, 0).sum() / 120
            totals["ended_player_episodes"] += reset.sum()
            totals["episodes_with_touch"] += (reset & self.episode_has_touch).sum()
            # Both self-play agents are trainable: every physical goal is one
            # player goal and one concede. Do not count both as two world goals.
            totals["goals"] += transition.terminated.sum()
            totals["concedes"] += transition.terminated.sum()
            no_touch = transition.truncated & (native["no_touch_ticks"] >= 1800)
            totals["no_touch"] += no_touch.sum()
            totals["time_limit"] += (transition.truncated & ~no_touch).sum()
            totals["resets"] += transition.reset_mask.sum()
            totals["speed"] += obs[..., 12:15].norm(dim=-1).sum() * 2300
            totals["entropy"] += entropy.sum()
            for j, key in enumerate(("jump", "boost", "handbrake"), 5):
                totals[key] += action[..., j].sum()
            for key, component in env.last_components.items():
                components[key] += component.sum(dtype=torch.float64)
            self.hidden = hidden.reshape_as(self.hidden).masked_fill(reset[None, ..., None], 0)
            self.reset_before = reset.clone()
            self.episode_has_touch.masked_fill_(reset, False)
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
        samples = horizon * n * 2
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
            current_selfplay_only=True,
            kl_telemetry_only=True,
        )
        return buffer


def sequence_data(rollout, config):
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
    data["initial_hidden"] = rollout.initial_hidden.reshape(1, -1, rollout.initial_hidden.shape[-1])
    selected = data["advantages"][data["train_mask"]]
    data["normalized_advantage"] = (data["advantages"] - selected.mean()) / selected.std(
        unbiased=False
    ).clamp_min(1e-8)
    return data


def joint_sequence_loss(model, data, index, config):
    logits, value, _ = model(
        data["observations"][index],
        data["initial_hidden"][:, index],
        reset_before=data["reset_before"][index],
    )
    logp, entropy = categorical_statistics(logits, data["action_indices"][index])
    mask = data["train_mask"][index]
    log_ratio = logp[mask] - data["old_log_probability"][index][mask]
    ratio = log_ratio.exp()
    adv = data["normalized_advantage"][index][mask]
    policy = -torch.minimum(
        ratio * adv, ratio.clamp(1 - config.clip_range, 1 + config.clip_range) * adv
    ).mean()
    # The critic is independent, so this value loss has no actor/GRU/entity path.
    value_loss = 0.5 * (value[mask] - data["returns"][index][mask]).square().mean()
    ent = entropy[mask].mean()
    total = policy + config.value_loss_coefficient * value_loss - config.entropy_coefficient * ent
    return total, dict(
        policy_loss=policy.detach(),
        value_loss=value_loss.detach(),
        entropy=ent.detach(),
        approx_kl=((ratio - 1) - log_ratio).mean().detach(),
        clip_fraction=((ratio - 1).abs() > config.clip_range).float().mean().detach(),
    )


def fresh_entity_optimizer(model):
    """Explicit fresh optimizer for the changed action/attention parameterization.

    No silently projected Adam moments, no BC or old-lineage optimizer reuse.
    This differs from reset/noise pilots and must be bound in the authority.
    """
    return torch.optim.Adam(
        [
            dict(
                params=[p for n, p in model.named_parameters() if not n.startswith("critic.")],
                lr=1e-4,
                name="entity_actor_and_recurrent_trunk",
            ),
            dict(params=model.critic.parameters(), lr=3e-4, name="independent_critic"),
        ]
    )


def joint_ppo_update(model, optimizer, rollout, config, generator):
    """Complete-sequence PPO with full update rollback for corruption, not KL.

    Categorical endpoint actions are intentional; hybrid saturation thresholds
    must not be misapplied to this parser. Entropy/action occupancy are reported.
    """
    old_model = {n: v.detach().clone() for n, v in model.state_dict().items()}
    old_optimizer = copy.deepcopy(optimizer.state_dict())
    rng = generator.get_state().clone()
    try:
        data = sequence_data(rollout, config)
        if not all(
            bool(torch.isfinite(data[k]).all())
            for k in ("returns", "advantages", "normalized_advantage")
        ):
            raise Rival2RecurrentPPOCorruption({"reason": "nonfinite_entity_gae"})
        sequences = data["observations"].shape[0]
        size = config.minibatch_size // rollout.horizon
        metrics = []
        model.train()
        for _epoch in range(config.epochs):
            order = torch.randperm(sequences, device=rollout.device, generator=generator)
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
                checks = [torch.isfinite(p).all() for p in model.parameters()]
                checks += [
                    torch.isfinite(v).all()
                    for s in optimizer.state.values()
                    for v in s.values()
                    if torch.is_tensor(v)
                ]
                if not bool(torch.stack(checks).all()):
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
                index = torch.arange(start, min(start + size, sequences), device=rollout.device)
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
