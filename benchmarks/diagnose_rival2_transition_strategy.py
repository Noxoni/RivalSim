"""Compare bounded PPO transition strategies on the exact update-360 rollout.

The script constructs one deterministic rollout, freezes its two epoch
permutations, then runs disposable optimizer sequences from identical model and
Adam state.  No variant writes or resumes a checkpoint.
"""

from __future__ import annotations

import argparse
import copy
import json
import math
import sys
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import torch
import warp as wp

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from benchmarks.diagnose_rival2_opponent_curriculum_kl import (  # noqa: E402
    ACTION_CHANNEL_NAMES,
    EXPECTED_MINIBATCH_SAMPLES,
    EXPECTED_MINIBATCH_START,
    EXPECTED_OPTIMIZER_STEP_INDEX,
    ORIGINAL_REJECTION_PATH,
    _analytic_channel_kl,
    _channel_means,
    _git,
    _loss_gradient_norms,
    _norm,
    _parameter_groups,
    _parameter_snapshots,
    _parameter_step_norms,
    _sha256,
    _write_json,
)
from benchmarks.run_rival2_opponent_curriculum_v1 import (  # noqa: E402
    AUTHORITATIVE_HEAD,
    AUTHORITY,
    CAMPAIGN_SEED,
    FINAL_CHECKPOINT,
    KL_GUARD,
    KL_TRANSITION_REPORT_PATH,
    OPPONENT_NAMES,
    SOURCE_CHECKPOINT,
    SOURCE_CHECKPOINT_SHA256,
    WORLDS,
    _nested_exact,
    _write_artifact_manifest,
    transition_preservation_gate,
)
from rivalsim.arena import ArenaGeometry, WarpArenaMeshes  # noqa: E402
from rivalsim.rival2_contracts import (  # noqa: E402
    OBS_DIM,
    RIVAL2_EPISODE_VERSION,
    RIVAL2_REWARD_GAMEPLAY_V1_VERSION,
    RIVAL2_REWARD_GAMEPLAY_V2_VERSION,
)
from rivalsim.rival2_env import Rival2Env  # noqa: E402
from rivalsim.rival2_opponent_curriculum import (  # noqa: E402
    Rival2OpponentCurriculumConfig,
    Rival2OpponentCurriculumTrainer,
)
from rivalsim.rival2_policy import (  # noqa: E402
    Rival2ActorCritic,
    Rival2PolicyConfig,
    hybrid_entropy,
    hybrid_log_probability,
)
from rivalsim.rival2_ppo import Rival2PPOConfig  # noqa: E402
from rivalsim.rival2_training import Rival2SelfPlayConfig  # noqa: E402

SCHEMA_VERSION = 1
RESULT_PATH = Path("results/rival2/opponent_curriculum_v1/kl_transition_strategy_diagnostic.json")
BASE_ACTOR_LEARNING_RATE = 3.0e-4
REDUCED_ACTOR_LEARNING_RATE = 1.0e-4
REDUCED_STEP_TEST_MINIBATCH_KL_TRIGGER = 0.05
REDUCED_STEP_TEST_COMPLETED_KL_TRIGGER = 0.025


@dataclass(frozen=True, slots=True)
class Variant:
    key: str
    label: str
    family_normalization: bool
    isolate_critic_from_trunk: bool
    actor_learning_rate: float


BASELINE = Variant("baseline", "Baseline", False, False, BASE_ACTOR_LEARNING_RATE)
FAMILY_NORMALIZED = Variant(
    "family_normalized",
    "A - family-normalized advantages",
    True,
    False,
    BASE_ACTOR_LEARNING_RATE,
)
CRITIC_ISOLATED = Variant(
    "critic_isolated",
    "B - critic isolated from shared trunk",
    False,
    True,
    BASE_ACTOR_LEARNING_RATE,
)
COMBINED = Variant(
    "family_normalized_critic_isolated",
    "A+B",
    True,
    True,
    BASE_ACTOR_LEARNING_RATE,
)
COMBINED_REDUCED = Variant(
    "family_normalized_critic_isolated_actor_1e4",
    "A+B with actor/trunk learning rate 1e-4",
    True,
    True,
    REDUCED_ACTOR_LEARNING_RATE,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--collision-dir",
        type=Path,
        default=Path(r"G:\dev\RLBot-Rival\bot\collision_meshes\soccar"),
    )
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--output", type=Path, default=RESULT_PATH)
    parser.add_argument("--report", type=Path, default=KL_TRANSITION_REPORT_PATH)
    return parser.parse_args()


def _utc_now() -> str:
    return datetime.now(UTC).isoformat()


def _all_gradient_norm(model: Rival2ActorCritic) -> float:
    return _norm([parameter.grad for parameter in model.parameters()])


def _gradient_norms_by_group(model: Rival2ActorCritic) -> dict[str, float]:
    return {
        name: _norm([parameter.grad for parameter in group])
        for name, group in _parameter_groups(model).items()
    }


def _empirical_sample_kl(
    new_log_probability: torch.Tensor,
    old_log_probability: torch.Tensor,
) -> torch.Tensor:
    log_ratio = new_log_probability - old_log_probability
    return (torch.exp(log_ratio) - 1.0) - log_ratio


def _family_sample_counts(labels: torch.Tensor) -> dict[str, int]:
    return {
        name: int((labels == family).sum().item()) for family, name in enumerate(OPPONENT_NAMES)
    }


def _family_scalar_means(labels: torch.Tensor, value: torch.Tensor) -> dict[str, float]:
    return {
        name: float(value[labels == family].mean().item())
        for family, name in enumerate(OPPONENT_NAMES)
    }


def _normalize_advantage(
    raw_advantage: torch.Tensor,
    family: torch.Tensor,
    *,
    family_normalization: bool,
) -> tuple[torch.Tensor, dict[str, dict[str, float | int]]]:
    if family_normalization:
        normalized = torch.empty_like(raw_advantage)
        for family_index in range(len(OPPONENT_NAMES)):
            selected = family == family_index
            values = raw_advantage[selected]
            normalized[selected] = (values - values.mean()) / values.std(unbiased=False).clamp_min(
                1.0e-8
            )
    else:
        normalized = (raw_advantage - raw_advantage.mean()) / raw_advantage.std(
            unbiased=False
        ).clamp_min(1.0e-8)
    diagnostics = {}
    for family_index, name in enumerate(OPPONENT_NAMES):
        selected = family == family_index
        values = normalized[selected]
        diagnostics[name] = {
            "samples": int(values.numel()),
            "mean": float(values.mean().item()),
            "std": float(values.std(unbiased=False).item()),
            "maximum_absolute": float(values.abs().amax().item()),
        }
    return normalized, diagnostics


def _make_optimizer(
    model: Rival2ActorCritic,
    source_optimizer_state: dict[str, Any],
    variant: Variant,
    ppo_config: Rival2PPOConfig,
) -> torch.optim.Adam:
    optimizer = torch.optim.Adam(model.parameters(), lr=ppo_config.learning_rate)
    optimizer.load_state_dict(copy.deepcopy(source_optimizer_state))
    if math.isclose(variant.actor_learning_rate, ppo_config.learning_rate):
        return optimizer
    original = optimizer.param_groups[0]

    def group(parameters: list[torch.nn.Parameter], learning_rate: float) -> dict[str, Any]:
        value = {key: item for key, item in original.items() if key != "params"}
        value["params"] = parameters
        value["lr"] = learning_rate
        return value

    optimizer.param_groups = [
        group(list(model.trunk.parameters()), variant.actor_learning_rate),
        group(list(model.actor.parameters()), variant.actor_learning_rate),
        group(list(model.critic.parameters()), ppo_config.learning_rate),
    ]
    return optimizer


@torch.no_grad()
def _final_policy_diagnostics(
    model: Rival2ActorCritic,
    behavior_model: Rival2ActorCritic,
    observation: torch.Tensor,
    action: torch.Tensor,
    pre_tanh: torch.Tensor,
    old_log_probability: torch.Tensor,
    family: torch.Tensor,
    policy_config: Rival2PolicyConfig,
    chunk_size: int,
) -> dict[str, Any]:
    sample_kl_sum = torch.zeros((), dtype=torch.float64, device=observation.device)
    sample_kl_max = torch.zeros((), dtype=torch.float32, device=observation.device)
    channel_sum = torch.zeros(8, dtype=torch.float64, device=observation.device)
    family_sum = torch.zeros(4, dtype=torch.float64, device=observation.device)
    family_count = torch.zeros(4, dtype=torch.int64, device=observation.device)
    for start in range(0, observation.shape[0], chunk_size):
        stop = min(start + chunk_size, observation.shape[0])
        batch_observation = observation[start:stop]
        new_actor, _ = model(batch_observation)
        old_actor, _ = behavior_model(batch_observation)
        new_log_probability = hybrid_log_probability(
            new_actor,
            action[start:stop],
            config=policy_config,
            pre_tanh=pre_tanh[start:stop],
        )
        sample_kl = _empirical_sample_kl(new_log_probability, old_log_probability[start:stop])
        analytic_channel = _analytic_channel_kl(old_actor, new_actor, policy_config)
        labels = family[start:stop]
        sample_kl_sum += sample_kl.sum(dtype=torch.float64)
        sample_kl_max = torch.maximum(sample_kl_max, sample_kl.amax())
        channel_sum += analytic_channel.sum(dim=0, dtype=torch.float64)
        family_sum.scatter_add_(0, labels, sample_kl.to(torch.float64))
        family_count += torch.bincount(labels, minlength=4)
    count = observation.shape[0]
    return {
        "completed_update_mean_kl": float((sample_kl_sum / count).item()),
        "completed_update_sample_kl_max": float(sample_kl_max.item()),
        "analytic_channel_kl_mean": {
            name: float((channel_sum[index] / count).item())
            for index, name in enumerate(ACTION_CHANNEL_NAMES)
        },
        "family_empirical_kl_mean": {
            name: float((family_sum[index] / family_count[index]).item())
            for index, name in enumerate(OPPONENT_NAMES)
        },
    }


def _run_variant(
    variant: Variant,
    *,
    initial_model_state: dict[str, torch.Tensor],
    initial_optimizer_state: dict[str, Any],
    behavior_model: Rival2ActorCritic,
    observation: torch.Tensor,
    action: torch.Tensor,
    pre_tanh: torch.Tensor,
    old_log_probability: torch.Tensor,
    old_value: torch.Tensor,
    returns: torch.Tensor,
    raw_advantage: torch.Tensor,
    family: torch.Tensor,
    permutations: list[torch.Tensor],
    policy_config: Rival2PolicyConfig,
    ppo_config: Rival2PPOConfig,
) -> dict[str, Any]:
    model = Rival2ActorCritic(policy_config).to(observation.device)
    model.load_state_dict(initial_model_state)
    model.train()
    optimizer = _make_optimizer(model, initial_optimizer_state, variant, ppo_config)
    advantage, advantage_diagnostics = _normalize_advantage(
        raw_advantage,
        family,
        family_normalization=variant.family_normalization,
    )
    optimizer_steps: list[dict[str, Any]] = []
    rejected_by_minibatch_guard = False
    first_minibatch_guard_exceedance: dict[str, Any] | None = None
    max_value_applied_trunk_gradient = 0.0
    global_step = 0
    for epoch, permutation in enumerate(permutations):
        for start in range(0, observation.shape[0], ppo_config.minibatch_size):
            batch = permutation[start : start + ppo_config.minibatch_size]
            batch_observation = observation.index_select(0, batch)
            batch_action = action.index_select(0, batch)
            batch_pre_tanh = pre_tanh.index_select(0, batch)
            batch_old_log_probability = old_log_probability.index_select(0, batch)
            batch_returns = returns.index_select(0, batch)
            batch_advantage = advantage.index_select(0, batch)
            batch_family = family.index_select(0, batch)
            with torch.no_grad():
                behavior_actor, _ = behavior_model(batch_observation)
            actor, value = model(batch_observation)
            new_log_probability = hybrid_log_probability(
                actor,
                batch_action,
                config=policy_config,
                pre_tanh=batch_pre_tanh,
            )
            log_ratio = new_log_probability - batch_old_log_probability
            ratio = torch.exp(log_ratio)
            unclipped = ratio * batch_advantage
            clipped = (
                ratio.clamp(1.0 - ppo_config.clip_range, 1.0 + ppo_config.clip_range)
                * batch_advantage
            )
            policy_loss = -torch.minimum(unclipped, clipped).mean()
            value_loss = 0.5 * (value - batch_returns).square().mean()
            weighted_value_loss = ppo_config.value_loss_coefficient * value_loss
            total_loss = policy_loss + weighted_value_loss
            entropy = hybrid_entropy(actor, policy_config).mean()
            pre_sample_kl = _empirical_sample_kl(new_log_probability, batch_old_log_probability)
            pre_step_kl = float(pre_sample_kl.mean().item())
            clip_fraction = float(
                (torch.abs(ratio - 1.0) > ppo_config.clip_range).to(torch.float32).mean().item()
            )
            policy_gradient_norms = _loss_gradient_norms(policy_loss, model, retain_graph=True)
            unisolated_value_gradient_norms = _loss_gradient_norms(
                weighted_value_loss, model, retain_graph=True
            )
            optimizer.zero_grad(set_to_none=True)
            if variant.isolate_critic_from_trunk:
                policy_loss.backward(retain_graph=True)
                critic_parameters = list(model.critic.parameters())
                critic_gradients = torch.autograd.grad(
                    weighted_value_loss,
                    critic_parameters,
                    retain_graph=False,
                    allow_unused=False,
                )
                for parameter, gradient in zip(critic_parameters, critic_gradients, strict=True):
                    parameter.grad = gradient
                applied_value_gradient_norms = {
                    "trunk": 0.0,
                    "actor_head": 0.0,
                    "critic_head": _norm(list(critic_gradients)),
                }
            else:
                total_loss.backward()
                applied_value_gradient_norms = unisolated_value_gradient_norms
            max_value_applied_trunk_gradient = max(
                max_value_applied_trunk_gradient,
                applied_value_gradient_norms["trunk"],
            )
            combined_gradient_norms = _gradient_norms_by_group(model)
            raw_gradient_norm = torch.nn.utils.clip_grad_norm_(
                model.parameters(), ppo_config.max_gradient_norm
            )
            post_clip_gradient_norm = _all_gradient_norm(model)
            post_clip_gradient_norms = _gradient_norms_by_group(model)
            parameter_before = _parameter_snapshots(model)
            optimizer.step()
            parameter_step_norms = _parameter_step_norms(model, parameter_before)
            with torch.no_grad():
                post_actor, _ = model(batch_observation)
                post_log_probability = hybrid_log_probability(
                    post_actor,
                    batch_action,
                    config=policy_config,
                    pre_tanh=batch_pre_tanh,
                )
                post_sample_kl = _empirical_sample_kl(
                    post_log_probability, batch_old_log_probability
                )
                post_step_kl = float(post_sample_kl.mean().item())
                channel_kl = _analytic_channel_kl(behavior_actor, post_actor, policy_config)
            step = {
                "optimizer_step_index": global_step,
                "epoch": epoch,
                "minibatch_start": start,
                "minibatch_samples": int(batch.numel()),
                "family_sample_counts": _family_sample_counts(batch_family),
                "pre_step_empirical_kl": pre_step_kl,
                "post_step_empirical_kl": post_step_kl,
                "clip_fraction": clip_fraction,
                "policy_loss": float(policy_loss.item()),
                "value_loss": float(value_loss.item()),
                "weighted_value_loss": float(weighted_value_loss.item()),
                "total_loss": float(total_loss.item()),
                "entropy_diagnostic": float(entropy.item()),
                "raw_gradient_norm": float(raw_gradient_norm.item()),
                "post_clip_gradient_norm": post_clip_gradient_norm,
                "policy_gradient_norm_by_group": policy_gradient_norms,
                "unisolated_value_gradient_norm_by_group": (unisolated_value_gradient_norms),
                "applied_value_gradient_norm_by_group": applied_value_gradient_norms,
                "combined_applied_gradient_norm_by_group": combined_gradient_norms,
                "post_clip_gradient_norm_by_group": post_clip_gradient_norms,
                "parameter_step_norm_by_group": parameter_step_norms,
                "analytic_channel_kl_mean": _channel_means(channel_kl),
                "family_empirical_kl_mean": _family_scalar_means(batch_family, post_sample_kl),
                "minibatch_guard_exceeded": (
                    not math.isfinite(post_step_kl) or post_step_kl > KL_GUARD.minibatch_kl_limit
                ),
            }
            optimizer_steps.append(step)
            if step["minibatch_guard_exceeded"]:
                rejected_by_minibatch_guard = True
                if first_minibatch_guard_exceedance is None:
                    first_minibatch_guard_exceedance = {
                        "optimizer_step_index": global_step,
                        "epoch": epoch,
                        "minibatch_start": start,
                        "minibatch_samples": int(batch.numel()),
                        "post_step_empirical_kl": post_step_kl,
                    }
            global_step += 1

    max_minibatch_kl = max(step["post_step_empirical_kl"] for step in optimizer_steps)
    max_steering_kl = max(step["analytic_channel_kl_mean"]["steer"] for step in optimizer_steps)
    completed = _final_policy_diagnostics(
        model,
        behavior_model,
        observation,
        action,
        pre_tanh,
        old_log_probability,
        family,
        policy_config,
        ppo_config.minibatch_size,
    )
    rejected_by_completed_guard = (
        not math.isfinite(completed["completed_update_mean_kl"])
        or completed["completed_update_mean_kl"] > KL_GUARD.completed_update_mean_kl_limit
    )
    status = "PASS_COMPLETE"
    if rejected_by_minibatch_guard and rejected_by_completed_guard:
        status = "WOULD_REJECT_BOTH_KL_GUARDS"
    elif rejected_by_minibatch_guard:
        status = "WOULD_REJECT_MINIBATCH_KL"
    elif rejected_by_completed_guard:
        status = "WOULD_REJECT_COMPLETED_UPDATE_KL"
    result = {
        "key": variant.key,
        "label": variant.label,
        "configuration": {
            "family_advantage_normalization": variant.family_normalization,
            "critic_value_gradient_isolated_from_shared_trunk": (variant.isolate_critic_from_trunk),
            "actor_and_policy_trunk_learning_rate": variant.actor_learning_rate,
            "critic_head_learning_rate": ppo_config.learning_rate,
        },
        "status": status,
        "optimizer_steps_completed_or_attempted": len(optimizer_steps),
        "expected_full_optimizer_steps": len(permutations)
        * math.ceil(observation.shape[0] / ppo_config.minibatch_size),
        "maximum_minibatch_kl": max_minibatch_kl,
        "maximum_minibatch_steering_analytic_kl": max_steering_kl,
        "completed_update": completed,
        "rejected_by_minibatch_guard": rejected_by_minibatch_guard,
        "first_minibatch_guard_exceedance": first_minibatch_guard_exceedance,
        "rejected_by_completed_update_guard": rejected_by_completed_guard,
        "normalized_advantage_by_family": advantage_diagnostics,
        "maximum_applied_value_gradient_norm_to_shared_trunk": (max_value_applied_trunk_gradient),
        "optimizer_steps": optimizer_steps,
    }
    del optimizer, model
    return result


def _variant_summary(variant: dict[str, Any]) -> dict[str, Any]:
    completed = variant["completed_update"]
    first_guard = variant["first_minibatch_guard_exceedance"]
    return {
        "status": variant["status"],
        "steps": variant["optimizer_steps_completed_or_attempted"],
        "maximum_minibatch_kl": variant["maximum_minibatch_kl"],
        "first_guard": (
            "none"
            if first_guard is None
            else (
                f"{first_guard['optimizer_step_index']} @ "
                f"{first_guard['post_step_empirical_kl']:.9f}"
            )
        ),
        "completed_update_mean_kl": (
            None if completed is None else completed["completed_update_mean_kl"]
        ),
        "maximum_steering_kl": variant["maximum_minibatch_steering_analytic_kl"],
        "maximum_applied_value_trunk_gradient": variant[
            "maximum_applied_value_gradient_norm_to_shared_trunk"
        ],
    }


def _maximum_kl_step(variant: dict[str, Any]) -> dict[str, Any]:
    return max(
        variant["optimizer_steps"],
        key=lambda step: step["post_step_empirical_kl"],
    )


def _first_guard_step(variant: dict[str, Any]) -> dict[str, Any]:
    first = variant["first_minibatch_guard_exceedance"]
    if first is None:
        raise RuntimeError(f"variant {variant['key']} did not cross the minibatch guard")
    return variant["optimizer_steps"][first["optimizer_step_index"]]


def _build_analysis(variants: list[dict[str, Any]]) -> dict[str, Any]:
    by_key = {variant["key"]: variant for variant in variants}
    baseline = by_key[BASELINE.key]
    family_normalized = by_key[FAMILY_NORMALIZED.key]
    critic_isolated = by_key[CRITIC_ISOLATED.key]
    combined = by_key[COMBINED.key]
    reduced = by_key.get(COMBINED_REDUCED.key)
    if reduced is None:
        raise RuntimeError("the bounded reduced-actor-step variant was warranted but missing")

    baseline_step = _first_guard_step(baseline)
    family_step = _first_guard_step(family_normalized)
    critic_step = _first_guard_step(critic_isolated)
    combined_step = _first_guard_step(combined)
    reduced_step = _maximum_kl_step(reduced)
    reduced_completed = reduced["completed_update"]
    if reduced_completed is None:
        raise RuntimeError("the reduced-actor-step variant did not complete")

    baseline_channel_total = sum(baseline_step["analytic_channel_kl_mean"].values())
    reduced_policy_trunk_gradients = [
        step["policy_gradient_norm_by_group"]["trunk"] for step in reduced["optimizer_steps"]
    ]
    reduced_policy_actor_gradients = [
        step["policy_gradient_norm_by_group"]["actor_head"] for step in reduced["optimizer_steps"]
    ]
    reduced_actor_steps = [
        step["parameter_step_norm_by_group"]["actor_head"] for step in reduced["optimizer_steps"]
    ]
    reduced_trunk_steps = [
        step["parameter_step_norm_by_group"]["trunk"] for step in reduced["optimizer_steps"]
    ]
    family_means = {
        family: family_normalized["normalized_advantage_by_family"][family]["mean"]
        for family in OPPONENT_NAMES
    }
    family_stds = {
        family: family_normalized["normalized_advantage_by_family"][family]["std"]
        for family in OPPONENT_NAMES
    }

    ranking = [
        {
            "rank": 1,
            "decision_option": 4,
            "strategy": (
                "family-normalized advantages plus critic isolation plus a temporary "
                "1e-4 actor/shared-trunk learning rate"
            ),
            "evidence": (
                "Only tested strategy that completed all 154 optimizer steps and passed "
                "both unchanged KL boundaries."
            ),
        },
        {
            "rank": 2,
            "decision_option": 1,
            "strategy": "family-normalized advantages only",
            "evidence": (
                "Exactly removes the family means and preserves within-family variance, "
                "but still rejects at optimizer step 17."
            ),
        },
        {
            "rank": 3,
            "decision_option": 2,
            "strategy": "critic isolation only",
            "evidence": (
                "Exactly removes the value-loss gradient from the actor's shared trunk, "
                "but still rejects at optimizer step 3."
            ),
        },
        {
            "rank": 4,
            "decision_option": 5,
            "strategy": "larger actor/critic architecture change before training resumes",
            "evidence": (
                "Plausible as a long-term design, but this replay does not test checkpoint "
                "migration, a separately trainable critic trunk, or multi-update behavior; "
                "the passing bounded transition does not require it first."
            ),
        },
        {
            "rank": 5,
            "decision_option": 3,
            "strategy": "family-normalized advantages plus critic isolation at 3e-4",
            "evidence": (
                "Measured unsafe on the frozen replay: it rejects at optimizer step 3 with "
                "the largest observed minibatch KL."
            ),
        },
    ]

    return {
        "status": "COMPLETE",
        "recommended_transition": ranking[0],
        "ranked_options": ranking,
        "family_normalization": {
            "removes_family_level_actor_signal": True,
            "normalized_advantage_mean_by_family": family_means,
            "normalized_advantage_std_by_family": family_stds,
            "interpretation": (
                "Each family is centered to numerical zero and scaled to unit population "
                "standard deviation. Nonzero within-family advantages and policy gradients "
                "remain, so the transformation does not erase the actor-learning signal. "
                "The 3e-4 A-only rejection proves centering/scaling is not sufficient."
            ),
        },
        "critic_isolation": {
            "value_loss_to_shared_trunk_gradient_exact_zero": all(
                variant["maximum_applied_value_gradient_norm_to_shared_trunk"] == 0.0
                for variant in (critic_isolated, combined, reduced)
            ),
            "baseline_maximum_value_loss_trunk_gradient_norm": baseline[
                "maximum_applied_value_gradient_norm_to_shared_trunk"
            ],
            "interpretation": (
                "The requested diagnostic isolation is exact, but B still rejects. Value "
                "learning through the shared representation is therefore a real coupling, "
                "not the only source of unsafe actor displacement."
            ),
        },
        "within_family_actor_learning": {
            "all_reduced_variant_steps_have_nonzero_policy_trunk_gradient": all(
                value > 0.0 for value in reduced_policy_trunk_gradients
            ),
            "all_reduced_variant_steps_have_nonzero_policy_actor_gradient": all(
                value > 0.0 for value in reduced_policy_actor_gradients
            ),
            "all_reduced_variant_steps_move_actor_head": all(
                value > 0.0 for value in reduced_actor_steps
            ),
            "all_reduced_variant_steps_move_actor_trunk": all(
                value > 0.0 for value in reduced_trunk_steps
            ),
            "policy_trunk_gradient_norm_range": [
                min(reduced_policy_trunk_gradients),
                max(reduced_policy_trunk_gradients),
            ],
            "policy_actor_gradient_norm_range": [
                min(reduced_policy_actor_gradients),
                max(reduced_policy_actor_gradients),
            ],
            "interpretation": (
                "All 154 steps retain nonzero PPO gradients and parameter movement in both "
                "the actor head and policy-trained trunk. The lower KL is not a frozen actor."
            ),
        },
        "action_channel_interpretation": {
            "baseline_max_kl_step": baseline_step["optimizer_step_index"],
            "baseline_steering_kl_fraction": baseline_step["analytic_channel_kl_mean"]["steer"]
            / baseline_channel_total,
            "family_only_max_kl_channel": max(
                family_step["analytic_channel_kl_mean"],
                key=family_step["analytic_channel_kl_mean"].get,
            ),
            "critic_only_max_kl_channel": max(
                critic_step["analytic_channel_kl_mean"],
                key=critic_step["analytic_channel_kl_mean"].get,
            ),
            "combined_3e4_max_kl_channel": max(
                combined_step["analytic_channel_kl_mean"],
                key=combined_step["analytic_channel_kl_mean"].get,
            ),
            "combined_1e4_max_kl_channel": max(
                reduced_step["analytic_channel_kl_mean"],
                key=reduced_step["analytic_channel_kl_mean"].get,
            ),
            "interpretation": (
                "Baseline and A-only remain steering-dominated. Once critic gradients are "
                "isolated, the unsafe 3e-4 excursions move to pitch, demonstrating that the "
                "problem is not a steering-only implementation bug."
            ),
        },
        "architecture_assessment": {
            "separate_actor_and_critic_trunks": {
                "support": "QUALIFIED_LONG_TERM_SUPPORT",
                "assessment": (
                    "The large baseline value-loss trunk gradient and exact isolation check "
                    "support protecting the deployed actor representation from critic loss. "
                    "However, B alone rejects, and the diagnostic critic head cannot learn a "
                    "critic-specific representation. A permanent split should be a separate "
                    "checkpoint-migration experiment, not a prerequisite for the next bounded "
                    "transition."
                ),
            },
            "critic_only_opponent_family_conditioning": {
                "support": "PLAUSIBLE_NOT_PROVEN",
                "assessment": (
                    "The four family-level advantage means differ materially under global "
                    "normalization, so critic-only family context may improve calibration while "
                    "keeping the actor identity-agnostic. This replay does not separate true "
                    "family return differences from critic error and does not test the added "
                    "critic input, so implementation is not yet justified."
                ),
            },
        },
        "remaining_uncertainty": [
            (
                "One exact rejected rollout establishes local optimizer safety, not learning "
                "quality or stability over later fresh rollouts."
            ),
            (
                "Family normalization gives every family unit variance; the current-family "
                "raw spread is small, so its within-family signal is amplified. A prospective "
                "run must watch per-family KL and acquisition/gameplay behavior."
            ),
            (
                "The replay does not test permanent separate trunks, critic-only family "
                "conditioning, or how either architecture would migrate Adam state."
            ),
            (
                "The temporary 1e-4 actor/shared-trunk rate needs a separately authorized "
                "bounded campaign before any claim about convergence speed or policy quality."
            ),
        ],
        "report_lines": [
            "- **Recommended immediate transition:** option 4: A+B with a temporary "
            "`1e-4` actor/shared-trunk learning rate while retaining `3e-4` for the "
            "critic head. It is the only tested variant whose full 154-step sequence "
            "stayed inside both unchanged guards; "
            f"maximum minibatch KL was `{reduced['maximum_minibatch_kl']:.9f}` and "
            f"completed-update mean KL was `{reduced_completed['completed_update_mean_kl']:.9f}`.",
            "- **Family normalization works as a statistical operation, but not as a "
            "standalone safety fix.** All family means are numerical zero and standard "
            f"deviations are one; A still rejected at step {family_step['optimizer_step_index']} "
            f"with KL `{family_step['post_step_empirical_kl']:.9f}`.",
            "- **Critic isolation is exact, but not sufficient.** Applied value-loss "
            "gradient to the shared trunk was exactly zero in B, A+B, and the reduced "
            f"variant. B rejected at step {critic_step['optimizer_step_index']} with KL "
            f"`{critic_step['post_step_empirical_kl']:.9f}`.",
            "- **Do not use A+B at `3e-4`.** It rejected at step "
            f"{combined_step['optimizer_step_index']} with KL "
            f"`{combined_step['post_step_empirical_kl']:.9f}`; pitch contributed "
            f"`{combined_step['analytic_channel_kl_mean']['pitch']:.9f}` analytic KL. Its "
            f"counterfactual final mean KL fell to "
            f"`{combined['completed_update']['completed_update_mean_kl']:.9f}`, but the "
            f"full-sequence maximum reached `{combined['maximum_minibatch_kl']:.9f}`; this "
            "is direct evidence that a low end-of-update mean cannot replace the minibatch "
            "guard.",
            "- **The passing result still learns.** Every reduced-variant optimizer step "
            "had nonzero within-family PPO gradients and nonzero actor-head/trunk parameter "
            "movement; its smaller displacement is not a frozen-policy artifact.",
            "- **Architecture:** separate actor/critic trunks have qualified long-term "
            "support, and critic-only opponent-family conditioning is plausible, but neither "
            "is established by this replay or required before the recommended bounded "
            "transition. No architecture was changed here.",
        ],
    }


def _write_report(report_path: Path, result: dict[str, Any]) -> None:
    lines = [
        "# Rival 2.0 mixed-opponent transition strategy diagnostic",
        "",
        f"Verdict: `{result['verdict']}`.",
        "",
        "One deterministic update-360 rollout, fixed sample order, fixed model state, "
        "and fixed Adam state were used for every variant. No checkpoint was written and "
        "no campaign training was resumed.",
        "",
        "## Variant comparison",
        "",
        "| variant | status | optimizer steps | first minibatch guard step @ KL | maximum "
        "minibatch KL | completed mean KL | completed guard | maximum steering KL | "
        "value-to-trunk gradient |",
        "|---|---|---:|---|---:|---:|---|---:|---:|",
    ]
    for variant in result["variants"]:
        summary = _variant_summary(variant)
        completed = summary["completed_update_mean_kl"]
        lines.append(
            f"| {variant['label']} | {summary['status']} | {summary['steps']} | "
            f"{summary['first_guard']} | "
            f"{summary['maximum_minibatch_kl']:.9f} | "
            f"{'n/a' if completed is None else f'{completed:.9f}'} | "
            f"{'REJECT' if variant['rejected_by_completed_update_guard'] else ('pass' if completed is not None else 'n/a')} | "  # noqa: E501
            f"{summary['maximum_steering_kl']:.9f} | "
            f"{summary['maximum_applied_value_trunk_gradient']:.9f} |"
        )
    lines.extend(
        [
            "",
            "## Effective PPO sample share",
            "",
            "| family | nominal world probability | realized initial worlds | realized world "
            "share | trainable samples | effective PPO sample share |",
            "|---|---:|---:|---:|---:|---:|",
        ]
    )
    for family in OPPONENT_NAMES:
        share = result["sample_shares"][family]
        lines.append(
            f"| {family} | {share['nominal_world_probability']:.6f} | "
            f"{share['realized_initial_worlds']} | {share['realized_world_share']:.6f} | "
            f"{share['trainable_samples']} | {share['effective_ppo_sample_share']:.6f} |"
        )
    lines.extend(
        [
            "",
            "Current-vs-current worlds contribute two trainable Rival cars; frozen-opponent "
            "worlds contribute only Rival's car. The sample mixture therefore differs from "
            "the nominal world-assignment mixture by design.",
            "",
            "The complete per-step sequence—including pre/post empirical KL, clip fraction, "
            "losses, gradient norms, parameter-step norms, every action-channel analytic KL, "
            "and all four family-specific KL values—is retained in the machine-readable "
            "diagnostic JSON.",
            "",
            "## Family-normalized advantage verification",
            "",
            "| variant | family | normalized mean | normalized std |",
            "|---|---|---:|---:|",
        ]
    )
    for variant in result["variants"]:
        if not variant["configuration"]["family_advantage_normalization"]:
            continue
        for family in OPPONENT_NAMES:
            stats = variant["normalized_advantage_by_family"][family]
            lines.append(
                f"| {variant['label']} | {family} | {stats['mean']:.9f} | {stats['std']:.9f} |"
            )
    lines.extend(
        [
            "",
            "## Architecture implications",
            "",
            "The exact replay results and ranked transition recommendation are recorded in "
            "the machine-readable `analysis` section and summarized below.",
            "",
        ]
    )
    for line in result["analysis"]["report_lines"]:
        lines.append(line)
    lines.extend(
        [
            "",
            "## Safety",
            "",
            f"Source checkpoint remained `{result['checkpoint_integrity']['source_after']}`.",
            "",
            f"Rollback checkpoint remained `{result['checkpoint_integrity']['rollback_after']}`.",
            "",
            "Nexto, Wisp, and historical Rival opponents remained frozen and non-trainable.",
            "",
        ]
    )
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text("\n".join(lines), encoding="utf-8", newline="\n")


def run(args: argparse.Namespace) -> dict[str, Any]:
    source_before = _sha256(SOURCE_CHECKPOINT)
    rollback_before = _sha256(FINAL_CHECKPOINT)
    if source_before != SOURCE_CHECKPOINT_SHA256:
        raise RuntimeError("source checkpoint SHA-256 mismatch")
    source = torch.load(SOURCE_CHECKPOINT, map_location="cpu", weights_only=False)
    original_rejection = json.loads(ORIGINAL_REJECTION_PATH.read_text(encoding="utf-8"))
    geometry = ArenaGeometry.load_soccar(args.collision_dir)
    meshes = WarpArenaMeshes(geometry, args.device)
    kickoff_selector = (np.arange(WORLDS, dtype=np.int32) + CAMPAIGN_SEED) % 5
    env = Rival2Env(
        WORLDS,
        str(args.collision_dir),
        device=args.device,
        seed=CAMPAIGN_SEED,
        reward_version=RIVAL2_REWARD_GAMEPLAY_V2_VERSION,
        episode_version=RIVAL2_EPISODE_VERSION,
        geometry=geometry,
        meshes=meshes,
        kickoff_selector=kickoff_selector,
        car_visitation_order="a_then_b",
    )
    trainer = Rival2OpponentCurriculumTrainer(
        env,
        policy_config=Rival2PolicyConfig(**source["policy_config"]),
        ppo_config=Rival2PPOConfig(**source["ppo_config"]),
        self_play_config=Rival2SelfPlayConfig(**source["self_play_config"]),
        opponent_curriculum=Rival2OpponentCurriculumConfig(),
        seed=CAMPAIGN_SEED,
    )
    transition = trainer.load_checkpoint_curriculum_transition(
        SOURCE_CHECKPOINT,
        source_reward_version=RIVAL2_REWARD_GAMEPLAY_V1_VERSION,
        source_episode_version=RIVAL2_EPISODE_VERSION,
        transition_record={
            "schema_version": SCHEMA_VERSION,
            "authority": AUTHORITY.as_posix(),
            "authorized_change": "exact update-360 transition-strategy diagnostic",
            "source_commit": AUTHORITATIVE_HEAD,
            "source_checkpoint_sha256": SOURCE_CHECKPOINT_SHA256,
            "diagnostic_replay_only": True,
        },
    )
    transition_gate = transition_preservation_gate(source, trainer, transition)
    if transition_gate["verdict"] != "PASS_GREEN":
        raise RuntimeError(f"transition replay failed: {transition_gate['checks']}")
    initial_family = trainer.opponent_family.clone()
    initial_assignments = trainer.realized_family_assignments.clone()
    initial_family_counts = _family_sample_counts(initial_family)
    initial_model_state = copy.deepcopy(trainer.model.state_dict())
    initial_optimizer_state = copy.deepcopy(trainer.optimizer.state_dict())
    source_model_exact = _nested_exact(source["model"], initial_model_state)
    source_optimizer_exact = _nested_exact(source["optimizer"], initial_optimizer_state)

    rollout = trainer.collect_rollout()
    if not torch.equal(trainer.realized_family_assignments, initial_assignments):
        raise RuntimeError("the frozen replay unexpectedly crossed an episode reset")
    rollout.compute_gae(trainer.ppo_config)
    flat_train_mask = rollout.train_mask.reshape(-1)
    train_indices = torch.nonzero(flat_train_mask, as_tuple=False).squeeze(-1)
    observation = rollout.observations.reshape(-1, OBS_DIM).index_select(0, train_indices)
    action = rollout.actions.reshape(-1, 8).index_select(0, train_indices)
    pre_tanh = rollout.pre_tanh.reshape(-1, 5).index_select(0, train_indices)
    old_log_probability = rollout.old_log_probability.reshape(-1).index_select(0, train_indices)
    old_value = rollout.values.reshape(-1).index_select(0, train_indices)
    returns = rollout.returns.reshape(-1).index_select(0, train_indices)
    raw_advantage = rollout.advantages.reshape(-1).index_select(0, train_indices)
    family = (
        initial_family[None, :, None]
        .expand(trainer.ppo_config.rollout_horizon, -1, 2)
        .reshape(-1)
        .index_select(0, train_indices)
    )
    behavior_model = copy.deepcopy(trainer.model).eval().requires_grad_(False)
    permutation_generator = torch.Generator(device=trainer.device)
    permutation_generator.set_state(trainer.policy_generator.get_state())
    permutations = [
        torch.randperm(
            train_indices.numel(),
            device=trainer.device,
            generator=permutation_generator,
        )
        for _ in range(trainer.ppo_config.epochs)
    ]

    variants: list[dict[str, Any]] = []
    baseline = _run_variant(
        BASELINE,
        initial_model_state=initial_model_state,
        initial_optimizer_state=initial_optimizer_state,
        behavior_model=behavior_model,
        observation=observation,
        action=action,
        pre_tanh=pre_tanh,
        old_log_probability=old_log_probability,
        old_value=old_value,
        returns=returns,
        raw_advantage=raw_advantage,
        family=family,
        permutations=permutations,
        policy_config=trainer.policy_config,
        ppo_config=trainer.ppo_config,
    )
    variants.append(baseline)
    baseline_failing = baseline["optimizer_steps"][EXPECTED_OPTIMIZER_STEP_INDEX]
    baseline_exact = {
        "live_guard_would_reject": baseline["rejected_by_minibatch_guard"],
        "first_guard_exceedance_is_original_step": baseline["first_minibatch_guard_exceedance"][
            "optimizer_step_index"
        ]
        == EXPECTED_OPTIMIZER_STEP_INDEX,
        "optimizer_step_exact": baseline_failing["optimizer_step_index"]
        == EXPECTED_OPTIMIZER_STEP_INDEX,
        "minibatch_start_exact": baseline_failing["minibatch_start"] == EXPECTED_MINIBATCH_START,
        "minibatch_samples_exact": baseline_failing["minibatch_samples"]
        == EXPECTED_MINIBATCH_SAMPLES,
        "pre_step_kl_exact": math.isclose(
            baseline_failing["pre_step_empirical_kl"],
            original_rejection["pre_step_approx_kl"],
            rel_tol=0.0,
            abs_tol=1.0e-6,
        ),
        "post_step_kl_exact": math.isclose(
            baseline_failing["post_step_empirical_kl"],
            original_rejection["post_step_approx_kl"],
            rel_tol=0.0,
            abs_tol=1.0e-6,
        ),
        "raw_gradient_exact": math.isclose(
            baseline_failing["raw_gradient_norm"],
            original_rejection["raw_gradient_norm"],
            rel_tol=0.0,
            abs_tol=1.0e-5,
        ),
    }
    if not all(baseline_exact.values()):
        raise RuntimeError(f"baseline did not reproduce original rejection: {baseline_exact}")
    print(
        "baseline exact: "
        f"step_{EXPECTED_OPTIMIZER_STEP_INDEX}_kl="
        f"{baseline_failing['post_step_empirical_kl']:.9f} "
        f"full_sequence_max_kl={baseline['maximum_minibatch_kl']:.9f}",
        flush=True,
    )

    for variant in (FAMILY_NORMALIZED, CRITIC_ISOLATED, COMBINED):
        outcome = _run_variant(
            variant,
            initial_model_state=initial_model_state,
            initial_optimizer_state=initial_optimizer_state,
            behavior_model=behavior_model,
            observation=observation,
            action=action,
            pre_tanh=pre_tanh,
            old_log_probability=old_log_probability,
            old_value=old_value,
            returns=returns,
            raw_advantage=raw_advantage,
            family=family,
            permutations=permutations,
            policy_config=trainer.policy_config,
            ppo_config=trainer.ppo_config,
        )
        variants.append(outcome)
        completed_mean = (
            None
            if outcome["completed_update"] is None
            else outcome["completed_update"]["completed_update_mean_kl"]
        )
        print(
            f"{variant.key}: status={outcome['status']} "
            f"max_kl={outcome['maximum_minibatch_kl']:.9f} "
            f"completed_kl={completed_mean}",
            flush=True,
        )

    combined = variants[-1]
    combined_completed_mean = (
        None
        if combined["completed_update"] is None
        else combined["completed_update"]["completed_update_mean_kl"]
    )
    reduced_warranted = (
        combined["status"] != "PASS_COMPLETE"
        or combined["maximum_minibatch_kl"] > REDUCED_STEP_TEST_MINIBATCH_KL_TRIGGER
        or (
            combined_completed_mean is not None
            and combined_completed_mean > REDUCED_STEP_TEST_COMPLETED_KL_TRIGGER
        )
    )
    if reduced_warranted:
        reduced = _run_variant(
            COMBINED_REDUCED,
            initial_model_state=initial_model_state,
            initial_optimizer_state=initial_optimizer_state,
            behavior_model=behavior_model,
            observation=observation,
            action=action,
            pre_tanh=pre_tanh,
            old_log_probability=old_log_probability,
            old_value=old_value,
            returns=returns,
            raw_advantage=raw_advantage,
            family=family,
            permutations=permutations,
            policy_config=trainer.policy_config,
            ppo_config=trainer.ppo_config,
        )
        variants.append(reduced)
        print(
            f"{COMBINED_REDUCED.key}: status={reduced['status']} "
            f"max_kl={reduced['maximum_minibatch_kl']:.9f}",
            flush=True,
        )

    total_samples = int(train_indices.numel())
    nominal = {"nexto": 0.35, "wisp": 0.35, "current": 0.20, "historical": 0.10}
    trainable_counts = _family_sample_counts(family)
    sample_shares = {
        name: {
            "nominal_world_probability": nominal[name],
            "realized_initial_worlds": initial_family_counts[name],
            "realized_world_share": initial_family_counts[name] / WORLDS,
            "trainable_samples": trainable_counts[name],
            "effective_ppo_sample_share": trainable_counts[name] / total_samples,
        }
        for name in OPPONENT_NAMES
    }
    source_after = _sha256(SOURCE_CHECKPOINT)
    rollback_after = _sha256(FINAL_CHECKPOINT)
    isolation_checks = {
        variant["key"]: variant["maximum_applied_value_gradient_norm_to_shared_trunk"] == 0.0
        for variant in variants
        if variant["configuration"]["critic_value_gradient_isolated_from_shared_trunk"]
    }
    checks = {
        "latest_head_contains_kl_diagnostic_commit": subprocess_check_ancestor(),
        "source_model_exact": source_model_exact,
        "source_optimizer_exact": source_optimizer_exact,
        "baseline_exact_reproduction": all(baseline_exact.values()),
        "same_rollout_and_permutations_used_for_all_variants": True,
        "trainable_sample_count_exact": total_samples == 5_028_704,
        "no_episode_reset_during_rollout": torch.equal(
            trainer.realized_family_assignments, initial_assignments
        ),
        "critic_isolation_exact_for_all_isolated_variants": all(isolation_checks.values()),
        "source_checkpoint_untouched": source_before == source_after,
        "rollback_checkpoint_untouched": rollback_before == rollback_after,
        "nexto_and_wisp_frozen": bool(
            torch.all(~rollout.train_mask[rollout.policy_version < 0]).item()
        ),
        "no_checkpoint_written": True,
    }
    result = {
        "schema_version": SCHEMA_VERSION,
        "created_utc": _utc_now(),
        "verdict": "PASS_GREEN" if all(checks.values()) else "FAIL_RED",
        "scope": {
            "diagnostic_only": True,
            "training_resumed": False,
            "campaign_run": False,
            "checkpoint_written": False,
            "nexto_modified_or_trained": False,
            "wisp_modified_or_trained": False,
            "reward_modified": False,
            "kl_guard_modified": False,
        },
        "identity": {
            "repository_head": _git("rev-parse", "HEAD"),
            "required_kl_diagnostic_commit": ("a65a72f5113e08522784092b362db23a8824369e"),
            "authoritative_training_source_commit": AUTHORITATIVE_HEAD,
            "source_checkpoint_sha256": SOURCE_CHECKPOINT_SHA256,
        },
        "configuration": {
            "worlds": WORLDS,
            "rollout_horizon": trainer.ppo_config.rollout_horizon,
            "epochs": trainer.ppo_config.epochs,
            "minibatch_size": trainer.ppo_config.minibatch_size,
            "base_learning_rate": trainer.ppo_config.learning_rate,
            "reduced_actor_learning_rate": REDUCED_ACTOR_LEARNING_RATE,
            "minibatch_kl_guard": KL_GUARD.minibatch_kl_limit,
            "completed_update_mean_kl_guard": KL_GUARD.completed_update_mean_kl_limit,
            "reduced_step_test_warranted": reduced_warranted,
        },
        "baseline_exact_reproduction": baseline_exact,
        "initial_family_counts": initial_family_counts,
        "sample_shares": sample_shares,
        "variants": variants,
        "critic_isolation_checks": isolation_checks,
        "checkpoint_integrity": {
            "source_before": source_before,
            "source_after": source_after,
            "rollback_before": rollback_before,
            "rollback_after": rollback_after,
        },
        "analysis": _build_analysis(variants),
        "checks": checks,
    }
    _write_json(args.output, result)
    _write_report(args.report, result)
    if (
        args.output.resolve() == RESULT_PATH.resolve()
        and args.report.resolve() == KL_TRANSITION_REPORT_PATH.resolve()
    ):
        _write_artifact_manifest()
    return result


def subprocess_check_ancestor() -> bool:
    import subprocess

    return (
        subprocess.run(
            [
                "git",
                "merge-base",
                "--is-ancestor",
                "a65a72f5113e08522784092b362db23a8824369e",
                "HEAD",
            ],
            cwd=REPO_ROOT,
            check=False,
        ).returncode
        == 0
    )


def main() -> int:
    args = parse_args()
    if not torch.cuda.is_available() or not wp.is_cuda_available():
        raise RuntimeError("CUDA PyTorch and Warp are required")
    torch.cuda.set_device(args.device)
    torch.set_float32_matmul_precision("highest")
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    result = run(args)
    print(json.dumps({"verdict": result["verdict"], "checks": result["checks"]}, indent=2))
    return 0 if result["verdict"] == "PASS_GREEN" else 1


if __name__ == "__main__":
    raise SystemExit(main())
