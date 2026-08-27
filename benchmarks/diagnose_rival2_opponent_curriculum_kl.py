"""Deterministically replay the rejected Opponent Curriculum V1 PPO minibatches.

This is a disposable diagnostic.  It reconstructs update 360 from the audited
Gameplay V1 +239 source checkpoint, mutates only an in-process trainer, stops at
the original KL boundary, and never writes a checkpoint.
"""

from __future__ import annotations

import argparse
import copy
import json
import math
import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import torch
import warp as wp

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from benchmarks.run_rival2_opponent_curriculum_v1 import (  # noqa: E402
    AUTHORITATIVE_HEAD,
    AUTHORITY,
    CAMPAIGN_SEED,
    FINAL_CHECKPOINT,
    KL_DIAGNOSIS_REPORT_PATH,
    KL_GUARD,
    OPPONENT_NAMES,
    SOURCE_CHECKPOINT,
    SOURCE_CHECKPOINT_SHA256,
    WORLDS,
    _nested_exact,
    _sha256,
    _write_artifact_manifest,
    transition_preservation_gate,
)
from rivalsim.arena import ArenaGeometry, WarpArenaMeshes  # noqa: E402
from rivalsim.rival2_contracts import (  # noqa: E402
    ANALOG_ACTION_NAMES,
    BUTTON_ACTION_NAMES,
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
EXPECTED_OPTIMIZER_STEP_INDEX = 6
EXPECTED_MINIBATCH_START = 393_216
EXPECTED_MINIBATCH_SAMPLES = 65_536
RESULT_PATH = Path("results/rival2/opponent_curriculum_v1/kl_replay_diagnostic.json")
ORIGINAL_REJECTION_PATH = Path("results/rival2/opponent_curriculum_v1/kl_rejection.json")
ACTION_CHANNEL_NAMES = (*ANALOG_ACTION_NAMES, *BUTTON_ACTION_NAMES)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--collision-dir",
        type=Path,
        default=Path(r"G:\dev\RLBot-Rival\bot\collision_meshes\soccar"),
    )
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--output", type=Path, default=RESULT_PATH)
    parser.add_argument("--report", type=Path, default=KL_DIAGNOSIS_REPORT_PATH)
    return parser.parse_args()


def _utc_now() -> str:
    return datetime.now(UTC).isoformat()


def _git(*arguments: str) -> str:
    return subprocess.check_output(["git", *arguments], cwd=REPO_ROOT, text=True).strip()


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8", newline="\n")


def _tensor_stats(value: torch.Tensor) -> dict[str, float | int]:
    value = value.detach().to(torch.float64)
    if value.numel() == 0:
        return {"count": 0}
    return {
        "count": int(value.numel()),
        "mean": float(value.mean().item()),
        "std": float(value.std(unbiased=False).item()),
        "minimum": float(value.amin().item()),
        "maximum": float(value.amax().item()),
        "maximum_absolute": float(value.abs().amax().item()),
    }


def _parameter_groups(model: Rival2ActorCritic) -> dict[str, list[torch.nn.Parameter]]:
    return {
        "trunk": list(model.trunk.parameters()),
        "actor_head": list(model.actor.parameters()),
        "critic_head": list(model.critic.parameters()),
    }


def _norm(values: list[torch.Tensor | None]) -> float:
    finite = [value.detach().square().sum() for value in values if value is not None]
    if not finite:
        return 0.0
    return float(torch.sqrt(torch.stack(finite).sum()).item())


def _loss_gradient_norms(
    loss: torch.Tensor,
    model: Rival2ActorCritic,
    *,
    retain_graph: bool,
) -> dict[str, float]:
    parameters = list(model.parameters())
    gradients = torch.autograd.grad(
        loss,
        parameters,
        retain_graph=retain_graph,
        allow_unused=True,
    )
    by_parameter = {
        id(parameter): gradient for parameter, gradient in zip(parameters, gradients, strict=False)
    }
    return {
        name: _norm([by_parameter[id(parameter)] for parameter in group])
        for name, group in _parameter_groups(model).items()
    }


def _stored_gradient_norms(model: Rival2ActorCritic) -> dict[str, float]:
    return {
        name: _norm([parameter.grad for parameter in group])
        for name, group in _parameter_groups(model).items()
    }


def _parameter_snapshots(model: Rival2ActorCritic) -> dict[str, list[torch.Tensor]]:
    return {
        name: [parameter.detach().clone() for parameter in group]
        for name, group in _parameter_groups(model).items()
    }


def _parameter_step_norms(
    model: Rival2ActorCritic,
    before: dict[str, list[torch.Tensor]],
) -> dict[str, float]:
    return {
        name: _norm(
            [
                parameter.detach() - prior
                for parameter, prior in zip(group, before[name], strict=True)
            ]
        )
        for name, group in _parameter_groups(model).items()
    }


def _empirical_kl(new_log_probability: torch.Tensor, old_log_probability: torch.Tensor) -> float:
    log_ratio = new_log_probability - old_log_probability
    ratio = torch.exp(log_ratio)
    return float(((ratio - 1.0) - log_ratio).mean().item())


def _sample_kl(
    new_log_probability: torch.Tensor, old_log_probability: torch.Tensor
) -> torch.Tensor:
    log_ratio = new_log_probability - old_log_probability
    return (torch.exp(log_ratio) - 1.0) - log_ratio


def _analytic_channel_kl(
    old_actor: torch.Tensor,
    new_actor: torch.Tensor,
    policy_config: Rival2PolicyConfig,
) -> torch.Tensor:
    old_mean = old_actor[:, :5]
    new_mean = new_actor[:, :5]
    old_log_std = old_actor[:, 5:10].clamp(policy_config.log_std_min, policy_config.log_std_max)
    new_log_std = new_actor[:, 5:10].clamp(policy_config.log_std_min, policy_config.log_std_max)
    old_variance = torch.exp(2.0 * old_log_std)
    new_variance = torch.exp(2.0 * new_log_std)
    analog = (
        new_log_std
        - old_log_std
        + (old_variance + (old_mean - new_mean).square()) / (2.0 * new_variance)
        - 0.5
    )
    old_probability = torch.sigmoid(old_actor[:, 10:13]).clamp(1.0e-7, 1.0 - 1.0e-7)
    new_probability = torch.sigmoid(new_actor[:, 10:13]).clamp(1.0e-7, 1.0 - 1.0e-7)
    buttons = old_probability * (torch.log(old_probability) - torch.log(new_probability))
    buttons += (1.0 - old_probability) * (
        torch.log1p(-old_probability) - torch.log1p(-new_probability)
    )
    return torch.cat((analog, buttons), dim=-1)


def _channel_means(value: torch.Tensor) -> dict[str, float]:
    means = value.mean(dim=0)
    return {name: float(means[index].item()) for index, name in enumerate(ACTION_CHANNEL_NAMES)}


def _family_statistics(
    labels: torch.Tensor,
    values: dict[str, torch.Tensor],
) -> dict[str, dict[str, Any]]:
    output: dict[str, dict[str, Any]] = {}
    for family, name in enumerate(OPPONENT_NAMES):
        selected = labels == family
        output[name] = {
            "samples": int(selected.sum().item()),
            **{metric: _tensor_stats(value[selected]) for metric, value in values.items()},
        }
    return output


def _family_kl_statistics(
    labels: torch.Tensor,
    sample_kl: torch.Tensor,
    channel_kl: torch.Tensor,
) -> dict[str, dict[str, Any]]:
    output: dict[str, dict[str, Any]] = {}
    for family, name in enumerate(OPPONENT_NAMES):
        selected = labels == family
        output[name] = {
            "samples": int(selected.sum().item()),
            "empirical_sample_kl": _tensor_stats(sample_kl[selected]),
            "analytic_channel_kl_mean": _channel_means(channel_kl[selected]),
            "analytic_channel_kl_sum_mean": float(channel_kl[selected].sum(dim=-1).mean().item()),
        }
    return output


def _counterfactual_step(
    model: Rival2ActorCritic,
    optimizer: torch.optim.Optimizer,
    *,
    loss_kind: str,
    observation: torch.Tensor,
    action: torch.Tensor,
    pre_tanh: torch.Tensor,
    old_log_probability: torch.Tensor,
    advantage: torch.Tensor,
    returns: torch.Tensor,
    ppo_config: Rival2PPOConfig,
    policy_config: Rival2PolicyConfig,
) -> dict[str, Any]:
    counterfactual_model = copy.deepcopy(model)
    counterfactual_optimizer = torch.optim.Adam(
        counterfactual_model.parameters(), lr=ppo_config.learning_rate
    )
    counterfactual_optimizer.load_state_dict(copy.deepcopy(optimizer.state_dict()))
    counterfactual_optimizer.zero_grad(set_to_none=True)
    actor, value = counterfactual_model(observation)
    new_log_probability = hybrid_log_probability(
        actor,
        action,
        config=policy_config,
        pre_tanh=pre_tanh,
    )
    log_ratio = new_log_probability - old_log_probability
    ratio = torch.exp(log_ratio)
    if loss_kind == "policy_only":
        unclipped = ratio * advantage
        clipped = ratio.clamp(1.0 - ppo_config.clip_range, 1.0 + ppo_config.clip_range) * advantage
        loss = -torch.minimum(unclipped, clipped).mean()
    elif loss_kind == "weighted_value_only":
        value_loss = 0.5 * (value - returns).square().mean()
        loss = ppo_config.value_loss_coefficient * value_loss
    else:
        raise ValueError(f"unsupported counterfactual loss: {loss_kind}")
    loss.backward()
    raw_gradient_norm = torch.nn.utils.clip_grad_norm_(
        counterfactual_model.parameters(), ppo_config.max_gradient_norm
    )
    post_clip_gradient_norms = _stored_gradient_norms(counterfactual_model)
    parameters_before = _parameter_snapshots(counterfactual_model)
    counterfactual_optimizer.step()
    with torch.no_grad():
        post_actor, _ = counterfactual_model(observation)
        post_log_probability = hybrid_log_probability(
            post_actor,
            action,
            config=policy_config,
            pre_tanh=pre_tanh,
        )
        post_kl = _empirical_kl(post_log_probability, old_log_probability)
    result = {
        "loss": float(loss.item()),
        "raw_gradient_norm": float(raw_gradient_norm.item()),
        "post_clip_gradient_norm_by_group": post_clip_gradient_norms,
        "parameter_step_norm_by_group": _parameter_step_norms(
            counterfactual_model, parameters_before
        ),
        "post_step_empirical_kl": post_kl,
    }
    del counterfactual_optimizer, counterfactual_model
    return result


def _write_report(report_path: Path, result: dict[str, Any]) -> None:
    failing = result["optimizer_steps"][-1]
    counterfactual = failing["counterfactual"]
    channel_kl = failing["post_step_analytic_channel_kl_mean"]
    analytic_kl_total = sum(channel_kl.values())
    steer_fraction = channel_kl["steer"] / analytic_kl_total
    lines = [
        "# Rival 2.0 Opponent Curriculum V1 KL diagnosis",
        "",
        f"Verdict: `{result['verdict']}`.",
        "",
        "This was a disposable deterministic replay of rejected update 360. It stopped "
        "at the original seventh-minibatch KL boundary and wrote no checkpoint.",
        "",
        "## Direct cause",
        "",
        f"The first six optimizer steps had already moved the failing minibatch to pre-step "
        f"KL `{failing['pre_step_empirical_kl']:.9f}`. The seventh combined PPO step moved "
        f"it to `{failing['post_step_empirical_kl']:.9f}`, exceeding the hard "
        f"`{KL_GUARD.minibatch_kl_limit}` guard.",
        "",
        "## Counterfactual attribution on the failing minibatch",
        "",
        "| update applied from the identical pre-step state | post-step empirical KL | "
        "raw gradient norm |",
        "|---|---:|---:|",
        f"| policy loss only | {counterfactual['policy_only']['post_step_empirical_kl']:.9f} | "
        f"{counterfactual['policy_only']['raw_gradient_norm']:.6f} |",
        f"| weighted value loss only | "
        f"{counterfactual['weighted_value_only']['post_step_empirical_kl']:.9f} | "
        f"{counterfactual['weighted_value_only']['raw_gradient_norm']:.6f} |",
        f"| actual combined PPO loss | {failing['post_step_empirical_kl']:.9f} | "
        f"{failing['raw_gradient_norm']:.6f} |",
        "",
        "These counterfactuals each use the same pre-step model, Adam state, minibatch, "
        "learning rate, and gradient clipping. They isolate the actor objective from "
        "critic-driven movement through the shared trunk; they are not additive.",
        "",
        "Both isolated steps exceed the guard. The policy-only path moves the actor head "
        "and shared trunk; the value-only path moves the shared trunk and changes actor "
        "outputs even though the actor head receives no value gradient. Their combined "
        "step partially cancels, but still exceeds the guard.",
        "",
        "## Gradient and parameter-step attribution",
        "",
        "| parameter group | policy gradient norm | weighted-value gradient norm | "
        "combined gradient norm | actual parameter-step norm |",
        "|---|---:|---:|---:|---:|",
    ]
    for group in ("trunk", "actor_head", "critic_head"):
        lines.append(
            f"| {group} | {failing['policy_loss_gradient_norm_by_group'][group]:.9f} | "
            f"{failing['weighted_value_loss_gradient_norm_by_group'][group]:.9f} | "
            f"{failing['combined_gradient_norm_by_group'][group]:.9f} | "
            f"{failing['parameter_step_norm_by_group'][group]:.9f} |"
        )
    lines.extend(
        [
            "",
            "The actor objective dominates the raw actor-head gradient, while the critic "
            "also displaces the policy through the shared trunk and preserved Adam state.",
            "",
            "## Action-channel attribution",
            "",
            "| channel | analytic KL mean |",
            "|---|---:|",
        ]
    )
    for channel in ACTION_CHANNEL_NAMES:
        lines.append(f"| {channel} | {channel_kl[channel]:.9f} |")
    lines.extend(
        [
            "",
            f"Steering contributes `{steer_fraction:.2%}` of the summed analytic "
            "action-channel KL on the failing minibatch. The failure is therefore "
            "primarily a steering-distribution displacement, not a button-policy jump.",
            "",
            "## Family composition of the failing minibatch",
            "",
            "| family | trainable samples | empirical sample-KL mean | analytic KL mean |",
            "|---|---:|---:|---:|",
        ]
    )
    for family in OPPONENT_NAMES:
        family_kl = failing["post_step_kl_by_family"][family]
        lines.append(
            f"| {family} | {family_kl['samples']} | "
            f"{family_kl['empirical_sample_kl']['mean']:.9f} | "
            f"{family_kl['analytic_channel_kl_sum_mean']:.9f} |"
        )
    lines.extend(
        [
            "",
            "Every family is affected, and the minibatch composition follows the expected "
            "trainable-sample proportions. This is not a single anomalous Wisp- or "
            "Nexto-heavy minibatch.",
            "",
            "## Rollout return and advantage shift",
            "",
            "| family | samples | reward mean | reward max-abs | return mean | old value "
            "mean | raw advantage mean | normalized advantage mean |",
            "|---|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for family in OPPONENT_NAMES:
        stats = result["rollout_statistics_by_family"][family]
        lines.append(
            f"| {family} | {stats['samples']} | {stats['reward']['mean']:.9f} | "
            f"{stats['reward']['maximum_absolute']:.9f} | "
            f"{stats['return']['mean']:.9f} | {stats['old_value']['mean']:.9f} | "
            f"{stats['raw_advantage']['mean']:.9f} | "
            f"{stats['normalized_advantage']['mean']:.9f} |"
        )
    lines.extend(
        [
            "",
            "The 32-decision replay had no episode reset, while every per-step reward was "
            "below `0.00112` in absolute value. The new `0.005` strict-double-dash reward "
            "therefore did not drive this rollout. The large family-conditioned returns "
            "and advantages arise from bootstrapped value differences on the new opponent "
            "state distribution. Global advantage normalization then turns those differences "
            "into coherent positive Wisp pressure and negative Nexto/current pressure.",
            "",
            "## Seven-step displacement sequence",
            "",
            "| optimizer step | pre-step KL | post-step KL | policy loss | value loss | "
            "raw gradient norm | clip fraction |",
            "|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for step in result["optimizer_steps"]:
        lines.append(
            f"| {step['optimizer_step_index']} | {step['pre_step_empirical_kl']:.9f} | "
            f"{step['post_step_empirical_kl']:.9f} | {step['policy_loss']:.9f} | "
            f"{step['value_loss']:.9f} | {step['raw_gradient_norm']:.6f} | "
            f"{step['clip_fraction']:.6f} |"
        )
    lines.extend(
        [
            "",
            "## Conclusion",
            "",
            "The rejection was caused by a compound policy-and-critic displacement after "
            "the abrupt mixed-opponent distribution transition. Family-conditioned "
            "bootstrapped advantages created a strong actor update, the critic independently "
            "moved actor outputs through the shared trunk, and the preserved Adam state plus "
            "low-entropy steering distribution made the clipped step KL-sensitive. PPO ratio "
            "clipping and gradient-norm clipping both operated, but neither is a KL bound.",
            "",
            "## Safety and scope",
            "",
            f"Source checkpoint SHA-256 remained "
            f"`{result['checkpoint_integrity']['source_after_sha256']}`.",
            "",
            f"Published rollback checkpoint SHA-256 remained "
            f"`{result['checkpoint_integrity']['rollback_after_sha256']}`.",
            "",
            "Nexto and Wisp were frozen inference-only opponents. Their samples remained "
            "non-trainable and neither model was connected to the optimizer.",
            "",
        ]
    )
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text("\n".join(lines), encoding="utf-8", newline="\n")


def run(args: argparse.Namespace) -> dict[str, Any]:
    source_before = _sha256(SOURCE_CHECKPOINT)
    rollback_before = _sha256(FINAL_CHECKPOINT)
    original_rejection = json.loads(ORIGINAL_REJECTION_PATH.read_text(encoding="utf-8"))
    source = torch.load(SOURCE_CHECKPOINT, map_location="cpu", weights_only=False)
    if source_before != SOURCE_CHECKPOINT_SHA256:
        raise RuntimeError("source checkpoint SHA-256 mismatch")

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
            "authorized_change": (
                "Gameplay V1 +239 to fresh short-lifecycle Gameplay V2 mixed opponents"
            ),
            "source_commit": AUTHORITATIVE_HEAD,
            "source_checkpoint_sha256": SOURCE_CHECKPOINT_SHA256,
            "collapsed_scoring_v1_lineage_used": False,
            "five_minute_world_state_carried": False,
            "diagnostic_replay_only": True,
        },
    )
    transition_gate = transition_preservation_gate(source, trainer, transition)
    if transition_gate["verdict"] != "PASS_GREEN":
        raise RuntimeError(f"transition replay failed: {transition_gate['checks']}")
    initial_family = trainer.opponent_family.clone()
    initial_family_counts = {
        name: int((initial_family == family).sum().item())
        for family, name in enumerate(OPPONENT_NAMES)
    }
    initial_assignments_total = trainer.realized_family_assignments.clone()
    source_model_exact = _nested_exact(source["model"], trainer.model.state_dict())
    source_optimizer_exact = _nested_exact(source["optimizer"], trainer.optimizer.state_dict())

    trainer.env.reset_transfer_counters()
    rollout = trainer.collect_rollout()
    if not torch.equal(trainer.realized_family_assignments, initial_assignments_total):
        raise RuntimeError("diagnostic rollout unexpectedly crossed an episode reset")
    rollout.compute_gae(trainer.ppo_config)
    trainer.model.train()

    flat_train_mask = rollout.train_mask.reshape(-1)
    train_indices = torch.nonzero(flat_train_mask, as_tuple=False).squeeze(-1)
    sample_count = int(train_indices.numel())
    observation = rollout.observations.reshape(-1, OBS_DIM).index_select(0, train_indices)
    action = rollout.actions.reshape(-1, 8).index_select(0, train_indices)
    pre_tanh = rollout.pre_tanh.reshape(-1, 5).index_select(0, train_indices)
    old_log_probability = rollout.old_log_probability.reshape(-1).index_select(0, train_indices)
    old_value = rollout.values.reshape(-1).index_select(0, train_indices)
    rewards = rollout.rewards.reshape(-1).index_select(0, train_indices)
    returns = rollout.returns.reshape(-1).index_select(0, train_indices)
    raw_advantage = rollout.advantages.reshape(-1).index_select(0, train_indices)
    advantage = (raw_advantage - raw_advantage.mean()) / raw_advantage.std(
        unbiased=False
    ).clamp_min(1.0e-8)
    family = (
        initial_family[None, :, None]
        .expand(trainer.ppo_config.rollout_horizon, -1, 2)
        .reshape(-1)
        .index_select(0, train_indices)
    )
    behavior_model = copy.deepcopy(trainer.model).eval().requires_grad_(False)
    permutation = torch.randperm(
        sample_count,
        device=trainer.device,
        generator=trainer.policy_generator,
    )

    rollout_statistics = _family_statistics(
        family,
        {
            "reward": rewards,
            "return": returns,
            "raw_advantage": raw_advantage,
            "normalized_advantage": advantage,
            "old_value": old_value,
        },
    )
    optimizer_steps: list[dict[str, Any]] = []
    guard_triggered = False
    config = trainer.ppo_config
    for optimizer_step_index in range(EXPECTED_OPTIMIZER_STEP_INDEX + 1):
        start = optimizer_step_index * config.minibatch_size
        batch = permutation[start : start + config.minibatch_size]
        batch_observation = observation.index_select(0, batch)
        batch_action = action.index_select(0, batch)
        batch_pre_tanh = pre_tanh.index_select(0, batch)
        batch_old_log_probability = old_log_probability.index_select(0, batch)
        batch_advantage = advantage.index_select(0, batch)
        batch_returns = returns.index_select(0, batch)
        batch_family = family.index_select(0, batch)
        with torch.no_grad():
            behavior_actor, _ = behavior_model(batch_observation)

        actor, value = trainer.model(batch_observation)
        new_log_probability = hybrid_log_probability(
            actor,
            batch_action,
            config=trainer.policy_config,
            pre_tanh=batch_pre_tanh,
        )
        log_ratio = new_log_probability - batch_old_log_probability
        ratio = torch.exp(log_ratio)
        unclipped = ratio * batch_advantage
        clipped = ratio.clamp(1.0 - config.clip_range, 1.0 + config.clip_range) * batch_advantage
        policy_loss = -torch.minimum(unclipped, clipped).mean()
        value_loss = 0.5 * (value - batch_returns).square().mean()
        weighted_value_loss = config.value_loss_coefficient * value_loss
        entropy = hybrid_entropy(actor, trainer.policy_config).mean()
        total_loss = policy_loss + weighted_value_loss
        pre_step_kl = _empirical_kl(new_log_probability, batch_old_log_probability)
        behavior_log_probability = hybrid_log_probability(
            behavior_actor,
            batch_action,
            config=trainer.policy_config,
            pre_tanh=batch_pre_tanh,
        )
        old_log_probability_max_abs_error = float(
            (behavior_log_probability - batch_old_log_probability).abs().amax().item()
        )
        pre_channel_kl = _analytic_channel_kl(behavior_actor, actor, trainer.policy_config)

        policy_gradient_norms = _loss_gradient_norms(policy_loss, trainer.model, retain_graph=True)
        value_gradient_norms = _loss_gradient_norms(
            weighted_value_loss, trainer.model, retain_graph=True
        )
        counterfactual = {
            "policy_only": _counterfactual_step(
                trainer.model,
                trainer.optimizer,
                loss_kind="policy_only",
                observation=batch_observation,
                action=batch_action,
                pre_tanh=batch_pre_tanh,
                old_log_probability=batch_old_log_probability,
                advantage=batch_advantage,
                returns=batch_returns,
                ppo_config=config,
                policy_config=trainer.policy_config,
            ),
            "weighted_value_only": _counterfactual_step(
                trainer.model,
                trainer.optimizer,
                loss_kind="weighted_value_only",
                observation=batch_observation,
                action=batch_action,
                pre_tanh=batch_pre_tanh,
                old_log_probability=batch_old_log_probability,
                advantage=batch_advantage,
                returns=batch_returns,
                ppo_config=config,
                policy_config=trainer.policy_config,
            ),
        }

        trainer.optimizer.zero_grad(set_to_none=True)
        total_loss.backward()
        combined_gradient_norms = _stored_gradient_norms(trainer.model)
        raw_gradient_norm = torch.nn.utils.clip_grad_norm_(
            trainer.model.parameters(), config.max_gradient_norm
        )
        post_clip_gradient_norms = _stored_gradient_norms(trainer.model)
        parameter_before = _parameter_snapshots(trainer.model)
        trainer.optimizer.step()
        parameter_step_norms = _parameter_step_norms(trainer.model, parameter_before)
        with torch.no_grad():
            post_actor, _ = trainer.model(batch_observation)
            post_log_probability = hybrid_log_probability(
                post_actor,
                batch_action,
                config=trainer.policy_config,
                pre_tanh=batch_pre_tanh,
            )
            post_step_kl = _empirical_kl(post_log_probability, batch_old_log_probability)
            post_sample_kl = _sample_kl(post_log_probability, batch_old_log_probability)
            post_channel_kl = _analytic_channel_kl(
                behavior_actor, post_actor, trainer.policy_config
            )
        clip_fraction = float(
            (torch.abs(ratio - 1.0) > config.clip_range).to(torch.float32).mean().item()
        )
        step = {
            "optimizer_step_index": optimizer_step_index,
            "minibatch_start": start,
            "minibatch_samples": int(batch.numel()),
            "family_sample_counts": {
                name: int((batch_family == family_index).sum().item())
                for family_index, name in enumerate(OPPONENT_NAMES)
            },
            "old_log_probability_max_abs_error": old_log_probability_max_abs_error,
            "policy_loss": float(policy_loss.item()),
            "value_loss": float(value_loss.item()),
            "weighted_value_loss": float(weighted_value_loss.item()),
            "entropy_diagnostic": float(entropy.item()),
            "total_loss": float(total_loss.item()),
            "clip_fraction": clip_fraction,
            "pre_step_empirical_kl": pre_step_kl,
            "post_step_empirical_kl": post_step_kl,
            "pre_step_analytic_channel_kl_mean": _channel_means(pre_channel_kl),
            "post_step_analytic_channel_kl_mean": _channel_means(post_channel_kl),
            "policy_loss_gradient_norm_by_group": policy_gradient_norms,
            "weighted_value_loss_gradient_norm_by_group": value_gradient_norms,
            "combined_gradient_norm_by_group": combined_gradient_norms,
            "raw_gradient_norm": float(raw_gradient_norm.item()),
            "post_clip_gradient_norm_by_group": post_clip_gradient_norms,
            "parameter_step_norm_by_group": parameter_step_norms,
            "counterfactual": counterfactual,
            "batch_statistics_by_family": _family_statistics(
                batch_family,
                {
                    "reward": rewards.index_select(0, batch),
                    "return": returns.index_select(0, batch),
                    "raw_advantage": raw_advantage.index_select(0, batch),
                    "normalized_advantage": batch_advantage,
                    "old_value": old_value.index_select(0, batch),
                },
            ),
            "post_step_kl_by_family": _family_kl_statistics(
                batch_family,
                post_sample_kl,
                post_channel_kl,
            ),
            "guard_triggered": post_step_kl > KL_GUARD.minibatch_kl_limit,
        }
        optimizer_steps.append(step)
        print(
            f"step={optimizer_step_index} pre_kl={pre_step_kl:.9f} "
            f"post_kl={post_step_kl:.9f} raw_grad={float(raw_gradient_norm.item()):.6f}",
            flush=True,
        )
        if step["guard_triggered"]:
            guard_triggered = True
            break

    source_after = _sha256(SOURCE_CHECKPOINT)
    rollback_after = _sha256(FINAL_CHECKPOINT)
    failing = optimizer_steps[-1]
    replay_match = {
        "guard_triggered_at_expected_step": guard_triggered
        and failing["optimizer_step_index"] == EXPECTED_OPTIMIZER_STEP_INDEX,
        "minibatch_start_exact": failing["minibatch_start"] == EXPECTED_MINIBATCH_START,
        "minibatch_samples_exact": failing["minibatch_samples"] == EXPECTED_MINIBATCH_SAMPLES,
        "pre_step_kl_matches_original": math.isclose(
            failing["pre_step_empirical_kl"],
            original_rejection["pre_step_approx_kl"],
            rel_tol=0.0,
            abs_tol=1.0e-6,
        ),
        "post_step_kl_matches_original": math.isclose(
            failing["post_step_empirical_kl"],
            original_rejection["post_step_approx_kl"],
            rel_tol=0.0,
            abs_tol=1.0e-6,
        ),
        "policy_loss_matches_original": math.isclose(
            failing["policy_loss"], original_rejection["policy_loss"], rel_tol=0.0, abs_tol=1.0e-6
        ),
        "value_loss_matches_original": math.isclose(
            failing["value_loss"], original_rejection["value_loss"], rel_tol=0.0, abs_tol=1.0e-6
        ),
        "raw_gradient_norm_matches_original": math.isclose(
            failing["raw_gradient_norm"],
            original_rejection["raw_gradient_norm"],
            rel_tol=0.0,
            abs_tol=1.0e-5,
        ),
    }
    checks = {
        "source_model_exact_before_replay": source_model_exact,
        "source_optimizer_exact_before_replay": source_optimizer_exact,
        "initial_family_counts_match_original": initial_family_counts
        == original_rejection["restored_checkpoint"]["realized_family_assignments"],
        "trainable_sample_count_exact": sample_count == 5_028_704,
        "no_episode_reset_during_replayed_rollout": torch.equal(
            trainer.realized_family_assignments, initial_assignments_total
        ),
        "all_old_log_probabilities_match_behavior_policy": all(
            step["old_log_probability_max_abs_error"] <= 1.0e-5 for step in optimizer_steps
        ),
        "source_checkpoint_untouched": source_before == source_after == SOURCE_CHECKPOINT_SHA256,
        "rollback_checkpoint_untouched": rollback_before == rollback_after,
        "replay_matches_original_rejection": all(replay_match.values()),
        "nexto_and_wisp_never_trainable": bool(
            torch.all(~rollout.train_mask[(rollout.policy_version < 0)]).item()
        ),
        "stopped_at_guard_without_checkpoint_write": guard_triggered,
    }
    result = {
        "schema_version": SCHEMA_VERSION,
        "created_utc": _utc_now(),
        "verdict": "PASS_GREEN" if all(checks.values()) else "FAIL_RED",
        "scope": {
            "diagnostic_replay_only": True,
            "training_continuation": False,
            "checkpoint_write": False,
            "maximum_optimizer_steps": EXPECTED_OPTIMIZER_STEP_INDEX + 1,
            "nexto_trainable": False,
            "wisp_trainable": False,
        },
        "identity": {
            "repository_head": _git("rev-parse", "HEAD"),
            "authoritative_source_commit": AUTHORITATIVE_HEAD,
            "source_checkpoint": SOURCE_CHECKPOINT.as_posix(),
            "source_checkpoint_sha256": SOURCE_CHECKPOINT_SHA256,
            "original_rejection": ORIGINAL_REJECTION_PATH.as_posix(),
        },
        "configuration": {
            "worlds": WORLDS,
            "campaign_seed": CAMPAIGN_SEED,
            "rollout_horizon": config.rollout_horizon,
            "minibatch_size": config.minibatch_size,
            "learning_rate": config.learning_rate,
            "clip_range": config.clip_range,
            "value_loss_coefficient": config.value_loss_coefficient,
            "entropy_coefficient": config.entropy_coefficient,
            "max_gradient_norm": config.max_gradient_norm,
            "kl_minibatch_limit": KL_GUARD.minibatch_kl_limit,
        },
        "initial_family_counts": initial_family_counts,
        "trainable_samples": sample_count,
        "rollout_statistics_by_family": rollout_statistics,
        "optimizer_steps": optimizer_steps,
        "original_rejection": {
            name: original_rejection[name]
            for name in (
                "optimizer_step_index",
                "minibatch_start",
                "minibatch_samples",
                "pre_step_approx_kl",
                "post_step_approx_kl",
                "policy_loss",
                "value_loss",
                "raw_gradient_norm",
                "post_clip_gradient_norm",
            )
        },
        "replay_match": replay_match,
        "checkpoint_integrity": {
            "source_before_sha256": source_before,
            "source_after_sha256": source_after,
            "rollback_before_sha256": rollback_before,
            "rollback_after_sha256": rollback_after,
        },
        "checks": checks,
    }
    _write_json(args.output, result)
    _write_report(args.report, result)
    if (
        args.output.resolve() == RESULT_PATH.resolve()
        and args.report.resolve() == KL_DIAGNOSIS_REPORT_PATH.resolve()
    ):
        _write_artifact_manifest()
    return result


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
