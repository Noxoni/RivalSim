"""Train and evaluate Rival's physical capability curriculum V1."""

from __future__ import annotations

import argparse
import copy
import gc
import hashlib
import json
import math
import os
import shutil
import sys
from dataclasses import asdict
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from benchmarks import run_rival2_codex_autonomous_v1 as human_base  # noqa: E402
from rivalsim.arena import ArenaGeometry, WarpArenaMeshes  # noqa: E402
from rivalsim.human_demo.bc_observation_bridge import hybrid_actor_channel_kl  # noqa: E402
from rivalsim.human_demo.behavior_cloning import (  # noqa: E402
    MechanicHierarchySampler,
    action_metric_summary,
    human_behavior_cloning_objective,
)
from rivalsim.rival2_capability_curriculum import (  # noqa: E402
    SCENARIO_NAMES,
    CapabilityRewardTracker,
    build_capability_scenarios,
)
from rivalsim.rival2_contracts import (  # noqa: E402
    REWARD_GAMEPLAY_120_V2_CONTRACT_HASH,
    RIVAL2_REWARD_GAMEPLAY_120_V2_VERSION,
)
from rivalsim.rival2_env import Rival2Env  # noqa: E402
from rivalsim.rival2_policy import (  # noqa: E402
    HybridDistributionOverride,
    Rival2ActorCritic,
    Rival2PolicyConfig,
    deterministic_hybrid_action,
    hybrid_entropy,
    hybrid_log_probability,
    sample_hybrid_action,
)
from rivalsim.rival2_ppo import (  # noqa: E402
    Rival2RolloutBuffer,
    rival2_ppo_120hz_config,
)

AUTHORITY = ROOT / "results/rival2/capability_curriculum_v1/authority.json"
RESULTS = ROOT / "results/rival2/capability_curriculum_v1"
CHECKPOINTS = ROOT / "checkpoints/rival2/capability_curriculum_v1"
SOURCE = ROOT / "checkpoints/rival2/codex_autonomous_v23/rival2_blue.pt"
SOURCE_SHA256 = "0263546263285384D2D9A0CE55A471C41A41A8B7D4870DD9504D0ACCEA76723C"
ORANGE_SHA256 = "56E4ECA5075EB5748402BA3C5D8D51AC91FC1AFF55219E64EA5CE688DAD3491A"
DEFAULT_RUN_DIR = Path("G:/dev/RivalSim-runs/capability-curriculum-v1")
DEFAULT_COLLISION_DIR = Path("G:/dev/RLBot-Rival/bot/collision_meshes/soccar")
TARGET_LABELS = (
    "aerialdribble",
    "ceilingpinch",
    "flipreset",
    "groundtoairdribble",
    "walldash",
    "wavedash",
)
SEED = 2_026_090_271


def utc_now() -> str:
    return datetime.now(UTC).isoformat()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest().upper()


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    os.replace(temporary, path)


def append_jsonl(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8", newline="\n") as handle:
        handle.write(json.dumps(payload, sort_keys=True, allow_nan=False) + "\n")
        handle.flush()
        os.fsync(handle.fileno())


def load_authority() -> dict[str, Any]:
    authority = json.loads(AUTHORITY.read_text(encoding="utf-8"))
    checks = {
        "format": authority.get("format") == "RIVAL2_CAPABILITY_CURRICULUM_V1_AUTHORITY",
        "blue_source": authority["protected_parents"]["blue"]["sha256"] == SOURCE_SHA256,
        "orange_source": authority["protected_parents"]["orange"]["sha256"] == ORANGE_SHA256,
        "base_reward": authority["contracts"]["base_reward_sha256"]
        == REWARD_GAMEPLAY_120_V2_CONTRACT_HASH,
        "labels": tuple(authority["human_rehearsal"]["train_only_labels"])
        == (
            "aerialdribble",
            "groundtoairdribble",
            "flipreset",
            "ceilingpinch",
            "wavedash",
            "walldash",
        ),
        "kl_telemetry": authority["scenario_curriculum"]["ppo"]["kl_telemetry_only"] is True,
        "no_named_classifier": authority["physical_training_overlay"][
            "named_mechanic_classifier_used"
        ]
        is False,
    }
    if not all(checks.values()):
        raise RuntimeError(f"capability authority mismatch: {checks}")
    bound_files = (
        authority["protected_parents"]["blue"],
        authority["protected_parents"]["orange"],
        authority["human_rehearsal"]["dataset_manifest"],
        authority["human_rehearsal"]["review_candidates"],
        authority["human_rehearsal"]["observation_adapter"],
    )
    for identity in bound_files:
        path = ROOT / identity["path"]
        if sha256_file(path) != identity["sha256"]:
            raise RuntimeError(f"bound capability input changed: {path}")
    return authority


@torch.no_grad()
def evaluate_rows(
    model: Rival2ActorCritic,
    observation: torch.Tensor,
    action: torch.Tensor,
    *,
    device: str,
) -> dict[str, Any]:
    outputs: list[torch.Tensor] = []
    model.eval()
    for start in range(0, observation.shape[0], 8192):
        actor, _ = model(observation[start : start + 8192].to(device))
        outputs.append(actor.cpu())
    actor = torch.cat(outputs) if outputs else torch.empty((0, 13))
    result = action_metric_summary(actor, action)
    result["finite"] = bool(torch.isfinite(actor).all())
    return result


def desired_validation(
    model: Rival2ActorCritic,
    validation: Any,
    *,
    device: str,
) -> dict[str, Any]:
    selected = torch.tensor(
        [label in TARGET_LABELS for label in validation.mechanic_label], dtype=torch.bool
    )
    gameplay = evaluate_rows(
        model,
        validation.gameplay_observation,
        validation.gameplay_action,
        device=device,
    )
    target = evaluate_rows(
        model,
        validation.mechanic_observation[selected],
        validation.mechanic_action[selected],
        device=device,
    )
    per_label: dict[str, Any] = {}
    labels = np.asarray(validation.mechanic_label)
    for label in TARGET_LABELS:
        mask = torch.from_numpy(labels == label)
        per_label[label] = evaluate_rows(
            model,
            validation.mechanic_observation[mask],
            validation.mechanic_action[mask],
            device=device,
        )
    return {
        "gameplay": gameplay,
        "target_mechanics": target,
        "per_label": per_label,
        "selection_score": 0.5
        * (gameplay["complete_action_rmse"] + target["complete_action_rmse"]),
        "eligible": bool(
            gameplay["complete_action_rmse"] <= 0.590379 and gameplay["finite"] and target["finite"]
        ),
    }


def human_warm_start(
    source: dict[str, Any],
    authority: dict[str, Any],
    *,
    device: str,
    run_dir: Path,
) -> tuple[dict[str, Any], dict[str, Any]]:
    human_base.SOURCE = SOURCE
    human_base.SOURCE_SHA256 = SOURCE_SHA256
    train, validation, teacher, identities = human_base.load_human_data(device=device)
    config = Rival2PolicyConfig(**source["policy_config"])
    student = Rival2ActorCritic(config).to(device)
    student.load_state_dict(source["model"])
    student.critic.requires_grad_(False)
    critic_before = {
        name: value.detach().cpu().clone() for name, value in student.critic.state_dict().items()
    }
    baseline = desired_validation(student, validation, device=device)
    target_mask = torch.tensor(
        [label in TARGET_LABELS for label in train.mechanic_label], dtype=torch.bool
    )
    target_observation = train.mechanic_observation[target_mask]
    target_action = train.mechanic_action[target_mask]
    target_labels = [
        label
        for label, keep in zip(train.mechanic_label, target_mask.tolist(), strict=True)
        if keep
    ]
    target_attempts = [
        attempt
        for attempt, keep in zip(train.mechanic_attempt, target_mask.tolist(), strict=True)
        if keep
    ]
    generator = torch.Generator().manual_seed(SEED ^ 0xBC)
    sampler = MechanicHierarchySampler(
        target_labels,
        target_attempts,
        uniform_label_fraction=0.25,
        maximum_oversampling_ratio=20.0,
        generator=generator,
    )
    human = authority["human_rehearsal"]
    optimizer = torch.optim.AdamW(
        [*student.trunk.parameters(), *student.actor.parameters()],
        lr=float(human["actor_trunk_learning_rate"]),
        weight_decay=float(human["weight_decay"]),
    )
    retention = source["opponent_curriculum"]["adaptive_ppo"]["retention_observations"].to(
        torch.float32
    )
    best_state = copy.deepcopy(student.state_dict())
    best = {"step": 0, "validation": baseline}
    stale = 0
    curve = RESULTS / "human_rehearsal_curve.jsonl"
    if curve.exists():
        curve.unlink()
    maximum = int(human["maximum_steps"])
    interval = int(human["validation_interval_steps"])
    for step in range(1, maximum + 1):
        gameplay_index = torch.randint(
            train.gameplay_observation.shape[0],
            (int(human["gameplay_frames_per_step"]),),
            generator=generator,
        )
        mechanic_index = sampler.sample(int(human["target_mechanic_frames_per_step"]))
        gameplay_observation = train.gameplay_observation.index_select(0, gameplay_index).to(device)
        gameplay_action = train.gameplay_action.index_select(0, gameplay_index).to(device)
        mechanic_observation = target_observation.index_select(0, mechanic_index).to(device)
        mechanic_action = target_action.index_select(0, mechanic_index).to(device)
        with torch.no_grad():
            teacher_gameplay, _ = teacher(gameplay_observation)
            teacher_mechanic, _ = teacher(mechanic_observation)
            retention_index = torch.randint(retention.shape[0], (512,), generator=generator)
            retention_observation = retention.index_select(0, retention_index).to(device)
            teacher_retention, _ = teacher(retention_observation)
        student_gameplay, _ = student(gameplay_observation)
        student_mechanic, _ = student(mechanic_observation)
        student_retention, _ = student(retention_observation)
        gameplay_loss = human_behavior_cloning_objective(
            student_gameplay,
            teacher_gameplay,
            gameplay_action,
            smooth_l1_beta=0.1,
            analog_weight=1.0,
            button_weight=0.25,
            log_std_weight=float(human["log_std_retention_coefficient"]),
            policy_config=config,
        )
        mechanic_loss = human_behavior_cloning_objective(
            student_mechanic,
            teacher_mechanic,
            mechanic_action,
            smooth_l1_beta=0.1,
            analog_weight=1.0,
            button_weight=0.25,
            log_std_weight=float(human["log_std_retention_coefficient"]),
            policy_config=config,
        )
        retention_kl = (
            hybrid_actor_channel_kl(teacher_retention, student_retention, policy_config=config)
            .sum(dim=-1)
            .mean()
        )
        loss = 0.5 * (gameplay_loss.loss + mechanic_loss.loss)
        loss = loss + float(human["teacher_actor_retention_coefficient"]) * retention_kl
        if not bool(torch.isfinite(loss)):
            raise RuntimeError("nonfinite human capability rehearsal loss")
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        gradient = torch.nn.utils.clip_grad_norm_(
            [*student.trunk.parameters(), *student.actor.parameters()], 0.5
        )
        if not bool(torch.isfinite(gradient)):
            raise RuntimeError("nonfinite human capability rehearsal gradient")
        optimizer.step()
        if step % interval != 0:
            continue
        validation_metrics = desired_validation(student, validation, device=device)
        row = {
            "step": step,
            "loss": float(loss.detach()),
            "retention_kl": float(retention_kl.detach()),
            "gradient_norm": float(gradient.detach()),
            "validation": validation_metrics,
        }
        append_jsonl(curve, row)
        improved = validation_metrics["eligible"] and (
            validation_metrics["selection_score"] < best["validation"]["selection_score"] - 1.0e-5
        )
        if improved:
            best_state = copy.deepcopy(student.state_dict())
            best = {"step": step, "validation": validation_metrics}
            stale = 0
        else:
            stale += 1
        print(
            json.dumps(
                {
                    "stage": "human_rehearsal",
                    "step": step,
                    "score": validation_metrics["selection_score"],
                    "gameplay": validation_metrics["gameplay"]["complete_action_rmse"],
                    "target": validation_metrics["target_mechanics"]["complete_action_rmse"],
                    "best_step": best["step"],
                },
                sort_keys=True,
            ),
            flush=True,
        )
        if stale >= int(human["patience_boundaries"]):
            break
    student.load_state_dict(best_state)
    if any(
        not torch.equal(critic_before[name], value.detach().cpu())
        for name, value in student.critic.state_dict().items()
    ):
        raise RuntimeError("human capability rehearsal changed the critic")
    payload = copy.deepcopy(source)
    payload["model"] = {name: value.detach().cpu() for name, value in student.state_dict().items()}
    payload["optimizer"] = optimizer.state_dict()
    payload["curriculum_transition"] = {
        "identity": "RIVAL2_CAPABILITY_CURRICULUM_V1_HUMAN_REHEARSAL",
        "created_utc": utc_now(),
        "authority": {
            "path": AUTHORITY.relative_to(ROOT).as_posix(),
            "sha256": sha256_file(AUTHORITY),
        },
        "source": {"path": SOURCE.relative_to(ROOT).as_posix(), "sha256": SOURCE_SHA256},
        "target_labels": list(TARGET_LABELS),
        "selected_step": best["step"],
        "validation": best["validation"],
        "critic_byte_identical": True,
        "human_test_loaded": False,
        "input_identities": identities,
    }
    path = run_dir / "human_rehearsal_selected.pt"
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, path)
    result = {
        "baseline": baseline,
        "selected": best,
        "checkpoint": {
            "path": str(path),
            "sha256": sha256_file(path),
            "model_tensor_sha256": human_base.tensor_tree_sha256(payload["model"]),
        },
        "target_training_frames": int(target_mask.sum()),
        "target_labels": list(TARGET_LABELS),
        "sampler_maximum_realized_oversampling_ratio": (
            sampler.maximum_realized_oversampling_ratio
        ),
        "human_test_loaded": False,
    }
    write_json(RESULTS / "human_rehearsal_result.json", result)
    return payload, result


def _build_optimizers(
    model: Rival2ActorCritic,
    *,
    policy_lr: float,
    critic_lr: float,
) -> tuple[torch.optim.Adam, torch.optim.Adam]:
    return (
        torch.optim.Adam([*model.trunk.parameters(), *model.actor.parameters()], lr=policy_lr),
        torch.optim.Adam(model.critic.parameters(), lr=critic_lr),
    )


def isolated_ppo_update(
    model: Rival2ActorCritic,
    policy_optimizer: torch.optim.Optimizer,
    critic_optimizer: torch.optim.Optimizer,
    rollout: Rival2RolloutBuffer,
    *,
    generator: torch.Generator,
    distribution_override: HybridDistributionOverride,
) -> dict[str, float]:
    config = rival2_ppo_120hz_config()
    rollout.compute_gae(config)
    index = torch.nonzero(rollout.train_mask.reshape(-1), as_tuple=False).squeeze(-1)
    observation = rollout.observations.reshape(-1, 182).index_select(0, index)
    action = rollout.actions.reshape(-1, 8).index_select(0, index)
    pre_tanh = rollout.pre_tanh.reshape(-1, 5).index_select(0, index)
    old_log_probability = rollout.old_log_probability.reshape(-1).index_select(0, index)
    returns = rollout.returns.reshape(-1).index_select(0, index)
    advantage = rollout.advantages.reshape(-1).index_select(0, index)
    advantage = (advantage - advantage.mean()) / advantage.std(unbiased=False).clamp_min(1e-8)
    sums = {
        "policy_loss": 0.0,
        "value_loss": 0.0,
        "entropy": 0.0,
        "approx_kl": 0.0,
        "clip_fraction": 0.0,
        "policy_gradient_norm": 0.0,
        "critic_gradient_norm": 0.0,
    }
    proposals = 0
    model.train()
    for _epoch in range(config.epochs):
        permutation = torch.randperm(index.numel(), device=index.device, generator=generator)
        for start in range(0, index.numel(), config.minibatch_size):
            batch = permutation[start : start + config.minibatch_size]
            batch_observation = observation.index_select(0, batch)
            features = model.trunk(batch_observation)
            actor = model.actor(features)
            new_log_probability = hybrid_log_probability(
                actor,
                action.index_select(0, batch),
                config=model.config,
                pre_tanh=pre_tanh.index_select(0, batch),
                distribution_override=distribution_override,
            )
            old = old_log_probability.index_select(0, batch)
            log_ratio = new_log_probability - old
            ratio = torch.exp(log_ratio)
            local_advantage = advantage.index_select(0, batch)
            policy_loss = -torch.minimum(
                ratio * local_advantage,
                ratio.clamp(1.0 - config.clip_range, 1.0 + config.clip_range) * local_advantage,
            ).mean()
            entropy = hybrid_entropy(
                actor,
                model.config,
                distribution_override=distribution_override,
            ).mean()
            policy_optimizer.zero_grad(set_to_none=True)
            policy_loss.backward()
            policy_gradient = torch.nn.utils.clip_grad_norm_(
                [*model.trunk.parameters(), *model.actor.parameters()],
                config.max_gradient_norm,
            )
            if not bool(torch.isfinite(policy_gradient)):
                raise RuntimeError("nonfinite capability PPO policy gradient")
            policy_optimizer.step()

            with torch.no_grad():
                detached_features = model.trunk(batch_observation)
            value = model.critic(detached_features).squeeze(-1)
            value_loss = 0.5 * (value - returns.index_select(0, batch)).square().mean()
            critic_optimizer.zero_grad(set_to_none=True)
            (config.value_loss_coefficient * value_loss).backward()
            critic_gradient = torch.nn.utils.clip_grad_norm_(
                model.critic.parameters(), config.max_gradient_norm
            )
            if not bool(torch.isfinite(critic_gradient)):
                raise RuntimeError("nonfinite capability PPO critic gradient")
            critic_optimizer.step()
            with torch.no_grad():
                approx_kl = ((ratio - 1.0) - log_ratio).mean()
                clip_fraction = (ratio.sub(1.0).abs() > config.clip_range).float().mean()
            values = (
                policy_loss,
                value_loss,
                entropy,
                approx_kl,
                clip_fraction,
                policy_gradient,
                critic_gradient,
            )
            if not all(bool(torch.isfinite(value)) for value in values):
                raise RuntimeError("nonfinite capability PPO optimizer telemetry")
            for name, value in zip(sums, values, strict=True):
                sums[name] += float(value.detach())
            proposals += 1
    if not all(bool(torch.isfinite(parameter).all()) for parameter in model.parameters()):
        raise RuntimeError("nonfinite capability PPO model state")
    return {**{name: value / proposals for name, value in sums.items()}, "steps": proposals}


def collect_scenario_rollout(
    model: Rival2ActorCritic,
    geometry: ArenaGeometry,
    meshes: WarpArenaMeshes,
    *,
    collision_dir: Path,
    worlds: int,
    seed: int,
    device: str,
    generator: torch.Generator,
    distribution_override: HybridDistributionOverride,
    deterministic: bool = False,
    horizon: int = 128,
) -> tuple[Rival2RolloutBuffer | None, dict[str, Any]]:
    batch = build_capability_scenarios(worlds, seed=seed)
    env = Rival2Env(
        worlds,
        str(collision_dir),
        device=device,
        seed=seed,
        reward_version=RIVAL2_REWARD_GAMEPLAY_120_V2_VERSION,
        initial=batch.state,
        geometry=geometry,
        meshes=meshes,
        car_visitation_order="a_then_b",
    )
    scenario = torch.from_numpy(batch.scenario).to(device)
    attacker = torch.from_numpy(batch.attacker_side).to(device)
    scripted = torch.from_numpy(batch.scripted_action).to(device)
    rows = torch.arange(worlds, device=device)
    tracker = CapabilityRewardTracker(scenario, attacker)
    active = torch.ones(worlds, dtype=torch.bool, device=device)
    rollout = None if deterministic else Rival2RolloutBuffer(horizon, worlds, device)
    goals = 0
    high_goal = 0
    touched_elevated = torch.zeros(worlds, dtype=torch.bool, device=device)
    action_saturation = torch.zeros(5, dtype=torch.float64, device=device)
    action_count = torch.zeros((), dtype=torch.float64, device=device)
    model.eval()
    observation = env.observation
    for tick in range(horizon):
        with torch.no_grad():
            actor_output, value_flat = model(observation.reshape(-1, 182))
            actor_output = actor_output.reshape(worlds, 2, 13)
            value = value_flat.reshape(worlds, 2)
            if deterministic:
                action = deterministic_hybrid_action(actor_output, model.config)
                pre_tanh = actor_output[..., :5]
                log_probability = hybrid_log_probability(
                    actor_output,
                    action,
                    config=model.config,
                    pre_tanh=pre_tanh,
                    distribution_override=distribution_override,
                )
            else:
                sample = sample_hybrid_action(
                    actor_output,
                    generator=generator,
                    config=model.config,
                    distribution_override=distribution_override,
                )
                action = sample.action
                pre_tanh = sample.pre_tanh
                log_probability = sample.log_probability
            opponent = 1 - attacker
            action[rows, opponent] = scripted[rows, opponent]
            action = torch.where(active[:, None, None], action, torch.zeros_like(action))
            transition = env.step(action)
            overlay = tracker.step(
                observation,
                transition.transition_observation,
                tick=tick,
            )
            selected_reward = transition.reward[rows, attacker] + overlay
            reward = torch.zeros_like(transition.reward)
            reward[rows, attacker] = torch.where(active, selected_reward, 0.0)
            elevated_now = (
                (transition.transition_observation[rows, attacker, 176] > 0.5)
                & (transition.transition_observation[rows, attacker, 2] * 2044.0 >= 300.0)
                & (transition.transition_observation[rows, attacker, 11] * 2044.0 >= 250.0)
            )
            touched_elevated |= elevated_now
            scoring_team = env.bridge.views["rival2.scoring_team_latched"].to(torch.int64)
            attacker_goal = active & transition.terminated & (scoring_team == attacker)
            goals += int(attacker_goal.sum())
            high_goal += int((attacker_goal & touched_elevated).sum())
            train_mask = torch.zeros((worlds, 2), dtype=torch.bool, device=device)
            train_mask[rows, attacker] = active
            if not deterministic:
                _, next_value_flat = model(transition.transition_observation.reshape(-1, 182))
                next_value = next_value_flat.reshape(worlds, 2)
                version = torch.zeros((worlds, 2), dtype=torch.int64, device=device)
                rollout.add(
                    observation=observation,
                    action=transition.emitted_action,
                    pre_tanh=pre_tanh,
                    old_log_probability=log_probability,
                    value=value,
                    reward=reward,
                    terminated=transition.terminated[:, None].expand(-1, 2),
                    truncated=transition.truncated[:, None].expand(-1, 2),
                    next_value=next_value,
                    policy_version=version,
                    opponent_version=torch.full_like(version, -99),
                    train_mask=train_mask,
                )
            selected_action = transition.emitted_action[rows, attacker, :5]
            action_saturation += (selected_action.abs() > 0.95).sum(dim=0, dtype=torch.float64)
            action_count += active.sum(dtype=torch.float64)
            active &= ~(transition.terminated | transition.truncated)
            observation = transition.observation
    torch.cuda.synchronize()
    metrics = {
        "scenario_counts": {
            name: int((scenario == index).sum()) for index, name in enumerate(SCENARIO_NAMES)
        },
        "active_worlds_at_end": int(active.sum()),
        "attacker_goals": goals,
        "high_aerial_goals": high_goal,
        "telemetry": asdict(tracker.telemetry),
        "analog_saturation_fraction": (action_saturation / action_count.clamp_min(1.0))
        .cpu()
        .tolist(),
        "finite_observation": bool(torch.isfinite(observation).all()),
    }
    del env
    gc.collect()
    torch.cuda.empty_cache()
    return rollout, metrics


def checkpoint_payload(
    source: dict[str, Any],
    model: Rival2ActorCritic,
    policy_optimizer: torch.optim.Optimizer,
    critic_optimizer: torch.optim.Optimizer,
    *,
    update: int,
    authority: dict[str, Any],
    human_result: dict[str, Any],
    validation: dict[str, Any],
) -> dict[str, Any]:
    payload = copy.deepcopy(source)
    payload["model"] = {name: value.detach().cpu() for name, value in model.state_dict().items()}
    payload["optimizer"] = {
        "format": "RIVAL2_CAPABILITY_CURRICULUM_V1_ISOLATED_OPTIMIZERS",
        "policy": policy_optimizer.state_dict(),
        "critic": critic_optimizer.state_dict(),
    }
    payload["policy_version"] = int(source["policy_version"]) + update
    payload["iteration"] = int(source["iteration"]) + update
    payload["curriculum_transition"] = {
        "identity": "RIVAL2_CAPABILITY_CURRICULUM_V1",
        "created_utc": utc_now(),
        "authority": {
            "path": AUTHORITY.relative_to(ROOT).as_posix(),
            "sha256": sha256_file(AUTHORITY),
        },
        "source": {"path": SOURCE.relative_to(ROOT).as_posix(), "sha256": SOURCE_SHA256},
        "human_rehearsal": human_result,
        "scenario_ppo_updates": update,
        "scenario_validation": validation,
        "kl_policy": "telemetry_only",
        "value_loss_to_policy_trunk": "isolated_zero_by_detached_features",
        "base_reward": authority["contracts"]["base_reward"],
        "training_overlay": authority["physical_training_overlay"],
        "named_mechanic_classifier_used": False,
    }
    return payload


def run(args: argparse.Namespace) -> int:
    authority = load_authority()
    if sha256_file(SOURCE) != SOURCE_SHA256:
        raise RuntimeError("capability source checkpoint changed")
    source = torch.load(SOURCE, map_location="cpu", weights_only=False)
    preflight = {
        "format": "RIVAL2_CAPABILITY_CURRICULUM_V1_PREFLIGHT",
        "created_utc": utc_now(),
        "authority_sha256": sha256_file(AUTHORITY),
        "source_sha256": SOURCE_SHA256,
        "source_model_tensor_sha256": human_base.tensor_tree_sha256(source["model"]),
        "contracts": source["contract_hashes"],
        "physics_hz": source["physics_hz"],
        "policy_hz": source["policy_hz"],
        "named_mechanic_classifier_used": False,
        "kl_telemetry_only": True,
        "human_test_loaded": False,
        "verdict": "PASS",
    }
    write_json(RESULTS / "preflight.json", preflight)
    if args.preflight_only:
        print(json.dumps(preflight, indent=2, sort_keys=True))
        return 0

    run_dir = Path(args.run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    human_payload, human_result = human_warm_start(
        source, authority, device=args.device, run_dir=run_dir
    )
    model = Rival2ActorCritic(Rival2PolicyConfig(**source["policy_config"])).to(args.device)
    model.load_state_dict(human_payload["model"])
    ppo = authority["scenario_curriculum"]["ppo"]
    policy_optimizer, critic_optimizer = _build_optimizers(
        model,
        policy_lr=float(ppo["actor_trunk_learning_rate"]),
        critic_lr=float(ppo["critic_learning_rate"]),
    )
    generator = torch.Generator(device=args.device).manual_seed(SEED ^ 0xA11)
    exploration = authority["scenario_curriculum"]["exploration"]
    distribution = HybridDistributionOverride(
        analog_log_std=math.log(float(exploration["analog_sigma"])),
        button_temperature=float(exploration["button_temperature"]),
    )
    geometry = ArenaGeometry.load_soccar(args.collision_dir)
    meshes = WarpArenaMeshes(geometry, args.device)
    baseline_validation = desired_validation(
        model, _load_validation(args.device), device=args.device
    )
    _, baseline_scenario = collect_scenario_rollout(
        model,
        geometry,
        meshes,
        collision_dir=args.collision_dir,
        worlds=args.eval_worlds,
        seed=SEED ^ 0xE00,
        device=args.device,
        generator=generator,
        distribution_override=distribution,
        deterministic=True,
        horizon=256,
    )
    best_rank = capability_rank(baseline_scenario, baseline_validation)
    best_payload = checkpoint_payload(
        human_payload,
        model,
        policy_optimizer,
        critic_optimizer,
        update=0,
        authority=authority,
        human_result=human_result,
        validation={"scenario": baseline_scenario, "human": baseline_validation},
    )
    best = {
        "update": 0,
        "rank": list(best_rank),
        "scenario": baseline_scenario,
        "human": baseline_validation,
    }
    curve = RESULTS / "scenario_curve.jsonl"
    if curve.exists():
        curve.unlink()
    for update in range(1, args.scenario_updates + 1):
        rollout, rollout_metrics = collect_scenario_rollout(
            model,
            geometry,
            meshes,
            collision_dir=args.collision_dir,
            worlds=args.worlds,
            seed=SEED + update,
            device=args.device,
            generator=generator,
            distribution_override=distribution,
            deterministic=False,
            horizon=128,
        )
        if rollout is None:
            raise RuntimeError("training rollout was not materialized")
        optimizer_metrics = isolated_ppo_update(
            model,
            policy_optimizer,
            critic_optimizer,
            rollout,
            generator=generator,
            distribution_override=distribution,
        )
        del rollout
        row: dict[str, Any] = {
            "update": update,
            "rollout": rollout_metrics,
            "optimizer": optimizer_metrics,
        }
        if (
            update % int(authority["scenario_curriculum"]["evaluation_interval_updates"]) == 0
            or update == args.scenario_updates
        ):
            validation = _load_validation(args.device)
            human_metrics = desired_validation(model, validation, device=args.device)
            _, scenario_metrics = collect_scenario_rollout(
                model,
                geometry,
                meshes,
                collision_dir=args.collision_dir,
                worlds=args.eval_worlds,
                seed=SEED ^ 0xE00,
                device=args.device,
                generator=generator,
                distribution_override=distribution,
                deterministic=True,
                horizon=256,
            )
            rank = capability_rank(scenario_metrics, human_metrics)
            row["validation"] = {
                "human": human_metrics,
                "scenario": scenario_metrics,
                "rank": list(rank),
            }
            if rank > best_rank:
                best_rank = rank
                best_payload = checkpoint_payload(
                    human_payload,
                    model,
                    policy_optimizer,
                    critic_optimizer,
                    update=update,
                    authority=authority,
                    human_result=human_result,
                    validation={"scenario": scenario_metrics, "human": human_metrics},
                )
                best = {
                    "update": update,
                    "rank": list(rank),
                    "scenario": scenario_metrics,
                    "human": human_metrics,
                }
            print(
                json.dumps(
                    {
                        "stage": "scenario_ppo",
                        "update": update,
                        "rank": list(rank),
                        "best_update": best["update"],
                        "telemetry": scenario_metrics["telemetry"],
                        "goals": scenario_metrics["attacker_goals"],
                        "high_goals": scenario_metrics["high_aerial_goals"],
                        "human_gameplay": human_metrics["gameplay"]["complete_action_rmse"],
                    },
                    sort_keys=True,
                ),
                flush=True,
            )
        append_jsonl(curve, row)
        rolling = run_dir / "rolling.pt"
        rolling_payload = checkpoint_payload(
            human_payload,
            model,
            policy_optimizer,
            critic_optimizer,
            update=update,
            authority=authority,
            human_result=human_result,
            validation=row.get("validation", {}),
        )
        torch.save(rolling_payload, rolling)
    selected = run_dir / "selected.pt"
    torch.save(best_payload, selected)
    CHECKPOINTS.mkdir(parents=True, exist_ok=True)
    promoted = CHECKPOINTS / "rival2_capability_curriculum_v1.pt"
    shutil.copy2(selected, promoted)
    result = {
        "format": "RIVAL2_CAPABILITY_CURRICULUM_V1_RESULT",
        "created_utc": utc_now(),
        "authority_sha256": sha256_file(AUTHORITY),
        "human_rehearsal": human_result,
        "baseline_scenario": baseline_scenario,
        "best": best,
        "checkpoint": {
            "path": promoted.relative_to(ROOT).as_posix(),
            "sha256": sha256_file(promoted),
            "model_tensor_sha256": human_base.tensor_tree_sha256(best_payload["model"]),
        },
        "human_test_loaded": False,
        "full_nexto_not_run_yet": True,
    }
    write_json(RESULTS / "result.json", result)
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


_VALIDATION_CACHE: Any | None = None


def _load_validation(device: str) -> Any:
    global _VALIDATION_CACHE
    if _VALIDATION_CACHE is None:
        human_base.SOURCE = SOURCE
        human_base.SOURCE_SHA256 = SOURCE_SHA256
        _train, _VALIDATION_CACHE, _teacher, _identity = human_base.load_human_data(device=device)
    return _VALIDATION_CACHE


def capability_rank(
    scenario: dict[str, Any], human: dict[str, Any]
) -> tuple[int, int, int, int, int, float]:
    telemetry = scenario["telemetry"]
    eligible = int(
        human["gameplay"]["complete_action_rmse"] <= 0.61
        and human["gameplay"]["finite"]
        and human["target_mechanics"]["finite"]
        and scenario["finite_observation"]
        and max(scenario["analog_saturation_fraction"]) < 0.98
    )
    aerial = int(telemetry["elevated_contacts"]) + 5 * int(scenario["high_aerial_goals"])
    dash = (
        int(telemetry["productive_floor_landings"])
        + 2 * int(telemetry["productive_wall_landings"])
        + 3 * int(telemetry["productive_dash_chains"])
    )
    demos = int(telemetry["actual_demos"])
    broad = int(aerial > 0) + int(dash > 0) + int(demos > 0)
    return (
        eligible,
        broad,
        aerial,
        dash,
        demos,
        -float(human["target_mechanics"]["complete_action_rmse"]),
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--collision-dir", type=Path, default=DEFAULT_COLLISION_DIR)
    parser.add_argument("--run-dir", type=Path, default=DEFAULT_RUN_DIR)
    parser.add_argument("--worlds", type=int, default=8192)
    parser.add_argument("--eval-worlds", type=int, default=4096)
    parser.add_argument("--scenario-updates", type=int, default=120)
    parser.add_argument("--preflight-only", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    raise SystemExit(run(parse_args()))
