"""Train side-specialized actor-only Rival capability curriculum V2."""

from __future__ import annotations

import argparse
import copy
import gc
import json
import math
import os
import shutil
import subprocess
import sys
from dataclasses import asdict
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
import warp as wp

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from benchmarks import run_rival2_capability_curriculum_v1 as v1  # noqa: E402
from benchmarks import run_rival2_codex_autonomous_v1 as human_base  # noqa: E402
from benchmarks.analyze_rival2_v23_physical_telemetry import (  # noqa: E402
    DualCheckpointPhysicalTelemetryRunner,
)
from rivalsim.arena import ArenaGeometry, WarpArenaMeshes  # noqa: E402
from rivalsim.full_match import FullMatchRunner  # noqa: E402
from rivalsim.human_demo.bc_observation_bridge import hybrid_actor_channel_kl  # noqa: E402
from rivalsim.human_demo.behavior_cloning import (  # noqa: E402
    MechanicHierarchySampler,
    human_behavior_cloning_objective,
)
from rivalsim.rival2_capability_curriculum_v2 import (  # noqa: E402
    SCENARIO_NAMES,
    CapabilityRewardTrackerV2,
    build_capability_scenarios_v2,
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
from rivalsim.rival2_ppo import Rival2RolloutBuffer, rival2_ppo_120hz_config  # noqa: E402

AUTHORITY = ROOT / "results/rival2/capability_curriculum_v2/authority.json"
RESULTS = ROOT / "results/rival2/capability_curriculum_v2"
CHECKPOINTS = ROOT / "checkpoints/rival2/capability_curriculum_v2"
BLUE = ROOT / "checkpoints/rival2/codex_autonomous_v23/rival2_blue.pt"
ORANGE = ROOT / "checkpoints/rival2/codex_autonomous_v23/rival2_orange.pt"
BLUE_SHA256 = "0263546263285384D2D9A0CE55A471C41A41A8B7D4870DD9504D0ACCEA76723C"
ORANGE_SHA256 = "56E4ECA5075EB5748402BA3C5D8D51AC91FC1AFF55219E64EA5CE688DAD3491A"
DEFAULT_RUN_DIR = Path("G:/dev/RivalSim-runs/capability-curriculum-v2")
DEFAULT_COLLISION_DIR = Path("G:/dev/RLBot-Rival/bot/collision_meshes/soccar")
DEFAULT_COLLISION_ROOT = Path("G:/dev/RLBot-Rival/bot/collision_meshes")
SEED = 2_026_090_221
TARGET_LABELS = v1.TARGET_LABELS


def utc_now() -> str:
    return datetime.now(UTC).isoformat()


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
        "format": authority.get("format") == "RIVAL2_CAPABILITY_CURRICULUM_V2_AUTHORITY",
        "blue": authority["protected_parents"]["blue"]["sha256"] == BLUE_SHA256,
        "orange": authority["protected_parents"]["orange"]["sha256"] == ORANGE_SHA256,
        "base_reward": authority["contracts"]["base_reward_sha256"]
        == REWARD_GAMEPLAY_120_V2_CONTRACT_HASH,
        "actor_only": authority["training_boundary"]["actor"] == "only trainable module",
        "pre_airborne_zero": authority["physical_scenarios"][
            "pre_airborne_intercept_training_fraction"
        ]
        == 0.0,
        "no_named_classifier": authority["physical_overlay"]["named_mechanic_classifier"]
        is False,
    }
    if not all(checks.values()):
        raise RuntimeError(f"V2 authority mismatch: {checks}")
    identities = [
        authority["protected_parents"]["blue"],
        authority["protected_parents"]["orange"],
        authority["v1_negative_evidence"]["failure_attribution"],
        authority["v1_negative_evidence"]["natural_verdict"],
        authority["human_auxiliary"]["dataset_manifest"],
        authority["human_auxiliary"]["review_candidates"],
        authority["human_auxiliary"]["observation_adapter"],
    ]
    for identity in identities:
        path = ROOT / identity["path"]
        observed = v1.sha256_file(path)
        if observed != identity["sha256"]:
            raise RuntimeError(f"V2 bound input changed: {path}: {observed}")
    return authority


def module_hash(model: Rival2ActorCritic, module: str) -> str:
    target = getattr(model, module)
    return human_base.tensor_tree_sha256(
        {name: value.detach().cpu() for name, value in target.state_dict().items()}
    )


def make_model(payload: dict[str, Any], device: str) -> Rival2ActorCritic:
    model = Rival2ActorCritic(Rival2PolicyConfig(**payload["policy_config"])).to(device)
    model.load_state_dict(payload["model"], strict=True)
    model.trunk.requires_grad_(False)
    model.critic.requires_grad_(False)
    model.actor.requires_grad_(True)
    return model


def save_side_checkpoint(
    source: dict[str, Any],
    model: Rival2ActorCritic,
    optimizer: torch.optim.Optimizer,
    path: Path,
    *,
    side: int,
    block: int,
    authority_sha256: str,
    validation: dict[str, Any],
) -> dict[str, Any]:
    payload = copy.deepcopy(source)
    payload["model"] = {
        name: value.detach().cpu().clone() for name, value in model.state_dict().items()
    }
    payload["optimizer"] = {
        "format": "RIVAL2_CAPABILITY_CURRICULUM_V2_ACTOR_ONLY_ADAMW",
        "actor": optimizer.state_dict(),
    }
    payload["iteration"] = int(source.get("iteration", 0)) + block
    payload["policy_version"] = int(source.get("policy_version", 0)) + block
    payload["curriculum_transition"] = {
        "identity": "RIVAL2_CAPABILITY_CURRICULUM_V2",
        "created_utc": utc_now(),
        "authority": {
            "path": AUTHORITY.relative_to(ROOT).as_posix(),
            "sha256": authority_sha256,
        },
        "source": {
            "path": (BLUE if side == 0 else ORANGE).relative_to(ROOT).as_posix(),
            "sha256": BLUE_SHA256 if side == 0 else ORANGE_SHA256,
        },
        "deployment_side": side,
        "accepted_blocks": block,
        "training_boundary": "actor_only; trunk_and_critic_frozen",
        "validation": validation,
        "named_mechanic_classifier_used": False,
        "kl_policy": "objective_and_telemetry_only",
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, path)
    return {
        "path": str(path),
        "sha256": v1.sha256_file(path),
        "model_tensor_sha256": human_base.tensor_tree_sha256(payload["model"]),
        "bytes": path.stat().st_size,
        "block": block,
    }


def collect_on_policy_pool(
    model: Rival2ActorCritic,
    source_path: Path,
    *,
    side: int,
    collision_root: Path,
    device: str,
    seed: int,
    worlds: int = 2048,
    snapshots: int = 32,
    ticks_between: int = 112,
) -> torch.Tensor:
    rng = np.random.default_rng(seed)
    runner = FullMatchRunner(
        worlds,
        str(collision_root),
        source_path,
        starting_layout=rng.integers(0, 5, size=worlds, dtype=np.int32),
        rival_side=np.full(worlds, side, dtype=np.int32),
        stochastic_rival=False,
        evaluation_seed=seed,
        device=device,
    )
    runner.rival_policy = model
    rows: list[torch.Tensor] = []
    for _ in range(snapshots):
        runner.run_ticks(ticks_between)
        rows.append(
            runner.rival_observation[
                runner.batch_index, runner.rival_side
            ].detach().cpu()
        )
    result = torch.cat(rows)
    del runner
    gc.collect()
    torch.cuda.empty_cache()
    if result.shape != (worlds * snapshots, 182) or not bool(torch.isfinite(result).all()):
        raise RuntimeError("invalid V2 on-policy preservation pool")
    return result


def actor_auxiliary_update(
    student: Rival2ActorCritic,
    teacher: Rival2ActorCritic,
    optimizer: torch.optim.Optimizer,
    *,
    fixed_retention: torch.Tensor,
    on_policy_pool: torch.Tensor,
    human_train: Any,
    mechanic_observation: torch.Tensor,
    mechanic_action: torch.Tensor,
    mechanic_sampler: MechanicHierarchySampler,
    cpu_generator: torch.Generator,
    cuda_generator: torch.Generator,
    device: str,
    authority: dict[str, Any],
) -> dict[str, Any]:
    natural = authority["natural_teacher_preservation"]
    human = authority["human_auxiliary"]
    fixed_index = torch.randint(
        fixed_retention.shape[0],
        (int(natural["fixed_parent_retention_samples_per_block"]),),
        generator=cpu_generator,
    )
    on_policy_index = torch.randint(
        on_policy_pool.shape[0],
        (int(natural["on_policy_nexto_state_samples_per_block"]),),
        generator=cpu_generator,
    )
    natural_observation = torch.cat(
        (
            fixed_retention.index_select(0, fixed_index),
            on_policy_pool.index_select(0, on_policy_index),
        )
    ).to(device)
    gameplay_index = torch.randint(
        human_train.gameplay_observation.shape[0],
        (int(human["gameplay_samples_per_block"]),),
        generator=cpu_generator,
    )
    mechanic_index = mechanic_sampler.sample(int(human["mechanic_samples_per_block"]))
    gameplay_observation = human_train.gameplay_observation.index_select(
        0, gameplay_index
    ).to(device)
    gameplay_action = human_train.gameplay_action.index_select(0, gameplay_index).to(device)
    target_observation = mechanic_observation.index_select(0, mechanic_index).to(device)
    target_action = mechanic_action.index_select(0, mechanic_index).to(device)
    with torch.no_grad():
        teacher_natural, _ = teacher(natural_observation)
        teacher_gameplay, _ = teacher(gameplay_observation)
        teacher_mechanic, _ = teacher(target_observation)
    student_natural, _ = student(natural_observation)
    student_gameplay, _ = student(gameplay_observation)
    student_mechanic, _ = student(target_observation)
    channel_kl = hybrid_actor_channel_kl(
        teacher_natural, student_natural, policy_config=student.config
    )
    retention_kl = channel_kl.sum(dim=-1).mean()
    action_loss = F.smooth_l1_loss(
        torch.tanh(student_natural[:, :5]), torch.tanh(teacher_natural[:, :5]), beta=0.1
    ) + F.binary_cross_entropy_with_logits(
        student_natural[:, 10:13], torch.sigmoid(teacher_natural[:, 10:13])
    )
    gameplay_bc = human_behavior_cloning_objective(
        student_gameplay,
        teacher_gameplay,
        gameplay_action,
        smooth_l1_beta=0.1,
        analog_weight=1.0,
        button_weight=0.25,
        log_std_weight=float(human["log_std_parent_retention_weight"]),
        policy_config=student.config,
    )
    mechanic_bc = human_behavior_cloning_objective(
        student_mechanic,
        teacher_mechanic,
        target_action,
        smooth_l1_beta=0.1,
        analog_weight=1.0,
        button_weight=0.25,
        log_std_weight=float(human["log_std_parent_retention_weight"]),
        policy_config=student.config,
    )
    loss = (
        float(natural["teacher_actor_kl_weight"]) * retention_kl
        + float(natural["teacher_action_smooth_l1_weight"]) * action_loss
        + float(human["gameplay_loss_weight"]) * gameplay_bc.loss
        + float(human["mechanic_loss_weight"]) * mechanic_bc.loss
    )
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    gradient = torch.nn.utils.clip_grad_norm_(student.actor.parameters(), 0.5)
    if not bool(torch.isfinite(loss) and torch.isfinite(gradient)):
        raise RuntimeError("nonfinite V2 actor auxiliary update")
    optimizer.step()
    if not all(bool(torch.isfinite(value).all()) for value in student.actor.parameters()):
        raise RuntimeError("nonfinite V2 actor state")
    return {
        "loss": float(loss.detach()),
        "retention_mean_kl": float(retention_kl.detach()),
        "retention_per_channel_kl": channel_kl.mean(dim=0).detach().cpu().tolist(),
        "teacher_action_loss": float(action_loss.detach()),
        "gameplay_bc_loss": float(gameplay_bc.loss.detach()),
        "mechanic_bc_loss": float(mechanic_bc.loss.detach()),
        "gradient_norm": float(gradient.detach()),
    }


def collect_scenario_rollout(
    model: Rival2ActorCritic,
    geometry: ArenaGeometry,
    meshes: WarpArenaMeshes,
    *,
    side: int,
    collision_dir: Path,
    worlds: int,
    horizon: int,
    seed: int,
    device: str,
    generator: torch.Generator,
    distribution: HybridDistributionOverride,
    deterministic: bool,
) -> tuple[Rival2RolloutBuffer | None, dict[str, Any]]:
    batch = build_capability_scenarios_v2(
        worlds, seed=seed, attacker_side=side
    )
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
    scripted = torch.from_numpy(batch.scripted_action).to(device)
    tracker = CapabilityRewardTrackerV2(scenario, attacker_side=side)
    world_normal = wp.to_torch(env.world.vehicle.world_contact_normal).reshape(worlds, 2, 3)
    rows = torch.arange(worlds, device=device)
    opponent = 1 - side
    active = torch.ones(worlds, dtype=torch.bool, device=device)
    rollout = None if deterministic else Rival2RolloutBuffer(horizon, worlds, device)
    saturation = torch.zeros(5, dtype=torch.float64, device=device)
    action_count = torch.zeros((), dtype=torch.float64, device=device)
    observation = env.observation
    model.eval()
    for tick in range(horizon):
        with torch.no_grad():
            actor_flat, value_flat = model(observation.reshape(-1, 182))
            actor = actor_flat.reshape(worlds, 2, 13)
            value = value_flat.reshape(worlds, 2)
            if deterministic:
                action = deterministic_hybrid_action(actor, model.config)
                pre_tanh = actor[..., :5]
                log_probability = hybrid_log_probability(
                    actor,
                    action,
                    config=model.config,
                    pre_tanh=pre_tanh,
                    distribution_override=distribution,
                )
            else:
                sample = sample_hybrid_action(
                    actor,
                    generator=generator,
                    config=model.config,
                    distribution_override=distribution,
                )
                action = sample.action
                pre_tanh = sample.pre_tanh
                log_probability = sample.log_probability
            action[:, opponent] = scripted[:, opponent]
            action = torch.where(active[:, None, None], action, torch.zeros_like(action))
            transition = env.step(action)
            scoring_team = env.bridge.views["rival2.scoring_team_latched"].to(torch.int64)
            goal_for = active & transition.terminated & (scoring_team == side)
            overlay = tracker.step(
                observation,
                transition.transition_observation,
                tick=tick,
                world_contact_normal=world_normal[:, side],
                goal_for_attacker=goal_for,
            )
            reward = torch.zeros_like(transition.reward)
            reward[:, side] = torch.where(
                active, transition.reward[:, side] + overlay, 0.0
            )
            train_mask = torch.zeros((worlds, 2), dtype=torch.bool, device=device)
            train_mask[:, side] = active
            if rollout is not None:
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
            selected_action = transition.emitted_action[:, side, :5]
            saturation += (selected_action.abs() > 0.95).sum(dim=0, dtype=torch.float64)
            action_count += active.sum(dtype=torch.float64)
            active &= ~(transition.terminated | transition.truncated)
            observation = transition.observation
    torch.cuda.synchronize()
    metrics = {
        "side": side,
        "scenario_counts": {
            name: int((scenario == index).sum()) for index, name in enumerate(SCENARIO_NAMES)
        },
        "active_worlds_at_end": int(active.sum()),
        "telemetry": asdict(tracker.telemetry),
        "analog_saturation_fraction": (
            saturation / action_count.clamp_min(1.0)
        ).cpu().tolist(),
        "finite_observation": bool(torch.isfinite(observation).all()),
    }
    del env
    gc.collect()
    torch.cuda.empty_cache()
    return rollout, metrics


def actor_ppo_update(
    model: Rival2ActorCritic,
    optimizer: torch.optim.Optimizer,
    rollout: Rival2RolloutBuffer,
    *,
    generator: torch.Generator,
    distribution: HybridDistributionOverride,
) -> dict[str, float]:
    config = rival2_ppo_120hz_config()
    rollout.compute_gae(config)
    index = torch.nonzero(rollout.train_mask.reshape(-1), as_tuple=False).squeeze(-1)
    observation = rollout.observations.reshape(-1, 182).index_select(0, index)
    action = rollout.actions.reshape(-1, 8).index_select(0, index)
    pre_tanh = rollout.pre_tanh.reshape(-1, 5).index_select(0, index)
    old_log_probability = rollout.old_log_probability.reshape(-1).index_select(0, index)
    advantage = rollout.advantages.reshape(-1).index_select(0, index)
    advantage = (advantage - advantage.mean()) / advantage.std(unbiased=False).clamp_min(1e-8)
    sums = {"policy_loss": 0.0, "entropy": 0.0, "approx_kl": 0.0, "clip_fraction": 0.0, "gradient_norm": 0.0}
    proposals = 0
    permutation = torch.randperm(index.numel(), device=index.device, generator=generator)
    for start in range(0, index.numel(), 65_536):
        batch = permutation[start : start + 65_536]
        local_observation = observation.index_select(0, batch)
        with torch.no_grad():
            features = model.trunk(local_observation)
        actor = model.actor(features)
        new_log_probability = hybrid_log_probability(
            actor,
            action.index_select(0, batch),
            config=model.config,
            pre_tanh=pre_tanh.index_select(0, batch),
            distribution_override=distribution,
        )
        old = old_log_probability.index_select(0, batch)
        log_ratio = new_log_probability - old
        ratio = torch.exp(log_ratio)
        local_advantage = advantage.index_select(0, batch)
        policy_loss = -torch.minimum(
            ratio * local_advantage,
            ratio.clamp(0.8, 1.2) * local_advantage,
        ).mean()
        entropy = hybrid_entropy(
            actor, model.config, distribution_override=distribution
        ).mean()
        optimizer.zero_grad(set_to_none=True)
        policy_loss.backward()
        gradient = torch.nn.utils.clip_grad_norm_(model.actor.parameters(), 0.5)
        if not bool(torch.isfinite(policy_loss) and torch.isfinite(gradient)):
            raise RuntimeError("nonfinite V2 scenario PPO update")
        optimizer.step()
        with torch.no_grad():
            approx_kl = ((ratio - 1.0) - log_ratio).mean()
            clip_fraction = (ratio.sub(1.0).abs() > 0.2).float().mean()
        for name, value in zip(
            sums,
            (policy_loss, entropy, approx_kl, clip_fraction, gradient),
            strict=True,
        ):
            sums[name] += float(value.detach())
        proposals += 1
    if not all(bool(torch.isfinite(value).all()) for value in model.actor.parameters()):
        raise RuntimeError("nonfinite V2 actor after PPO")
    return {**{name: value / proposals for name, value in sums.items()}, "steps": proposals}


def short_screen(
    blue_path: Path,
    blue_sha: str,
    orange_path: Path,
    orange_sha: str,
    *,
    collision_root: Path,
    seed: int,
    ticks: int = 3600,
) -> dict[str, Any]:
    DualCheckpointPhysicalTelemetryRunner.orange_checkpoint = orange_path.resolve()
    DualCheckpointPhysicalTelemetryRunner.orange_sha256 = orange_sha
    layout = np.repeat(np.arange(5, dtype=np.int32), 2)
    side = np.tile(np.asarray([0, 1], dtype=np.int32), 5)
    runner = DualCheckpointPhysicalTelemetryRunner(
        10,
        str(collision_root),
        blue_path,
        starting_layout=layout,
        rival_side=side,
        stochastic_rival=False,
        evaluation_seed=seed,
        trace_ticks=ticks,
    )
    runner.run_ticks(ticks)
    trace = runner.trace_numpy()
    raw = runner.export()["raw"]
    status = runner.phase_status()
    touch = raw["touch_count"][np.arange(10), side]
    rival_goals = np.where(side == 0, status["blue_score"], status["orange_score"])
    nexto_goals = np.where(side == 0, status["orange_score"], status["blue_score"])
    result = {
        "ticks": ticks,
        "rival_goals": int(rival_goals.sum()),
        "nexto_goals": int(nexto_goals.sum()),
        "touches": int(touch.sum()),
        "no_touch_worlds": int((touch == 0).sum()),
        "mean_speed_uu_per_second": float(trace["speed_3d"].mean()),
        "finite_actions": bool(np.isfinite(trace["action"]).all()),
        "analog_saturation_fraction": (
            np.abs(trace["action"][..., :5]) > 0.95
        ).mean(axis=(0, 1)).tolist(),
    }
    # FullMatchRunner activates a Torch wrapper around its Warp-owned stream.
    # Restore a process-owned default stream before releasing the runner;
    # otherwise the next Rival2Env can inherit a stream whose Warp handle was
    # destroyed with the evaluation world.
    torch.cuda.synchronize(runner.device)
    default_stream = torch.cuda.default_stream(runner.device)
    torch.cuda.set_stream(default_stream)
    wp.set_stream(
        wp.stream_from_torch(default_stream),
        device=runner.world.device,
        sync=False,
    )
    del runner
    gc.collect()
    torch.cuda.empty_cache()
    return result


def screen_eligible(candidate: dict[str, Any], baseline: dict[str, Any]) -> bool:
    ratios = (
        candidate["rival_goals"] / max(1, baseline["rival_goals"]),
        candidate["touches"] / max(1, baseline["touches"]),
        candidate["mean_speed_uu_per_second"] / max(1.0, baseline["mean_speed_uu_per_second"]),
    )
    return bool(
        min(ratios) >= 0.80
        and candidate["no_touch_worlds"] <= max(1, baseline["no_touch_worlds"] + 1)
        and candidate["finite_actions"]
        and max(candidate["analog_saturation_fraction"]) < 0.98
    )


def combined_scenario_metrics(rows: list[dict[str, Any]]) -> dict[str, Any]:
    names = rows[0]["telemetry"].keys()
    return {
        "telemetry": {name: sum(row["telemetry"][name] for row in rows) for name in names},
        "finite": all(row["finite_observation"] for row in rows),
        "maximum_analog_saturation": max(
            max(row["analog_saturation_fraction"]) for row in rows
        ),
        "by_side": rows,
    }


def pair_rank(
    scenario: dict[str, Any],
    human: dict[str, Any],
    screen: dict[str, Any],
    baseline_screen: dict[str, Any],
    baseline_human: dict[str, Any],
) -> tuple[int, int, int, int, int, int, float]:
    telemetry = scenario["telemetry"]
    gates = (
        telemetry["ground_origin_high_contacts"] >= 20,
        telemetry["high_contact_goals"] >= 2,
        telemetry["productive_floor_landings"] >= 20,
        telemetry["productive_wall_landings"] >= 10,
        telemetry["productive_landing_chains"] >= 5,
        telemetry["actual_demos"] >= 10,
        telemetry["offensive_context_demos"] / max(1, telemetry["actual_demos"]) >= 0.6,
    )
    human_gameplay = max(
        side["gameplay"]["complete_action_rmse"] for side in human["by_side"]
    )
    human_target = sum(
        side["target_mechanics"]["complete_action_rmse"] for side in human["by_side"]
    ) / 2.0
    baseline_target = sum(
        side["target_mechanics"]["complete_action_rmse"]
        for side in baseline_human["by_side"]
    ) / 2.0
    eligible = int(
        screen_eligible(screen, baseline_screen)
        and human_gameplay <= 0.59
        and human_target < baseline_target
        and scenario["finite"]
        and scenario["maximum_analog_saturation"] < 0.98
    )
    return (
        eligible,
        sum(gates),
        int(telemetry["high_contact_goals"]),
        int(telemetry["ground_origin_high_contacts"]),
        int(
            telemetry["productive_floor_landings"]
            + telemetry["productive_wall_landings"]
            + 2 * telemetry["productive_landing_chains"]
        ),
        int(
            telemetry["actual_demos"]
            + telemetry["demo_followup_touches"]
            + 2 * telemetry["demo_followup_goals"]
        ),
        -human_target,
    )


def run(args: argparse.Namespace) -> int:
    authority = load_authority()
    authority_sha = v1.sha256_file(AUTHORITY)
    blue_source = torch.load(BLUE, map_location="cpu", weights_only=False)
    orange_source = torch.load(ORANGE, map_location="cpu", weights_only=False)
    if human_base.tensor_tree_sha256(blue_source["model"]) != authority["protected_parents"]["blue"]["model_tensor_sha256"]:
        raise RuntimeError("V2 Blue source model changed")
    if human_base.tensor_tree_sha256(orange_source["model"]) != authority["protected_parents"]["orange"]["model_tensor_sha256"]:
        raise RuntimeError("V2 Orange source model changed")
    sources = [blue_source, orange_source]
    models = [make_model(source, args.device) for source in sources]
    teachers = [make_model(source, args.device).eval().requires_grad_(False) for source in sources]
    frozen_hashes = [
        {"trunk": module_hash(model, "trunk"), "critic": module_hash(model, "critic")}
        for model in models
    ]
    optimizers = [
        torch.optim.AdamW(
            model.actor.parameters(),
            lr=float(authority["training_boundary"]["actor_learning_rate"]),
            weight_decay=float(authority["training_boundary"]["weight_decay"]),
        )
        for model in models
    ]
    human_base.SOURCE = BLUE
    human_base.SOURCE_SHA256 = BLUE_SHA256
    train, validation, _human_teacher, human_identity = human_base.load_human_data(
        device=args.device
    )
    if human_identity["test_loaded"]:
        raise RuntimeError("V2 human test split must remain unopened")
    target_mask = torch.tensor(
        [label in TARGET_LABELS for label in train.mechanic_label], dtype=torch.bool
    )
    target_observation = train.mechanic_observation[target_mask]
    target_action = train.mechanic_action[target_mask]
    target_labels = np.asarray(train.mechanic_label)[target_mask.numpy()]
    target_attempts = np.asarray(train.mechanic_attempt)[target_mask.numpy()]
    cpu_generators = [
        torch.Generator(device="cpu").manual_seed(SEED ^ (0xB100 + side))
        for side in (0, 1)
    ]
    cuda_generators = [
        torch.Generator(device=args.device).manual_seed(SEED ^ (0xC100 + side))
        for side in (0, 1)
    ]
    samplers = [
        MechanicHierarchySampler(
            target_labels.tolist(),
            target_attempts.tolist(),
            uniform_label_fraction=0.10,
            maximum_oversampling_ratio=6.0,
            generator=cpu_generators[side],
        )
        for side in (0, 1)
    ]
    fixed_retention = [
        source["opponent_curriculum"]["adaptive_ppo"]["retention_observations"].clone()
        for source in sources
    ]
    baseline_human = {
        "by_side": [
            v1.desired_validation(model, validation, device=args.device) for model in models
        ]
    }
    baseline_screen = short_screen(
        BLUE,
        BLUE_SHA256,
        ORANGE,
        ORANGE_SHA256,
        collision_root=args.collision_root,
        seed=SEED,
    )
    preflight = {
        "format": "RIVAL2_CAPABILITY_CURRICULUM_V2_PREFLIGHT",
        "created_utc": utc_now(),
        "authority_sha256": authority_sha,
        "source_sha256": [BLUE_SHA256, ORANGE_SHA256],
        "frozen_hashes": frozen_hashes,
        "baseline_human": baseline_human,
        "baseline_short_screen": baseline_screen,
        "human_test_loaded": False,
        "named_mechanic_classifier_used": False,
        "verdict": "PASS",
    }
    RESULTS.mkdir(parents=True, exist_ok=True)
    write_json(RESULTS / "preflight.json", preflight)
    if args.preflight_only:
        print(json.dumps(preflight, indent=2, sort_keys=True))
        return 0

    run_dir = Path(args.run_dir)
    if run_dir.exists() and any(run_dir.iterdir()):
        raise RuntimeError("V2 run directory must be fresh")
    run_dir.mkdir(parents=True, exist_ok=True)
    geometry = ArenaGeometry.load_soccar(args.collision_dir)
    meshes = WarpArenaMeshes(geometry, args.device)
    distribution = HybridDistributionOverride(
        analog_log_std=math.log(float(authority["physical_scenarios"]["exploration"]["analog_sigma"])),
        button_temperature=float(authority["physical_scenarios"]["exploration"]["button_temperature"]),
    )
    on_policy_pools: list[torch.Tensor] = []
    best_rank = (0, 0, 0, 0, 0, 0, float("-inf"))
    best: dict[str, Any] | None = None
    curve = RESULTS / "training_curve.jsonl"
    if curve.exists():
        curve.unlink()
    consecutive_pass = 0
    stop_reason = "maximum_blocks"
    maximum = min(int(args.blocks), int(authority["optimization"]["maximum_blocks"]))
    for block in range(1, maximum + 1):
        if block == 1 or (block - 1) % int(authority["natural_teacher_preservation"]["screen"]["frequency_blocks"]) == 0:
            on_policy_pools = [
                collect_on_policy_pool(
                    models[side],
                    BLUE if side == 0 else ORANGE,
                    side=side,
                    collision_root=args.collision_root,
                    device=args.device,
                    seed=SEED + block * 10 + side,
                )
                for side in (0, 1)
            ]
        side_rows: list[dict[str, Any]] = []
        for side in (0, 1):
            rollout, rollout_metrics = collect_scenario_rollout(
                models[side],
                geometry,
                meshes,
                side=side,
                collision_dir=args.collision_dir,
                worlds=args.worlds_per_side,
                horizon=int(authority["physical_scenarios"]["horizon_ticks"]),
                seed=SEED + block * 100 + side,
                device=args.device,
                generator=cuda_generators[side],
                distribution=distribution,
                deterministic=False,
            )
            assert rollout is not None
            ppo = actor_ppo_update(
                models[side],
                optimizers[side],
                rollout,
                generator=cuda_generators[side],
                distribution=distribution,
            )
            del rollout
            auxiliary = actor_auxiliary_update(
                models[side],
                teachers[side],
                optimizers[side],
                fixed_retention=fixed_retention[side],
                on_policy_pool=on_policy_pools[side],
                human_train=train,
                mechanic_observation=target_observation,
                mechanic_action=target_action,
                mechanic_sampler=samplers[side],
                cpu_generator=cpu_generators[side],
                cuda_generator=cuda_generators[side],
                device=args.device,
                authority=authority,
            )
            if module_hash(models[side], "trunk") != frozen_hashes[side]["trunk"] or module_hash(models[side], "critic") != frozen_hashes[side]["critic"]:
                raise RuntimeError("V2 frozen trunk or critic changed")
            side_rows.append(
                {"side": side, "rollout": rollout_metrics, "ppo": ppo, "auxiliary": auxiliary}
            )
        row: dict[str, Any] = {"block": block, "sides": side_rows}
        interval = int(authority["optimization"]["selection_interval_blocks"])
        if block % interval == 0 or block == maximum:
            candidate_records = []
            for side in (0, 1):
                path = run_dir / f"candidate_b{block:04d}_{'blue' if side == 0 else 'orange'}.pt"
                candidate_records.append(
                    save_side_checkpoint(
                        sources[side],
                        models[side],
                        optimizers[side],
                        path,
                        side=side,
                        block=block,
                        authority_sha256=authority_sha,
                        validation={},
                    )
                )
            screen = short_screen(
                Path(candidate_records[0]["path"]),
                candidate_records[0]["sha256"],
                Path(candidate_records[1]["path"]),
                candidate_records[1]["sha256"],
                collision_root=args.collision_root,
                seed=SEED,
            )
            human = {
                "by_side": [
                    v1.desired_validation(model, validation, device=args.device)
                    for model in models
                ]
            }
            eval_sides = []
            for side in (0, 1):
                _unused, metrics = collect_scenario_rollout(
                    models[side],
                    geometry,
                    meshes,
                    side=side,
                    collision_dir=args.collision_dir,
                    worlds=args.eval_worlds_per_side,
                    horizon=int(authority["physical_scenarios"]["horizon_ticks"]),
                    seed=SEED ^ (0xE200 + side),
                    device=args.device,
                    generator=cuda_generators[side],
                    distribution=distribution,
                    deterministic=True,
                )
                eval_sides.append(metrics)
            scenario = combined_scenario_metrics(eval_sides)
            rank = pair_rank(scenario, human, screen, baseline_screen, baseline_human)
            validation_row = {
                "checkpoint": candidate_records,
                "short_screen": screen,
                "short_screen_eligible": screen_eligible(screen, baseline_screen),
                "human": human,
                "scenario": scenario,
                "rank": list(rank),
            }
            row["validation"] = validation_row
            if rank > best_rank:
                best_rank = rank
                best = copy.deepcopy(validation_row)
                write_json(RESULTS / "best.json", best)
            if rank[0] and rank[1] == 7:
                consecutive_pass += 1
            else:
                consecutive_pass = 0
            print(
                json.dumps(
                    {
                        "block": block,
                        "rank": list(rank),
                        "best_rank": list(best_rank),
                        "screen": screen,
                        "telemetry": scenario["telemetry"],
                    },
                    sort_keys=True,
                ),
                flush=True,
            )
            if consecutive_pass >= 2:
                stop_reason = "two_consecutive_capability_passes"
                append_jsonl(curve, row)
                break
        append_jsonl(curve, row)
        rolling_dir = run_dir / "rolling"
        for side in (0, 1):
            save_side_checkpoint(
                sources[side],
                models[side],
                optimizers[side],
                rolling_dir / ("blue.pt" if side == 0 else "orange.pt"),
                side=side,
                block=block,
                authority_sha256=authority_sha,
                validation=row.get("validation", {}),
            )

    if best is None:
        raise RuntimeError("V2 produced no selection boundary")
    selected_records = best["checkpoint"]
    CHECKPOINTS.mkdir(parents=True, exist_ok=True)
    selected_blue = CHECKPOINTS / "rival2_blue.pt"
    selected_orange = CHECKPOINTS / "rival2_orange.pt"
    shutil.copy2(Path(selected_records[0]["path"]), selected_blue)
    shutil.copy2(Path(selected_records[1]["path"]), selected_orange)
    selected_sha = [v1.sha256_file(selected_blue), v1.sha256_file(selected_orange)]
    physical_output = RESULTS / "selected_physical_behavior_telemetry.json"
    command = [
        str(ROOT / ".venv/Scripts/python.exe"),
        "-u",
        str(ROOT / "benchmarks/analyze_rival2_v23_physical_telemetry.py"),
        "--checkpoint",
        str(selected_blue),
        "--checkpoint-sha256",
        selected_sha[0],
        "--orange-checkpoint",
        str(selected_orange),
        "--orange-checkpoint-sha256",
        selected_sha[1],
        "--collision-root",
        str(args.collision_root),
        "--output",
        str(physical_output),
        "--seed",
        str(SEED ^ 0xF100),
    ]
    completed = subprocess.run(command, cwd=ROOT, check=False)
    if completed.returncode != 0:
        raise RuntimeError("V2 final physical evaluation failed")
    physical = json.loads(physical_output.read_text(encoding="utf-8"))
    overall = physical["overall"]
    score = physical["evaluation"]["score"]
    final_pass = bool(
        score["wins"] >= 6
        and score["rival_goals"] - score["nexto_goals"] >= 0
        and overall["touches"]["high_aerial_proxy"] >= 1
        and overall["scoring"]["goals_from_high_aerial_proxy"] >= 1
        and overall["jump_flip_recovery"]["productive_floor_landing_proxy"] >= 1
        and overall["jump_flip_recovery"]["productive_wall_landing_proxy"] >= 1
        and overall["jump_flip_recovery"]["productive_dash_chain_proxy"] >= 1
        and overall["demolitions"]["total"] >= 1
        and (
            overall["demolitions"]["followed_by_rival_touch_within_5_seconds"] >= 1
            or overall["demolitions"]["followed_by_rival_goal_within_5_seconds"] >= 1
        )
    )
    result = {
        "format": "RIVAL2_CAPABILITY_CURRICULUM_V2_RESULT",
        "created_utc": utc_now(),
        "authority_sha256": authority_sha,
        "stop_reason": stop_reason,
        "best": best,
        "selected": {
            "blue": {
                "path": selected_blue.relative_to(ROOT).as_posix(),
                "sha256": selected_sha[0],
            },
            "orange": {
                "path": selected_orange.relative_to(ROOT).as_posix(),
                "sha256": selected_sha[1],
            },
        },
        "final_physical_evaluation": physical_output.relative_to(ROOT).as_posix(),
        "final_pass": final_pass,
        "promoted": False,
        "human_test_loaded": False,
        "trunk_critic_frozen": True,
    }
    write_json(RESULTS / "result.json", result)
    print(json.dumps(result, indent=2, sort_keys=True), flush=True)
    return 0 if final_pass else 2


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--collision-dir", type=Path, default=DEFAULT_COLLISION_DIR)
    parser.add_argument("--collision-root", type=Path, default=DEFAULT_COLLISION_ROOT)
    parser.add_argument("--run-dir", type=Path, default=DEFAULT_RUN_DIR)
    parser.add_argument("--worlds-per-side", type=int, default=4096)
    parser.add_argument("--eval-worlds-per-side", type=int, default=2048)
    parser.add_argument("--blocks", type=int, default=160)
    parser.add_argument("--preflight-only", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    raise SystemExit(run(parse_args()))
