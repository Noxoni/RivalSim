"""Distill the source-exact observation-only aerial intercept teacher into Rival."""

from __future__ import annotations

import argparse
import copy
import gc
import json
import sys
from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from benchmarks import run_rival2_aerial_option_v1 as option_v1  # noqa: E402
from benchmarks import run_rival2_aerial_training_pack_v1 as pack_runner  # noqa: E402
from benchmarks import run_rival2_codex_autonomous_v1 as human_base  # noqa: E402
from rivalsim.arena import ArenaGeometry, WarpArenaMeshes  # noqa: E402
from rivalsim.human_demo.behavior_cloning import (  # noqa: E402
    MechanicHierarchySampler,
    action_metric_summary,
    human_behavior_cloning_objective,
)
from rivalsim.rival2_120hz_transition import tensor_tree_sha256  # noqa: E402
from rivalsim.rival2_aerial_intercept_distillation import (  # noqa: E402
    DISTILLATION_VERSION,
    DistillationDataset,
    physical_gate,
)
from rivalsim.rival2_aerial_intercept_teacher import plan_aerial_intercept  # noqa: E402
from rivalsim.rival2_aerial_option import FIELD  # noqa: E402
from rivalsim.rival2_aerial_option_v2 import apply_fast_aerial_initiation  # noqa: E402
from rivalsim.rival2_aerial_training_pack import (  # noqa: E402
    PACK_NAMES,
    AerialTrainingPackTracker,
    build_training_pack_scenarios,
)
from rivalsim.rival2_contracts import (  # noqa: E402
    BALL_LINEAR_SPEED_SCALE,
    RIVAL2_REWARD_GAMEPLAY_120_V2_VERSION,
)
from rivalsim.rival2_env import Rival2Env  # noqa: E402
from rivalsim.rival2_policy import (  # noqa: E402
    Rival2ActorCritic,
)

AUTHORITY = ROOT / "results/rival2/aerial_intercept_distillation_v1/authority.json"
AUTHORITY_SHA256 = "593DCF72E8A9720479A3A972A429444D05822FBDFBDF36FF79A4D389586FA659"
RESULTS = ROOT / "results/rival2/aerial_intercept_distillation_v1"
BLUE = ROOT / "checkpoints/rival2/aerial_option_v2/rival2_aerial_option_v2_selected_blue.pt"
ORANGE = ROOT / "checkpoints/rival2/aerial_option_v2/rival2_aerial_option_v2_selected_orange.pt"
DEFAULT_RUN_DIR = Path("G:/dev/RivalSim-runs/aerial-intercept-distillation-v1")
TARGET_LABELS = ("aerialdribble", "groundtoairdribble")


def load_authority() -> dict[str, Any]:
    if human_base.sha256_file(AUTHORITY) != AUTHORITY_SHA256:
        raise RuntimeError("aerial intercept distillation authority changed")
    authority = json.loads(AUTHORITY.read_text(encoding="utf-8"))
    identities = (
        authority["protected_competitive_base"]["blue"],
        authority["protected_competitive_base"]["orange"],
        authority["initial_option"]["blue"],
        authority["initial_option"]["orange"],
        authority["teacher"]["implementation"],
        authority["teacher"]["calibration_runner"],
        authority["teacher"]["calibration"],
        authority["scenario_authority"]["training_pack"],
        authority["scenario_authority"]["scoring_geometry_correction"],
    )
    for identity in identities:
        path = ROOT / identity["path"]
        if human_base.sha256_file(path) != identity["sha256"]:
            raise RuntimeError(f"bound distillation input changed: {path}")
    if not authority["integrity"]["no_general_policy_ppo"]:
        raise RuntimeError("distillation authority unexpectedly permits PPO")
    if authority["optimization"]["frozen"] != [
        "option.critic",
        "protected competitive base",
    ]:
        raise RuntimeError("distillation critic/competitive-base freeze changed")
    return authority


def _make_model(payload: dict[str, Any], device: str) -> Rival2ActorCritic:
    return option_v1.make_model(payload, device)


def _make_optimizer(model: Rival2ActorCritic, authority: dict[str, Any]) -> torch.optim.AdamW:
    optimization = authority["optimization"]
    return torch.optim.AdamW(
        [
            {"params": model.trunk.parameters(), "lr": float(optimization["trunk_learning_rate"])},
            {"params": model.actor.parameters(), "lr": float(optimization["actor_learning_rate"])},
        ],
        weight_decay=float(optimization["weight_decay"]),
    )


def _outcome_weight(
    pack: int, *, goal: bool, goalward: bool, authority: dict[str, Any]
) -> tuple[str, float]:
    selection = authority["trajectory_selection"]
    row = selection[PACK_NAMES[pack]]
    if pack == 0:
        if goal:
            return "qualified_goal", float(row["qualified_goal_world_weight"])
        return "qualified_high_touch", float(row["qualified_high_touch_non_goal_world_weight"])
    if goalward:
        return "goalward_high_touch", float(row["goalward_high_touch_world_weight"])
    return "other_high_touch", float(row["other_high_touch_world_weight"])


@torch.no_grad()
def collect_teacher_category(
    geometry: ArenaGeometry,
    meshes: WarpArenaMeshes,
    *,
    pack: int,
    side: int,
    worlds: int,
    seed: int,
    device: str,
    collision_dir: Path,
    horizon: int,
    deadline: int,
    authority: dict[str, Any],
) -> tuple[dict[str, Any], dict[str, Any]]:
    batch = build_training_pack_scenarios(worlds, seed=seed, attacker_side=side, pack=pack)
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
    tracker = AerialTrainingPackTracker(
        worlds,
        attacker_side=side,
        pack=pack,
        first_touch_deadline=deadline,
        horizon=horizon,
    )
    stored_observation = torch.empty((horizon, worlds, 182), dtype=torch.float32, device=device)
    stored_action = torch.empty((horizon, worlds, 8), dtype=torch.float32, device=device)
    stored_valid = torch.zeros((horizon, worlds), dtype=torch.bool, device=device)
    active = torch.ones(worlds, dtype=torch.bool, device=device)
    option_age = torch.zeros(worlds, dtype=torch.int64, device=device)
    goalward_world = torch.zeros(worlds, dtype=torch.bool, device=device)
    observation = env.observation
    for tick in range(horizon):
        active_before = active.clone()
        first_before = (
            tracker.first_high_touch.clone()
            if tracker.initialized
            else torch.zeros(worlds, dtype=torch.bool, device=device)
        )
        plan = plan_aerial_intercept(observation[:, side])
        selected, primitive = apply_fast_aerial_initiation(plan.action, option_age, active_before)
        stored_observation[tick].copy_(observation[:, side])
        stored_action[tick].copy_(plan.action)
        stored_valid[tick].copy_(active_before & ~primitive & ~first_before)
        action = torch.zeros((worlds, 2, 8), dtype=torch.float32, device=device)
        action[:, side] = torch.where(active_before[:, None], selected, 0.0)
        transition = env.step(action)
        after = transition.transition_observation
        scoring = env.bridge.views["rival2.scoring_team_latched"].to(torch.int64)
        any_goal = active_before & transition.terminated & (scoring >= 0)
        goal_for = any_goal & (scoring == side)
        _reward, pack_done = tracker.step(
            observation,
            after,
            tick=tick,
            goal_for_attacker=goal_for,
            any_goal=any_goal,
            active=active_before,
        )
        first_now = tracker.first_high_touch & ~first_before
        before_forward = observation[:, side, FIELD["ball.linear_velocity.y"]]
        after_forward = after[:, side, FIELD["ball.linear_velocity.y"]]
        transfer = (after_forward - before_forward) * BALL_LINEAR_SPEED_SCALE
        goalward_world |= first_now & (transfer >= 150.0)
        terminal = pack_done | transition.terminated | transition.truncated
        active &= ~terminal
        option_age += active_before.to(torch.int64)
        observation = transition.observation
        if not bool(active.any()):
            break
    if device.startswith("cuda"):
        torch.cuda.synchronize()
    success = tracker.first_high_touch
    observations: list[torch.Tensor] = []
    actions: list[torch.Tensor] = []
    ticks: list[torch.Tensor] = []
    trajectories: list[dict[str, Any]] = []
    offset = 0
    for world in success.nonzero(as_tuple=False).flatten().cpu().tolist():
        world_ticks = stored_valid[:, world].nonzero(as_tuple=False).flatten()
        if world_ticks.numel() == 0:
            raise RuntimeError("successful teacher world has no post-primitive action")
        expected = torch.arange(
            int(world_ticks[0]),
            int(world_ticks[0]) + world_ticks.numel(),
            device=device,
        )
        if not torch.equal(world_ticks, expected):
            raise RuntimeError("teacher trajectory is not tick-contiguous")
        world_observation = stored_observation.index_select(0, world_ticks)[:, world].cpu()
        world_action = stored_action.index_select(0, world_ticks)[:, world].cpu()
        world_tick = world_ticks.cpu().to(torch.int64)
        outcome, weight = _outcome_weight(
            pack,
            goal=bool(tracker.goal_paid[world]),
            goalward=bool(goalward_world[world]),
            authority=authority,
        )
        length = int(world_ticks.numel())
        observations.append(world_observation)
        actions.append(world_action)
        ticks.append(world_tick)
        trajectories.append(
            {
                "pack": pack,
                "side": side,
                "world": world,
                "seed": seed,
                "offset": offset,
                "length": length,
                "first_touch_tick": int(tracker.first_touch_tick[world]),
                "outcome": outcome,
                "weight": weight,
            }
        )
        offset += length
    telemetry = asdict(tracker.telemetry)
    payload = {
        "observation": torch.cat(observations) if observations else torch.empty((0, 182)),
        "action": torch.cat(actions) if actions else torch.empty((0, 8)),
        "tick": torch.cat(ticks) if ticks else torch.empty((0,), dtype=torch.int64),
        "trajectories": trajectories,
    }
    metrics = {
        "pack": PACK_NAMES[pack],
        "pack_index": pack,
        "side": side,
        "seed": seed,
        "worlds": worlds,
        "successful_trajectories": len(trajectories),
        "samples": offset,
        "telemetry": telemetry,
        "fractions": {
            "high_touch": telemetry["first_high_touches"] / worlds,
            "goalward_first_touch": telemetry["goalward_first_touches"] / worlds,
            "goal": telemetry["goals"] / worlds,
        },
        "outcomes": {
            name: sum(row["outcome"] == name for row in trajectories)
            for name in sorted({row["outcome"] for row in trajectories})
        },
    }
    del env, stored_observation, stored_action, stored_valid
    gc.collect()
    if device.startswith("cuda"):
        torch.cuda.empty_cache()
    return payload, metrics


def build_trajectory_artifact(
    geometry: ArenaGeometry,
    meshes: WarpArenaMeshes,
    *,
    split: str,
    seed_base: int,
    worlds: int,
    device: str,
    collision_dir: Path,
    authority: dict[str, Any],
    pack_authority: dict[str, Any],
    path: Path,
) -> tuple[DistillationDataset, dict[str, Any]]:
    all_observation: list[torch.Tensor] = []
    all_action: list[torch.Tensor] = []
    all_tick: list[torch.Tensor] = []
    all_trajectories: list[dict[str, Any]] = []
    metrics: list[dict[str, Any]] = []
    offset = 0
    for pack, pack_row in enumerate(pack_authority["physical_training"]["packs"]):
        for side in (0, 1):
            seed = seed_base ^ (pack * 16 + side)
            payload, row = collect_teacher_category(
                geometry,
                meshes,
                pack=pack,
                side=side,
                worlds=worlds,
                seed=seed,
                device=device,
                collision_dir=collision_dir,
                horizon=int(pack_row["horizon_ticks"]),
                deadline=int(pack_row["first_high_touch_deadline_tick"]),
                authority=authority,
            )
            for trajectory in payload["trajectories"]:
                trajectory["offset"] += offset
                all_trajectories.append(trajectory)
            all_observation.append(payload["observation"])
            all_action.append(payload["action"])
            all_tick.append(payload["tick"])
            offset += int(payload["observation"].shape[0])
            metrics.append(row)
            print(json.dumps({"stage": f"build_{split}", **row}, sort_keys=True), flush=True)
    artifact = {
        "format": f"{DISTILLATION_VERSION}_TRAJECTORIES",
        "split": split,
        "seed_base": seed_base,
        "worlds_per_pack_side": worlds,
        "observation": torch.cat(all_observation),
        "action": torch.cat(all_action),
        "tick": torch.cat(all_tick),
        "trajectories": all_trajectories,
        "category_metrics": metrics,
        "optimizer_steps": 0,
    }
    dataset = DistillationDataset(artifact)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(artifact, path)
    manifest = {
        "format": f"{DISTILLATION_VERSION}_TRAJECTORY_MANIFEST",
        "split": split,
        "path": str(path),
        "sha256": human_base.sha256_file(path),
        "tensor_sha256": tensor_tree_sha256(
            {
                "observation": artifact["observation"],
                "action": artifact["action"],
                "tick": artifact["tick"],
                "trajectories": artifact["trajectories"],
            }
        ),
        "samples": int(dataset.observation.shape[0]),
        "trajectories": len(dataset.trajectories),
        "categories": metrics,
        "optimizer_steps": 0,
    }
    return dataset, manifest


@torch.no_grad()
def evaluate_teacher_actions(
    model: Rival2ActorCritic,
    dataset: DistillationDataset,
    *,
    device: str,
    batch_size: int = 8192,
) -> dict[str, Any]:
    actors: list[torch.Tensor] = []
    for start in range(0, dataset.observation.shape[0], batch_size):
        actor, _ = model(dataset.observation[start : start + batch_size].to(device))
        actors.append(actor.cpu())
    result = action_metric_summary(torch.cat(actors), dataset.action)
    result["finite"] = all(
        bool(torch.isfinite(parameter).all()) for parameter in model.parameters()
    )
    return result


def evaluate_physical(
    models: list[Rival2ActorCritic],
    geometry: ArenaGeometry,
    meshes: WarpArenaMeshes,
    *,
    worlds: int,
    device: str,
    collision_dir: Path,
    pack_authority: dict[str, Any],
    validation_seed_base: int,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    generators = [
        torch.Generator(device=device).manual_seed(validation_seed_base ^ (0xCA00 + side))
        for side in (0, 1)
    ]
    for pack, pack_row in enumerate(pack_authority["physical_training"]["packs"]):
        distribution = pack_runner.distribution_override(pack_authority, pack)
        for side in (0, 1):
            _unused, metrics = pack_runner.collect_pack(
                models[side],
                geometry,
                meshes,
                side=side,
                pack=pack,
                worlds=worlds,
                horizon=int(pack_row["horizon_ticks"]),
                first_touch_deadline=int(pack_row["first_high_touch_deadline_tick"]),
                seed=validation_seed_base ^ (pack * 16 + side),
                device=device,
                generator=generators[side],
                distribution=distribution,
                deterministic=True,
                collision_dir=collision_dir,
            )
            rows.append(metrics)
    return rows


def save_checkpoint(
    source: dict[str, Any],
    model: Rival2ActorCritic,
    optimizer: torch.optim.AdamW,
    path: Path,
    *,
    side: int,
    block: int,
    optimizer_steps: int,
    evaluation: dict[str, Any],
    trajectory_manifests: dict[str, Any],
) -> dict[str, Any]:
    payload = copy.deepcopy(source)
    payload["model"] = {
        name: value.detach().cpu().clone() for name, value in model.state_dict().items()
    }
    payload["optimizer"] = {
        "format": f"{DISTILLATION_VERSION}_FRESH_ADAMW",
        "option": optimizer.state_dict(),
    }
    payload["curriculum_transition"] = {
        "identity": DISTILLATION_VERSION,
        "authority": {"path": AUTHORITY.relative_to(ROOT).as_posix(), "sha256": AUTHORITY_SHA256},
        "deployment_side": side,
        "accepted_block": block,
        "accepted_supervised_optimizer_steps": optimizer_steps,
        "evaluation": evaluation,
        "trajectory_manifests": trajectory_manifests,
        "fast_aerial_final_tick": 28,
        "protected_competitive_base_unchanged": True,
        "production_reward_unchanged": True,
        "ppo_resumable_as_general_policy": False,
        "promotion_authorized": False,
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, path)
    return {
        "path": str(path),
        "sha256": human_base.sha256_file(path),
        "model_tensor_sha256": tensor_tree_sha256(payload["model"]),
        "side": side,
        "block": block,
        "optimizer_steps": optimizer_steps,
    }


def run(args: argparse.Namespace) -> int:
    authority = load_authority()
    pack_authority = pack_runner.load_authority()
    sources = [
        torch.load(BLUE, map_location="cpu", weights_only=False),
        torch.load(ORANGE, map_location="cpu", weights_only=False),
    ]
    models = [_make_model(source, args.device) for source in sources]
    parents = [_make_model(source, args.device).eval().requires_grad_(False) for source in sources]
    optimizers = [_make_optimizer(model, authority) for model in models]
    critic_hashes = [tensor_tree_sha256(model.critic.state_dict()) for model in models]
    geometry = ArenaGeometry.load_soccar(args.collision_dir)
    meshes = WarpArenaMeshes(geometry, args.device)

    run_dir = Path(args.run_dir)
    if run_dir.exists() and any(run_dir.iterdir()) and not args.reuse_trajectories:
        raise RuntimeError("distillation V1 requires a fresh run directory")
    run_dir.mkdir(parents=True, exist_ok=True)
    train_path = run_dir / "train_trajectories.pt"
    validation_path = run_dir / "validation_trajectories.pt"
    if args.reuse_trajectories:
        train_artifact = torch.load(train_path, map_location="cpu", weights_only=False)
        validation_artifact = torch.load(validation_path, map_location="cpu", weights_only=False)
        train_dataset = DistillationDataset(train_artifact)
        validation_dataset = DistillationDataset(validation_artifact)
        trajectory_manifests = json.loads(
            (run_dir / "trajectory_manifests.json").read_text(encoding="utf-8")
        )
    else:
        train_dataset, train_manifest = build_trajectory_artifact(
            geometry,
            meshes,
            split="train",
            seed_base=int(authority["scenario_authority"]["training_seed_base"]),
            worlds=args.training_worlds_per_pack_side,
            device=args.device,
            collision_dir=args.collision_dir,
            authority=authority,
            pack_authority=pack_authority,
            path=train_path,
        )
        validation_dataset, validation_manifest = build_trajectory_artifact(
            geometry,
            meshes,
            split="validation",
            seed_base=int(authority["scenario_authority"]["validation_seed_base"]),
            worlds=args.validation_worlds_per_pack_side,
            device=args.device,
            collision_dir=args.collision_dir,
            authority=authority,
            pack_authority=pack_authority,
            path=validation_path,
        )
        trajectory_manifests = {"train": train_manifest, "validation": validation_manifest}
        pack_runner.v1.write_json(run_dir / "trajectory_manifests.json", trajectory_manifests)

    human_base.SOURCE = ROOT / authority["protected_competitive_base"]["blue"]["path"]
    human_base.SOURCE_SHA256 = authority["protected_competitive_base"]["blue"]["sha256"]
    human_train, human_validation, _unused, human_identity = human_base.load_human_data(
        device=args.device
    )
    if human_identity["test_loaded"]:
        raise RuntimeError("aerial distillation must not open human test")
    labels = np.asarray(human_train.mechanic_label)
    mask = torch.from_numpy(np.isin(labels, TARGET_LABELS))
    human_observation = human_train.mechanic_observation[mask]
    human_action = human_train.mechanic_action[mask]
    human_labels = labels[mask.numpy()].tolist()
    human_attempts = np.asarray(human_train.mechanic_attempt)[mask.numpy()].tolist()
    human_samplers = [
        MechanicHierarchySampler(
            human_labels,
            human_attempts,
            uniform_label_fraction=0.5,
            maximum_oversampling_ratio=8.0,
            generator=torch.Generator(device="cpu").manual_seed(2_026_090_317 ^ (0xB500 + side)),
        )
        for side in (0, 1)
    ]
    baseline_teacher_action = [
        evaluate_teacher_actions(model, validation_dataset, device=args.device) for model in models
    ]
    baseline_human = [
        option_v1.aerial_validation(model, human_validation, device=args.device) for model in models
    ]
    validation_seed = int(authority["scenario_authority"]["validation_seed_base"])
    baseline_physical = evaluate_physical(
        models,
        geometry,
        meshes,
        worlds=args.evaluation_worlds_per_pack_side,
        device=args.device,
        collision_dir=args.collision_dir,
        pack_authority=pack_authority,
        validation_seed_base=validation_seed,
    )
    preflight = {
        "format": f"{DISTILLATION_VERSION}_PREFLIGHT",
        "authority_sha256": AUTHORITY_SHA256,
        "trajectory_manifests": trajectory_manifests,
        "human_identity": human_identity,
        "human_test_loaded": False,
        "baseline_teacher_action": baseline_teacher_action,
        "baseline_human": baseline_human,
        "baseline_physical": baseline_physical,
        "critic_hashes": critic_hashes,
        "optimizer_steps": 0,
        "verdict": "PASS",
    }
    pack_runner.v1.write_json(RESULTS / "preflight.json", preflight)
    if args.preflight_only:
        print(json.dumps(preflight, indent=2, sort_keys=True), flush=True)
        return 0

    optimization = authority["optimization"]
    batch_size = int(optimization["batch_size"])
    steps_per_block = int(optimization["supervised_steps_per_block"])
    human_per_block = int(optimization["human_aerial_auxiliary_samples_per_block"])
    maximum_blocks = min(int(optimization["maximum_blocks"]), args.maximum_blocks)
    generators = [
        torch.Generator(device="cpu").manual_seed(2_026_090_317 ^ (0xD500 + side))
        for side in (0, 1)
    ]
    curve = RESULTS / "training_curve.jsonl"
    if curve.exists():
        curve.unlink()
    accepted_steps = [0, 0]
    best_score: tuple[float, float] | None = None
    best_records: list[dict[str, Any]] | None = None
    best_evaluation: dict[str, Any] | None = None
    boundaries_without_improvement = 0
    consecutive_passes = 0
    stop_reason = "maximum_blocks"
    for block in range(1, maximum_blocks + 1):
        block_rows = []
        for side in (0, 1):
            losses = []
            gradients = []
            block_index = train_dataset.sample(
                batch_size * steps_per_block,
                generator=generators[side],
                maximum_samples_per_world=int(
                    authority["trajectory_selection"]["maximum_samples_per_world_per_block"]
                ),
            )
            block_human_index = human_samplers[side].sample(human_per_block)
            human_cursor = 0
            for local_step in range(steps_per_block):
                index = block_index[local_step * batch_size : (local_step + 1) * batch_size]
                observation = train_dataset.observation.index_select(0, index).to(args.device)
                action = train_dataset.action.index_select(0, index).to(args.device)
                with torch.no_grad():
                    parent_actor, _ = parents[side](observation)
                student_actor, _ = models[side](observation)
                distillation = human_behavior_cloning_objective(
                    student_actor,
                    parent_actor,
                    action,
                    smooth_l1_beta=float(optimization["smooth_l1_beta"]),
                    analog_weight=float(optimization["analog_weight"]),
                    button_weight=float(optimization["button_weight"]),
                    log_std_weight=float(
                        optimization["parent_log_standard_deviation_retention_weight"]
                    ),
                    policy_config=models[side].config,
                )
                human_count = human_per_block // steps_per_block + int(
                    local_step < human_per_block % steps_per_block
                )
                human_index = block_human_index[human_cursor : human_cursor + human_count]
                human_cursor += human_count
                auxiliary_observation = human_observation.index_select(0, human_index).to(
                    args.device
                )
                auxiliary_action = human_action.index_select(0, human_index).to(args.device)
                with torch.no_grad():
                    auxiliary_parent, _ = parents[side](auxiliary_observation)
                auxiliary_student, _ = models[side](auxiliary_observation)
                auxiliary = human_behavior_cloning_objective(
                    auxiliary_student,
                    auxiliary_parent,
                    auxiliary_action,
                    smooth_l1_beta=float(optimization["smooth_l1_beta"]),
                    analog_weight=float(optimization["analog_weight"]),
                    button_weight=float(optimization["button_weight"]),
                    log_std_weight=float(
                        optimization["parent_log_standard_deviation_retention_weight"]
                    ),
                    policy_config=models[side].config,
                )
                loss = (
                    distillation.loss
                    + float(optimization["human_aerial_auxiliary_weight"]) * auxiliary.loss
                )
                optimizers[side].zero_grad(set_to_none=True)
                loss.backward()
                gradient = torch.nn.utils.clip_grad_norm_(
                    [*models[side].trunk.parameters(), *models[side].actor.parameters()],
                    float(optimization["maximum_gradient_norm"]),
                )
                if not bool(torch.isfinite(loss) and torch.isfinite(gradient)):
                    raise RuntimeError("nonfinite aerial intercept distillation update")
                optimizers[side].step()
                accepted_steps[side] += 1
                losses.append(float(loss.detach()))
                gradients.append(float(gradient.detach()))
            observed_critic = tensor_tree_sha256(models[side].critic.state_dict())
            if observed_critic != critic_hashes[side]:
                raise RuntimeError("frozen aerial-option critic changed")
            block_rows.append(
                {
                    "side": side,
                    "accepted_steps": accepted_steps[side],
                    "loss_mean": float(np.mean(losses)),
                    "gradient_norm_mean": float(np.mean(gradients)),
                }
            )
        row: dict[str, Any] = {"block": block, "sides": block_rows}
        interval = int(optimization["validation_interval_blocks"])
        if block % interval == 0 or block == maximum_blocks:
            teacher_validation = [
                evaluate_teacher_actions(model, validation_dataset, device=args.device)
                for model in models
            ]
            human_validation_rows = [
                option_v1.aerial_validation(model, human_validation, device=args.device)
                for model in models
            ]
            physical = evaluate_physical(
                models,
                geometry,
                meshes,
                worlds=args.evaluation_worlds_per_pack_side,
                device=args.device,
                collision_dir=args.collision_dir,
                pack_authority=pack_authority,
                validation_seed_base=validation_seed,
            )
            human_rmse = [value["mean_complete_action_rmse"] for value in human_validation_rows]
            passed = physical_gate(physical, human_rmse, authority)
            evaluation = {
                "block": block,
                "passed": passed,
                "teacher_action_validation": teacher_validation,
                "human_validation": human_validation_rows,
                "physical": physical,
            }
            score = (
                min(
                    min(
                        item["fractions"]["high_touch"]
                        for item in physical
                        if item["pack"] == "center_pop"
                    )
                    / authority["selection"]["center_pop"]["high_touch_fraction_min"],
                    min(
                        item["fractions"]["goal"]
                        for item in physical
                        if item["pack"] == "center_pop"
                    )
                    / authority["selection"]["center_pop"]["qualified_goal_fraction_min"],
                    min(
                        item["fractions"]["high_touch"]
                        for item in physical
                        if item["pack"] == "lateral_pop"
                    )
                    / authority["selection"]["lateral_pop"]["high_touch_fraction_min"],
                    min(
                        item["fractions"]["goalward_first_touch"]
                        for item in physical
                        if item["pack"] == "lateral_pop"
                    )
                    / authority["selection"]["lateral_pop"]["goalward_first_touch_fraction_min"],
                    min(
                        item["fractions"]["high_touch"]
                        for item in physical
                        if item["pack"] == "airborne_possession"
                    )
                    / authority["selection"]["airborne_possession"]["high_touch_fraction_min"],
                    min(
                        item["fractions"]["goalward_first_touch"]
                        for item in physical
                        if item["pack"] == "airborne_possession"
                    )
                    / authority["selection"]["airborne_possession"][
                        "goalward_first_touch_fraction_min"
                    ],
                ),
                -float(np.mean([item["complete_action_rmse"] for item in teacher_validation])),
            )
            records = [
                save_checkpoint(
                    sources[side],
                    models[side],
                    optimizers[side],
                    run_dir
                    / "candidates"
                    / f"block_{block:04d}_{'blue' if side == 0 else 'orange'}.pt",
                    side=side,
                    block=block,
                    optimizer_steps=accepted_steps[side],
                    evaluation=evaluation,
                    trajectory_manifests=trajectory_manifests,
                )
                for side in (0, 1)
            ]
            evaluation["checkpoints"] = records
            row["evaluation"] = evaluation
            if best_score is None or score > best_score:
                best_score = score
                best_records = records
                best_evaluation = evaluation
                boundaries_without_improvement = 0
            else:
                boundaries_without_improvement += 1
            consecutive_passes = consecutive_passes + 1 if passed else 0
            print(
                json.dumps(
                    {
                        "stage": DISTILLATION_VERSION,
                        "block": block,
                        "passed": passed,
                        "teacher_rmse": [
                            item["complete_action_rmse"] for item in teacher_validation
                        ],
                        "human_rmse": human_rmse,
                        "physical": [
                            {
                                "pack": item["pack"],
                                "side": item["side"],
                                "fractions": item["fractions"],
                            }
                            for item in physical
                        ],
                    },
                    sort_keys=True,
                ),
                flush=True,
            )
            if consecutive_passes >= int(authority["selection"]["consecutive_boundaries"]):
                stop_reason = "two_consecutive_physical_gate_passes"
                pack_runner.v1.append_jsonl(curve, row)
                break
            if boundaries_without_improvement >= int(optimization["plateau_patience_boundaries"]):
                stop_reason = "validation_plateau"
                pack_runner.v1.append_jsonl(curve, row)
                break
        pack_runner.v1.append_jsonl(curve, row)

    controlled_pass = bool(
        best_evaluation is not None
        and best_evaluation["passed"]
        and consecutive_passes >= int(authority["selection"]["consecutive_boundaries"])
    )
    selected_records: list[dict[str, Any]] = []
    if controlled_pass and best_records is not None:
        output_dir = ROOT / "checkpoints/rival2/aerial_intercept_distillation_v1"
        for side, record in enumerate(best_records):
            source_path = Path(record["path"])
            payload = torch.load(source_path, map_location="cpu", weights_only=False)
            output_path = (
                output_dir
                / f"rival2_aerial_intercept_distillation_v1_{'blue' if side == 0 else 'orange'}.pt"
            )
            output_path.parent.mkdir(parents=True, exist_ok=True)
            torch.save(payload, output_path)
            selected_records.append(
                {
                    "path": output_path.relative_to(ROOT).as_posix(),
                    "sha256": human_base.sha256_file(output_path),
                    "model_tensor_sha256": tensor_tree_sha256(payload["model"]),
                    "side": side,
                }
            )
    result = {
        "format": f"{DISTILLATION_VERSION}_RESULT",
        "authority_sha256": AUTHORITY_SHA256,
        "trajectory_manifests": trajectory_manifests,
        "baseline_teacher_action": baseline_teacher_action,
        "baseline_human": baseline_human,
        "baseline_physical": baseline_physical,
        "accepted_supervised_optimizer_steps": accepted_steps,
        "best_evaluation": best_evaluation,
        "selected_checkpoints": selected_records,
        "controlled_pass": controlled_pass,
        "stop_reason": stop_reason,
        "protected_competitive_base_unchanged": True,
        "production_reward_unchanged": True,
        "human_test_loaded": False,
        "promoted": False,
    }
    pack_runner.v1.write_json(RESULTS / "result.json", result)
    print(json.dumps(result, indent=2, sort_keys=True), flush=True)
    return 0 if controlled_pass else 3


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--collision-dir", type=Path, default=option_v1.DEFAULT_COLLISION_DIR)
    parser.add_argument("--run-dir", type=Path, default=DEFAULT_RUN_DIR)
    parser.add_argument("--training-worlds-per-pack-side", type=int, default=2048)
    parser.add_argument("--validation-worlds-per-pack-side", type=int, default=2048)
    parser.add_argument("--evaluation-worlds-per-pack-side", type=int, default=2048)
    parser.add_argument("--maximum-blocks", type=int, default=200)
    parser.add_argument("--reuse-trajectories", action="store_true")
    parser.add_argument("--preflight-only", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    raise SystemExit(run(parse_args()))
