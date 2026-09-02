"""On-policy frozen-teacher correction for Rival's learned aerial intercept option."""

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

from benchmarks import run_rival2_aerial_intercept_distillation_v1 as distill  # noqa: E402
from benchmarks import run_rival2_aerial_option_v1 as option_v1  # noqa: E402
from benchmarks import run_rival2_aerial_training_pack_v1 as pack_runner  # noqa: E402
from benchmarks import run_rival2_codex_autonomous_v1 as human_base  # noqa: E402
from rivalsim.arena import ArenaGeometry, WarpArenaMeshes  # noqa: E402
from rivalsim.human_demo.behavior_cloning import (  # noqa: E402
    MechanicHierarchySampler,
    human_behavior_cloning_objective,
)
from rivalsim.rival2_120hz_transition import tensor_tree_sha256  # noqa: E402
from rivalsim.rival2_aerial_intercept_dagger import (  # noqa: E402
    DAGGER_VERSION,
    CorrectionDataset,
    evenly_spaced_indices,
)
from rivalsim.rival2_aerial_intercept_distillation import (  # noqa: E402
    DistillationDataset,
    physical_gate,
)
from rivalsim.rival2_aerial_intercept_teacher import (  # noqa: E402
    BOOST_ACCELERATION,
    plan_aerial_intercept,
)
from rivalsim.rival2_aerial_option import FIELD  # noqa: E402
from rivalsim.rival2_aerial_option_v2 import apply_fast_aerial_initiation  # noqa: E402
from rivalsim.rival2_aerial_training_pack import (  # noqa: E402
    PACK_NAMES,
    AerialTrainingPackTracker,
    build_training_pack_scenarios,
)
from rivalsim.rival2_contracts import (  # noqa: E402
    POSITION_SCALE,
    RIVAL2_REWARD_GAMEPLAY_120_V2_VERSION,
)
from rivalsim.rival2_env import Rival2Env  # noqa: E402
from rivalsim.rival2_policy import Rival2ActorCritic, deterministic_hybrid_action  # noqa: E402

AUTHORITY = ROOT / "results/rival2/aerial_intercept_dagger_v1/authority.json"
AUTHORITY_SHA256 = "6CA5397095DFADC5D941167CEBC27C3EC600A8FCB9CB2147C2A5C1292E4C2C5B"
RESULTS = ROOT / "results/rival2/aerial_intercept_dagger_v1"
BLUE = ROOT / "checkpoints/rival2/aerial_intercept_distillation_v1/diagnostic_best_blue.pt"
ORANGE = ROOT / "checkpoints/rival2/aerial_intercept_distillation_v1/diagnostic_best_orange.pt"
DEFAULT_RUN_DIR = Path("G:/dev/RivalSim-runs/aerial-intercept-dagger-v1")
DEFAULT_SOURCE_RUN_DIR = Path("G:/dev/RivalSim-runs/aerial-intercept-distillation-v1-source-exact")
TARGET_LABELS = ("aerialdribble", "groundtoairdribble")


def load_authority() -> dict[str, Any]:
    if human_base.sha256_file(AUTHORITY) != AUTHORITY_SHA256:
        raise RuntimeError("aerial intercept DAgger authority changed")
    authority = json.loads(AUTHORITY.read_text(encoding="utf-8"))
    identities = (
        authority["protected_competitive_base"]["blue"],
        authority["protected_competitive_base"]["orange"],
        authority["diagnostic_parent"]["blue"],
        authority["diagnostic_parent"]["orange"],
        authority["failure_evidence"]["result"],
        authority["failure_evidence"]["analysis"],
        authority["frozen_teacher"]["implementation"],
        authority["source_exact_corpus"]["rebuild_authority"],
    )
    for identity in identities:
        path = ROOT / identity["path"]
        if human_base.sha256_file(path) != identity["sha256"]:
            raise RuntimeError(f"bound DAgger input changed: {path}")
    if authority["integrity"]["optimizer_steps_before_authority_commit"] != 0:
        raise RuntimeError("DAgger authority was not prospective")
    if authority["optimization"]["analog_loss"] != (
        "SmoothL1(tanh(actor_mean), teacher_action), beta=0.1"
    ):
        raise RuntimeError("DAgger SmoothL1 authority changed")
    return authority


def make_optimizer(model: Rival2ActorCritic, authority: dict[str, Any]) -> torch.optim.AdamW:
    row = authority["optimization"]
    return torch.optim.AdamW(
        [
            {"params": model.trunk.parameters(), "lr": float(row["trunk_learning_rate"])},
            {"params": model.actor.parameters(), "lr": float(row["actor_learning_rate"])},
        ],
        weight_decay=float(row["weight_decay"]),
    )


def _source_tensor_hash(payload: dict[str, Any]) -> str:
    return tensor_tree_sha256(
        {
            "observation": payload["observation"],
            "action": payload["action"],
            "tick": payload["tick"],
            "trajectories": payload["trajectories"],
        }
    )


@torch.no_grad()
def collect_correction_category(
    model: Rival2ActorCritic,
    geometry: ArenaGeometry,
    meshes: WarpArenaMeshes,
    *,
    round_index: int,
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
    stored_observation = torch.empty((horizon, worlds, 182), device=device)
    stored_action = torch.empty((horizon, worlds, 8), device=device)
    stored_eligible = torch.zeros((horizon, worlds), dtype=torch.bool, device=device)
    active = torch.ones(worlds, dtype=torch.bool, device=device)
    option_age = torch.zeros(worlds, dtype=torch.int64, device=device)
    observation = env.observation
    collection = authority["on_policy_collection"]
    model.eval()
    for tick in range(horizon):
        active_before = active.clone()
        first_before = (
            tracker.first_high_touch.clone()
            if tracker.initialized
            else torch.zeros(worlds, dtype=torch.bool, device=device)
        )
        actor, _ = model(observation[:, side])
        learned = deterministic_hybrid_action(actor, model.config)
        plan = plan_aerial_intercept(observation[:, side])
        selected, primitive = apply_fast_aerial_initiation(learned, option_age, active_before)
        reachable = 0.5 * BOOST_ACCELERATION * plan.intercept_time.square()
        residual = (plan.predicted_distance - reachable).abs()
        ball_height = observation[:, side, FIELD["ball.position.z"]] * POSITION_SCALE[2]
        eligible = (
            active_before
            & ~primitive
            & ~first_before
            & (ball_height >= float(collection["minimum_current_ball_height_uu"]))
            & (residual <= float(collection["maximum_teacher_reach_residual_uu"]))
        )
        stored_observation[tick].copy_(observation[:, side])
        stored_action[tick].copy_(plan.action)
        stored_eligible[tick].copy_(eligible)
        action = torch.zeros((worlds, 2, 8), device=device)
        action[:, side] = torch.where(active_before[:, None], selected, 0.0)
        transition = env.step(action)
        scoring = env.bridge.views["rival2.scoring_team_latched"].to(torch.int64)
        any_goal = active_before & transition.terminated & (scoring >= 0)
        goal_for = any_goal & (scoring == side)
        _unused_reward, done = tracker.step(
            observation,
            transition.transition_observation,
            tick=tick,
            goal_for_attacker=goal_for,
            any_goal=any_goal,
            active=active_before,
        )
        active &= ~(done | transition.terminated | transition.truncated)
        option_age += active_before.to(torch.int64)
        observation = transition.observation
        if not bool(active.any()):
            break
    if device.startswith("cuda"):
        torch.cuda.synchronize()
    observations: list[torch.Tensor] = []
    actions: list[torch.Tensor] = []
    ticks: list[torch.Tensor] = []
    trajectories: list[dict[str, Any]] = []
    offset = 0
    maximum = int(collection["maximum_samples_per_world_per_round"])
    for world in range(worlds):
        world_ticks = stored_eligible[:, world].nonzero(as_tuple=False).flatten()
        if world_ticks.numel() == 0:
            continue
        world_ticks = evenly_spaced_indices(world_ticks, maximum)
        success = bool(tracker.first_high_touch[world])
        length = int(world_ticks.numel())
        observations.append(stored_observation.index_select(0, world_ticks)[:, world].cpu())
        actions.append(stored_action.index_select(0, world_ticks)[:, world].cpu())
        ticks.append(world_ticks.cpu())
        trajectories.append(
            {
                "pack": pack,
                "side": side,
                "world": world,
                "round": round_index,
                "seed": seed,
                "offset": offset,
                "length": length,
                "success": success,
                "weight": float(
                    collection[
                        "successful_world_sampling_weight"
                        if success
                        else "failed_world_sampling_weight"
                    ]
                ),
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
        "round": round_index,
        "pack": PACK_NAMES[pack],
        "pack_index": pack,
        "side": side,
        "seed": seed,
        "worlds": worlds,
        "correction_trajectories": len(trajectories),
        "successful_correction_trajectories": sum(row["success"] for row in trajectories),
        "failed_correction_trajectories": sum(not row["success"] for row in trajectories),
        "samples": offset,
        "telemetry": telemetry,
        "fractions": {
            "high_touch": telemetry["first_high_touches"] / worlds,
            "goalward_first_touch": telemetry["goalward_first_touches"] / worlds,
            "goal": telemetry["goals"] / worlds,
        },
    }
    del env, stored_observation, stored_action, stored_eligible
    gc.collect()
    if device.startswith("cuda"):
        torch.cuda.empty_cache()
    return payload, metrics


def collect_round(
    models: list[Rival2ActorCritic],
    geometry: ArenaGeometry,
    meshes: WarpArenaMeshes,
    *,
    round_index: int,
    worlds: int,
    device: str,
    collision_dir: Path,
    authority: dict[str, Any],
    pack_authority: dict[str, Any],
    path: Path,
) -> tuple[dict[str, Any], dict[str, Any]]:
    observations: list[torch.Tensor] = []
    actions: list[torch.Tensor] = []
    ticks: list[torch.Tensor] = []
    trajectories: list[dict[str, Any]] = []
    metrics: list[dict[str, Any]] = []
    offset = 0
    base = int(authority["on_policy_collection"]["seed_base"])
    stride = int(authority["on_policy_collection"]["seed_stride_per_round"])
    for pack, pack_row in enumerate(pack_authority["physical_training"]["packs"]):
        for side in (0, 1):
            seed = base + (round_index - 1) * stride
            seed ^= pack * 16 + side
            payload, row = collect_correction_category(
                models[side],
                geometry,
                meshes,
                round_index=round_index,
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
                trajectories.append(trajectory)
            observations.append(payload["observation"])
            actions.append(payload["action"])
            ticks.append(payload["tick"])
            offset += payload["observation"].shape[0]
            metrics.append(row)
            print(json.dumps({"stage": "collect_correction", **row}, sort_keys=True), flush=True)
    artifact = {
        "format": f"{DAGGER_VERSION}_CORRECTIONS",
        "round": round_index,
        "observation": torch.cat(observations),
        "action": torch.cat(actions),
        "tick": torch.cat(ticks),
        "trajectories": trajectories,
        "collection_metrics": metrics,
        "optimizer_steps_before_collection": (round_index - 1)
        * int(authority["optimization"]["supervised_steps_per_round"]),
    }
    CorrectionDataset([artifact])
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(artifact, path)
    manifest = {
        "round": round_index,
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
        "samples": int(artifact["observation"].shape[0]),
        "trajectories": len(trajectories),
        "metrics": metrics,
    }
    return artifact, manifest


def _gate_authority(authority: dict[str, Any]) -> dict[str, Any]:
    return {
        "optimization": {
            "human_validation_rmse_max": authority["optimization"]["human_validation_rmse_max"]
        },
        "selection": authority["validation"]["physical_gate"],
    }


def _score(physical: list[dict[str, Any]], authority: dict[str, Any]) -> float:
    threshold = authority["validation"]["physical_gate"]
    values = []
    for row in physical:
        pack = row["pack"]
        values.append(row["fractions"]["high_touch"] / threshold[pack]["high_touch_fraction_min"])
        if pack == "center_pop":
            values.append(row["fractions"]["goal"] / threshold[pack]["qualified_goal_fraction_min"])
        else:
            values.append(
                row["fractions"]["goalward_first_touch"]
                / threshold[pack]["goalward_first_touch_fraction_min"]
            )
    return min(values)


def save_checkpoint(
    source: dict[str, Any],
    model: Rival2ActorCritic,
    optimizer: torch.optim.AdamW,
    path: Path,
    *,
    side: int,
    round_index: int,
    accepted_steps: int,
    evaluation: dict[str, Any],
    correction_manifests: list[dict[str, Any]],
) -> dict[str, Any]:
    payload = copy.deepcopy(source)
    payload["model"] = {
        name: value.detach().cpu().clone() for name, value in model.state_dict().items()
    }
    payload["optimizer"] = {
        "format": f"{DAGGER_VERSION}_FRESH_ADAMW",
        "option": optimizer.state_dict(),
    }
    payload["curriculum_transition"] = {
        "identity": DAGGER_VERSION,
        "authority": {
            "path": AUTHORITY.relative_to(ROOT).as_posix(),
            "sha256": AUTHORITY_SHA256,
        },
        "deployment_side": side,
        "round": round_index,
        "accepted_supervised_optimizer_steps": accepted_steps,
        "evaluation": evaluation,
        "correction_manifests": correction_manifests,
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
        "round": round_index,
        "accepted_steps": accepted_steps,
    }


def run(args: argparse.Namespace) -> int:
    authority = load_authority()
    pack_authority = pack_runner.load_authority()
    run_dir = Path(args.run_dir)
    if run_dir.exists() and any(run_dir.iterdir()):
        raise RuntimeError("DAgger V1 requires a fresh run directory")
    run_dir.mkdir(parents=True, exist_ok=True)
    source_dir = Path(args.source_exact_run_dir)
    train_payload = torch.load(
        source_dir / "train_trajectories.pt", map_location="cpu", weights_only=False
    )
    validation_payload = torch.load(
        source_dir / "validation_trajectories.pt", map_location="cpu", weights_only=False
    )
    source = authority["source_exact_corpus"]
    if _source_tensor_hash(train_payload) != source["training_tensor_sha256"]:
        raise RuntimeError("source-exact training corpus changed")
    if _source_tensor_hash(validation_payload) != source["validation_tensor_sha256"]:
        raise RuntimeError("source-exact validation corpus changed")
    train_dataset = DistillationDataset(train_payload)
    validation_dataset = DistillationDataset(validation_payload)
    sources = [
        torch.load(BLUE, map_location="cpu", weights_only=False),
        torch.load(ORANGE, map_location="cpu", weights_only=False),
    ]
    models = [distill._make_model(payload, args.device) for payload in sources]
    parents = [
        distill._make_model(payload, args.device).eval().requires_grad_(False)
        for payload in sources
    ]
    optimizers = [make_optimizer(model, authority) for model in models]
    critic_hashes = [tensor_tree_sha256(model.critic.state_dict()) for model in models]
    human_base.SOURCE = ROOT / authority["protected_competitive_base"]["blue"]["path"]
    human_base.SOURCE_SHA256 = authority["protected_competitive_base"]["blue"]["sha256"]
    human_train, human_validation, _unused, human_identity = human_base.load_human_data(
        device=args.device
    )
    if human_identity["test_loaded"]:
        raise RuntimeError("DAgger must not open human test")
    labels = np.asarray(human_train.mechanic_label)
    mask = torch.from_numpy(np.isin(labels, TARGET_LABELS))
    human_observation = human_train.mechanic_observation[mask]
    human_action = human_train.mechanic_action[mask]
    human_labels = labels[mask.numpy()].tolist()
    human_attempts = np.asarray(human_train.mechanic_attempt)[mask.numpy()].tolist()
    generators = [
        torch.Generator(device="cpu").manual_seed(2_026_092_317 ^ (0xDA00 + side))
        for side in (0, 1)
    ]
    human_samplers = [
        MechanicHierarchySampler(
            human_labels,
            human_attempts,
            uniform_label_fraction=0.5,
            maximum_oversampling_ratio=8.0,
            generator=torch.Generator(device="cpu").manual_seed(2_026_092_317 ^ (0xDB00 + side)),
        )
        for side in (0, 1)
    ]
    geometry = ArenaGeometry.load_soccar(args.collision_dir)
    meshes = WarpArenaMeshes(geometry, args.device)
    validation_seed = int(authority["validation"]["physical_seed_base"])
    baseline_teacher = [
        distill.evaluate_teacher_actions(model, validation_dataset, device=args.device)
        for model in models
    ]
    baseline_human = [
        option_v1.aerial_validation(model, human_validation, device=args.device) for model in models
    ]
    baseline_physical = distill.evaluate_physical(
        models,
        geometry,
        meshes,
        worlds=int(authority["validation"]["physical_worlds_per_pack_side"]),
        device=args.device,
        collision_dir=args.collision_dir,
        pack_authority=pack_authority,
        validation_seed_base=validation_seed,
    )
    preflight = {
        "format": f"{DAGGER_VERSION}_PREFLIGHT",
        "authority_sha256": AUTHORITY_SHA256,
        "source_training_tensor_sha256": _source_tensor_hash(train_payload),
        "source_validation_tensor_sha256": _source_tensor_hash(validation_payload),
        "baseline_teacher_action": baseline_teacher,
        "baseline_human": baseline_human,
        "baseline_physical": baseline_physical,
        "critic_hashes": critic_hashes,
        "human_test_loaded": False,
        "optimizer_steps": 0,
        "verdict": "PASS",
    }
    pack_runner.v1.write_json(RESULTS / "preflight.json", preflight)
    if args.preflight_only:
        print(json.dumps(preflight, indent=2, sort_keys=True), flush=True)
        return 0

    optimization = authority["optimization"]
    batch_size = int(optimization["batch_size"])
    source_count = round(batch_size * float(optimization["source_exact_batch_fraction"]))
    correction_count = batch_size - source_count
    steps_per_round = int(optimization["supervised_steps_per_round"])
    maximum_rounds = min(int(optimization["maximum_rounds"]), args.maximum_rounds)
    human_per_round = int(optimization["human_aerial_auxiliary_samples_per_round"])
    curve = RESULTS / "training_curve.jsonl"
    if curve.exists():
        curve.unlink()
    correction_payloads: list[dict[str, Any]] = []
    correction_manifests: list[dict[str, Any]] = []
    accepted_steps = [0, 0]
    best_score = _score(baseline_physical, authority)
    best_evaluation: dict[str, Any] | None = None
    best_records: list[dict[str, Any]] | None = None
    no_improvement = 0
    consecutive_passes = 0
    stop_reason = "maximum_rounds"
    for round_index in range(1, maximum_rounds + 1):
        artifact, manifest = collect_round(
            models,
            geometry,
            meshes,
            round_index=round_index,
            worlds=int(authority["on_policy_collection"]["worlds_per_pack_side_per_round"]),
            device=args.device,
            collision_dir=args.collision_dir,
            authority=authority,
            pack_authority=pack_authority,
            path=run_dir / "corrections" / f"round_{round_index:02d}.pt",
        )
        correction_payloads.append(artifact)
        correction_manifests.append(manifest)
        replay = int(authority["on_policy_collection"]["replay_rounds"])
        correction_dataset = CorrectionDataset(correction_payloads[-replay:])
        side_rows = []
        for side in (0, 1):
            losses = []
            gradients = []
            human_index_round = human_samplers[side].sample(human_per_round)
            human_cursor = 0
            for local_step in range(steps_per_round):
                source_index = train_dataset.sample(
                    source_count,
                    generator=generators[side],
                    maximum_samples_per_world=96,
                )
                correction_index = correction_dataset.sample(
                    correction_count, generator=generators[side]
                )
                observation = torch.cat(
                    (
                        train_dataset.observation.index_select(0, source_index),
                        correction_dataset.observation.index_select(0, correction_index),
                    )
                )
                action = torch.cat(
                    (
                        train_dataset.action.index_select(0, source_index),
                        correction_dataset.action.index_select(0, correction_index),
                    )
                )
                permutation = torch.randperm(batch_size, generator=generators[side])
                observation = observation.index_select(0, permutation).to(args.device)
                action = action.index_select(0, permutation).to(args.device)
                with torch.no_grad():
                    parent_actor, _ = parents[side](observation)
                student_actor, _ = models[side](observation)
                objective = human_behavior_cloning_objective(
                    student_actor,
                    parent_actor,
                    action,
                    smooth_l1_beta=0.1,
                    analog_weight=float(optimization["analog_weight"]),
                    button_weight=float(optimization["button_weight"]),
                    log_std_weight=float(
                        optimization["diagnostic_parent_log_standard_deviation_retention_weight"]
                    ),
                    policy_config=models[side].config,
                )
                human_count = human_per_round // steps_per_round + int(
                    local_step < human_per_round % steps_per_round
                )
                human_index = human_index_round[human_cursor : human_cursor + human_count]
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
                    smooth_l1_beta=0.1,
                    analog_weight=float(optimization["analog_weight"]),
                    button_weight=float(optimization["button_weight"]),
                    log_std_weight=float(
                        optimization["diagnostic_parent_log_standard_deviation_retention_weight"]
                    ),
                    policy_config=models[side].config,
                )
                loss = (
                    objective.loss
                    + float(optimization["human_aerial_auxiliary_weight"]) * auxiliary.loss
                )
                optimizers[side].zero_grad(set_to_none=True)
                loss.backward()
                gradient = torch.nn.utils.clip_grad_norm_(
                    [*models[side].trunk.parameters(), *models[side].actor.parameters()],
                    float(optimization["maximum_gradient_norm"]),
                )
                if not bool(torch.isfinite(loss) and torch.isfinite(gradient)):
                    raise RuntimeError("nonfinite DAgger optimizer step")
                optimizers[side].step()
                accepted_steps[side] += 1
                losses.append(float(loss.detach()))
                gradients.append(float(gradient.detach()))
            if tensor_tree_sha256(models[side].critic.state_dict()) != critic_hashes[side]:
                raise RuntimeError("frozen DAgger critic changed")
            side_rows.append(
                {
                    "side": side,
                    "accepted_steps": accepted_steps[side],
                    "loss_mean": float(np.mean(losses)),
                    "gradient_norm_mean": float(np.mean(gradients)),
                }
            )
        teacher_validation = [
            distill.evaluate_teacher_actions(model, validation_dataset, device=args.device)
            for model in models
        ]
        human_validation_rows = [
            option_v1.aerial_validation(model, human_validation, device=args.device)
            for model in models
        ]
        physical = distill.evaluate_physical(
            models,
            geometry,
            meshes,
            worlds=int(authority["validation"]["physical_worlds_per_pack_side"]),
            device=args.device,
            collision_dir=args.collision_dir,
            pack_authority=pack_authority,
            validation_seed_base=validation_seed,
        )
        human_rmse = [row["mean_complete_action_rmse"] for row in human_validation_rows]
        passed = physical_gate(physical, human_rmse, _gate_authority(authority))
        evaluation = {
            "round": round_index,
            "passed": passed,
            "teacher_action_validation": teacher_validation,
            "human_validation": human_validation_rows,
            "physical": physical,
            "correction_manifest": manifest,
        }
        records = [
            save_checkpoint(
                sources[side],
                models[side],
                optimizers[side],
                run_dir
                / "candidates"
                / f"round_{round_index:02d}_{'blue' if side == 0 else 'orange'}.pt",
                side=side,
                round_index=round_index,
                accepted_steps=accepted_steps[side],
                evaluation=evaluation,
                correction_manifests=correction_manifests,
            )
            for side in (0, 1)
        ]
        evaluation["checkpoints"] = records
        score = _score(physical, authority)
        if score > best_score + 1.0e-9:
            best_score = score
            best_evaluation = evaluation
            best_records = records
            no_improvement = 0
        else:
            no_improvement += 1
        consecutive_passes = consecutive_passes + 1 if passed else 0
        row = {
            "round": round_index,
            "sides": side_rows,
            "evaluation": evaluation,
            "physical_gate_score": score,
        }
        pack_runner.v1.append_jsonl(curve, row)
        print(
            json.dumps(
                {
                    "stage": DAGGER_VERSION,
                    "round": round_index,
                    "passed": passed,
                    "score": score,
                    "teacher_rmse": [value["complete_action_rmse"] for value in teacher_validation],
                    "human_rmse": human_rmse,
                    "physical": [
                        {
                            "pack": value["pack"],
                            "side": value["side"],
                            "fractions": value["fractions"],
                        }
                        for value in physical
                    ],
                },
                sort_keys=True,
            ),
            flush=True,
        )
        if consecutive_passes >= int(
            authority["validation"]["physical_gate"]["consecutive_rounds"]
        ):
            stop_reason = "two_consecutive_physical_gate_passes"
            break
        if no_improvement >= int(authority["validation"]["plateau_patience_rounds"]):
            stop_reason = "validation_plateau"
            break

    controlled_pass = bool(
        best_evaluation is not None
        and best_evaluation["passed"]
        and consecutive_passes
        >= int(authority["validation"]["physical_gate"]["consecutive_rounds"])
    )
    selected: list[dict[str, Any]] = []
    if controlled_pass and best_records is not None:
        output_dir = ROOT / "checkpoints/rival2/aerial_intercept_dagger_v1"
        for side, record in enumerate(best_records):
            payload = torch.load(record["path"], map_location="cpu", weights_only=False)
            output_path = output_dir / (
                f"rival2_aerial_intercept_dagger_v1_{'blue' if side == 0 else 'orange'}.pt"
            )
            output_path.parent.mkdir(parents=True, exist_ok=True)
            torch.save(payload, output_path)
            selected.append(
                {
                    "path": output_path.relative_to(ROOT).as_posix(),
                    "sha256": human_base.sha256_file(output_path),
                    "model_tensor_sha256": tensor_tree_sha256(payload["model"]),
                    "side": side,
                }
            )
    result = {
        "format": f"{DAGGER_VERSION}_RESULT",
        "authority_sha256": AUTHORITY_SHA256,
        "baseline_teacher_action": baseline_teacher,
        "baseline_human": baseline_human,
        "baseline_physical": baseline_physical,
        "accepted_supervised_optimizer_steps": accepted_steps,
        "correction_manifests": correction_manifests,
        "best_evaluation": best_evaluation,
        "selected_checkpoints": selected,
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
    parser.add_argument("--source-exact-run-dir", type=Path, default=DEFAULT_SOURCE_RUN_DIR)
    parser.add_argument("--maximum-rounds", type=int, default=12)
    parser.add_argument("--preflight-only", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    raise SystemExit(run(parse_args()))
