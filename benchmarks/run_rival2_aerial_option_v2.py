"""Train the fast-aerial-initiated post-launch controller for Aerial Option V2."""

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

from benchmarks import run_rival2_aerial_option_v1 as v1  # noqa: E402
from benchmarks import run_rival2_codex_autonomous_v1 as human_base  # noqa: E402
from rivalsim.arena import ArenaGeometry, WarpArenaMeshes  # noqa: E402
from rivalsim.human_demo.behavior_cloning import MechanicHierarchySampler  # noqa: E402
from rivalsim.rival2_aerial_option import build_aerial_scenarios  # noqa: E402
from rivalsim.rival2_aerial_option_v2 import (  # noqa: E402
    PHASE_GOAL_DIRECTED,
    PHASE_MOVING_INTERCEPT,
    PHASE_NAMES,
    AerialRewardTrackerV2,
    apply_fast_aerial_initiation,
)
from rivalsim.rival2_contracts import RIVAL2_REWARD_GAMEPLAY_120_V2_VERSION  # noqa: E402
from rivalsim.rival2_env import Rival2Env  # noqa: E402
from rivalsim.rival2_policy import (  # noqa: E402
    HybridDistributionOverride,
    Rival2ActorCritic,
    deterministic_hybrid_action,
    hybrid_log_probability,
    sample_hybrid_action,
)

AUTHORITY = ROOT / "results/rival2/aerial_option_v2/authority.json"
AUTHORITY_SHA256 = "3AE030402525049317F9B221BEC288401A936B178CE37EBAA7C93090CA630860"
RESULTS = ROOT / "results/rival2/aerial_option_v2"
BLUE = ROOT / "checkpoints/rival2/aerial_option_v1/rival2_aerial_option_v1_final_blue.pt"
ORANGE = ROOT / "checkpoints/rival2/aerial_option_v1/rival2_aerial_option_v1_final_orange.pt"
BLUE_SHA256 = "75C4572C5601D6753D00A5EC80112FB68AFAAA2446AA21DA9FF1C0C9BE8D5F51"
ORANGE_SHA256 = "E2741EDA8513ED250212C3A9AD7FFA1D0A39A46CF06584954889E511D8378DA1"
DEFAULT_RUN_DIR = Path("G:/dev/RivalSim-runs/aerial-option-v2")
SEED = 2_026_090_302
TARGET_LABELS = ("aerialdribble", "groundtoairdribble")


def load_authority() -> dict[str, Any]:
    if v1.v1.sha256_file(AUTHORITY) != AUTHORITY_SHA256:
        raise RuntimeError("aerial-option V2 authority changed")
    authority = json.loads(AUTHORITY.read_text(encoding="utf-8"))
    for identity in (
        authority["protected_competitive_base"]["blue"],
        authority["protected_competitive_base"]["orange"],
        authority["initial_option"]["blue"],
        authority["initial_option"]["orange"],
        authority["initial_option"]["v1_result"],
        authority["initial_option"]["v1_failure_attribution"],
        authority["human_auxiliary"]["dataset_manifest"],
        authority["human_auxiliary"]["observation_adapter"],
    ):
        path = ROOT / identity["path"]
        if v1.v1.sha256_file(path) != identity["sha256"]:
            raise RuntimeError(f"aerial-option V2 bound input changed: {path}")
    return authority


def make_optimizer(
    model: Rival2ActorCritic, authority: dict[str, Any]
) -> torch.optim.AdamW:
    boundary = authority["training_boundary"]
    return torch.optim.AdamW(
        [
            {
                "params": model.trunk.parameters(),
                "lr": float(boundary["trunk_learning_rate"]),
            },
            {
                "params": model.actor.parameters(),
                "lr": float(boundary["actor_learning_rate"]),
            },
        ],
        weight_decay=float(boundary["weight_decay"]),
    )


def compatibility_authority(authority: dict[str, Any]) -> dict[str, Any]:
    physical = authority["physical_training"]
    return {
        "option_training_boundary": {
            "maximum_gradient_norm": authority["training_boundary"][
                "maximum_gradient_norm"
            ]
        },
        "physical_curriculum": {
            "discount_gamma": physical["discount_gamma"],
            "ppo_clip": physical["ppo_clip"],
            "epochs": physical["epochs"],
            "minibatch_size": physical["minibatch_size"],
            "human_auxiliary_samples_per_block": authority["human_auxiliary"][
                "samples_per_block"
            ],
            "human_auxiliary_weight": authority["human_auxiliary"]["weight"],
        },
        "human_launch_rehearsal": {"log_std_parent_retention_weight": 0.05},
    }


def distribution_override(
    authority: dict[str, Any], phase: int
) -> HybridDistributionOverride:
    row = authority["physical_training"]["exploration"][PHASE_NAMES[phase]]
    return HybridDistributionOverride(
        analog_log_std=float(np.log(row["analog_sigma"])),
        button_temperature=float(row["button_temperature"]),
    )


def collect_rollout(
    model: Rival2ActorCritic,
    geometry: ArenaGeometry,
    meshes: WarpArenaMeshes,
    *,
    side: int,
    phase: int,
    worlds: int,
    horizon: int,
    seed: int,
    device: str,
    generator: torch.Generator,
    distribution: HybridDistributionOverride,
    deterministic: bool,
    collision_dir: Path,
) -> tuple[v1.OptionRollout | None, dict[str, Any]]:
    scenario_phase = phase + 1
    batch = build_aerial_scenarios(
        worlds, seed=seed, attacker_side=side, phase=scenario_phase
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
    tracker = AerialRewardTrackerV2(worlds, attacker_side=side, phase=phase)
    rollout = None if deterministic else v1.OptionRollout(horizon, worlds, device)
    active = torch.ones(worlds, dtype=torch.bool, device=device)
    option_age = torch.zeros(worlds, dtype=torch.int64, device=device)
    opponent = 1 - side
    observation = env.observation
    saturation = torch.zeros(5, dtype=torch.float64, device=device)
    action_count = torch.zeros((), dtype=torch.float64, device=device)
    model.eval()
    for tick in range(horizon):
        active_before = active.clone()
        with torch.no_grad():
            actor, _ = model(observation[:, side])
            if deterministic:
                learned = deterministic_hybrid_action(actor, model.config)
                pre_tanh = actor[:, :5]
            else:
                sample = sample_hybrid_action(
                    actor,
                    generator=generator,
                    config=model.config,
                    distribution_override=distribution,
                )
                learned = sample.action
                pre_tanh = sample.pre_tanh
            old_log_probability = hybrid_log_probability(
                actor,
                learned,
                config=model.config,
                pre_tanh=pre_tanh,
                distribution_override=distribution,
            )
        selected, primitive_mask = apply_fast_aerial_initiation(
            learned, option_age, active_before
        )
        action = torch.zeros((worlds, 2, 8), device=device)
        action[:, side] = torch.where(active_before[:, None], selected, 0.0)
        action[:, opponent] = 0.0
        transition = env.step(action)
        scoring_team = env.bridge.views["rival2.scoring_team_latched"].to(torch.int64)
        goal_for = active_before & transition.terminated & (scoring_team == side)
        reward, skill_done = tracker.step(
            observation,
            transition.transition_observation,
            tick=tick,
            goal_for_attacker=goal_for,
            active=active_before,
        )
        terminal = skill_done | transition.terminated | transition.truncated
        if rollout is not None:
            rollout.observation[tick].copy_(observation[:, side])
            rollout.action[tick].copy_(transition.emitted_action[:, side])
            rollout.pre_tanh[tick].copy_(pre_tanh)
            rollout.old_log_probability[tick].copy_(old_log_probability)
            rollout.reward[tick].copy_(reward)
            rollout.done[tick].copy_(terminal)
            rollout.mask[tick].copy_(active_before & ~primitive_mask)
        saturation += (
            (transition.emitted_action[:, side, :5].abs() > 0.95)
            & active_before[:, None]
        ).sum(dim=0, dtype=torch.float64)
        action_count += active_before.sum(dtype=torch.float64)
        active &= ~terminal
        option_age += active_before.to(torch.int64)
        observation = transition.observation
        if not bool(active.any()):
            break
    if rollout is not None and bool(active.any()):
        rollout.done[min(horizon - 1, tick)] |= active
    torch.cuda.synchronize()
    telemetry = asdict(tracker.telemetry)
    metrics = {
        "side": side,
        "phase": PHASE_NAMES[phase],
        "horizon": horizon,
        "active_worlds_at_end": int(active.sum()),
        "telemetry": telemetry,
        "fractions": {
            "launch": telemetry["launches"] / worlds,
            "reached_150uu": telemetry["reached_150uu"] / worlds,
            "reached_250uu": telemetry["reached_250uu"] / worlds,
            "reached_350uu": telemetry["reached_350uu"] / worlds,
            "elevated_contact": telemetry["elevated_contacts"] / worlds,
            "high_contact": telemetry["high_contacts"] / worlds,
            "forward_high_contact": telemetry["forward_high_contacts"] / worlds,
            "aerial_origin_goal": telemetry["aerial_origin_goals"] / worlds,
        },
        "analog_saturation_fraction": (
            saturation / action_count.clamp_min(1.0)
        ).cpu().tolist(),
        "finite_observation": bool(torch.isfinite(observation).all()),
    }
    del env
    gc.collect()
    torch.cuda.empty_cache()
    return rollout, metrics


def phase_gate(
    physical: list[dict[str, Any]],
    human: list[dict[str, Any]],
    phase: int,
    authority: dict[str, Any],
) -> bool:
    phase_authority = authority["physical_training"]["phases"][phase]
    if any(
        row["mean_complete_action_rmse"]
        > authority["human_auxiliary"]["validation_rmse_max"]
        for row in human
    ):
        return False
    for row in physical:
        fractions = row["fractions"]
        threshold = (
            phase_authority["advance"]
            if phase == PHASE_MOVING_INTERCEPT
            else phase_authority["select"]
        )
        if phase == PHASE_MOVING_INTERCEPT and fractions["launch"] < threshold[
            "launch_fraction_min"
        ]:
            return False
        if fractions["high_contact"] < threshold["high_contact_fraction_min"]:
            return False
        if fractions["forward_high_contact"] < threshold[
            "forward_high_contact_fraction_min"
        ]:
            return False
        if phase == PHASE_GOAL_DIRECTED and fractions["aerial_origin_goal"] < threshold[
            "aerial_origin_goal_fraction_min"
        ]:
            return False
        if not row["finite_observation"] or max(row["analog_saturation_fraction"]) >= 0.98:
            return False
    return True


def save_checkpoint(
    source: dict[str, Any],
    model: Rival2ActorCritic,
    optimizer: torch.optim.AdamW,
    path: Path,
    *,
    side: int,
    phase: int,
    block: int,
    evaluation: dict[str, Any],
) -> dict[str, Any]:
    payload = copy.deepcopy(source)
    payload["model"] = {
        name: value.detach().cpu().clone() for name, value in model.state_dict().items()
    }
    payload["optimizer"] = {
        "format": "RIVAL2_AERIAL_OPTION_V2_FRESH_ADAMW",
        "option": optimizer.state_dict(),
    }
    payload["curriculum_transition"] = {
        "identity": "RIVAL2_AERIAL_OPTION_V2",
        "created_utc": v1.utc_now(),
        "authority": {
            "path": AUTHORITY.relative_to(ROOT).as_posix(),
            "sha256": AUTHORITY_SHA256,
        },
        "deployment_side": side,
        "phase": PHASE_NAMES[phase],
        "accepted_phase_block": block,
        "evaluation": evaluation,
        "fast_aerial_final_tick": 28,
        "protected_competitive_base_unchanged": True,
        "production_reward_unchanged": True,
        "ppo_resumable_as_general_policy": False,
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, path)
    return {
        "path": str(path),
        "sha256": v1.v1.sha256_file(path),
        "model_tensor_sha256": human_base.tensor_tree_sha256(payload["model"]),
        "bytes": path.stat().st_size,
        "side": side,
        "phase": PHASE_NAMES[phase],
        "block": block,
    }


def run(args: argparse.Namespace) -> int:
    authority = load_authority()
    sources = [
        torch.load(BLUE, map_location="cpu", weights_only=False),
        torch.load(ORANGE, map_location="cpu", weights_only=False),
    ]
    for side, color in ((0, "blue"), (1, "orange")):
        expected = authority["initial_option"][color]
        if human_base.tensor_tree_sha256(sources[side]["model"]) != expected[
            "model_tensor_sha256"
        ]:
            raise RuntimeError("aerial-option V2 initial model changed")
    models = [v1.make_model(source, args.device) for source in sources]
    teachers = [
        v1.make_model(source, args.device).eval().requires_grad_(False)
        for source in sources
    ]
    optimizers = [make_optimizer(model, authority) for model in models]
    human_base.SOURCE = ROOT / authority["protected_competitive_base"]["blue"]["path"]
    human_base.SOURCE_SHA256 = authority["protected_competitive_base"]["blue"]["sha256"]
    train, validation, _unused, human_identity = human_base.load_human_data(
        device=args.device
    )
    if human_identity["test_loaded"]:
        raise RuntimeError("aerial-option V2 must not open human test")
    baseline_human = [
        v1.aerial_validation(model, validation, device=args.device) for model in models
    ]
    labels = np.asarray(train.mechanic_label)
    target_mask = torch.from_numpy(np.isin(labels, TARGET_LABELS))
    observations = train.mechanic_observation[target_mask]
    actions = train.mechanic_action[target_mask]
    target_labels = labels[target_mask.numpy()].tolist()
    attempts = np.asarray(train.mechanic_attempt)[target_mask.numpy()].tolist()
    samplers = [
        MechanicHierarchySampler(
            target_labels,
            attempts,
            uniform_label_fraction=0.5,
            maximum_oversampling_ratio=8.0,
            generator=torch.Generator(device="cpu").manual_seed(SEED ^ (0xB200 + side)),
        )
        for side in (0, 1)
    ]
    geometry = ArenaGeometry.load_soccar(args.collision_dir)
    meshes = WarpArenaMeshes(geometry, args.device)
    generators = [
        torch.Generator(device=args.device).manual_seed(SEED ^ (0xC200 + side))
        for side in (0, 1)
    ]
    baseline_physical: dict[str, Any] = {}
    for phase in range(len(PHASE_NAMES)):
        phase_authority = authority["physical_training"]["phases"][phase]
        rows = []
        for side in (0, 1):
            _unused_rollout, metrics = collect_rollout(
                models[side],
                geometry,
                meshes,
                side=side,
                phase=phase,
                worlds=args.evaluation_worlds_per_side,
                horizon=int(phase_authority["horizon_ticks"]),
                seed=SEED ^ (0xE200 + phase * 16 + side),
                device=args.device,
                generator=generators[side],
                distribution=distribution_override(authority, phase),
                deterministic=True,
                collision_dir=args.collision_dir,
            )
            rows.append(metrics)
        baseline_physical[PHASE_NAMES[phase]] = rows
    preflight = {
        "format": "RIVAL2_AERIAL_OPTION_V2_PREFLIGHT",
        "created_utc": v1.utc_now(),
        "authority_sha256": AUTHORITY_SHA256,
        "protected_competitive_base_unchanged": True,
        "initial_option_hashes_verified": True,
        "human_test_loaded": False,
        "baseline_human": baseline_human,
        "baseline_physical": baseline_physical,
        "verdict": "PASS",
    }
    v1.write_json(RESULTS / "preflight.json", preflight)
    if args.preflight_only:
        print(json.dumps(preflight, indent=2, sort_keys=True), flush=True)
        return 0

    run_dir = Path(args.run_dir)
    if run_dir.exists() and any(run_dir.iterdir()):
        raise RuntimeError("aerial-option V2 requires a fresh run directory")
    run_dir.mkdir(parents=True, exist_ok=True)
    curve = RESULTS / "training_curve.jsonl"
    if curve.exists():
        curve.unlink()
    compat = compatibility_authority(authority)
    completed: list[str] = []
    selected: dict[str, Any] | None = None
    stop_reason = "maximum_blocks"
    for phase in range(len(PHASE_NAMES)):
        phase_authority = authority["physical_training"]["phases"][phase]
        maximum = min(int(phase_authority["maximum_blocks"]), args.maximum_blocks_per_phase)
        distribution = distribution_override(authority, phase)
        consecutive = 0
        best_score = (-1.0, -1.0, -1.0)
        best: dict[str, Any] | None = None
        for block in range(1, maximum + 1):
            sides = []
            for side in (0, 1):
                rollout, rollout_metrics = collect_rollout(
                    models[side],
                    geometry,
                    meshes,
                    side=side,
                    phase=phase,
                    worlds=args.worlds_per_side,
                    horizon=int(phase_authority["horizon_ticks"]),
                    seed=SEED + phase * 1_000_000 + block * 100 + side,
                    device=args.device,
                    generator=generators[side],
                    distribution=distribution,
                    deterministic=False,
                    collision_dir=args.collision_dir,
                )
                assert rollout is not None
                ppo = v1.option_ppo_update(
                    models[side],
                    optimizers[side],
                    rollout,
                    generator=generators[side],
                    distribution=distribution,
                    authority=compat,
                )
                del rollout
                auxiliary = v1.human_auxiliary_update(
                    models[side],
                    teachers[side],
                    optimizers[side],
                    observations,
                    actions,
                    samplers[side],
                    authority=compat,
                    device=args.device,
                )
                sides.append(
                    {
                        "side": side,
                        "rollout": rollout_metrics,
                        "ppo": ppo,
                        "human_auxiliary": auxiliary,
                    }
                )
            row: dict[str, Any] = {
                "phase": PHASE_NAMES[phase],
                "phase_index": phase,
                "block": block,
                "sides": sides,
            }
            interval = int(authority["physical_training"]["evaluation_interval_blocks"])
            if block % interval == 0 or block == maximum:
                physical = []
                for side in (0, 1):
                    _unused_rollout, metrics = collect_rollout(
                        models[side],
                        geometry,
                        meshes,
                        side=side,
                        phase=phase,
                        worlds=args.evaluation_worlds_per_side,
                        horizon=int(phase_authority["horizon_ticks"]),
                        seed=SEED ^ (0xE200 + phase * 16 + side),
                        device=args.device,
                        generator=generators[side],
                        distribution=distribution,
                        deterministic=True,
                        collision_dir=args.collision_dir,
                    )
                    physical.append(metrics)
                human = [
                    v1.aerial_validation(model, validation, device=args.device)
                    for model in models
                ]
                passed = phase_gate(physical, human, phase, authority)
                evaluation: dict[str, Any] = {
                    "passed": passed,
                    "physical": physical,
                    "human": human,
                }
                records = []
                for side in (0, 1):
                    records.append(
                        save_checkpoint(
                            sources[side],
                            models[side],
                            optimizers[side],
                            run_dir
                            / f"{PHASE_NAMES[phase]}_b{block:04d}_{'blue' if side == 0 else 'orange'}.pt",
                            side=side,
                            phase=phase,
                            block=block,
                            evaluation=evaluation,
                        )
                    )
                evaluation["checkpoint"] = records
                row["evaluation"] = evaluation
                minimum_goal = min(x["fractions"]["aerial_origin_goal"] for x in physical)
                minimum_high = min(x["fractions"]["high_contact"] for x in physical)
                minimum_forward = min(
                    x["fractions"]["forward_high_contact"] for x in physical
                )
                score = (minimum_goal, minimum_high, minimum_forward)
                if score > best_score:
                    best_score = score
                    best = copy.deepcopy(evaluation)
                consecutive = consecutive + 1 if passed else 0
                print(
                    json.dumps(
                        {
                            "stage": "aerial_option_v2",
                            "phase": PHASE_NAMES[phase],
                            "block": block,
                            "passed": passed,
                            "fractions": [x["fractions"] for x in physical],
                            "human_rmse": [x["mean_complete_action_rmse"] for x in human],
                        },
                        sort_keys=True,
                    ),
                    flush=True,
                )
                required = (
                    int(phase_authority["advance"]["consecutive_boundaries"])
                    if phase == PHASE_MOVING_INTERCEPT
                    else 1
                )
                if consecutive >= required:
                    v1.append_jsonl(curve, row)
                    completed.append(PHASE_NAMES[phase])
                    if phase == PHASE_GOAL_DIRECTED:
                        selected = copy.deepcopy(evaluation)
                        stop_reason = "goal_directed_gate_passed"
                    break
            v1.append_jsonl(curve, row)
            for side in (0, 1):
                save_checkpoint(
                    sources[side],
                    models[side],
                    optimizers[side],
                    run_dir / "rolling" / ("blue.pt" if side == 0 else "orange.pt"),
                    side=side,
                    phase=phase,
                    block=block,
                    evaluation=row.get("evaluation", {}),
                )
        else:
            stop_reason = f"{PHASE_NAMES[phase]}_maximum_blocks_without_gate"
        if phase == PHASE_GOAL_DIRECTED and selected is not None:
            break
        if PHASE_NAMES[phase] not in completed:
            selected = best
            break
    result = {
        "format": "RIVAL2_AERIAL_OPTION_V2_RESULT",
        "created_utc": v1.utc_now(),
        "authority_sha256": AUTHORITY_SHA256,
        "baseline_human": baseline_human,
        "baseline_physical": baseline_physical,
        "completed_phases": completed,
        "stop_reason": stop_reason,
        "selected": selected,
        "controlled_pass": bool(
            selected is not None
            and selected.get("passed")
            and len(completed) == len(PHASE_NAMES)
        ),
        "protected_competitive_base_unchanged": True,
        "human_test_loaded": False,
        "promoted": False,
    }
    v1.write_json(RESULTS / "result.json", result)
    print(json.dumps(result, indent=2, sort_keys=True), flush=True)
    return 0 if result["controlled_pass"] else 2


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--collision-dir", type=Path, default=v1.DEFAULT_COLLISION_DIR)
    parser.add_argument("--run-dir", type=Path, default=DEFAULT_RUN_DIR)
    parser.add_argument("--worlds-per-side", type=int, default=2048)
    parser.add_argument("--evaluation-worlds-per-side", type=int, default=2048)
    parser.add_argument("--maximum-blocks-per-phase", type=int, default=160)
    parser.add_argument("--preflight-only", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    raise SystemExit(run(parse_args()))
