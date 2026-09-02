"""Train Rival's aerial option on discrete resettable shot packs."""

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
from benchmarks import run_rival2_aerial_option_v2 as v2  # noqa: E402
from benchmarks import run_rival2_codex_autonomous_v1 as human_base  # noqa: E402
from rivalsim.arena import ArenaGeometry, WarpArenaMeshes  # noqa: E402
from rivalsim.human_demo.behavior_cloning import MechanicHierarchySampler  # noqa: E402
from rivalsim.rival2_aerial_option_v2 import apply_fast_aerial_initiation  # noqa: E402
from rivalsim.rival2_aerial_training_pack import (  # noqa: E402
    PACK_NAMES,
    AerialTrainingPackTracker,
    build_training_pack_scenarios,
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

AUTHORITY = ROOT / "results/rival2/aerial_training_pack_v1/authority.json"
AUTHORITY_SHA256 = "C7CD065A5AB5FCC26CBB6C62F65DD99BDD07F812BDB5492CFFDE657578A6ED72"
RESULTS = ROOT / "results/rival2/aerial_training_pack_v1"
BLUE = ROOT / "checkpoints/rival2/aerial_option_v2/rival2_aerial_option_v2_selected_blue.pt"
ORANGE = ROOT / "checkpoints/rival2/aerial_option_v2/rival2_aerial_option_v2_selected_orange.pt"
DEFAULT_RUN_DIR = Path("G:/dev/RivalSim-runs/aerial-training-pack-v1")
SEED = 2_026_090_303
TARGET_LABELS = ("aerialdribble", "groundtoairdribble")


def load_authority() -> dict[str, Any]:
    if v1.v1.sha256_file(AUTHORITY) != AUTHORITY_SHA256:
        raise RuntimeError("aerial training-pack authority changed")
    authority = json.loads(AUTHORITY.read_text(encoding="utf-8"))
    for identity in (
        authority["protected_competitive_base"]["blue"],
        authority["protected_competitive_base"]["orange"],
        authority["initial_option"]["blue"],
        authority["initial_option"]["orange"],
        authority["initial_option"]["v2_result"],
        authority["initial_option"]["v2_failure_attribution"],
    ):
        path = ROOT / identity["path"]
        if v1.v1.sha256_file(path) != identity["sha256"]:
            raise RuntimeError(f"training-pack input changed: {path}")
    return authority


def make_optimizer(
    model: Rival2ActorCritic, authority: dict[str, Any]
) -> torch.optim.AdamW:
    boundary = authority["training_boundary"]
    return torch.optim.AdamW(
        [
            {"params": model.trunk.parameters(), "lr": boundary["trunk_learning_rate"]},
            {"params": model.actor.parameters(), "lr": boundary["actor_learning_rate"]},
        ],
        weight_decay=boundary["weight_decay"],
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
    authority: dict[str, Any], pack: int
) -> HybridDistributionOverride:
    row = authority["physical_training"]["exploration"][PACK_NAMES[pack]]
    return HybridDistributionOverride(
        analog_log_std=float(np.log(row["analog_sigma"])),
        button_temperature=float(row["button_temperature"]),
    )


def collect_pack(
    model: Rival2ActorCritic,
    geometry: ArenaGeometry,
    meshes: WarpArenaMeshes,
    *,
    side: int,
    pack: int,
    worlds: int,
    horizon: int,
    first_touch_deadline: int,
    seed: int,
    device: str,
    generator: torch.Generator,
    distribution: HybridDistributionOverride,
    deterministic: bool,
    collision_dir: Path,
) -> tuple[v1.OptionRollout | None, dict[str, Any]]:
    batch = build_training_pack_scenarios(
        worlds, seed=seed, attacker_side=side, pack=pack
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
    tracker = AerialTrainingPackTracker(
        worlds,
        attacker_side=side,
        pack=pack,
        first_touch_deadline=first_touch_deadline,
        horizon=horizon,
    )
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
        transition = env.step(action)
        scoring = env.bridge.views["rival2.scoring_team_latched"].to(torch.int64)
        any_goal = active_before & transition.terminated & (scoring >= 0)
        goal_for = any_goal & (scoring == side)
        reward, pack_done = tracker.step(
            observation,
            transition.transition_observation,
            tick=tick,
            goal_for_attacker=goal_for,
            any_goal=any_goal,
            active=active_before,
        )
        terminal = pack_done | transition.terminated | transition.truncated
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
        "pack": PACK_NAMES[pack],
        "horizon": horizon,
        "active_worlds_at_end": int(active.sum()),
        "telemetry": telemetry,
        "fractions": {
            "launch": telemetry["launches"] / worlds,
            "high_touch": telemetry["first_high_touches"] / worlds,
            "goalward_first_touch": telemetry["goalward_first_touches"] / worlds,
            "second_airborne_touch": telemetry["second_airborne_touches"] / worlds,
            "goal": telemetry["goals"] / worlds,
            "no_launch_failure": telemetry["no_launch_failures"] / worlds,
            "missed_intercept_failure": telemetry["missed_intercept_failures"] / worlds,
            "ball_ground_failure": telemetry["ball_ground_failures"] / worlds,
            "timeout_after_touch_failure": telemetry[
                "timeout_after_touch_failures"
            ]
            / worlds,
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


def pack_gate(
    physical: list[dict[str, Any]],
    human: list[dict[str, Any]],
    pack: int,
    authority: dict[str, Any],
) -> bool:
    if any(
        row["mean_complete_action_rmse"]
        > authority["human_auxiliary"]["validation_rmse_max"]
        for row in human
    ):
        return False
    pack_authority = authority["physical_training"]["packs"][pack]
    threshold = pack_authority.get("advance", pack_authority.get("select"))
    for row in physical:
        fractions = row["fractions"]
        if fractions["high_touch"] < threshold["high_touch_fraction_min"]:
            return False
        if fractions["goal"] < threshold["goal_fraction_min"]:
            return False
        if pack == 2 and fractions["second_airborne_touch"] < threshold[
            "second_airborne_touch_fraction_min"
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
    pack: int,
    block: int,
    evaluation: dict[str, Any],
) -> dict[str, Any]:
    payload = copy.deepcopy(source)
    payload["model"] = {
        name: value.detach().cpu().clone() for name, value in model.state_dict().items()
    }
    payload["optimizer"] = {
        "format": "RIVAL2_AERIAL_TRAINING_PACK_V1_FRESH_ADAMW",
        "option": optimizer.state_dict(),
    }
    payload["curriculum_transition"] = {
        "identity": "RIVAL2_AERIAL_TRAINING_PACK_V1",
        "created_utc": v1.utc_now(),
        "authority": {
            "path": AUTHORITY.relative_to(ROOT).as_posix(),
            "sha256": AUTHORITY_SHA256,
        },
        "deployment_side": side,
        "pack": PACK_NAMES[pack],
        "accepted_pack_block": block,
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
        "pack": PACK_NAMES[pack],
        "block": block,
    }


def run(args: argparse.Namespace) -> int:
    authority = load_authority()
    sources = [
        torch.load(BLUE, map_location="cpu", weights_only=False),
        torch.load(ORANGE, map_location="cpu", weights_only=False),
    ]
    for side, color in ((0, "blue"), (1, "orange")):
        if human_base.tensor_tree_sha256(sources[side]["model"]) != authority[
            "initial_option"
        ][color]["model_tensor_sha256"]:
            raise RuntimeError("training-pack initial option changed")
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
        raise RuntimeError("training pack must not open human test")
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
            generator=torch.Generator(device="cpu").manual_seed(SEED ^ (0xB300 + side)),
        )
        for side in (0, 1)
    ]
    geometry = ArenaGeometry.load_soccar(args.collision_dir)
    meshes = WarpArenaMeshes(geometry, args.device)
    generators = [
        torch.Generator(device=args.device).manual_seed(SEED ^ (0xC300 + side))
        for side in (0, 1)
    ]
    baseline_physical: dict[str, Any] = {}
    for pack in range(len(PACK_NAMES)):
        pack_authority = authority["physical_training"]["packs"][pack]
        rows = []
        for side in (0, 1):
            _unused_rollout, metrics = collect_pack(
                models[side],
                geometry,
                meshes,
                side=side,
                pack=pack,
                worlds=args.evaluation_worlds_per_side,
                horizon=int(pack_authority["horizon_ticks"]),
                first_touch_deadline=int(pack_authority["first_high_touch_deadline_tick"]),
                seed=SEED ^ (0xE300 + pack * 16 + side),
                device=args.device,
                generator=generators[side],
                distribution=distribution_override(authority, pack),
                deterministic=True,
                collision_dir=args.collision_dir,
            )
            rows.append(metrics)
        baseline_physical[PACK_NAMES[pack]] = rows
    physical_rows = [
        row
        for pack_name in PACK_NAMES
        for row in baseline_physical[pack_name]
    ]
    training_signal_verified = bool(
        all(row["fractions"]["launch"] == 1.0 for row in physical_rows)
        and all(row["fractions"]["high_touch"] > 0.0 for row in physical_rows)
        and all(row["finite_observation"] for row in physical_rows)
        and all(
            row["mean_complete_action_rmse"]
            <= authority["human_auxiliary"]["validation_rmse_max"]
            for row in baseline_human
        )
    )
    preflight = {
        "format": "RIVAL2_AERIAL_TRAINING_PACK_V1_PREFLIGHT",
        "created_utc": v1.utc_now(),
        "authority_sha256": AUTHORITY_SHA256,
        "protected_competitive_base_unchanged": True,
        "initial_option_verified": True,
        "human_test_loaded": False,
        "baseline_human": baseline_human,
        "baseline_physical": baseline_physical,
        "training_signal_verified": training_signal_verified,
        "verdict": "PASS" if training_signal_verified else "FAIL",
    }
    v1.write_json(RESULTS / "preflight.json", preflight)
    if args.preflight_only:
        print(json.dumps(preflight, indent=2, sort_keys=True), flush=True)
        return 0 if training_signal_verified else 2
    if not training_signal_verified:
        raise RuntimeError("aerial training-pack preflight did not expose usable signal")

    run_dir = Path(args.run_dir)
    if run_dir.exists() and any(run_dir.iterdir()):
        raise RuntimeError("training-pack V1 requires a fresh run directory")
    run_dir.mkdir(parents=True, exist_ok=True)
    curve = RESULTS / "training_curve.jsonl"
    if curve.exists():
        curve.unlink()
    compat = compatibility_authority(authority)
    completed: list[str] = []
    selected: dict[str, Any] | None = None
    stop_reason = "maximum_blocks"
    for pack in range(len(PACK_NAMES)):
        pack_authority = authority["physical_training"]["packs"][pack]
        maximum = min(int(pack_authority["maximum_blocks"]), args.maximum_blocks_per_pack)
        distribution = distribution_override(authority, pack)
        consecutive = 0
        best_score = (-1.0, -1.0, -1.0)
        best: dict[str, Any] | None = None
        for block in range(1, maximum + 1):
            sides = []
            for side in (0, 1):
                rollout, rollout_metrics = collect_pack(
                    models[side],
                    geometry,
                    meshes,
                    side=side,
                    pack=pack,
                    worlds=args.worlds_per_side,
                    horizon=int(pack_authority["horizon_ticks"]),
                    first_touch_deadline=int(pack_authority["first_high_touch_deadline_tick"]),
                    seed=SEED + pack * 1_000_000 + block * 100 + side,
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
                "pack": PACK_NAMES[pack],
                "pack_index": pack,
                "block": block,
                "sides": sides,
            }
            interval = int(authority["physical_training"]["evaluation_interval_blocks"])
            if block % interval == 0 or block == maximum:
                physical = []
                for side in (0, 1):
                    _unused_rollout, metrics = collect_pack(
                        models[side],
                        geometry,
                        meshes,
                        side=side,
                        pack=pack,
                        worlds=args.evaluation_worlds_per_side,
                        horizon=int(pack_authority["horizon_ticks"]),
                        first_touch_deadline=int(
                            pack_authority["first_high_touch_deadline_tick"]
                        ),
                        seed=SEED ^ (0xE300 + pack * 16 + side),
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
                passed = pack_gate(physical, human, pack, authority)
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
                            / f"{PACK_NAMES[pack]}_b{block:04d}_{'blue' if side == 0 else 'orange'}.pt",
                            side=side,
                            pack=pack,
                            block=block,
                            evaluation=evaluation,
                        )
                    )
                evaluation["checkpoint"] = records
                row["evaluation"] = evaluation
                score = (
                    min(x["fractions"]["goal"] for x in physical),
                    min(x["fractions"]["high_touch"] for x in physical),
                    min(x["fractions"]["second_airborne_touch"] for x in physical),
                )
                if score > best_score:
                    best_score = score
                    best = copy.deepcopy(evaluation)
                consecutive = consecutive + 1 if passed else 0
                print(
                    json.dumps(
                        {
                            "stage": "aerial_training_pack_v1",
                            "pack": PACK_NAMES[pack],
                            "block": block,
                            "passed": passed,
                            "fractions": [x["fractions"] for x in physical],
                            "human_rmse": [x["mean_complete_action_rmse"] for x in human],
                        },
                        sort_keys=True,
                    ),
                    flush=True,
                )
                required = int(
                    pack_authority.get("advance", {}).get("consecutive_boundaries", 1)
                )
                if consecutive >= required:
                    v1.append_jsonl(curve, row)
                    completed.append(PACK_NAMES[pack])
                    if pack == len(PACK_NAMES) - 1:
                        selected = copy.deepcopy(evaluation)
                        stop_reason = "all_training_packs_passed"
                    break
            v1.append_jsonl(curve, row)
            for side in (0, 1):
                save_checkpoint(
                    sources[side],
                    models[side],
                    optimizers[side],
                    run_dir / "rolling" / ("blue.pt" if side == 0 else "orange.pt"),
                    side=side,
                    pack=pack,
                    block=block,
                    evaluation=row.get("evaluation", {}),
                )
        else:
            stop_reason = f"{PACK_NAMES[pack]}_maximum_blocks_without_gate"
        if pack == len(PACK_NAMES) - 1 and selected is not None:
            break
        if PACK_NAMES[pack] not in completed:
            selected = best
            break
    result = {
        "format": "RIVAL2_AERIAL_TRAINING_PACK_V1_RESULT",
        "created_utc": v1.utc_now(),
        "authority_sha256": AUTHORITY_SHA256,
        "baseline_human": baseline_human,
        "baseline_physical": baseline_physical,
        "completed_packs": completed,
        "stop_reason": stop_reason,
        "selected": selected,
        "controlled_pass": bool(
            selected is not None
            and selected.get("passed")
            and len(completed) == len(PACK_NAMES)
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
    parser.add_argument("--maximum-blocks-per-pack", type=int, default=200)
    parser.add_argument("--preflight-only", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    raise SystemExit(run(parse_args()))
