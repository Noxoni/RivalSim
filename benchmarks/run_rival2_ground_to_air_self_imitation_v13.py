"""Make successful stochastic natural aerial continuations deterministic.

V13 starts from the selected V12 update-60 aerial option.  It runs the frozen
V23 composite in natural self-play with the same bounded exploration used by
V12, detects success from literal second airborne contacts/goals, and performs
actor-only rehearsal of the exact preceding option actions.  This is not PPO.
"""

from __future__ import annotations

import argparse
import copy
import gc
import json
import os
import subprocess
import sys
import time
from dataclasses import asdict
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from benchmarks import run_rival2_codex_autonomous_v1 as autonomous  # noqa: E402
from benchmarks import run_rival2_ground_to_air_selfplay_v12 as v12  # noqa: E402
from rivalsim.arena import ArenaGeometry, WarpArenaMeshes  # noqa: E402
from rivalsim.rival2_contracts import RIVAL2_REWARD_GAMEPLAY_120_V2_VERSION  # noqa: E402
from rivalsim.rival2_env import Rival2Env  # noqa: E402
from rivalsim.rival2_ground_to_air_self_imitation_v13 import (  # noqa: E402
    GROUND_TO_AIR_SELF_IMITATION_V13_VERSION,
    AerialSuccessfulSelfImitationV13,
    SelfImitationConfig,
)
from rivalsim.rival2_ground_to_air_selfplay_training_v12 import (  # noqa: E402
    AerialOptionSelfPlayTrainerV12,
)
from rivalsim.rival2_ground_to_air_selfplay_v12 import (  # noqa: E402
    AerialOptionRouterConfig,
    AerialSelfPlayRewardConfig,
)
from rivalsim.rival2_policy import HybridDistributionOverride  # noqa: E402
from rivalsim.rival2_ppo import Rival2PolicyDisplacementRejected  # noqa: E402

VERSION = GROUND_TO_AIR_SELF_IMITATION_V13_VERSION
AUTHORITY = ROOT / "results/rival2/ground_to_air_self_imitation_v13/authority.json"
AUTHORITY_SHA256 = "51BDCEB54AF0E714DB91403261645DA880179CFE94464062A670B390B71B77CF"
RESULTS = ROOT / "results/rival2/ground_to_air_self_imitation_v13"
CHECKPOINTS = ROOT / "checkpoints/rival2/ground_to_air_self_imitation_v13"
DEFAULT_RUN_DIR = Path("G:/dev/RivalSim-runs/ground-to-air-self-imitation-v13")
DEFAULT_COLLISION_DIR = Path("G:/dev/RLBot-Rival/bot/collision_meshes/soccar")
PARENT = (
    ROOT
    / "checkpoints/rival2/ground_to_air_selfplay_v12"
    / "rival2_ground_to_air_selfplay_v12_u0060.pt"
)
PARENT_SHA256 = "0A80DD35040D5FE354240D4E4E4F4B2CD50EB342CC95985647D3B0947DB092B2"


def utc_now() -> str:
    return datetime.now(UTC).replace(microsecond=0).isoformat().replace("+00:00", "Z")


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
    observed = v12.sha256_file(AUTHORITY)
    if observed != AUTHORITY_SHA256:
        raise RuntimeError(f"V13 authority changed: {observed}")
    authority = json.loads(AUTHORITY.read_text(encoding="utf-8"))
    if authority.get("format") != f"{VERSION}_AUTHORITY":
        raise RuntimeError("unexpected V13 authority format")
    for identity in authority["sources"].values():
        path = ROOT / identity["path"]
        if v12.sha256_file(path) != identity["sha256"]:
            raise RuntimeError(f"V13 bound source changed: {path}")
    return authority


def tensor_hash(module: torch.nn.Module) -> str:
    return autonomous.tensor_tree_sha256(
        {name: value.detach().cpu() for name, value in module.state_dict().items()}
    )


def build_campaign(
    authority: dict[str, Any],
    *,
    collision_dir: Path,
    worlds: int,
    device: str,
) -> tuple[
    AerialOptionSelfPlayTrainerV12,
    AerialSuccessfulSelfImitationV13,
    ArenaGeometry,
    WarpArenaMeshes,
    dict[str, Any],
]:
    source = authority["sources"]
    blue, _ = v12.load_model(
        ROOT / source["blue_v23"]["path"], source["blue_v23"]["sha256"], device
    )
    orange, _ = v12.load_model(
        ROOT / source["orange_v23"]["path"], source["orange_v23"]["sha256"], device
    )
    option, option_payload = v12.load_model(PARENT, PARENT_SHA256, device)
    teacher, _ = v12.load_model(PARENT, PARENT_SHA256, device)
    geometry = ArenaGeometry.load_soccar(collision_dir)
    meshes = WarpArenaMeshes(geometry, device)
    collection = authority["collection"]
    env = Rival2Env(
        worlds,
        str(collision_dir),
        device=device,
        seed=int(collection["candidate_seed"]),
        reward_version=RIVAL2_REWARD_GAMEPLAY_120_V2_VERSION,
        geometry=geometry,
        meshes=meshes,
        kickoff_selector=np.arange(worlds, dtype=np.int32) % 5,
        car_visitation_order="a_then_b",
    )
    exploration = HybridDistributionOverride(
        analog_log_std=float(np.log(float(collection["analog_sigma"]))),
        button_temperature=float(collection["button_temperature"]),
    )
    collector = AerialOptionSelfPlayTrainerV12(
        env,
        blue_base=blue,
        orange_base=orange,
        option=option,
        ppo_config=v12.rival2_ppo_120hz_config(),
        router_config=AerialOptionRouterConfig(**option_payload["router_config"]),
        reward_config=AerialSelfPlayRewardConfig(
            **option_payload["aerial_reward_config"]
        ),
        exploration=exploration,
        seed=int(collection["candidate_seed"]),
        actor_learning_rate=1.0e-12,
        critic_learning_rate=1.0e-12,
    )
    # The V12 collector optimizer is deliberately disabled. V13 owns the only
    # optimizer and it contains actor-head parameters exclusively.
    collector.optimizer = None  # type: ignore[assignment]
    optimization = authority["optimization"]
    config = SelfImitationConfig(
        history_ticks=int(optimization["history_ticks"]),
        maximum_success_samples=int(optimization["maximum_success_samples_per_block"]),
        maximum_retention_samples=int(
            optimization["maximum_retention_samples_per_block"]
        ),
        smooth_l1_beta=float(optimization["smooth_l1_beta"]),
        analog_weight=float(optimization["analog_weight"]),
        button_weight=float(optimization["button_weight"]),
        log_std_weight=float(optimization["log_std_weight"]),
        teacher_actor_kl_weight=float(optimization["teacher_actor_kl_weight"]),
        maximum_gradient_norm=float(optimization["maximum_gradient_norm"]),
    )
    campaign = AerialSuccessfulSelfImitationV13(
        collector,
        teacher=teacher,
        learning_rate=float(optimization["actor_learning_rate"]),
        weight_decay=float(optimization["weight_decay"]),
        config=config,
        seed=int(collection["candidate_seed"]) ^ 0x51A13,
    )
    provenance = {
        "authority": {
            "path": AUTHORITY.relative_to(ROOT).as_posix(),
            "sha256": AUTHORITY_SHA256,
        },
        "parent": {
            "path": PARENT.relative_to(ROOT).as_posix(),
            "sha256": PARENT_SHA256,
        },
        "blue_v23": copy.deepcopy(source["blue_v23"]),
        "orange_v23": copy.deepcopy(source["orange_v23"]),
    }
    return collector, campaign, geometry, meshes, provenance


def checkpoint_payload(
    collector: AerialOptionSelfPlayTrainerV12,
    campaign: AerialSuccessfulSelfImitationV13,
    provenance: dict[str, Any],
) -> dict[str, Any]:
    return {
        "format": f"{VERSION}_CHECKPOINT",
        "model": {
            name: value.detach().cpu().clone()
            for name, value in campaign.model.state_dict().items()
        },
        "optimizer": campaign.optimizer.state_dict(),
        "policy_config": asdict(campaign.model.config),
        "router_config": asdict(collector.router_config),
        "aerial_reward_config": asdict(collector.reward_config),
        "exploration": asdict(collector.exploration),
        "self_imitation_config": asdict(campaign.config),
        "accepted_blocks": campaign.accepted_blocks,
        "total_success_samples": campaign.total_success_samples,
        "total_option_samples": collector.total_option_samples,
        "total_physics_ticks": collector.total_physics_ticks,
        "collector_policy_generator_state": collector.policy_generator.get_state(),
        "self_imitation_generator_state": campaign.generator.get_state(),
        "router_telemetry": collector.router.telemetry(),
        "actor_only": True,
        "trunk_frozen": True,
        "critic_frozen": True,
        "ppo_resumable": False,
        "provenance": copy.deepcopy(provenance),
    }


def save_checkpoint(
    collector: AerialOptionSelfPlayTrainerV12,
    campaign: AerialSuccessfulSelfImitationV13,
    path: Path,
    provenance: dict[str, Any],
) -> dict[str, Any]:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(checkpoint_payload(collector, campaign, provenance), path)
    return {
        "accepted_block": campaign.accepted_blocks,
        "path": path.relative_to(ROOT).as_posix() if path.is_relative_to(ROOT) else str(path),
        "sha256": v12.sha256_file(path),
        "bytes": path.stat().st_size,
    }


def controlled_macro(evaluation: dict[str, Any]) -> dict[str, float]:
    setups = evaluation["summary"]["difficulty_0.000"]
    rows = list(setups.values())
    return {
        "entry_airborne_contact_fraction": float(
            np.mean([row["entry_airborne_contact"] for row in rows])
        ),
        "second_airborne_contact_fraction": float(
            np.mean([row["second_airborne_contact"] for row in rows])
        ),
        "goal_within_six_contacts_fraction": float(
            np.mean([row["goal_within_contact_budget"] for row in rows])
        ),
        "ball_ground_failure_fraction": float(
            np.mean([row["ball_ground_failure"] for row in rows])
        ),
    }


def validation_checks(
    controlled: dict[str, float], natural_result: dict[str, Any], authority: dict[str, Any]
) -> dict[str, bool]:
    gate = authority["acceptance"]
    counters = natural_result["router"]["counters"]
    return {
        "controlled_second": controlled["second_airborne_contact_fraction"]
        >= float(gate["controlled_second_airborne_contact_fraction_min"]),
        "controlled_goal": controlled["goal_within_six_contacts_fraction"]
        >= float(gate["controlled_goal_within_six_contacts_fraction_min"]),
        "controlled_floor": controlled["ball_ground_failure_fraction"]
        <= float(gate["controlled_ball_ground_failure_fraction_max"]),
        "natural_second": int(counters["second_airborne_contacts"])
        >= int(gate["natural_deterministic_second_airborne_contact_count_min"]),
        "natural_goal": int(counters["goals_within_contact_budget"])
        >= int(gate["natural_deterministic_goal_count_min"]),
        "natural_touch_health": int(natural_result["touches"]["players_without_touch"])
        <= int(gate["natural_players_without_touch_max"]),
    }


def selection_key(row: dict[str, Any]) -> tuple[Any, ...]:
    natural_counters = row["natural"]["router"]["counters"]
    controlled = row["controlled_macro"]
    return (
        bool(row["eligible"]),
        int(natural_counters["goals_within_contact_budget"]),
        int(natural_counters["second_airborne_contacts"]),
        controlled["goal_within_six_contacts_fraction"],
        controlled["second_airborne_contact_fraction"],
        -controlled["ball_ground_failure_fraction"],
    )


def evaluate_validation(
    collector: AerialOptionSelfPlayTrainerV12,
    campaign: AerialSuccessfulSelfImitationV13,
    geometry: ArenaGeometry,
    meshes: WarpArenaMeshes,
    checkpoint: dict[str, Any],
    *,
    authority: dict[str, Any],
    collision_dir: Path,
    device: str,
    natural_output: Path,
) -> dict[str, Any]:
    selection = authority["selection"]
    controlled = v12.controlled_evaluation(
        collector,
        geometry,
        meshes,
        worlds_per_row=int(selection["controlled_evaluation_worlds_per_row"]),
        collision_dir=collision_dir,
        seed=int(selection["controlled_seed_base"]),
    )
    macro = controlled_macro(controlled)
    checkpoint_path = Path(checkpoint["path"])
    if not checkpoint_path.is_absolute():
        checkpoint_path = ROOT / checkpoint_path
    command = [
        sys.executable,
        str(ROOT / "benchmarks/evaluate_rival2_ground_to_air_selfplay_v12_natural.py"),
        "--checkpoint",
        str(checkpoint_path),
        "--checkpoint-sha256",
        str(checkpoint["sha256"]),
        "--worlds",
        str(selection["natural_validation_worlds"]),
        "--ticks",
        str(selection["natural_validation_ticks"]),
        "--seed",
        str(selection["natural_validation_seed"]),
        "--device",
        device,
        "--collision-root",
        str(collision_dir),
        "--output",
        str(natural_output),
    ]
    completed = subprocess.run(
        command,
        cwd=ROOT,
        capture_output=True,
        text=True,
        timeout=600,
        check=False,
    )
    if completed.returncode != 0:
        raise RuntimeError(
            "isolated natural validation failed: "
            f"{completed.stderr[-2000:]} {completed.stdout[-2000:]}"
        )
    natural_result = json.loads(natural_output.read_text(encoding="utf-8"))
    checks = validation_checks(macro, natural_result, authority)
    return {
        "accepted_block": campaign.accepted_blocks,
        "checkpoint": checkpoint,
        "controlled": controlled,
        "controlled_macro": macro,
        "natural": natural_result,
        "checks": checks,
        "eligible": all(checks.values()),
    }


def restore_checkpoint(
    collector: AerialOptionSelfPlayTrainerV12,
    campaign: AerialSuccessfulSelfImitationV13,
    path: Path,
    provenance: dict[str, Any],
) -> None:
    payload = torch.load(path, map_location="cpu", weights_only=False)
    if payload.get("format") != f"{VERSION}_CHECKPOINT":
        raise RuntimeError("unsupported V13 resume checkpoint")
    if payload.get("provenance") != provenance:
        raise RuntimeError("V13 resume provenance differs from frozen authority")
    campaign.model.load_state_dict(payload["model"], strict=True)
    campaign.optimizer.load_state_dict(payload["optimizer"])
    campaign.accepted_blocks = int(payload["accepted_blocks"])
    campaign.total_success_samples = int(payload["total_success_samples"])
    collector.total_option_samples = int(payload["total_option_samples"])
    collector.total_physics_ticks = int(payload["total_physics_ticks"])
    collector.policy_generator.set_state(
        payload["collector_policy_generator_state"].to(dtype=torch.uint8, device="cpu")
    )
    campaign.generator.set_state(
        payload["self_imitation_generator_state"].to(dtype=torch.uint8, device="cpu")
    )


def run(args: argparse.Namespace) -> int:
    authority = load_authority()
    collector, campaign, geometry, meshes, provenance = build_campaign(
        authority,
        collision_dir=args.collision_dir,
        worlds=int(args.worlds),
        device=args.device,
    )
    source = torch.load(PARENT, map_location="cpu", weights_only=False)
    source_model = source["model"]
    trunk_before = autonomous.tensor_tree_sha256(
        {name: value for name, value in source_model.items() if name.startswith("trunk.")}
    )
    critic_before = autonomous.tensor_tree_sha256(
        {name: value for name, value in source_model.items() if name.startswith("critic.")}
    )
    checks = {
        "authority_hash": v12.sha256_file(AUTHORITY) == AUTHORITY_SHA256,
        "parent_hash": v12.sha256_file(PARENT) == PARENT_SHA256,
        "worlds": int(args.worlds) == int(authority["collection"]["worlds"]),
        "physics_and_policy_120_hz": collector.env.physics_hz
        == collector.env.policy_hz
        == 120,
        "v23_frozen": not any(
            parameter.requires_grad
            for model in (collector.blue_base, collector.orange_base)
            for parameter in model.parameters()
        ),
        "trunk_frozen": not any(
            parameter.requires_grad for parameter in campaign.model.trunk.parameters()
        ),
        "critic_frozen": not any(
            parameter.requires_grad for parameter in campaign.model.critic.parameters()
        ),
        "actor_only_optimizer": {
            id(parameter)
            for group in campaign.optimizer.param_groups
            for parameter in group["params"]
        }
        == {id(parameter) for parameter in campaign.model.actor.parameters()},
        "fresh_optimizer": len(campaign.optimizer.state) == 0,
        "v12_collector_optimizer_disabled": collector.optimizer is None,
        "no_named_mechanics_state": collector.env.world.gameplay_v3 is None,
        "raw_airtime_zero": collector.reward_config.raw_airtime_reward == 0.0,
    }
    preflight = {
        "format": f"{VERSION}_PREFLIGHT",
        "created_utc": utc_now(),
        "verdict": "PASS" if all(checks.values()) else "FAIL",
        "checks": checks,
        "provenance": provenance,
        "trunk_sha256": trunk_before,
        "critic_sha256": critic_before,
    }
    RESULTS.mkdir(parents=True, exist_ok=True)
    args.run_dir.mkdir(parents=True, exist_ok=True)
    write_json(RESULTS / "preflight.json", preflight)
    if preflight["verdict"] != "PASS":
        raise RuntimeError(f"V13 preflight failed: {checks}")
    if args.preflight_only:
        print(json.dumps(preflight, indent=2, sort_keys=True), flush=True)
        return 0

    curve = RESULTS / "training_curve.jsonl"
    if args.resume is None and curve.exists():
        raise RuntimeError("V13 training curve already exists")
    if args.resume is not None:
        restore_checkpoint(collector, campaign, args.resume, provenance)
    rolling = args.run_dir / "rolling.pt"
    manifest_path = RESULTS / "validation_manifest.json"
    if manifest_path.exists():
        validation_rows = json.loads(manifest_path.read_text(encoding="utf-8"))["rows"]
    else:
        validation_rows = []
    hard_failure: dict[str, Any] | None = None
    rollout_boundaries = 0
    stale_boundaries = 0
    best: dict[str, Any] | None = (
        max(validation_rows, key=selection_key) if validation_rows else None
    )
    started = time.monotonic()
    maximum_blocks = min(
        int(args.maximum_blocks), int(authority["optimization"]["accepted_block_ceiling"])
    )
    interval = int(authority["selection"]["validation_interval_blocks"])
    patience = int(authority["selection"]["early_stop_patience_boundaries"])

    pending_validation_path = RESULTS / f"validation_b{campaign.accepted_blocks:04d}.json"
    if (
        args.resume is not None
        and campaign.accepted_blocks > 0
        and campaign.accepted_blocks % interval == 0
        and not pending_validation_path.exists()
    ):
        resume_checkpoint = {
            "accepted_block": campaign.accepted_blocks,
            "path": str(args.resume.resolve()),
            "sha256": v12.sha256_file(args.resume),
            "bytes": args.resume.stat().st_size,
        }
        validation = evaluate_validation(
            collector,
            campaign,
            geometry,
            meshes,
            resume_checkpoint,
            authority=authority,
            collision_dir=args.collision_dir,
            device=args.device,
            natural_output=RESULTS
            / f"natural_validation_b{campaign.accepted_blocks:04d}.json",
        )
        write_json(pending_validation_path, validation)
        compact = {
            "accepted_block": campaign.accepted_blocks,
            "checkpoint": resume_checkpoint,
            "controlled_macro": validation["controlled_macro"],
            "natural": {
                "touches": validation["natural"]["touches"],
                "scoring": validation["natural"]["scoring"],
                "router": validation["natural"]["router"],
                "derived": validation["natural"]["derived"],
            },
            "checks": validation["checks"],
            "eligible": validation["eligible"],
        }
        validation_rows.append(compact)
        if best is None or selection_key(compact) > selection_key(best):
            best = compact
        write_json(
            manifest_path,
            {"format": f"{VERSION}_VALIDATION_MANIFEST", "rows": validation_rows},
        )
        print(
            json.dumps(
                {
                    "resumed_boundary": campaign.accepted_blocks,
                    "eligible": validation["eligible"],
                    "controlled": validation["controlled_macro"],
                    "natural_router": validation["natural"]["router"]["counters"],
                },
                sort_keys=True,
            ),
            flush=True,
        )

    while campaign.accepted_blocks < maximum_blocks:
        rollout = collector.collect_rollout()
        rollout_boundaries += 1
        try:
            metrics = campaign.update(rollout)
        except Rival2PolicyDisplacementRejected as error:
            hard_failure = {
                "created_utc": utc_now(),
                "last_accepted_block": campaign.accepted_blocks,
                "diagnostics": error.diagnostics,
            }
            write_json(RESULTS / "hard_failure.json", hard_failure)
            save_checkpoint(collector, campaign, rolling, provenance)
            break
        if not metrics["accepted"]:
            del rollout
            continue
        row = {
            "accepted_block": campaign.accepted_blocks,
            "created_utc": utc_now(),
            "elapsed_seconds": time.monotonic() - started,
            "rollout_boundary": rollout_boundaries,
            "collector": collector.last_rollout_metrics,
            "self_imitation": metrics,
        }
        append_jsonl(curve, row)
        save_checkpoint(collector, campaign, rolling, provenance)
        del rollout
        gc.collect()
        torch.cuda.empty_cache()
        if campaign.accepted_blocks % interval != 0:
            if campaign.accepted_blocks == 1 or campaign.accepted_blocks % 5 == 0:
                print(
                    json.dumps(
                        {
                            "accepted_block": campaign.accepted_blocks,
                            "success_events": metrics["success_events"],
                            "success_samples": metrics["success_samples"],
                            "retention_mean_kl": metrics["post_step_retention_mean_kl"],
                        }
                    ),
                    flush=True,
                )
            continue

        snapshot = save_checkpoint(
            collector,
            campaign,
            args.run_dir / f"snapshot_b{campaign.accepted_blocks:04d}.pt",
            provenance,
        )
        validation = evaluate_validation(
            collector,
            campaign,
            geometry,
            meshes,
            snapshot,
            authority=authority,
            collision_dir=args.collision_dir,
            device=args.device,
            natural_output=RESULTS
            / f"natural_validation_b{campaign.accepted_blocks:04d}.json",
        )
        write_json(
            RESULTS / f"validation_b{campaign.accepted_blocks:04d}.json", validation
        )
        validation_rows.append(
            {
                "accepted_block": campaign.accepted_blocks,
                "checkpoint": snapshot,
                "controlled_macro": validation["controlled_macro"],
                "natural": {
                    "touches": validation["natural"]["touches"],
                    "scoring": validation["natural"]["scoring"],
                    "router": validation["natural"]["router"],
                    "derived": validation["natural"]["derived"],
                },
                "checks": validation["checks"],
                "eligible": validation["eligible"],
            }
        )
        if best is None or selection_key(validation_rows[-1]) > selection_key(best):
            best = validation_rows[-1]
            stale_boundaries = 0
        else:
            stale_boundaries += 1
        write_json(
            RESULTS / "validation_manifest.json",
            {"format": f"{VERSION}_VALIDATION_MANIFEST", "rows": validation_rows},
        )
        print(
            json.dumps(
                {
                    "accepted_block": campaign.accepted_blocks,
                    "eligible": validation["eligible"],
                    "controlled": validation["controlled_macro"],
                    "natural_router": validation["natural"]["router"]["counters"],
                    "stale_boundaries": stale_boundaries,
                },
                sort_keys=True,
            ),
            flush=True,
        )
        if best is not None and best["eligible"] and stale_boundaries >= patience:
            break

    stop_reason = (
        "hard_nonfinite_failure"
        if hard_failure is not None
        else (
            "eligible_validation_plateau"
            if best is not None and best["eligible"] and stale_boundaries >= patience
            else "accepted_block_ceiling"
        )
    )
    trunk_after = tensor_hash(campaign.model.trunk)
    critic_after = tensor_hash(campaign.model.critic)
    # tensor_hash strips the module prefix, so compare directly against the
    # same normalized teacher submodule hashes here.
    teacher_trunk = tensor_hash(campaign.teacher.trunk)
    teacher_critic = tensor_hash(campaign.teacher.critic)
    summary = {
        "format": f"{VERSION}_TRAINING_SUMMARY",
        "created_utc": utc_now(),
        "authority_sha256": AUTHORITY_SHA256,
        "accepted_blocks": campaign.accepted_blocks,
        "rollout_boundaries": rollout_boundaries,
        "total_success_samples": campaign.total_success_samples,
        "hard_failure": hard_failure,
        "stop_reason": stop_reason,
        "best_validation": best,
        "trunk_unchanged": trunk_after == teacher_trunk,
        "critic_unchanged": critic_after == teacher_critic,
        "validation_rows": validation_rows,
    }
    write_json(RESULTS / "training_summary.json", summary)
    print(json.dumps(summary, indent=2, sort_keys=True), flush=True)
    return 2 if hard_failure is not None else 0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--worlds", type=int, default=32_768)
    parser.add_argument("--maximum-blocks", type=int, default=120)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--collision-dir", type=Path, default=DEFAULT_COLLISION_DIR)
    parser.add_argument("--run-dir", type=Path, default=DEFAULT_RUN_DIR)
    parser.add_argument("--resume", type=Path)
    parser.add_argument("--preflight-only", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    raise SystemExit(run(parse_args()))
