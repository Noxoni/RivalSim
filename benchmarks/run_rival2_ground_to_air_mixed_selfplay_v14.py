"""Train the exact protected aerial scorer in mixed natural self-play."""

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
import time
from dataclasses import asdict
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from benchmarks import run_rival2_ground_to_air_self_imitation_v13 as v13  # noqa: E402
from benchmarks import run_rival2_ground_to_air_selfplay_v12 as v12  # noqa: E402
from rivalsim.arena import ArenaGeometry, WarpArenaMeshes  # noqa: E402
from rivalsim.rival2_contracts import (  # noqa: E402
    REWARD_GAMEPLAY_120_V2_CONTRACT_HASH,
    RIVAL2_REWARD_GAMEPLAY_120_V2_VERSION,
    contract_hashes_for_reward,
)
from rivalsim.rival2_env import Rival2Env  # noqa: E402
from rivalsim.rival2_ground_to_air_mixed_selfplay_v14 import (  # noqa: E402
    GROUND_TO_AIR_MIXED_SELFPLAY_V14_VERSION,
    AerialContinuationFailureConfig,
    AerialOptionMixedSelfPlayTrainerV14,
    build_mixed_selfplay_initial_state,
    mixed_state_summary,
)
from rivalsim.rival2_ground_to_air_selfplay_v12 import (  # noqa: E402
    AerialOptionRouterConfig,
    AerialSelfPlayRewardConfig,
)
from rivalsim.rival2_policy import HybridDistributionOverride  # noqa: E402
from rivalsim.rival2_ppo import (  # noqa: E402
    RIVAL2_PPO_120HZ_CONTRACT_HASH,
    Rival2PolicyDisplacementRejected,
    rival2_ppo_120hz_config,
)

VERSION = GROUND_TO_AIR_MIXED_SELFPLAY_V14_VERSION
AUTHORITY = ROOT / "results/rival2/ground_to_air_mixed_selfplay_v14/authority.json"
AUTHORITY_SHA256 = "E3C8BAB7BB7229DD3C78A39CD05ADEC8529A04D0E604C6F06D8AB9CE0B53DD6C"
RESULTS = ROOT / "results/rival2/ground_to_air_mixed_selfplay_v14"
CHECKPOINTS = ROOT / "checkpoints/rival2/ground_to_air_mixed_selfplay_v14"
DEFAULT_RUN_DIR = Path("G:/dev/RivalSim-runs/ground-to-air-mixed-selfplay-v14")
DEFAULT_COLLISION_DIR = Path("G:/dev/RLBot-Rival/bot/collision_meshes/soccar")

BLUE = ROOT / "checkpoints/rival2/codex_autonomous_v23/rival2_blue.pt"
ORANGE = ROOT / "checkpoints/rival2/codex_autonomous_v23/rival2_orange.pt"
OPTION = ROOT / "checkpoints/rival2/ground_to_air_goal_v3/rival2_ground_to_air_goal_v3.pt"
BLUE_SHA256 = "0263546263285384D2D9A0CE55A471C41A41A8B7D4870DD9504D0ACCEA76723C"
ORANGE_SHA256 = "56E4ECA5075EB5748402BA3C5D8D51AC91FC1AFF55219E64EA5CE688DAD3491A"
OPTION_SHA256 = "F7049F8EF6CC4D1EE3F7303D6D9CE1AA2207A10F6651A33BC71B7C344CC77154"

WORLD_COUNT = 32_768
TARGET_UPDATES = 120
SNAPSHOT_INTERVAL = 30
SEED = 2_026_090_314
CONTROLLED_FRACTION = 0.25
SETUP_WEIGHTS = (0.35, 0.15, 0.20, 0.30)
DIFFICULTY = 0.0
ACTOR_LR = 2.0e-6
CRITIC_LR = 3.0e-4
EXPLORATION_SIGMA = 0.05
BUTTON_TEMPERATURE = 1.25


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


def router_config() -> AerialOptionRouterConfig:
    return AerialOptionRouterConfig()


def reward_config() -> AerialSelfPlayRewardConfig:
    return AerialSelfPlayRewardConfig()


def failure_config() -> AerialContinuationFailureConfig:
    return AerialContinuationFailureConfig()


def exploration_config() -> HybridDistributionOverride:
    return HybridDistributionOverride(
        analog_log_std=math.log(EXPLORATION_SIGMA),
        button_temperature=BUTTON_TEMPERATURE,
    )


def load_authority() -> dict[str, Any]:
    actual = v12.sha256_file(AUTHORITY)
    if AUTHORITY_SHA256 == "TO_BE_FROZEN" or actual != AUTHORITY_SHA256:
        raise RuntimeError(f"V14 authority is not frozen or changed: {actual}")
    authority = json.loads(AUTHORITY.read_text(encoding="utf-8"))
    checks = {
        "format": authority.get("format") == f"{VERSION}_AUTHORITY",
        "blue": authority["sources"]["blue_v23"]["sha256"] == BLUE_SHA256,
        "orange": authority["sources"]["orange_v23"]["sha256"] == ORANGE_SHA256,
        "option": authority["sources"]["aerial_scorer"]["sha256"] == OPTION_SHA256,
        "worlds": authority["campaign"]["worlds"] == WORLD_COUNT,
        "updates": authority["campaign"]["accepted_updates"] == TARGET_UPDATES,
        "fraction": authority["curriculum"]["controlled_world_fraction"]
        == CONTROLLED_FRACTION,
        "weights": tuple(authority["curriculum"]["setup_weights"])
        == SETUP_WEIGHTS,
        "failure": authority["continuation_failure_reward"]
        == asdict(failure_config()),
        "raw_airtime": authority["aerial_reward"]["raw_airtime_reward"] == 0.0,
        "named_mechanics": authority["integrity"]["named_mechanic_reward"] is False,
    }
    if not all(checks.values()):
        raise RuntimeError(f"V14 authority mismatch: {checks}")
    return authority


def build_environment(
    geometry: ArenaGeometry,
    meshes: WarpArenaMeshes,
    *,
    worlds: int,
    device: str,
    refresh_index: int,
) -> tuple[Rival2Env, dict[str, Any]]:
    batch = build_mixed_selfplay_initial_state(
        worlds,
        seed=SEED + refresh_index * 1_000_003,
        controlled_fraction=CONTROLLED_FRACTION,
        setup_weights=SETUP_WEIGHTS,
        difficulty=DIFFICULTY,
    )
    env = Rival2Env(
        worlds,
        str(DEFAULT_COLLISION_DIR),
        device=device,
        seed=SEED + refresh_index * 1_000_003,
        reward_version=RIVAL2_REWARD_GAMEPLAY_120_V2_VERSION,
        initial=batch.state,
        geometry=geometry,
        meshes=meshes,
        kickoff_selector=batch.kickoff_selector,
        car_visitation_order="a_then_b",
    )
    summary = mixed_state_summary(batch)
    summary["refresh_index"] = refresh_index
    summary["seed"] = SEED + refresh_index * 1_000_003
    return env, summary


def build_trainer(
    collision_dir: Path,
    *,
    worlds: int,
    device: str,
) -> tuple[
    AerialOptionMixedSelfPlayTrainerV14,
    dict[str, Any],
    ArenaGeometry,
    WarpArenaMeshes,
    dict[str, Any],
]:
    blue, blue_payload = v12.load_model(BLUE, BLUE_SHA256, device)
    orange, orange_payload = v12.load_model(ORANGE, ORANGE_SHA256, device)
    option, option_payload = v12.load_model(OPTION, OPTION_SHA256, device)
    if blue_payload.get("contract_hashes") != orange_payload.get("contract_hashes"):
        raise RuntimeError("V23 side policy contracts differ")
    if option_payload.get("contract_hashes") != blue_payload.get("contract_hashes"):
        raise RuntimeError("aerial scorer and V23 contracts differ")
    geometry = ArenaGeometry.load_soccar(collision_dir)
    meshes = WarpArenaMeshes(geometry, device)
    batch = build_mixed_selfplay_initial_state(
        worlds,
        seed=SEED,
        controlled_fraction=CONTROLLED_FRACTION,
        setup_weights=SETUP_WEIGHTS,
        difficulty=DIFFICULTY,
    )
    env = Rival2Env(
        worlds,
        str(collision_dir),
        device=device,
        seed=SEED,
        reward_version=RIVAL2_REWARD_GAMEPLAY_120_V2_VERSION,
        initial=batch.state,
        geometry=geometry,
        meshes=meshes,
        kickoff_selector=batch.kickoff_selector,
        car_visitation_order="a_then_b",
    )
    trainer = AerialOptionMixedSelfPlayTrainerV14(
        env,
        blue_base=blue,
        orange_base=orange,
        option=option,
        ppo_config=rival2_ppo_120hz_config(),
        router_config=router_config(),
        reward_config=reward_config(),
        exploration=exploration_config(),
        seed=SEED,
        actor_learning_rate=ACTOR_LR,
        critic_learning_rate=CRITIC_LR,
        failure_config=failure_config(),
    )
    provenance = {
        "blue_v23": {"path": BLUE.relative_to(ROOT).as_posix(), "sha256": BLUE_SHA256},
        "orange_v23": {
            "path": ORANGE.relative_to(ROOT).as_posix(),
            "sha256": ORANGE_SHA256,
        },
        "aerial_scorer": {
            "path": OPTION.relative_to(ROOT).as_posix(),
            "sha256": OPTION_SHA256,
        },
        "authority": {
            "path": AUTHORITY.relative_to(ROOT).as_posix(),
            "sha256": v12.sha256_file(AUTHORITY),
        },
    }
    return trainer, provenance, geometry, meshes, mixed_state_summary(batch)


def preflight(
    trainer: AerialOptionMixedSelfPlayTrainerV14,
    provenance: dict[str, Any],
    initial_summary: dict[str, Any],
) -> dict[str, Any]:
    inventory = trainer.env.world.gameplay_120.memory_inventory()
    expected_controlled = round(trainer.env.num_envs * CONTROLLED_FRACTION)
    expected_controlled -= expected_controlled % 2
    checks = {
        "source_hashes_exact": all(
            v12.sha256_file(ROOT / row["path"]) == row["sha256"]
            for name, row in provenance.items()
            if name != "authority"
        ),
        "reward_contract_exact": trainer.env.contract_hashes
        == contract_hashes_for_reward(RIVAL2_REWARD_GAMEPLAY_120_V2_VERSION),
        "reward_hash_exact": trainer.env.contract_hashes[
            RIVAL2_REWARD_GAMEPLAY_120_V2_VERSION
        ]
        == REWARD_GAMEPLAY_120_V2_CONTRACT_HASH,
        "ppo_hash_exact": json.loads(AUTHORITY.read_text(encoding="utf-8"))["ppo"][
            "contract_sha256"
        ]
        == RIVAL2_PPO_120HZ_CONTRACT_HASH,
        "physics_and_policy_120_hz": trainer.env.physics_hz == trainer.env.policy_hz == 120,
        "controlled_count_exact": initial_summary["controlled_worlds"]
        == expected_controlled,
        "ordinary_worlds_present": initial_summary["ordinary_kickoff_worlds"] > 0,
        "attacker_side_balanced": len(
            set(initial_summary["controlled_by_attacker_side"].values())
        )
        == 1,
        "all_entry_families_present": all(
            value > 0 for value in initial_summary["controlled_by_setup"].values()
        ),
        "base_policies_frozen": not any(
            parameter.requires_grad
            for model in (trainer.blue_base, trainer.orange_base)
            for parameter in model.parameters()
        ),
        "option_trunk_frozen": not any(
            parameter.requires_grad for parameter in trainer.model.trunk.parameters()
        ),
        "fresh_optimizer": len(trainer.optimizer.state) == 0,
        "no_named_mechanics_state": trainer.env.world.gameplay_v3 is None,
        "no_named_mechanics_arrays": inventory["named_mechanics_arrays"] == 0,
        "no_raw_airtime_reward": trainer.reward_config.raw_airtime_reward == 0.0,
        "failed_entry_cancels_entry_reward": (
            trainer.failure_config.landing_before_second_contact
            == -trainer.reward_config.entry_airborne_contact_event
            and trainer.failure_config.ball_ground_before_second_contact
            == -trainer.reward_config.entry_airborne_contact_event
        ),
        "six_contact_budget": trainer.router_config.maximum_distinct_contacts == 6,
        "kl_rejection_disabled": True,
    }
    return {
        "format": f"{VERSION}_PREFLIGHT",
        "created_utc": utc_now(),
        "verdict": "PASS" if all(checks.values()) else "FAIL",
        "checks": checks,
        "initial_distribution": initial_summary,
        "provenance": provenance,
        "router": asdict(trainer.router_config),
        "aerial_reward": asdict(trainer.reward_config),
        "continuation_failure_reward": asdict(trainer.failure_config),
        "ppo": asdict(trainer.ppo_config),
    }


def checkpoint_record(
    trainer: AerialOptionMixedSelfPlayTrainerV14,
    path: Path,
    provenance: dict[str, Any],
) -> dict[str, Any]:
    path.parent.mkdir(parents=True, exist_ok=True)
    trainer.save_checkpoint(path, provenance)
    return {
        "accepted_update": trainer.iteration,
        "path": path.relative_to(ROOT).as_posix() if path.is_relative_to(ROOT) else str(path),
        "sha256": v12.sha256_file(path),
        "bytes": path.stat().st_size,
        "total_option_samples": trainer.total_option_samples,
        "total_physics_ticks": trainer.total_physics_ticks,
    }


def restore(
    trainer: AerialOptionMixedSelfPlayTrainerV14,
    path: Path,
    provenance: dict[str, Any],
) -> None:
    payload = torch.load(path, map_location="cpu", weights_only=False)
    if payload.get("format") != f"{VERSION}_CHECKPOINT":
        raise RuntimeError("unsupported V14 resume checkpoint")
    if payload.get("provenance") != provenance:
        raise RuntimeError("V14 resume provenance differs from frozen authority")
    trainer.model.load_state_dict(payload["model"], strict=True)
    trainer.optimizer.load_state_dict(payload["optimizer"])
    trainer.iteration = int(payload["iteration"])
    trainer.policy_version = int(payload["policy_version"])
    trainer.total_option_samples = int(payload["total_option_samples"])
    trainer.total_physics_ticks = int(payload["total_physics_ticks"])
    trainer.policy_generator.set_state(
        payload["policy_generator_state"].to(device="cpu", dtype=torch.uint8)
    )
    trainer.cumulative_router_counts = {
        str(name): int(value)
        for name, value in payload.get("cumulative_router_counts", {}).items()
    }
    trainer.curriculum_refreshes = int(payload.get("curriculum_refreshes", 0))


def validation_checks(
    controlled_macro: dict[str, float],
    natural: dict[str, Any],
    authority: dict[str, Any],
) -> dict[str, bool]:
    gate = authority["acceptance"]
    counters = natural["router"]["counters"]
    return {
        "controlled_second": controlled_macro["second_airborne_contact_fraction"]
        >= gate["controlled_second_airborne_contact_fraction_min"],
        "controlled_goal": controlled_macro["goal_within_six_contacts_fraction"]
        >= gate["controlled_goal_within_six_contacts_fraction_min"],
        "controlled_floor": controlled_macro["ball_ground_failure_fraction"]
        <= gate["controlled_ball_ground_failure_fraction_max"],
        "natural_second": counters["second_airborne_contacts"]
        >= gate["natural_second_airborne_contact_count_min"],
        "natural_goal": counters["goals_within_contact_budget"]
        >= gate["natural_goal_within_contact_budget_count_min"],
        "natural_touch_health": natural["touches"]["players_without_touch"]
        <= gate["natural_players_without_touch_max"],
    }


def selection_key(row: dict[str, Any]) -> tuple[Any, ...]:
    counters = row["natural"]["router"]["counters"]
    controlled = row["controlled_macro"]
    return (
        bool(row["eligible"]),
        int(counters["goals_within_contact_budget"]),
        int(counters["second_airborne_contacts"]),
        controlled["goal_within_six_contacts_fraction"],
        controlled["second_airborne_contact_fraction"],
        -controlled["ball_ground_failure_fraction"],
    )


def evaluate_boundary(
    trainer: AerialOptionMixedSelfPlayTrainerV14,
    geometry: ArenaGeometry,
    meshes: WarpArenaMeshes,
    checkpoint: dict[str, Any],
    *,
    authority: dict[str, Any],
    collision_dir: Path,
    device: str,
) -> dict[str, Any]:
    evaluation = v12.controlled_evaluation(
        trainer,
        geometry,
        meshes,
        worlds_per_row=int(authority["selection"]["controlled_worlds_per_row"]),
        collision_dir=collision_dir,
        seed=int(authority["selection"]["controlled_seed"]),
    )
    macro = v13.controlled_macro(evaluation)
    checkpoint_path = Path(checkpoint["path"])
    if not checkpoint_path.is_absolute():
        checkpoint_path = ROOT / checkpoint_path
    natural_path = RESULTS / f"natural_validation_u{trainer.iteration:04d}.json"
    command = [
        sys.executable,
        str(ROOT / "benchmarks/evaluate_rival2_ground_to_air_selfplay_v12_natural.py"),
        "--checkpoint",
        str(checkpoint_path),
        "--checkpoint-sha256",
        str(checkpoint["sha256"]),
        "--worlds",
        str(authority["selection"]["natural_validation_worlds"]),
        "--ticks",
        str(authority["selection"]["natural_validation_ticks"]),
        "--seed",
        str(authority["selection"]["natural_validation_seed"]),
        "--device",
        device,
        "--collision-root",
        str(collision_dir),
        "--output",
        str(natural_path),
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
            "isolated V14 natural validation failed: "
            f"{completed.stderr[-2000:]} {completed.stdout[-2000:]}"
        )
    natural = json.loads(natural_path.read_text(encoding="utf-8"))
    checks = validation_checks(macro, natural, authority)
    return {
        "accepted_update": trainer.iteration,
        "checkpoint": checkpoint,
        "controlled": evaluation,
        "controlled_macro": macro,
        "natural": natural,
        "checks": checks,
        "eligible": all(checks.values()),
    }


def replace_curriculum_environment(
    trainer: AerialOptionMixedSelfPlayTrainerV14,
    geometry: ArenaGeometry,
    meshes: WarpArenaMeshes,
    *,
    worlds: int,
    device: str,
    refresh_index: int,
    collision_dir: Path,
) -> dict[str, Any]:
    trainer.env = None
    gc.collect()
    torch.cuda.empty_cache()
    batch = build_mixed_selfplay_initial_state(
        worlds,
        seed=SEED + refresh_index * 1_000_003,
        controlled_fraction=CONTROLLED_FRACTION,
        setup_weights=SETUP_WEIGHTS,
        difficulty=DIFFICULTY,
    )
    env = Rival2Env(
        worlds,
        str(collision_dir),
        device=device,
        seed=SEED + refresh_index * 1_000_003,
        reward_version=RIVAL2_REWARD_GAMEPLAY_120_V2_VERSION,
        initial=batch.state,
        geometry=geometry,
        meshes=meshes,
        kickoff_selector=batch.kickoff_selector,
        car_visitation_order="a_then_b",
    )
    trainer.replace_environment(env)
    summary = mixed_state_summary(batch)
    summary.update(
        {"refresh_index": refresh_index, "seed": SEED + refresh_index * 1_000_003}
    )
    return summary


def run(args: argparse.Namespace) -> int:
    authority = load_authority()
    trainer, provenance, geometry, meshes, initial_summary = build_trainer(
        args.collision_dir, worlds=args.worlds, device=args.device
    )
    RESULTS.mkdir(parents=True, exist_ok=True)
    args.run_dir.mkdir(parents=True, exist_ok=True)
    preflight_payload = preflight(trainer, provenance, initial_summary)
    write_json(RESULTS / "preflight.json", preflight_payload)
    if preflight_payload["verdict"] != "PASS":
        raise RuntimeError(f"V14 preflight failed: {preflight_payload['checks']}")
    if args.preflight_only:
        print(json.dumps(preflight_payload, indent=2, sort_keys=True), flush=True)
        return 0
    curve = RESULTS / "training_curve.jsonl"
    rolling = args.run_dir / "rolling.pt"
    if args.resume:
        restore(trainer, args.resume, provenance)
    elif curve.exists():
        raise RuntimeError("V14 training curve already exists; pass --resume")
    refresh_manifest = RESULTS / "curriculum_refresh_manifest.json"
    validation_manifest = RESULTS / "validation_manifest.json"
    refresh_rows = (
        json.loads(refresh_manifest.read_text(encoding="utf-8"))["rows"]
        if refresh_manifest.exists()
        else [initial_summary]
    )
    validation_rows: list[dict[str, Any]] = (
        json.loads(validation_manifest.read_text(encoding="utf-8"))["rows"]
        if validation_manifest.exists()
        else []
    )
    hard_failure: dict[str, Any] | None = None
    rollout_boundaries = 0
    start = time.monotonic()
    completed_boundaries = {
        int(row["accepted_update"]) for row in validation_rows
    }
    if (
        args.resume
        and trainer.iteration > 0
        and trainer.iteration % SNAPSHOT_INTERVAL == 0
        and trainer.iteration not in completed_boundaries
    ):
        snapshot_path = args.run_dir / f"snapshot_u{trainer.iteration:04d}.pt"
        snapshot = {
            "accepted_update": trainer.iteration,
            "path": str(snapshot_path),
            "sha256": v12.sha256_file(snapshot_path),
            "bytes": snapshot_path.stat().st_size,
            "total_option_samples": trainer.total_option_samples,
            "total_physics_ticks": trainer.total_physics_ticks,
        }
        validation = evaluate_boundary(
            trainer,
            geometry,
            meshes,
            snapshot,
            authority=authority,
            collision_dir=args.collision_dir,
            device=args.device,
        )
        validation_rows.append(validation)
        write_json(
            validation_manifest,
            {"format": f"{VERSION}_VALIDATION_MANIFEST", "rows": validation_rows},
        )
        if trainer.iteration < args.target_updates:
            refreshed = replace_curriculum_environment(
                trainer,
                geometry,
                meshes,
                worlds=args.worlds,
                device=args.device,
                refresh_index=trainer.iteration // SNAPSHOT_INTERVAL,
                collision_dir=args.collision_dir,
            )
            refresh_rows.append(refreshed)
            write_json(
                refresh_manifest,
                {"format": f"{VERSION}_REFRESH_MANIFEST", "rows": refresh_rows},
            )
        print(
            json.dumps(
                {
                    "resumed_boundary": trainer.iteration,
                    "controlled": validation["controlled_macro"],
                    "natural_router": validation["natural"]["router"]["counters"],
                    "eligible": validation["eligible"],
                },
                sort_keys=True,
            ),
            flush=True,
        )
    while trainer.iteration < args.target_updates:
        rollout_started = time.monotonic()
        rollout = trainer.collect_rollout()
        rollout_boundaries += 1
        rollout_metrics = copy.deepcopy(trainer.last_rollout_metrics or {})
        if int(rollout_metrics.get("option_samples", 0)) == 0:
            del rollout
            continue
        try:
            ppo = trainer.update(rollout)
        except Rival2PolicyDisplacementRejected as error:
            hard_failure = {
                "created_utc": utc_now(),
                "last_accepted_update": trainer.iteration,
                "diagnostics": error.diagnostics,
            }
            write_json(RESULTS / "hard_safety_failure.json", hard_failure)
            checkpoint_record(trainer, rolling, provenance)
            break
        row = {
            "accepted_update": trainer.iteration,
            "created_utc": utc_now(),
            "elapsed_seconds": time.monotonic() - start,
            "rollout_boundary": rollout_boundaries,
            "rollout_seconds": time.monotonic() - rollout_started,
            "total_option_samples": trainer.total_option_samples,
            "ppo": v12.scalar_metrics(ppo),
            "gameplay_and_option": rollout_metrics,
        }
        append_jsonl(curve, row)
        checkpoint_record(trainer, rolling, provenance)
        del rollout, ppo
        gc.collect()
        torch.cuda.empty_cache()
        if trainer.iteration % SNAPSHOT_INTERVAL == 0:
            snapshot = checkpoint_record(
                trainer,
                args.run_dir / f"snapshot_u{trainer.iteration:04d}.pt",
                provenance,
            )
            validation = evaluate_boundary(
                trainer,
                geometry,
                meshes,
                snapshot,
                authority=authority,
                collision_dir=args.collision_dir,
                device=args.device,
            )
            validation_rows.append(validation)
            write_json(
                RESULTS / f"validation_u{trainer.iteration:04d}.json", validation
            )
            write_json(
                RESULTS / "validation_manifest.json",
                {"format": f"{VERSION}_VALIDATION_MANIFEST", "rows": validation_rows},
            )
            if trainer.iteration < args.target_updates:
                refreshed = replace_curriculum_environment(
                    trainer,
                    geometry,
                    meshes,
                    worlds=args.worlds,
                    device=args.device,
                    refresh_index=trainer.iteration // SNAPSHOT_INTERVAL,
                    collision_dir=args.collision_dir,
                )
                refresh_rows.append(refreshed)
                write_json(
                    RESULTS / "curriculum_refresh_manifest.json",
                    {"format": f"{VERSION}_REFRESH_MANIFEST", "rows": refresh_rows},
                )
            print(
                json.dumps(
                    {
                        "accepted_update": trainer.iteration,
                        "controlled": validation["controlled_macro"],
                        "natural_router": validation["natural"]["router"]["counters"],
                        "eligible": validation["eligible"],
                    },
                    sort_keys=True,
                ),
                flush=True,
            )
        elif trainer.iteration == 1 or trainer.iteration % 5 == 0:
            print(
                json.dumps(
                    {
                        "accepted_update": trainer.iteration,
                        "option_samples": rollout_metrics["option_samples"],
                        "activations": rollout_metrics["router"]["activations"],
                        "entries": rollout_metrics["router"]["entry_airborne_contacts"],
                        "seconds": rollout_metrics["router"]["second_airborne_contacts"],
                        "aerial_goals": rollout_metrics["router"][
                            "goals_within_contact_budget"
                        ],
                        "landing_failures": rollout_metrics["router"].get(
                            "landing_before_second_contact", 0
                        ),
                        "ball_ground_failures": rollout_metrics["router"].get(
                            "ball_ground_before_second_contact", 0
                        ),
                    },
                    sort_keys=True,
                ),
                flush=True,
            )
    completed = trainer.iteration == args.target_updates and hard_failure is None
    best = max(validation_rows, key=selection_key) if validation_rows else None
    promoted: dict[str, Any] | None = None
    untouched_test: dict[str, Any] | None = None
    if completed and best is not None and best["eligible"]:
        source = Path(best["checkpoint"]["path"])
        if not source.is_absolute():
            source = ROOT / source
        target = CHECKPOINTS / "rival2_ground_to_air_mixed_selfplay_v14.pt"
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, target)
        promoted = {
            "path": target.relative_to(ROOT).as_posix(),
            "sha256": v12.sha256_file(target),
            "selected_update": best["accepted_update"],
        }
        test_path = RESULTS / "untouched_natural_test.json"
        test_command = [
            sys.executable,
            str(ROOT / "benchmarks/evaluate_rival2_ground_to_air_selfplay_v12_natural.py"),
            "--checkpoint",
            str(target),
            "--checkpoint-sha256",
            promoted["sha256"],
            "--worlds",
            str(authority["selection"]["untouched_test_worlds"]),
            "--ticks",
            str(authority["selection"]["untouched_test_ticks"]),
            "--seed",
            str(authority["selection"]["untouched_test_seed"]),
            "--device",
            args.device,
            "--collision-root",
            str(args.collision_dir),
            "--output",
            str(test_path),
        ]
        subprocess.run(test_command, cwd=ROOT, check=True, timeout=900)
        untouched_test = json.loads(test_path.read_text(encoding="utf-8"))
    summary = {
        "format": f"{VERSION}_TRAINING_SUMMARY",
        "created_utc": utc_now(),
        "authority_sha256": v12.sha256_file(AUTHORITY),
        "accepted_updates": trainer.iteration,
        "target_updates": args.target_updates,
        "rollout_boundaries": rollout_boundaries,
        "hard_failure": hard_failure,
        "completed": completed,
        "cumulative_router_counts": trainer.cumulative_router_counts,
        "curriculum_refreshes": refresh_rows,
        "best_validation": best,
        "promoted_checkpoint": promoted,
        "untouched_natural_test": untouched_test,
    }
    write_json(RESULTS / "training_summary.json", summary)
    print(json.dumps(summary, indent=2, sort_keys=True), flush=True)
    return 2 if hard_failure is not None else 0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--worlds", type=int, default=WORLD_COUNT)
    parser.add_argument("--target-updates", type=int, default=TARGET_UPDATES)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--collision-dir", type=Path, default=DEFAULT_COLLISION_DIR)
    parser.add_argument("--run-dir", type=Path, default=DEFAULT_RUN_DIR)
    parser.add_argument("--resume", type=Path)
    parser.add_argument("--preflight-only", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    raise SystemExit(run(parse_args()))
