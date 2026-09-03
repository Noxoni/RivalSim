"""Train the protected V3 aerial scorer in integrated V23 self-play."""

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
from rivalsim.rival2_ground_to_air_integrated_selfplay_v17 import (  # noqa: E402
    GROUND_TO_AIR_INTEGRATED_SELFPLAY_V17_VERSION,
    AerialOptionIntegratedSelfPlayTrainerV17,
    build_integrated_selfplay_initial_state,
    integrated_state_summary,
)
from rivalsim.rival2_ground_to_air_mixed_selfplay_v14 import (  # noqa: E402
    AerialContinuationFailureConfig,
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

VERSION = GROUND_TO_AIR_INTEGRATED_SELFPLAY_V17_VERSION
AUTHORITY = ROOT / "results/rival2/ground_to_air_integrated_selfplay_v17/authority.json"
AUTHORITY_SHA256 = "84D937E3DF2DAE5C0DF19B02E1A2D174BC83DDEBF7710F5B53DDC8D6B97A681A"
RESULTS = ROOT / "results/rival2/ground_to_air_integrated_selfplay_v17"
CHECKPOINTS = ROOT / "checkpoints/rival2/ground_to_air_integrated_selfplay_v17"
DEFAULT_RUN_DIR = Path("G:/dev/RivalSim-runs/ground-to-air-integrated-selfplay-v17")
DEFAULT_COLLISION_DIR = Path("G:/dev/RLBot-Rival/bot/collision_meshes/soccar")

BLUE = ROOT / "checkpoints/rival2/codex_autonomous_v23/rival2_blue.pt"
ORANGE = ROOT / "checkpoints/rival2/codex_autonomous_v23/rival2_orange.pt"
OPTION = ROOT / "checkpoints/rival2/ground_to_air_goal_v3/rival2_ground_to_air_goal_v3.pt"
BLUE_SHA256 = "0263546263285384D2D9A0CE55A471C41A41A8B7D4870DD9504D0ACCEA76723C"
ORANGE_SHA256 = "56E4ECA5075EB5748402BA3C5D8D51AC91FC1AFF55219E64EA5CE688DAD3491A"
OPTION_SHA256 = "F7049F8EF6CC4D1EE3F7303D6D9CE1AA2207A10F6651A33BC71B7C344CC77154"

WORLD_COUNT = 32_768
TARGET_UPDATES = 240
SNAPSHOT_INTERVAL = 30
SEED = 2_026_090_324
V11_FRACTION = 0.25
HIGH_SPEED_FRACTION = 0.25
SETUP_WEIGHTS = (0.35, 0.15, 0.20, 0.30)
DIFFICULTY = 0.0
ACTOR_LR = 5.0e-6
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


def build_batch(worlds: int, refresh_index: int):
    seed = SEED + refresh_index * 1_000_003
    return build_integrated_selfplay_initial_state(
        worlds,
        seed=seed,
        v11_fraction=V11_FRACTION,
        high_speed_fraction=HIGH_SPEED_FRACTION,
        setup_weights=SETUP_WEIGHTS,
        difficulty=DIFFICULTY,
    )


def load_authority() -> dict[str, Any]:
    actual = v12.sha256_file(AUTHORITY)
    if AUTHORITY_SHA256 == "TO_BE_FROZEN" or actual != AUTHORITY_SHA256:
        raise RuntimeError(f"V17 authority is not frozen or changed: {actual}")
    authority = json.loads(AUTHORITY.read_text(encoding="utf-8"))
    checks = {
        "format": authority.get("format") == f"{VERSION}_AUTHORITY",
        "blue": authority["sources"]["blue_v23"]["sha256"] == BLUE_SHA256,
        "orange": authority["sources"]["orange_v23"]["sha256"] == ORANGE_SHA256,
        "option": authority["sources"]["aerial_scorer_v3"]["sha256"]
        == OPTION_SHA256,
        "worlds": authority["campaign"]["worlds"] == WORLD_COUNT,
        "updates": authority["campaign"]["accepted_updates"] == TARGET_UPDATES,
        "v11_fraction": authority["curriculum"]["v11_fraction"] == V11_FRACTION,
        "high_speed_fraction": authority["curriculum"]["high_speed_fraction"]
        == HIGH_SPEED_FRACTION,
        "setup_weights": tuple(authority["curriculum"]["v11_setup_weights"])
        == SETUP_WEIGHTS,
        "actor_lr": authority["optimizer"]["actor_learning_rate"] == ACTOR_LR,
        "failure": authority["continuation_failure_reward"]
        == asdict(failure_config()),
        "raw_airtime": authority["aerial_reward"]["raw_airtime_reward"] == 0.0,
        "named_mechanics": authority["integrity"]["named_mechanic_reward"] is False,
        "natural_gate": authority["selection"]["natural_goal_for_count_min"] >= 1,
    }
    if not all(checks.values()):
        raise RuntimeError(f"V17 authority mismatch: {checks}")
    return authority


def make_environment(
    geometry: ArenaGeometry,
    meshes: WarpArenaMeshes,
    *,
    worlds: int,
    device: str,
    collision_dir: Path,
    refresh_index: int,
) -> tuple[Rival2Env, dict[str, Any]]:
    batch = build_batch(worlds, refresh_index)
    seed = SEED + refresh_index * 1_000_003
    env = Rival2Env(
        worlds,
        str(collision_dir),
        device=device,
        seed=seed,
        reward_version=RIVAL2_REWARD_GAMEPLAY_120_V2_VERSION,
        initial=batch.state,
        geometry=geometry,
        meshes=meshes,
        kickoff_selector=batch.kickoff_selector,
        car_visitation_order="a_then_b",
    )
    summary = integrated_state_summary(batch)
    summary.update({"refresh_index": refresh_index, "seed": seed})
    return env, summary


def build_trainer(
    collision_dir: Path,
    *,
    worlds: int,
    device: str,
) -> tuple[
    AerialOptionIntegratedSelfPlayTrainerV17,
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
        raise RuntimeError("V3 aerial scorer and V23 contracts differ")
    geometry = ArenaGeometry.load_soccar(collision_dir)
    meshes = WarpArenaMeshes(geometry, device)
    env, initial_summary = make_environment(
        geometry,
        meshes,
        worlds=worlds,
        device=device,
        collision_dir=collision_dir,
        refresh_index=0,
    )
    trainer = AerialOptionIntegratedSelfPlayTrainerV17(
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
        "aerial_scorer_v3": {
            "path": OPTION.relative_to(ROOT).as_posix(),
            "sha256": OPTION_SHA256,
        },
        "authority": {
            "path": AUTHORITY.relative_to(ROOT).as_posix(),
            "sha256": v12.sha256_file(AUTHORITY),
        },
    }
    return trainer, provenance, geometry, meshes, initial_summary


def preflight(
    trainer: AerialOptionIntegratedSelfPlayTrainerV17,
    provenance: dict[str, Any],
    initial_summary: dict[str, Any],
    authority: dict[str, Any],
) -> dict[str, Any]:
    inventory = trainer.env.world.gameplay_120.memory_inventory()
    categories = initial_summary["by_category"]
    source_hashes = {
        name: v12.sha256_file(ROOT / identity["path"])
        == identity["sha256"]
        for name, identity in authority["sources"].items()
    }
    checks = {
        "source_hashes_exact": all(
            v12.sha256_file(ROOT / row["path"]) == row["sha256"]
            for name, row in provenance.items()
            if name != "authority"
        ),
        "authority_hash_exact": v12.sha256_file(AUTHORITY) == AUTHORITY_SHA256,
        "all_authority_sources_exact": all(source_hashes.values()),
        "reward_contract_exact": trainer.env.contract_hashes
        == contract_hashes_for_reward(RIVAL2_REWARD_GAMEPLAY_120_V2_VERSION),
        "reward_hash_exact": trainer.env.contract_hashes[
            RIVAL2_REWARD_GAMEPLAY_120_V2_VERSION
        ]
        == REWARD_GAMEPLAY_120_V2_CONTRACT_HASH,
        "ppo_hash_exact": authority["ppo"]["contract_sha256"]
        == RIVAL2_PPO_120HZ_CONTRACT_HASH,
        "physics_and_policy_120_hz": trainer.env.physics_hz
        == trainer.env.policy_hz
        == 120,
        "ordinary_worlds_half": categories["ordinary_v23_selfplay"]
        == round(trainer.env.num_envs * 0.5),
        "v11_worlds_quarter": categories["v11_controlled"]
        == round(trainer.env.num_envs * V11_FRACTION),
        "high_speed_worlds_quarter": categories["v16_high_speed"]
        == round(trainer.env.num_envs * HIGH_SPEED_FRACTION),
        "both_curricula_side_balanced": all(
            len(set(row.values())) == 1
            for row in initial_summary["by_category_and_attacker_side"].values()
        ),
        "all_v11_setups_present": all(
            value > 0 for value in initial_summary["v11_by_setup"].values()
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
        "entry_failure_cancels_entry_reward": (
            trainer.failure_config.landing_before_second_contact
            == -trainer.reward_config.entry_airborne_contact_event
            and trainer.failure_config.ball_ground_before_second_contact
            == -trainer.reward_config.entry_airborne_contact_event
        ),
        "six_contact_budget": trainer.router_config.maximum_distinct_contacts == 6,
        "kl_is_telemetry_only": authority["ppo"]["kl_telemetry_only"] is True,
        "no_optimizer_steps_before_authority": authority["integrity"][
            "optimizer_steps_before_authority_commit"
        ]
        == 0,
    }
    return {
        "format": f"{VERSION}_PREFLIGHT",
        "created_utc": utc_now(),
        "verdict": "PASS" if all(checks.values()) else "FAIL",
        "checks": checks,
        "authority_source_checks": source_hashes,
        "initial_distribution": initial_summary,
        "provenance": provenance,
        "router": asdict(trainer.router_config),
        "aerial_reward": asdict(trainer.reward_config),
        "continuation_failure_reward": asdict(trainer.failure_config),
        "ppo": asdict(trainer.ppo_config),
    }


def checkpoint_record(
    trainer: AerialOptionIntegratedSelfPlayTrainerV17,
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
    trainer: AerialOptionIntegratedSelfPlayTrainerV17,
    path: Path,
    provenance: dict[str, Any],
) -> None:
    payload = torch.load(path, map_location="cpu", weights_only=False)
    if payload.get("format") != f"{VERSION}_CHECKPOINT":
        raise RuntimeError("unsupported V17 resume checkpoint")
    if payload.get("provenance") != provenance:
        raise RuntimeError("V17 resume provenance differs from frozen authority")
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


def _checkpoint_path(record: dict[str, Any]) -> Path:
    path = Path(record["path"])
    return path if path.is_absolute() else ROOT / path


def run_json_command(command: list[str], output: Path, *, timeout: int = 900) -> dict[str, Any]:
    completed = subprocess.run(
        command,
        cwd=ROOT,
        capture_output=True,
        text=True,
        timeout=timeout,
        check=False,
    )
    if completed.returncode != 0:
        raise RuntimeError(
            f"isolated evaluation failed ({completed.returncode}): "
            f"{completed.stderr[-3000:]} {completed.stdout[-3000:]}"
        )
    return json.loads(output.read_text(encoding="utf-8"))


def validation_checks(
    controlled: dict[str, float],
    high_speed: dict[str, Any],
    natural: dict[str, Any],
    authority: dict[str, Any],
) -> dict[str, bool]:
    gate = authority["selection"]
    high = high_speed["summary"]
    counters = natural["router"]["counters"]
    route_goals = natural["route_goal_outcomes"]
    natural_entries = max(int(counters["entry_airborne_contacts"]), 1)
    return {
        "controlled_second": controlled["second_airborne_contact_fraction"]
        >= gate["controlled_second_fraction_min"],
        "controlled_goal": controlled["goal_within_six_contacts_fraction"]
        >= gate["controlled_goal_fraction_min"],
        "controlled_floor": controlled["ball_ground_failure_fraction"]
        <= gate["controlled_ball_ground_fraction_max"],
        "high_speed_first": high["entry_airborne_contact"]
        >= gate["high_speed_first_fraction_min"],
        "high_speed_second": high["second_airborne_contact"]
        >= gate["high_speed_second_fraction_min"],
        "high_speed_goal": high["goal_within_contact_budget"]
        >= gate["high_speed_goal_fraction_min"],
        "high_speed_floor": high["ball_ground_failure"]
        <= gate["high_speed_ball_ground_fraction_max"],
        "high_speed_goal_differential": int(high["goals_for"])
        > int(high["goals_against"]),
        "natural_second": int(counters["second_airborne_contacts"])
        >= gate["natural_second_count_min"],
        "natural_goal_for": int(route_goals["goals_for_while_active"])
        >= gate["natural_goal_for_count_min"],
        "natural_concession_per_entry": int(
            route_goals["goals_against_while_active"]
        )
        / natural_entries
        <= gate["natural_concession_per_entry_max"],
        "natural_touch_health": natural["touches"]["players_without_touch"]
        <= gate["natural_players_without_touch_max"],
    }


def selection_key(row: dict[str, Any]) -> tuple[Any, ...]:
    controlled = row["controlled_macro"]
    high = row["high_speed"]["summary"]
    natural_counters = row["natural"]["router"]["counters"]
    natural_goals = row["natural"]["route_goal_outcomes"]
    return (
        bool(row["eligible"]),
        int(natural_goals["goals_for_while_active"]),
        int(natural_counters["second_airborne_contacts"]),
        -int(natural_goals["goals_against_while_active"]),
        float(high["goal_within_contact_budget"]),
        float(high["second_airborne_contact"]),
        float(controlled["goal_within_six_contacts_fraction"]),
        float(controlled["second_airborne_contact_fraction"]),
    )


def evaluate_boundary(
    trainer: AerialOptionIntegratedSelfPlayTrainerV17,
    geometry: ArenaGeometry,
    meshes: WarpArenaMeshes,
    checkpoint: dict[str, Any],
    *,
    authority: dict[str, Any],
    collision_dir: Path,
    device: str,
) -> dict[str, Any]:
    controlled_rows = v12.controlled_evaluation(
        trainer,
        geometry,
        meshes,
        worlds_per_row=int(authority["selection"]["controlled_worlds_per_row"]),
        collision_dir=collision_dir,
        seed=int(authority["selection"]["controlled_seed"]),
    )
    controlled_macro = v13.controlled_macro(controlled_rows)
    checkpoint_path = _checkpoint_path(checkpoint)

    high_path = RESULTS / f"high_speed_validation_u{trainer.iteration:04d}.json"
    high_command = [
        sys.executable,
        str(ROOT / "benchmarks/run_rival2_ground_to_air_high_speed_probe_v16.py"),
        "--checkpoint",
        str(checkpoint_path),
        "--checkpoint-sha256",
        checkpoint["sha256"],
        "--worlds-per-side",
        str(authority["selection"]["high_speed_worlds_per_side"]),
        "--horizon",
        str(authority["selection"]["high_speed_horizon_ticks"]),
        "--seed",
        str(authority["selection"]["high_speed_seed"]),
        "--device",
        device,
        "--collision-dir",
        str(collision_dir),
        "--output",
        str(high_path),
    ]
    high_speed = run_json_command(high_command, high_path)

    natural_path = RESULTS / f"natural_validation_u{trainer.iteration:04d}.json"
    natural_command = [
        sys.executable,
        str(ROOT / "benchmarks/evaluate_rival2_ground_to_air_selfplay_v12_natural.py"),
        "--checkpoint",
        str(checkpoint_path),
        "--checkpoint-sha256",
        checkpoint["sha256"],
        "--worlds",
        str(authority["selection"]["natural_worlds"]),
        "--ticks",
        str(authority["selection"]["natural_ticks"]),
        "--seed",
        str(authority["selection"]["natural_seed"]),
        "--device",
        device,
        "--collision-root",
        str(collision_dir),
        "--output",
        str(natural_path),
    ]
    natural = run_json_command(natural_command, natural_path)
    checks = validation_checks(
        controlled_macro, high_speed, natural, authority
    )
    return {
        "accepted_update": trainer.iteration,
        "checkpoint": checkpoint,
        "controlled": controlled_rows,
        "controlled_macro": controlled_macro,
        "high_speed": high_speed,
        "natural": natural,
        "checks": checks,
        "eligible": all(checks.values()),
    }


def replace_environment(
    trainer: AerialOptionIntegratedSelfPlayTrainerV17,
    geometry: ArenaGeometry,
    meshes: WarpArenaMeshes,
    *,
    worlds: int,
    device: str,
    collision_dir: Path,
    refresh_index: int,
) -> dict[str, Any]:
    trainer.env = None
    gc.collect()
    if torch.device(device).type == "cuda":
        torch.cuda.empty_cache()
    env, summary = make_environment(
        geometry,
        meshes,
        worlds=worlds,
        device=device,
        collision_dir=collision_dir,
        refresh_index=refresh_index,
    )
    trainer.replace_environment(env)
    return summary


def run(args: argparse.Namespace) -> int:
    authority = load_authority()
    trainer, provenance, geometry, meshes, initial_summary = build_trainer(
        args.collision_dir,
        worlds=args.worlds,
        device=args.device,
    )
    RESULTS.mkdir(parents=True, exist_ok=True)
    args.run_dir.mkdir(parents=True, exist_ok=True)
    preflight_payload = preflight(
        trainer, provenance, initial_summary, authority
    )
    write_json(RESULTS / "preflight.json", preflight_payload)
    if preflight_payload["verdict"] != "PASS":
        raise RuntimeError(f"V17 preflight failed: {preflight_payload['checks']}")
    if args.preflight_only:
        print(json.dumps(preflight_payload, indent=2, sort_keys=True), flush=True)
        return 0

    curve = RESULTS / "training_curve.jsonl"
    rolling = args.run_dir / "rolling.pt"
    if args.resume:
        restore(trainer, args.resume, provenance)
        if trainer.iteration < args.target_updates:
            refreshed = replace_environment(
                trainer,
                geometry,
                meshes,
                worlds=args.worlds,
                device=args.device,
                collision_dir=args.collision_dir,
                refresh_index=max(1, trainer.iteration // SNAPSHOT_INTERVAL),
            )
            initial_summary = refreshed
    elif curve.exists():
        raise RuntimeError("V17 training curve already exists; pass --resume")

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
    completed_boundaries = {
        int(row["accepted_update"]) for row in validation_rows
    }
    hard_failure: dict[str, Any] | None = None
    rollout_boundaries = 0
    start = time.monotonic()

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
        validation_rows.append(
            evaluate_boundary(
                trainer,
                geometry,
                meshes,
                snapshot,
                authority=authority,
                collision_dir=args.collision_dir,
                device=args.device,
            )
        )
        write_json(
            validation_manifest,
            {"format": f"{VERSION}_VALIDATION_MANIFEST", "rows": validation_rows},
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
        if torch.device(args.device).type == "cuda":
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
                RESULTS / f"validation_u{trainer.iteration:04d}.json",
                validation,
            )
            write_json(
                validation_manifest,
                {"format": f"{VERSION}_VALIDATION_MANIFEST", "rows": validation_rows},
            )
            if trainer.iteration < args.target_updates:
                refreshed = replace_environment(
                    trainer,
                    geometry,
                    meshes,
                    worlds=args.worlds,
                    device=args.device,
                    collision_dir=args.collision_dir,
                    refresh_index=trainer.iteration // SNAPSHOT_INTERVAL,
                )
                refresh_rows.append(refreshed)
                write_json(
                    refresh_manifest,
                    {"format": f"{VERSION}_REFRESH_MANIFEST", "rows": refresh_rows},
                )
            print(
                json.dumps(
                    {
                        "accepted_update": trainer.iteration,
                        "controlled": validation["controlled_macro"],
                        "high_speed": validation["high_speed"]["summary"],
                        "natural_router": validation["natural"]["router"]["counters"],
                        "natural_route_goals": validation["natural"]["route_goal_outcomes"],
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
                        "aerial_goals": rollout_metrics["router"]["goals_within_contact_budget"],
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
        source = _checkpoint_path(best["checkpoint"])
        target = CHECKPOINTS / "rival2_ground_to_air_integrated_selfplay_v17.pt"
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, target)
        promoted = {
            "path": target.relative_to(ROOT).as_posix(),
            "sha256": v12.sha256_file(target),
            "selected_update": best["accepted_update"],
        }
        test_path = RESULTS / "untouched_natural_test.json"
        command = [
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
        untouched_test = run_json_command(command, test_path, timeout=1200)

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
