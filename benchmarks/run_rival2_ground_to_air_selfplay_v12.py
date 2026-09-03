"""Train the protected multi-touch aerial scorer inside V23 self-play."""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import math
import os
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

from benchmarks import run_rival2_ground_to_air_entry_probe_v11 as entry_probe  # noqa: E402
from rivalsim.arena import ArenaGeometry, WarpArenaMeshes  # noqa: E402
from rivalsim.rival2_contracts import (  # noqa: E402
    REWARD_GAMEPLAY_120_V2_CONTRACT_HASH,
    RIVAL2_REWARD_GAMEPLAY_120_V2_VERSION,
    contract_hashes_for_reward,
)
from rivalsim.rival2_env import Rival2Env  # noqa: E402
from rivalsim.rival2_ground_to_air_entry_v11 import (  # noqa: E402
    DEFENDER_LIVE,
    SETUP_NAMES,
)
from rivalsim.rival2_ground_to_air_selfplay_training_v12 import (  # noqa: E402
    AerialOptionSelfPlayTrainerV12,
)
from rivalsim.rival2_ground_to_air_selfplay_v12 import (  # noqa: E402
    AerialOptionRouterConfig,
    AerialSelfPlayRewardConfig,
)
from rivalsim.rival2_policy import (  # noqa: E402
    HybridDistributionOverride,
    Rival2ActorCritic,
    Rival2PolicyConfig,
)
from rivalsim.rival2_ppo import (  # noqa: E402
    RIVAL2_PPO_120HZ_CONTRACT_HASH,
    Rival2PolicyDisplacementRejected,
    rival2_ppo_120hz_config,
)

VERSION = "RIVAL2_GROUND_TO_AIR_SELFPLAY_V12"
AUTHORITY = ROOT / "results/rival2/ground_to_air_selfplay_v12/authority.json"
AUTHORITY_SHA256 = "95595B08D03A7B839A5FBF132DEB3FEA88D4F698215DEDA1CE764AA2757029D5"
RESULTS = ROOT / "results/rival2/ground_to_air_selfplay_v12"
CHECKPOINTS = ROOT / "checkpoints/rival2/ground_to_air_selfplay_v12"
DEFAULT_RUN_DIR = Path("G:/dev/RivalSim-runs/ground-to-air-selfplay-v12")
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
SEED = 2_026_090_312
ACTOR_LR = 2.0e-6
CRITIC_LR = 3.0e-4
EXPLORATION_SIGMA = 0.05
BUTTON_TEMPERATURE = 1.25


def utc_now() -> str:
    return datetime.now(UTC).replace(microsecond=0).isoformat().replace("+00:00", "Z")


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


def scalar_metrics(metrics: dict[str, torch.Tensor]) -> dict[str, float]:
    return {
        name: float(value.detach().item())
        for name, value in metrics.items()
        if value.numel() == 1
    }


def load_model(
    path: Path, expected_sha256: str, device: str
) -> tuple[Rival2ActorCritic, dict[str, Any]]:
    actual = sha256_file(path)
    if actual != expected_sha256:
        raise RuntimeError(f"checkpoint identity mismatch for {path}: {actual}")
    payload = torch.load(path, map_location="cpu", weights_only=False)
    model = Rival2ActorCritic(Rival2PolicyConfig(**payload["policy_config"])).to(device)
    model.load_state_dict(payload["model"], strict=True)
    return model, payload


def router_config() -> AerialOptionRouterConfig:
    return AerialOptionRouterConfig()


def reward_config() -> AerialSelfPlayRewardConfig:
    return AerialSelfPlayRewardConfig()


def exploration_config() -> HybridDistributionOverride:
    return HybridDistributionOverride(
        analog_log_std=math.log(EXPLORATION_SIGMA),
        button_temperature=BUTTON_TEMPERATURE,
    )


def load_authority() -> dict[str, Any]:
    actual = sha256_file(AUTHORITY)
    if AUTHORITY_SHA256 == "TO_BE_FROZEN" or actual != AUTHORITY_SHA256:
        raise RuntimeError(f"V12 authority is not frozen or changed: {actual}")
    payload = json.loads(AUTHORITY.read_text(encoding="utf-8"))
    checks = {
        "format": payload.get("format") == f"{VERSION}_AUTHORITY",
        "blue": payload["sources"]["blue_v23"]["sha256"] == BLUE_SHA256,
        "orange": payload["sources"]["orange_v23"]["sha256"] == ORANGE_SHA256,
        "option": payload["sources"]["aerial_scorer"]["sha256"] == OPTION_SHA256,
        "router": payload["router"] == asdict(router_config()),
        "reward": payload["aerial_reward"] == asdict(reward_config()),
        "worlds": payload["campaign"]["worlds"] == WORLD_COUNT,
        "target": payload["campaign"]["accepted_updates"] == TARGET_UPDATES,
        "actor_lr": payload["optimizer"]["actor_learning_rate"] == ACTOR_LR,
        "critic_lr": payload["optimizer"]["critic_learning_rate"] == CRITIC_LR,
        "kl_telemetry": payload["ppo"]["kl_telemetry_only"] is True,
    }
    if not all(checks.values()):
        raise RuntimeError(f"V12 authority mismatch: {checks}")
    return payload


def build_trainer(
    collision_dir: Path,
    *,
    worlds: int,
    device: str,
) -> tuple[AerialOptionSelfPlayTrainerV12, dict[str, Any], ArenaGeometry, WarpArenaMeshes]:
    blue, blue_payload = load_model(BLUE, BLUE_SHA256, device)
    orange, orange_payload = load_model(ORANGE, ORANGE_SHA256, device)
    option, option_payload = load_model(OPTION, OPTION_SHA256, device)
    if blue_payload.get("contract_hashes") != orange_payload.get("contract_hashes"):
        raise RuntimeError("V23 side policy contracts differ")
    if option_payload.get("contract_hashes") != blue_payload.get("contract_hashes"):
        raise RuntimeError("aerial scorer and V23 contracts differ")
    geometry = ArenaGeometry.load_soccar(collision_dir)
    meshes = WarpArenaMeshes(geometry, device)
    kickoff = np.arange(worlds, dtype=np.int32) % 5
    env = Rival2Env(
        worlds,
        str(collision_dir),
        device=device,
        seed=SEED,
        reward_version=RIVAL2_REWARD_GAMEPLAY_120_V2_VERSION,
        geometry=geometry,
        meshes=meshes,
        kickoff_selector=kickoff,
        car_visitation_order="a_then_b",
    )
    trainer = AerialOptionSelfPlayTrainerV12(
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
            "sha256": sha256_file(AUTHORITY),
        },
    }
    return trainer, provenance, geometry, meshes


def preflight(
    trainer: AerialOptionSelfPlayTrainerV12,
    provenance: dict[str, Any],
) -> dict[str, Any]:
    inventory = trainer.env.world.gameplay_120.memory_inventory()
    checks = {
        "source_hashes_exact": all(
            sha256_file(ROOT / row["path"]) == row["sha256"]
            for name, row in provenance.items()
            if name != "authority"
        ),
        "reward_contract_exact": trainer.env.contract_hashes
        == contract_hashes_for_reward(RIVAL2_REWARD_GAMEPLAY_120_V2_VERSION),
        "reward_hash_exact": trainer.env.contract_hashes[
            RIVAL2_REWARD_GAMEPLAY_120_V2_VERSION
        ]
        == REWARD_GAMEPLAY_120_V2_CONTRACT_HASH,
        "ppo_hash_exact": json.loads(AUTHORITY.read_text(encoding="utf-8"))[
            "ppo"
        ]["contract_sha256"]
        == RIVAL2_PPO_120HZ_CONTRACT_HASH,
        "physics_and_policy_120_hz": trainer.env.physics_hz == trainer.env.policy_hz == 120,
        "base_policies_frozen": not any(
            parameter.requires_grad
            for model in (trainer.blue_base, trainer.orange_base)
            for parameter in model.parameters()
        ),
        "option_trunk_frozen": not any(
            parameter.requires_grad for parameter in trainer.model.trunk.parameters()
        ),
        "option_actor_trainable": all(
            parameter.requires_grad for parameter in trainer.model.actor.parameters()
        ),
        "option_critic_trainable": all(
            parameter.requires_grad for parameter in trainer.model.critic.parameters()
        ),
        "fresh_optimizer": len(trainer.optimizer.state) == 0,
        "no_named_mechanics_state": trainer.env.world.gameplay_v3 is None,
        "no_named_mechanics_arrays": inventory["named_mechanics_arrays"] == 0,
        "no_raw_airtime_reward": trainer.reward_config.raw_airtime_reward == 0.0,
        "six_contact_budget": trainer.router_config.maximum_distinct_contacts == 6,
        "kl_rejection_disabled": True,
    }
    return {
        "format": f"{VERSION}_PREFLIGHT",
        "created_utc": utc_now(),
        "verdict": "PASS" if all(checks.values()) else "FAIL",
        "checks": checks,
        "provenance": provenance,
        "router": asdict(trainer.router_config),
        "aerial_reward": asdict(trainer.reward_config),
        "ppo": asdict(trainer.ppo_config),
        "optimizer": [
            {"name": group["name"], "learning_rate": group["lr"]}
            for group in trainer.optimizer.param_groups
        ],
    }


@torch.inference_mode()
def controlled_evaluation(
    trainer: AerialOptionSelfPlayTrainerV12,
    geometry: ArenaGeometry,
    meshes: WarpArenaMeshes,
    *,
    worlds_per_row: int,
    collision_dir: Path,
    seed: int,
) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    defenders = {0: trainer.blue_base, 1: trainer.orange_base}
    for setup in range(len(SETUP_NAMES)):
        for side in (0, 1):
            rows.append(
                entry_probe.collect_row(
                    trainer.model,
                    defenders,
                    geometry,
                    meshes,
                    side=side,
                    setup=setup,
                    defender_mode=DEFENDER_LIVE,
                    difficulty=0.0,
                    worlds=worlds_per_row,
                    horizon=600,
                    seed=seed + setup * 100_000,
                    device=str(trainer.device),
                    collision_dir=collision_dir,
                )
            )
    return {
        "worlds_per_row": worlds_per_row,
        "rows": rows,
        "summary": entry_probe.summarize(rows),
    }


def checkpoint_record(
    trainer: AerialOptionSelfPlayTrainerV12,
    path: Path,
    provenance: dict[str, Any],
) -> dict[str, Any]:
    path.parent.mkdir(parents=True, exist_ok=True)
    trainer.save_checkpoint(path, provenance)
    return {
        "accepted_update": trainer.iteration,
        "path": path.relative_to(ROOT).as_posix() if path.is_relative_to(ROOT) else str(path),
        "sha256": sha256_file(path),
        "bytes": path.stat().st_size,
        "total_option_samples": trainer.total_option_samples,
        "total_physics_ticks": trainer.total_physics_ticks,
    }


def restore(
    trainer: AerialOptionSelfPlayTrainerV12,
    path: Path,
    provenance: dict[str, Any],
) -> None:
    payload = torch.load(path, map_location="cpu", weights_only=False)
    if payload.get("format") != "RIVAL2_GROUND_TO_AIR_SELFPLAY_V12_CHECKPOINT":
        raise RuntimeError("unsupported V12 resume checkpoint")
    if payload.get("provenance") != provenance:
        raise RuntimeError("V12 resume provenance differs from frozen sources")
    trainer.model.load_state_dict(payload["model"], strict=True)
    trainer.optimizer.load_state_dict(payload["optimizer"])
    trainer.iteration = int(payload["iteration"])
    trainer.policy_version = int(payload["policy_version"])
    trainer.total_option_samples = int(payload["total_option_samples"])
    trainer.total_physics_ticks = int(payload["total_physics_ticks"])
    trainer.policy_generator.set_state(payload["policy_generator_state"].to(trainer.device))


def run(args: argparse.Namespace) -> int:
    authority = load_authority()
    trainer, provenance, geometry, meshes = build_trainer(
        args.collision_dir, worlds=args.worlds, device=args.device
    )
    RESULTS.mkdir(parents=True, exist_ok=True)
    args.run_dir.mkdir(parents=True, exist_ok=True)
    preflight_payload = preflight(trainer, provenance)
    write_json(RESULTS / "preflight.json", preflight_payload)
    if preflight_payload["verdict"] != "PASS":
        raise RuntimeError(f"V12 preflight failed: {preflight_payload['checks']}")
    if args.preflight_only:
        print(json.dumps(preflight_payload, indent=2, sort_keys=True))
        return 0
    curve = RESULTS / "training_curve.jsonl"
    rolling = args.run_dir / "rolling.pt"
    if args.resume:
        restore(trainer, args.resume, provenance)
    elif curve.exists():
        raise RuntimeError("V12 training curve already exists; pass --resume")
    rollout_boundaries = 0
    hard_failure: dict[str, Any] | None = None
    manifest: list[dict[str, Any]] = []
    start = time.monotonic()
    while trainer.iteration < args.target_updates:
        rollout_started = time.monotonic()
        rollout = trainer.collect_rollout()
        rollout_boundaries += 1
        rollout_seconds = time.monotonic() - rollout_started
        rollout_metrics = trainer.last_rollout_metrics or {}
        if int(rollout_metrics.get("option_samples", 0)) == 0:
            if rollout_boundaries % 10 == 0:
                print(
                    json.dumps(
                        {
                            "accepted_update": trainer.iteration,
                            "rollout_boundary": rollout_boundaries,
                            "option_samples": 0,
                        }
                    ),
                    flush=True,
                )
            del rollout
            continue
        update_started = time.monotonic()
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
        update_seconds = time.monotonic() - update_started
        row = {
            "accepted_update": trainer.iteration,
            "created_utc": utc_now(),
            "elapsed_seconds": time.monotonic() - start,
            "rollout_boundary": rollout_boundaries,
            "rollout_seconds": rollout_seconds,
            "update_seconds": update_seconds,
            "total_option_samples": trainer.total_option_samples,
            "ppo": scalar_metrics(ppo),
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
            evaluation = controlled_evaluation(
                trainer,
                geometry,
                meshes,
                worlds_per_row=args.evaluation_worlds_per_row,
                collision_dir=args.collision_dir,
                seed=SEED + trainer.iteration * 1_000_000,
            )
            write_json(
                RESULTS / f"controlled_evaluation_u{trainer.iteration:04d}.json",
                evaluation,
            )
            snapshot["controlled_evaluation"] = (
                RESULTS / f"controlled_evaluation_u{trainer.iteration:04d}.json"
            ).relative_to(ROOT).as_posix()
            manifest.append(snapshot)
            write_json(
                RESULTS / "snapshot_manifest.json",
                {"format": f"{VERSION}_SNAPSHOT_MANIFEST", "snapshots": manifest},
            )
        if trainer.iteration == 1 or trainer.iteration % 5 == 0:
            print(
                json.dumps(
                    {
                        "accepted_update": trainer.iteration,
                        "option_samples": rollout_metrics["option_samples"],
                        "activations": rollout_metrics["router"]["activations"],
                        "entries": rollout_metrics["router"]["entry_airborne_contacts"],
                        "seconds": rollout_metrics["router"]["second_airborne_contacts"],
                        "aerial_goals": rollout_metrics["router"]["goals_within_contact_budget"],
                        "touches_per_minute": rollout_metrics["touches_per_player_minute"],
                        "kl": row["ppo"].get("completed_update_mean_kl"),
                    },
                    sort_keys=True,
                ),
                flush=True,
            )
    final = checkpoint_record(
        trainer,
        CHECKPOINTS / "rival2_ground_to_air_selfplay_v12.pt",
        provenance,
    )
    summary = {
        "format": f"{VERSION}_TRAINING_SUMMARY",
        "created_utc": utc_now(),
        "authority_sha256": sha256_file(AUTHORITY),
        "authority": authority,
        "accepted_updates": trainer.iteration,
        "target_updates": args.target_updates,
        "rollout_boundaries": rollout_boundaries,
        "hard_failure": hard_failure,
        "final_checkpoint": final,
        "router_telemetry": trainer.router.telemetry(),
        "completed": trainer.iteration == args.target_updates and hard_failure is None,
    }
    write_json(RESULTS / "training_summary.json", summary)
    print(json.dumps(summary, indent=2, sort_keys=True), flush=True)
    return 2 if hard_failure is not None else 0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--worlds", type=int, default=WORLD_COUNT)
    parser.add_argument("--target-updates", type=int, default=TARGET_UPDATES)
    parser.add_argument("--evaluation-worlds-per-row", type=int, default=64)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--collision-dir", type=Path, default=DEFAULT_COLLISION_DIR)
    parser.add_argument("--run-dir", type=Path, default=DEFAULT_RUN_DIR)
    parser.add_argument("--resume", type=Path)
    parser.add_argument("--preflight-only", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    raise SystemExit(run(parse_args()))
