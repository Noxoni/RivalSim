"""Pure-current 120 Hz PPO stage for the Fresh Human Seed v1 lineage."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import sys
import time
from dataclasses import asdict, replace
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from rivalsim.arena import ArenaGeometry, WarpArenaMeshes  # noqa: E402
from rivalsim.rival2_contracts import (  # noqa: E402
    REWARD_GAMEPLAY_120_V2_CONTRACT_HASH,
    RIVAL2_REWARD_GAMEPLAY_120_V2_VERSION,
    contract_hashes_for_reward,
)
from rivalsim.rival2_env import Rival2Env  # noqa: E402
from rivalsim.rival2_opponent_curriculum import (  # noqa: E402
    OPPONENT_CURRENT,
    Rival2OpponentCurriculumConfig,
    Rival2OpponentCurriculumTrainer,
)
from rivalsim.rival2_policy import Rival2PolicyConfig  # noqa: E402
from rivalsim.rival2_ppo import (  # noqa: E402
    RIVAL2_PPO_120HZ_CONTRACT_HASH,
    RIVAL2_PPO_120HZ_V1,
    Rival2KLGuardConfig,
    Rival2PolicyDisplacementRejected,
    rival2_ppo_120hz_config,
)
from rivalsim.rival2_training import Rival2SelfPlayConfig  # noqa: E402

FORMAT = "RIVAL2_FRESH_HUMAN_SEED_V1_PPO"
STAGE1_FORMAT = "RIVAL2_FRESH_HUMAN_SEED_V1_STAGE1_CHECKPOINT"
RESULTS = ROOT / "results/rival2/fresh_human_seed_v1"
AUTHORITY = RESULTS / "ppo_authority.json"
SOURCE = ROOT / "checkpoints/rival2/fresh_human_seed_v1/rival2_fresh_human_seed_v1.pt"
CHECKPOINT_DIR = ROOT / "checkpoints/rival2/fresh_human_seed_v1/ppo"
DEFAULT_RUN_DIR = Path("G:/dev/RivalSim-runs/fresh-human-seed-v1-ppo")
WORLD_COUNT = 32_768
TARGET_UPDATES = 600
LR_SCHEDULE = (1.0e-4, 5.0e-5, 2.5e-5)
SEED = 2026090104
CRITIC_SEED = 2026090105
SNAPSHOTS = frozenset([*range(30, 481, 30), 500, 510, 540, 570, 600])


def utc_now() -> str:
    return datetime.now(UTC).isoformat()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest().upper()


def state_dict_sha256(state: dict[str, torch.Tensor]) -> str:
    digest = hashlib.sha256()
    for name in sorted(state):
        value = state[name].detach().cpu().contiguous()
        digest.update(name.encode())
        digest.update(str(value.dtype).encode())
        digest.update(str(tuple(value.shape)).encode())
        digest.update(value.numpy().tobytes())
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
        name: float(value.detach().item()) for name, value in metrics.items() if value.numel() == 1
    }


def prepare_authority() -> dict[str, Any]:
    source = torch.load(SOURCE, map_location="cpu", weights_only=False)
    if source.get("format") != STAGE1_FORMAT:
        raise RuntimeError("Stage-1 source format mismatch")
    if source["lineage"].get("prior_rival_checkpoint_loaded") is not False:
        raise RuntimeError("Stage-1 source is not the fresh lineage")
    authority = {
        "format": f"{FORMAT}_AUTHORITY",
        "created_utc": utc_now(),
        "source": {
            "path": SOURCE.relative_to(ROOT).as_posix(),
            "sha256": sha256_file(SOURCE),
            "format": source["format"],
            "selected_step": source["selected_step"],
            "validation_rmse": source["validation_rmse"],
            "model_tensor_sha256": state_dict_sha256(source["model"]),
            "used_fields": ["model", "policy_config"],
            "stage1_optimizer_loaded": False,
        },
        "transition": {
            "critic_reinitialized": True,
            "critic_seed": CRITIC_SEED,
            "full_model_unfrozen": True,
            "fresh_ppo_optimizer": True,
            "fresh_ppo_rng": True,
            "fresh_ppo_counters": True,
            "historical_pool_empty": True,
        },
        "reward": {
            "version": RIVAL2_REWARD_GAMEPLAY_120_V2_VERSION,
            "contract_sha256": REWARD_GAMEPLAY_120_V2_CONTRACT_HASH,
        },
        "ppo": {
            "version": RIVAL2_PPO_120HZ_V1,
            "contract_sha256": RIVAL2_PPO_120HZ_CONTRACT_HASH,
            "worlds": WORLD_COUNT,
            "accepted_updates": TARGET_UPDATES,
            "initial_learning_rate": LR_SCHEDULE[0],
            "allowed_learning_rates": list(LR_SCHEDULE),
            "hard_minibatch_kl": 0.10,
            "hard_completed_update_mean_kl": 0.05,
        },
        "opponents": {
            "current_probability": 1.0,
            "historical_probability": 0.0,
            "nexto_probability": 0.0,
            "wisp_probability": 0.0,
            "both_current_sides_trainable": True,
        },
        "snapshots": sorted(SNAPSHOTS),
        "seed": SEED,
    }
    write_json(AUTHORITY, authority)
    return authority


def load_authority() -> dict[str, Any]:
    authority = json.loads(AUTHORITY.read_text(encoding="utf-8"))
    checks = {
        "format": authority.get("format") == f"{FORMAT}_AUTHORITY",
        "source": authority.get("source", {}).get("sha256") == sha256_file(SOURCE),
        "reward": authority.get("reward", {}).get("contract_sha256")
        == REWARD_GAMEPLAY_120_V2_CONTRACT_HASH,
        "ppo": authority.get("ppo", {}).get("contract_sha256") == RIVAL2_PPO_120HZ_CONTRACT_HASH,
        "worlds": authority.get("ppo", {}).get("worlds") == WORLD_COUNT,
        "updates": authority.get("ppo", {}).get("accepted_updates") == TARGET_UPDATES,
        "snapshots": authority.get("snapshots") == sorted(SNAPSHOTS),
    }
    if not all(checks.values()):
        raise RuntimeError(f"PPO authority mismatch: {checks}")
    return authority


def make_trainer(
    collision_root: Path, *, worlds: int
) -> tuple[Rival2OpponentCurriculumTrainer, dict[str, Any]]:
    authority = load_authority()
    source = torch.load(SOURCE, map_location="cpu", weights_only=False)
    policy_config = Rival2PolicyConfig(**source["policy_config"])
    geometry = ArenaGeometry.load_soccar(collision_root)
    env = Rival2Env(
        worlds,
        str(collision_root),
        geometry=geometry,
        meshes=WarpArenaMeshes(geometry),
        device="cuda:0",
        seed=SEED,
        car_visitation_order="a_then_b",
        reward_version=RIVAL2_REWARD_GAMEPLAY_120_V2_VERSION,
    )
    trainer = Rival2OpponentCurriculumTrainer(
        env,
        policy_config=policy_config,
        ppo_config=replace(rival2_ppo_120hz_config(), learning_rate=LR_SCHEDULE[0]),
        self_play_config=Rival2SelfPlayConfig(historical_chance=0.0, historical_pool_bound=1),
        opponent_curriculum=Rival2OpponentCurriculumConfig(
            nexto_probability=0.0,
            wisp_probability=0.0,
            current_probability=1.0,
            historical_probability=0.0,
            seed=SEED ^ 0x7171,
        ),
        seed=SEED,
    )
    if trainer.optimizer.state:
        raise RuntimeError("fresh PPO optimizer unexpectedly has state")
    trainer.model.load_state_dict(source["model"], strict=True)
    stage1_critic = {
        key: value.detach().cpu().clone()
        for key, value in trainer.model.critic.state_dict().items()
    }
    torch.manual_seed(CRITIC_SEED)
    replacement = nn.Linear(policy_config.hidden_dim, 1)
    nn.init.orthogonal_(replacement.weight, gain=0.01)
    nn.init.zeros_(replacement.bias)
    trainer.model.critic.load_state_dict(replacement.state_dict(), strict=True)
    if state_dict_sha256(stage1_critic) == state_dict_sha256(trainer.model.critic.state_dict()):
        raise RuntimeError("critic was not freshly reinitialized")
    trainer.model.requires_grad_(True)
    if not all(parameter.requires_grad for parameter in trainer.model.parameters()):
        raise RuntimeError("full Stage-2 model was not unfrozen")
    if trainer.optimizer.state or trainer.opponent_pool.versions:
        raise RuntimeError("fresh optimizer/pool precondition failed")
    trainer.curriculum_transition = {
        "format": f"{FORMAT}_TRANSITION",
        "source": authority["source"],
        "loaded_fields": ["model", "policy_config"],
        "stage1_optimizer_loaded": False,
        "critic_reinitialized": True,
        "stage1_critic_tensor_sha256": state_dict_sha256(stage1_critic),
        "fresh_critic_tensor_sha256": state_dict_sha256(trainer.model.critic.state_dict()),
        "fresh_optimizer_rng_counters": True,
        "contracts": dict(env.contract_hashes),
        "authority_sha256": sha256_file(AUTHORITY),
    }
    trainer.initialize_curriculum_assignments()
    if not bool((trainer.opponent_family == OPPONENT_CURRENT).all().item()):
        raise RuntimeError("pure-current self-play assignment failed")
    return trainer, source


def optimizer_lr(trainer: Rival2OpponentCurriculumTrainer) -> float:
    rates = {float(group["lr"]) for group in trainer.optimizer.param_groups}
    if len(rates) != 1:
        raise RuntimeError(f"unexpected PPO learning rates: {rates}")
    return rates.pop()


def set_optimizer_lr(trainer: Rival2OpponentCurriculumTrainer, rate: float) -> None:
    if rate not in LR_SCHEDULE:
        raise RuntimeError(f"unauthorized PPO LR: {rate}")
    for group in trainer.optimizer.param_groups:
        group["lr"] = rate


def checkpoint_record(trainer: Rival2OpponentCurriculumTrainer, path: Path) -> dict[str, Any]:
    path.parent.mkdir(parents=True, exist_ok=True)
    trainer.save_checkpoint(path)
    return {
        "accepted_update": trainer.iteration,
        "policy_version": trainer.policy_version,
        "path": path.relative_to(ROOT).as_posix() if path.is_relative_to(ROOT) else str(path),
        "sha256": sha256_file(path),
        "bytes": path.stat().st_size,
        "total_agent_samples": trainer.total_agent_samples,
        "learning_rate": optimizer_lr(trainer),
    }


def preflight(
    trainer: Rival2OpponentCurriculumTrainer, source: dict[str, Any], worlds: int
) -> dict[str, Any]:
    inventory = trainer.env.world.gameplay_120.memory_inventory()
    curriculum = trainer.opponent_curriculum
    checks = {
        "fresh_stage1_source": source.get("format") == STAGE1_FORMAT,
        "prior_rival_checkpoint_absent": source["lineage"]["prior_rival_checkpoint_loaded"]
        is False,
        "fresh_optimizer_empty": len(trainer.optimizer.state) == 0,
        "fresh_counters_zero": trainer.iteration == 0 and trainer.policy_version == 0,
        "critic_reinitialized": trainer.curriculum_transition["critic_reinitialized"],
        "full_model_trainable": all(
            parameter.requires_grad for parameter in trainer.model.parameters()
        ),
        "worlds_exact": worlds == WORLD_COUNT,
        "policy_and_physics_120_hz": trainer.env.policy_hz == trainer.env.physics_hz == 120,
        "reward_contract_exact": trainer.env.contract_hashes
        == contract_hashes_for_reward(RIVAL2_REWARD_GAMEPLAY_120_V2_VERSION),
        "pure_current": bool((trainer.opponent_family == OPPONENT_CURRENT).all().item()),
        "external_and_historical_zero": curriculum.nexto_probability
        == curriculum.wisp_probability
        == curriculum.historical_probability
        == 0.0,
        "historical_pool_empty": not trainer.opponent_pool.versions,
        "named_mechanics_hot_path_absent": trainer.env.world.gameplay_v3 is None,
        "named_mechanics_arrays_zero": inventory["named_mechanics_arrays"] == 0,
    }
    return {
        "format": f"{FORMAT}_PREFLIGHT",
        "created_utc": utc_now(),
        "checks": checks,
        "verdict": "PASS" if all(checks.values()) else "FAIL",
        "transition": trainer.curriculum_transition,
        "ppo_config": asdict(trainer.ppo_config),
        "opponent_curriculum": asdict(curriculum),
        "memory_inventory": inventory,
    }


def run(args: argparse.Namespace) -> int:
    load_authority()
    trainer, source = make_trainer(Path(args.collision_root), worlds=args.worlds)
    payload = preflight(trainer, source, args.worlds)
    write_json(RESULTS / "ppo_preflight.json", payload)
    if payload["verdict"] != "PASS":
        raise RuntimeError(f"PPO preflight failed: {payload['checks']}")
    if args.preflight_only:
        print(json.dumps(payload, indent=2, sort_keys=True))
        return 0

    run_dir = Path(args.run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
    curve = RESULTS / "ppo_curve.jsonl"
    rolling = run_dir / "fresh_human_seed_v1_ppo_rolling.pt"
    manifest_path = RESULTS / "ppo_snapshot_manifest.json"
    manifest = (
        json.loads(manifest_path.read_text(encoding="utf-8"))
        if manifest_path.is_file()
        else {
            "format": f"{FORMAT}_SNAPSHOT_MANIFEST",
            "source_sha256": sha256_file(SOURCE),
            "snapshots": [],
        }
    )
    if args.resume:
        trainer.load_checkpoint(Path(args.resume))
        if trainer.opponent_pool.versions:
            raise RuntimeError("resume checkpoint unexpectedly contains historical policies")
        if not bool((trainer.opponent_family == OPPONENT_CURRENT).all().item()):
            raise RuntimeError("resume checkpoint is not pure-current self-play")
    elif curve.exists():
        raise RuntimeError("PPO curve already exists; use --resume")

    guard = Rival2KLGuardConfig(0.10, 0.05)
    hard_failure: dict[str, Any] | None = None
    first_rollout_verified = trainer.iteration > 0
    started = time.monotonic()
    while trainer.iteration < TARGET_UPDATES:
        rollout_started = time.monotonic()
        rollout = trainer.collect_rollout()
        rollout_seconds = time.monotonic() - rollout_started
        curriculum_metrics = trainer.last_rollout_curriculum_metrics
        gameplay = trainer.last_rollout_gameplay_metrics
        if not first_rollout_verified:
            expected = trainer.env.num_envs * trainer.ppo_config.rollout_horizon * 2
            trainable = curriculum_metrics["trainable_agent_samples"]
            first = {
                "format": f"{FORMAT}_FIRST_ROLLOUT",
                "created_utc": utc_now(),
                "checks": {
                    "current_trainable_samples_exact": trainable["current"] == expected,
                    "historical_trainable_samples_zero": trainable["historical"] == 0,
                    "nexto_trainable_samples_zero": trainable["nexto"] == 0,
                    "wisp_trainable_samples_zero": trainable["wisp"] == 0,
                },
                "curriculum": curriculum_metrics,
                "gameplay": gameplay,
            }
            first["verdict"] = "PASS" if all(first["checks"].values()) else "FAIL"
            write_json(RESULTS / "ppo_first_rollout.json", first)
            if first["verdict"] != "PASS":
                raise RuntimeError(f"first rollout opponent check failed: {first}")
            first_rollout_verified = True

        rejected: list[dict[str, Any]] = []
        update_started = time.monotonic()
        while True:
            rate = optimizer_lr(trainer)
            try:
                metrics = trainer.update(rollout, kl_guard=guard)
                break
            except Rival2PolicyDisplacementRejected as error:
                diagnostics = dict(error.diagnostics)
                diagnostics["learning_rate"] = rate
                diagnostics["created_utc"] = utc_now()
                rejected.append(diagnostics)
                reason = str(diagnostics.get("reason", ""))
                index = LR_SCHEDULE.index(rate)
                if "nonfinite" in reason or index + 1 >= len(LR_SCHEDULE):
                    hard_failure = diagnostics
                    break
                set_optimizer_lr(trainer, LR_SCHEDULE[index + 1])
        if hard_failure is not None:
            write_json(RESULTS / "ppo_hard_safety_failure.json", hard_failure)
            checkpoint_record(trainer, rolling)
            break

        scalars = scalar_metrics(metrics)
        completed_kl = scalars.get("completed_update_mean_kl")
        if completed_kl is None or not math.isfinite(completed_kl):
            raise RuntimeError("accepted PPO update lacks finite completed KL")
        row = {
            "accepted_update": trainer.iteration,
            "policy_version": trainer.policy_version,
            "created_utc": utc_now(),
            "learning_rate": optimizer_lr(trainer),
            "rollout_seconds": rollout_seconds,
            "update_seconds": time.monotonic() - update_started,
            "elapsed_seconds": time.monotonic() - started,
            "total_agent_samples": trainer.total_agent_samples,
            "ppo": scalars,
            "rejected_proposals": rejected,
            "curriculum": curriculum_metrics,
            "gameplay": gameplay,
        }
        append_jsonl(curve, row)
        checkpoint_record(trainer, rolling)
        if trainer.iteration in SNAPSHOTS:
            path = CHECKPOINT_DIR / f"rival2_fresh_human_seed_v1_ppo_u{trainer.iteration:04d}.pt"
            record = checkpoint_record(trainer, path)
            manifest["snapshots"] = [
                prior
                for prior in manifest["snapshots"]
                if prior["accepted_update"] != trainer.iteration
            ] + [record]
            manifest["snapshots"].sort(key=lambda item: item["accepted_update"])
            write_json(manifest_path, manifest)
        if trainer.iteration % 10 == 0 or trainer.iteration == 1:
            print(
                json.dumps(
                    {
                        "accepted_update": trainer.iteration,
                        "lr": optimizer_lr(trainer),
                        "kl": completed_kl,
                        "touches_per_minute": gameplay.get("touches_per_minute"),
                        "goals": gameplay.get("goal_events"),
                        "no_touch": gameplay.get("no_touch_truncations"),
                    },
                    sort_keys=True,
                ),
                flush=True,
            )

    final = next(
        (
            row
            for row in manifest["snapshots"]
            if row["accepted_update"] == trainer.iteration
        ),
        None,
    )
    if final is None:
        final = checkpoint_record(
            trainer,
            CHECKPOINT_DIR / f"rival2_fresh_human_seed_v1_ppo_u{trainer.iteration:04d}.pt",
        )
    manifest["final"] = final
    write_json(manifest_path, manifest)
    summary = {
        "format": f"{FORMAT}_SUMMARY",
        "created_utc": utc_now(),
        "source_sha256": sha256_file(SOURCE),
        "authority_sha256": sha256_file(AUTHORITY),
        "accepted_updates": trainer.iteration,
        "exact_600_completed": trainer.iteration == TARGET_UPDATES,
        "total_agent_samples": trainer.total_agent_samples,
        "final_learning_rate": optimizer_lr(trainer),
        "hard_safety_failure": hard_failure,
        "final_checkpoint": final,
        "snapshots_complete": {row["accepted_update"] for row in manifest["snapshots"]}
        == SNAPSHOTS,
        "stop_reason": (
            "hard_safety_guard_at_last_accepted_checkpoint"
            if hard_failure is not None
            else "exact_600_accepted_updates_completed"
        ),
    }
    write_json(RESULTS / "ppo_summary.json", summary)
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 2 if hard_failure is not None else 0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--prepare-authority", action="store_true")
    parser.add_argument("--preflight-only", action="store_true")
    parser.add_argument("--resume")
    parser.add_argument("--worlds", type=int, default=WORLD_COUNT)
    parser.add_argument("--run-dir", default=str(DEFAULT_RUN_DIR))
    parser.add_argument(
        "--collision-root",
        default=os.environ.get("RIVALSIM_COLLISION_DIR", "G:/dev/RLBot-Rival/bot/collision_meshes"),
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.prepare_authority:
        print(json.dumps(prepare_authority(), indent=2, sort_keys=True))
        return 0
    return run(args)


if __name__ == "__main__":
    raise SystemExit(main())
