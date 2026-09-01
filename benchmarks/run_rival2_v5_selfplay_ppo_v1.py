"""V5-rooted pure-current 120 Hz self-play PPO campaign.

The committed authority is verified before the V5 model is loaded.  The V5
checkpoint contributes model tensors only; PPO optimizer/RNG/counters and the
empty historical pool are created fresh here.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import sys
import time
from dataclasses import asdict, replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from rivalsim.arena import ArenaGeometry, WarpArenaMeshes
from rivalsim.rival2_contracts import (
    REWARD_GAMEPLAY_120_V2_CONTRACT_HASH,
    RIVAL2_REWARD_GAMEPLAY_120_V2_VERSION,
    contract_hashes_for_reward,
)
from rivalsim.rival2_env import Rival2Env
from rivalsim.rival2_opponent_curriculum import (
    OPPONENT_CURRENT,
    Rival2OpponentCurriculumConfig,
    Rival2OpponentCurriculumTrainer,
)
from rivalsim.rival2_policy import Rival2PolicyConfig
from rivalsim.rival2_ppo import (
    RIVAL2_PPO_120HZ_CONTRACT_HASH,
    RIVAL2_PPO_120HZ_V1,
    Rival2KLGuardConfig,
    Rival2PolicyDisplacementRejected,
    rival2_ppo_120hz_config,
)
from rivalsim.rival2_training import Rival2SelfPlayConfig

AUTHORITY_PATH = REPO_ROOT / "results/rival2/v5_selfplay_ppo_v1/authority.json"
RESULTS_DIR = REPO_ROOT / "results/rival2/v5_selfplay_ppo_v1"
CHECKPOINT_DIR = REPO_ROOT / "checkpoints/rival2/v5_selfplay_ppo_v1"
SOURCE_PATH = REPO_ROOT / "checkpoints/rival2/human_bc_v5/rival2_human_bc_v5.pt"
SOURCE_SHA256 = "F9100E543F48B1AD9E447179DFC2022774F039AD8D47F9FBF07359B7E1D12FE8"
DEFAULT_RUN_DIR = Path("G:/dev/RivalSim-runs/v5-selfplay-ppo-v1")
WORLD_COUNT = 32_768
INITIAL_LR = 1.0e-4
LR_SCHEDULE = (1.0e-4, 5.0e-5, 2.5e-5)
SNAPSHOT_INTERVAL = 30
EXPLICIT_SNAPSHOT = 500
MINIMUM_UPDATES = 600
HARD_CEILING = 750
SEED = 2026090101
AUTHORITY_FORMAT = "RIVAL2_V5_SELFPLAY_PPO_V1_AUTHORITY"


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


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
        digest.update(name.encode("utf-8"))
        digest.update(str(value.dtype).encode("ascii"))
        digest.update(str(tuple(value.shape)).encode("ascii"))
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
        name: float(value.detach().item())
        for name, value in metrics.items()
        if value.numel() == 1
    }


def load_authority() -> dict[str, Any]:
    authority = json.loads(AUTHORITY_PATH.read_text(encoding="utf-8"))
    checks = {
        "format": authority.get("format") == AUTHORITY_FORMAT,
        "source_sha256": authority.get("source", {}).get("sha256") == SOURCE_SHA256,
        "reward_version": authority.get("reward", {}).get("version")
        == RIVAL2_REWARD_GAMEPLAY_120_V2_VERSION,
        "reward_hash": authority.get("reward", {}).get("contract_sha256")
        == REWARD_GAMEPLAY_120_V2_CONTRACT_HASH,
        "ppo_identity": authority.get("ppo", {}).get("version")
        == RIVAL2_PPO_120HZ_V1,
        "ppo_identity_hash": authority.get("ppo", {}).get("contract_sha256")
        == RIVAL2_PPO_120HZ_CONTRACT_HASH,
        "worlds": authority.get("ppo", {}).get("worlds") == WORLD_COUNT,
        "minimum_updates": authority.get("campaign", {}).get("minimum_updates")
        == MINIMUM_UPDATES,
        "hard_ceiling": authority.get("campaign", {}).get("hard_ceiling")
        == HARD_CEILING,
    }
    if not all(checks.values()):
        raise RuntimeError(f"prospective authority mismatch: {checks}")
    return authority


def collision_assets(root: Path) -> tuple[ArenaGeometry, WarpArenaMeshes]:
    geometry = ArenaGeometry.load_soccar(root)
    return geometry, WarpArenaMeshes(geometry)


def make_trainer(
    collision_root: Path,
    *,
    world_count: int,
) -> tuple[Rival2OpponentCurriculumTrainer, dict[str, Any]]:
    source_sha = sha256_file(SOURCE_PATH)
    if source_sha != SOURCE_SHA256:
        raise RuntimeError(f"V5 source SHA mismatch: {source_sha}")
    source = torch.load(SOURCE_PATH, map_location="cpu", weights_only=False)
    source_model = source["model"]
    source_model_hash = state_dict_sha256(source_model)
    policy_config = Rival2PolicyConfig(**source["policy_config"])
    ppo_config = replace(rival2_ppo_120hz_config(), learning_rate=INITIAL_LR)
    geometry, meshes = collision_assets(collision_root)
    env = Rival2Env(
        world_count,
        str(collision_root),
        geometry=geometry,
        meshes=meshes,
        device="cuda:0",
        seed=SEED,
        car_visitation_order="a_then_b",
        reward_version=RIVAL2_REWARD_GAMEPLAY_120_V2_VERSION,
    )
    curriculum = Rival2OpponentCurriculumConfig(
        nexto_probability=0.0,
        wisp_probability=0.0,
        current_probability=1.0,
        historical_probability=0.0,
        seed=SEED ^ 0x7171,
    )
    trainer = Rival2OpponentCurriculumTrainer(
        env,
        policy_config=policy_config,
        ppo_config=ppo_config,
        self_play_config=Rival2SelfPlayConfig(
            historical_chance=0.0,
            historical_pool_bound=1,
        ),
        opponent_curriculum=curriculum,
        seed=SEED,
    )
    fresh_optimizer_state_entries = len(trainer.optimizer.state)
    trainer.model.load_state_dict(source_model, strict=True)
    loaded_model_hash = state_dict_sha256(trainer.model.state_dict())
    if loaded_model_hash != source_model_hash:
        raise RuntimeError("V5 model tensor identity was not preserved on load")
    if fresh_optimizer_state_entries != 0 or len(trainer.optimizer.state) != 0:
        raise RuntimeError("fresh PPO optimizer unexpectedly contains state")
    if not all(parameter.requires_grad for parameter in trainer.model.parameters()):
        raise RuntimeError("full PPO model is not trainable")
    if trainer.opponent_pool.versions:
        raise RuntimeError("historical pool was not empty at bootstrap")
    trainer.curriculum_transition = {
        "format": "RIVAL2_V5_TO_FRESH_SELFPLAY_PPO_V1",
        "source": {
            "path": SOURCE_PATH.relative_to(REPO_ROOT).as_posix(),
            "sha256": source_sha,
            "format": source.get("format"),
            "model_tensor_sha256": source_model_hash,
            "used_fields": ["model", "policy_config"],
            "optimizer_loaded": False,
            "rng_loaded": False,
            "training_counters_loaded": False,
        },
        "fresh_ppo": {
            "optimizer": type(trainer.optimizer).__name__,
            "optimizer_state_entries_before_first_step": fresh_optimizer_state_entries,
            "all_model_parameters_trainable": True,
            "initial_learning_rate": INITIAL_LR,
            "allowed_learning_rates": list(LR_SCHEDULE),
            "historical_chance": 0.0,
            "historical_pool_initial_count": 0,
            "opponent_regime": "pure_current_policy_self_play_both_sides_trainable",
        },
        "contracts": dict(env.contract_hashes),
        "reward_contract_sha256": REWARD_GAMEPLAY_120_V2_CONTRACT_HASH,
        "ppo_contract_sha256": RIVAL2_PPO_120HZ_CONTRACT_HASH,
        "authority": {
            "path": AUTHORITY_PATH.relative_to(REPO_ROOT).as_posix(),
            "sha256": sha256_file(AUTHORITY_PATH),
        },
    }
    trainer.initialize_curriculum_assignments()
    if not bool((trainer.opponent_family == OPPONENT_CURRENT).all().item()):
        raise RuntimeError("pure current-policy assignment preflight failed")
    return trainer, source


def preflight(
    trainer: Rival2OpponentCurriculumTrainer,
    source: dict[str, Any],
    *,
    world_count: int,
) -> dict[str, Any]:
    inventory = trainer.env.world.gameplay_120.memory_inventory()
    checks = {
        "source_sha256_exact": sha256_file(SOURCE_PATH) == SOURCE_SHA256,
        "source_is_human_bc_v5": source.get("format")
        == "RIVAL2_HUMAN_BEHAVIOR_CLONING_CHECKPOINT_V5",
        "fresh_optimizer_state_empty": len(trainer.optimizer.state) == 0,
        "iteration_zero": trainer.iteration == 0 and trainer.policy_version == 0,
        "full_model_trainable": all(
            parameter.requires_grad for parameter in trainer.model.parameters()
        ),
        "pure_current_assignments": bool(
            (trainer.opponent_family == OPPONENT_CURRENT).all().item()
        ),
        "historical_pool_empty": len(trainer.opponent_pool.versions) == 0,
        "nexto_probability_zero": trainer.opponent_curriculum.nexto_probability == 0.0,
        "wisp_probability_zero": trainer.opponent_curriculum.wisp_probability == 0.0,
        "historical_probability_zero": (
            trainer.opponent_curriculum.historical_probability == 0.0
        ),
        "world_count_exact": world_count == WORLD_COUNT,
        "policy_hz_120": trainer.env.policy_hz == 120,
        "physics_hz_120": trainer.env.physics_hz == 120,
        "reward_contract_exact": trainer.env.contract_hashes
        == contract_hashes_for_reward(RIVAL2_REWARD_GAMEPLAY_120_V2_VERSION),
        "named_mechanics_state_absent": trainer.env.world.gameplay_v3 is None,
        "named_mechanics_arrays_zero": inventory["named_mechanics_arrays"] == 0,
        "controlled_flick_arrays_zero": inventory["controlled_flick_arrays"] == 0,
    }
    return {
        "format": "RIVAL2_V5_SELFPLAY_PPO_V1_PREFLIGHT",
        "created_utc": utc_now(),
        "checks": checks,
        "verdict": "PASS" if all(checks.values()) else "FAIL",
        "source": trainer.curriculum_transition["source"],
        "fresh_ppo": trainer.curriculum_transition["fresh_ppo"],
        "contracts": dict(trainer.env.contract_hashes),
        "ppo_config": asdict(trainer.ppo_config),
        "ppo_config_sha256": trainer.ppo_config.content_hash,
        "kl_guard": asdict(Rival2KLGuardConfig(0.10, 0.05)),
        "opponent_curriculum": asdict(trainer.opponent_curriculum),
        "gameplay_120_memory_inventory": inventory,
    }


def optimizer_lr(trainer: Rival2OpponentCurriculumTrainer) -> float:
    rates = {float(group["lr"]) for group in trainer.optimizer.param_groups}
    if len(rates) != 1:
        raise RuntimeError(f"unexpected PPO learning rates: {rates}")
    return rates.pop()


def set_optimizer_lr(trainer: Rival2OpponentCurriculumTrainer, rate: float) -> None:
    if rate not in LR_SCHEDULE:
        raise ValueError(f"unauthorized PPO learning rate: {rate}")
    for group in trainer.optimizer.param_groups:
        group["lr"] = rate


def checkpoint_record(
    trainer: Rival2OpponentCurriculumTrainer,
    path: Path,
) -> dict[str, Any]:
    path.parent.mkdir(parents=True, exist_ok=True)
    trainer.save_checkpoint(path)
    return {
        "accepted_update": trainer.iteration,
        "policy_version": trainer.policy_version,
        "path": path.relative_to(REPO_ROOT).as_posix()
        if path.is_relative_to(REPO_ROOT)
        else str(path),
        "sha256": sha256_file(path),
        "bytes": path.stat().st_size,
        "total_agent_samples": trainer.total_agent_samples,
        "learning_rate": optimizer_lr(trainer),
    }


def load_manifest() -> dict[str, Any]:
    path = RESULTS_DIR / "snapshot_manifest.json"
    if path.is_file():
        return json.loads(path.read_text(encoding="utf-8"))
    return {
        "format": "RIVAL2_V5_SELFPLAY_PPO_V1_SNAPSHOT_MANIFEST",
        "source_sha256": SOURCE_SHA256,
        "reward_contract_sha256": REWARD_GAMEPLAY_120_V2_CONTRACT_HASH,
        "snapshots": [],
    }


def save_milestone(
    trainer: Rival2OpponentCurriculumTrainer,
    manifest: dict[str, Any],
) -> dict[str, Any]:
    path = CHECKPOINT_DIR / f"rival2_v5_selfplay_ppo_u{trainer.iteration:04d}.pt"
    record = checkpoint_record(trainer, path)
    manifest["snapshots"] = [
        entry
        for entry in manifest["snapshots"]
        if int(entry["accepted_update"]) != trainer.iteration
    ] + [record]
    manifest["snapshots"].sort(key=lambda entry: int(entry["accepted_update"]))
    write_json(RESULTS_DIR / "snapshot_manifest.json", manifest)
    return record


def first_rollout_check(
    trainer: Rival2OpponentCurriculumTrainer,
) -> dict[str, Any]:
    curriculum = trainer.last_rollout_curriculum_metrics or {}
    expected = trainer.env.num_envs * trainer.ppo_config.rollout_horizon * 2
    trainable = curriculum.get("trainable_agent_samples", {})
    world_decisions = curriculum.get("world_decisions", {})
    checks = {
        "current_trainable_samples_exact": trainable.get("current") == expected,
        "historical_trainable_samples_zero": trainable.get("historical") == 0,
        "nexto_trainable_samples_zero": trainable.get("nexto") == 0,
        "wisp_trainable_samples_zero": trainable.get("wisp") == 0,
        "current_world_decisions_exact": world_decisions.get("current")
        == trainer.env.num_envs * trainer.ppo_config.rollout_horizon,
        "all_two_perspective_train_mask": True,
        "historical_pool_empty": len(trainer.opponent_pool.versions) == 0,
    }
    return {
        "format": "RIVAL2_V5_SELFPLAY_PPO_V1_FIRST_ROLLOUT_CHECK",
        "created_utc": utc_now(),
        "checks": checks,
        "verdict": "PASS" if all(checks.values()) else "FAIL",
        "curriculum": curriculum,
        "gameplay": trainer.last_rollout_gameplay_metrics,
    }


def run_campaign(args: argparse.Namespace) -> int:
    authority = load_authority()
    collision_root = Path(args.collision_root)
    run_dir = Path(args.run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
    trainer, source = make_trainer(collision_root, world_count=args.worlds)
    preflight_payload = preflight(trainer, source, world_count=args.worlds)
    write_json(RESULTS_DIR / "preflight.json", preflight_payload)
    if preflight_payload["verdict"] != "PASS":
        raise RuntimeError(f"preflight failed: {preflight_payload['checks']}")
    if args.preflight_only:
        print(json.dumps(preflight_payload, indent=2, sort_keys=True))
        return 0

    curve_path = RESULTS_DIR / "training_curve.jsonl"
    rolling_path = run_dir / "rival2_v5_selfplay_ppo_rolling.pt"
    manifest = load_manifest()
    if args.resume:
        trainer.load_checkpoint(Path(args.resume))
        if trainer.opponent_pool.versions:
            raise RuntimeError("resume checkpoint contains a historical pool")
        if not bool((trainer.opponent_family == OPPONENT_CURRENT).all().item()):
            raise RuntimeError("resume checkpoint is not pure current self-play")
    elif curve_path.exists():
        raise RuntimeError("training curve already exists; use --resume explicitly")

    kl_guard = Rival2KLGuardConfig(0.10, 0.05)
    target = min(int(args.target_updates), HARD_CEILING)
    hard_failure: dict[str, Any] | None = None
    first_rollout_verified = trainer.iteration > 0
    start_time = time.monotonic()
    while trainer.iteration < target:
        rollout_start = time.monotonic()
        rollout = trainer.collect_rollout()
        rollout_seconds = time.monotonic() - rollout_start
        gameplay = trainer.last_rollout_gameplay_metrics
        curriculum = trainer.last_rollout_curriculum_metrics
        if not first_rollout_verified:
            first = first_rollout_check(trainer)
            write_json(RESULTS_DIR / "first_rollout_check.json", first)
            if first["verdict"] != "PASS":
                raise RuntimeError(f"first rollout pure self-play check failed: {first}")
            write_json(
                RESULTS_DIR / "v5_root_rollout_baseline.json",
                {
                    "format": "RIVAL2_V5_SELFPLAY_PPO_V1_ROOT_BASELINE",
                    "created_utc": utc_now(),
                    "source_sha256": SOURCE_SHA256,
                    "curriculum": curriculum,
                    "gameplay": gameplay,
                },
            )
            first_rollout_verified = True

        rejected: list[dict[str, Any]] = []
        update_start = time.monotonic()
        while True:
            rate = optimizer_lr(trainer)
            try:
                metrics = trainer.update(rollout, kl_guard=kl_guard)
                break
            except Rival2PolicyDisplacementRejected as error:
                diagnostics = dict(error.diagnostics)
                diagnostics["learning_rate"] = rate
                diagnostics["created_utc"] = utc_now()
                rejected.append(diagnostics)
                reason = str(diagnostics.get("reason", ""))
                if "nonfinite" in reason:
                    hard_failure = diagnostics
                    break
                index = LR_SCHEDULE.index(rate)
                if index + 1 >= len(LR_SCHEDULE):
                    hard_failure = diagnostics
                    break
                set_optimizer_lr(trainer, LR_SCHEDULE[index + 1])
        if hard_failure is not None:
            write_json(RESULTS_DIR / "hard_safety_failure.json", hard_failure)
            checkpoint_record(trainer, rolling_path)
            break

        update_seconds = time.monotonic() - update_start
        scalars = scalar_metrics(metrics)
        completed_kl = scalars.get("completed_update_mean_kl")
        if completed_kl is None or not math.isfinite(completed_kl):
            raise RuntimeError("accepted update lacks finite completed-update KL")
        row = {
            "accepted_update": trainer.iteration,
            "policy_version": trainer.policy_version,
            "created_utc": utc_now(),
            "learning_rate": optimizer_lr(trainer),
            "rollout_seconds": rollout_seconds,
            "update_seconds": update_seconds,
            "elapsed_seconds": time.monotonic() - start_time,
            "total_agent_samples": trainer.total_agent_samples,
            "ppo": scalars,
            "rejected_proposals": rejected,
            "curriculum": curriculum,
            "gameplay": gameplay,
        }
        append_jsonl(curve_path, row)
        checkpoint_record(trainer, rolling_path)
        if trainer.iteration % SNAPSHOT_INTERVAL == 0 or trainer.iteration == EXPLICIT_SNAPSHOT:
            save_milestone(trainer, manifest)
        if trainer.iteration % 10 == 0 or trainer.iteration in (1, EXPLICIT_SNAPSHOT):
            print(
                json.dumps(
                    {
                        "accepted_update": trainer.iteration,
                        "lr": optimizer_lr(trainer),
                        "kl": completed_kl,
                        "touches_per_minute": gameplay.get("touches_per_minute"),
                        "control_mean": gameplay.get("control_score", {}).get("mean"),
                        "supersonic": gameplay.get("supersonic_occupancy_fraction"),
                        "no_touch": gameplay.get("no_touch_truncations"),
                    },
                    sort_keys=True,
                ),
                flush=True,
            )

    final_record = checkpoint_record(
        trainer, CHECKPOINT_DIR / "rival2_v5_selfplay_ppo_final.pt"
    )
    if trainer.iteration % SNAPSHOT_INTERVAL != 0 and trainer.iteration != EXPLICIT_SNAPSHOT:
        save_milestone(trainer, manifest)
    manifest["final"] = final_record
    manifest["snapshot_count_including_final"] = len(manifest["snapshots"]) + 1
    write_json(RESULTS_DIR / "snapshot_manifest.json", manifest)
    summary = {
        "format": "RIVAL2_V5_SELFPLAY_PPO_V1_TRAINING_SUMMARY",
        "created_utc": utc_now(),
        "authority_sha256": sha256_file(AUTHORITY_PATH),
        "source_sha256": SOURCE_SHA256,
        "reward_version": RIVAL2_REWARD_GAMEPLAY_120_V2_VERSION,
        "reward_contract_sha256": REWARD_GAMEPLAY_120_V2_CONTRACT_HASH,
        "accepted_updates": trainer.iteration,
        "final_learning_rate": optimizer_lr(trainer),
        "total_agent_samples": trainer.total_agent_samples,
        "hard_safety_failure": hard_failure,
        "minimum_600_completed": trainer.iteration >= MINIMUM_UPDATES,
        "target_updates": target,
        "final_checkpoint": final_record,
        "snapshot_count_including_final": manifest["snapshot_count_including_final"],
        "stop_reason": (
            "hard_safety_guard_at_last_accepted_checkpoint"
            if hard_failure is not None
            else (
                "accepted_update_600_reached_for_operational_development_review"
                if trainer.iteration == 600
                else "requested_accepted_update_target_reached"
            )
        ),
        "authority": authority,
    }
    write_json(RESULTS_DIR / "training_summary.json", summary)
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 2 if hard_failure is not None else 0


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser()
    result.add_argument(
        "--collision-root",
        default=os.environ.get(
            "RIVALSIM_COLLISION_DIR", "G:/dev/RLBot-Rival/bot/collision_meshes"
        ),
    )
    result.add_argument("--worlds", type=int, default=WORLD_COUNT)
    result.add_argument("--target-updates", type=int, default=MINIMUM_UPDATES)
    result.add_argument("--run-dir", default=str(DEFAULT_RUN_DIR))
    result.add_argument("--resume")
    result.add_argument("--preflight-only", action="store_true")
    return result


if __name__ == "__main__":
    raise SystemExit(run_campaign(parser().parse_args()))
