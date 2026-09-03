"""Train Unified V5 in conservative 120 Hz current-policy self-play.

This campaign intentionally uses the existing Gameplay 120 V2 reward instead
of assigning rewards to the long competency checklist.  Goals, signed
goalward progress, physical controlled possession, saves, boost pickups, and
competitive outcomes provide the learning signal.  Named mechanics remain
unrewarded and are not claimed by this campaign.
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
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from rivalsim.arena import ArenaGeometry, WarpArenaMeshes  # noqa: E402
from rivalsim.rival2_contracts import (  # noqa: E402
    REWARD_GAMEPLAY_120_V2_CONTRACT_HASH,
    RIVAL2_ACTION_V2_120HZ_VERSION,
    RIVAL2_EPISODE_VERSION,
    RIVAL2_OBS_V2_120HZ_VERSION,
    RIVAL2_REWARD_GAMEPLAY_120_V2_VERSION,
    contract_hashes_for_reward,
)
from rivalsim.rival2_env import Rival2Env  # noqa: E402
from rivalsim.rival2_exploration import FreshHumanSeedExploration  # noqa: E402
from rivalsim.rival2_ppo import (  # noqa: E402
    RIVAL2_PPO_120HZ_CONTRACT_HASH,
    RIVAL2_PPO_120HZ_V1,
    rival2_ppo_120hz_config,
)
from rivalsim.rival2_recurrent_ppo import Rival2RecurrentPPOCorruption  # noqa: E402
from rivalsim.rival2_recurrent_training import Rival2RecurrentTrainer  # noqa: E402
from rivalsim.rival2_unified_policy import (  # noqa: E402
    Rival2UnifiedActorCritic,
    Rival2UnifiedPolicyConfig,
)

FORMAT = "RIVAL2_UNIFIED_GROUND_SELFPLAY_PPO_V1"
CHECKPOINT_FORMAT = f"{FORMAT}_CHECKPOINT"
AUTHORITY = ROOT / "results/rival2/unified_ground_selfplay_ppo_v1/authority.json"
RESULTS = ROOT / "results/rival2/unified_ground_selfplay_ppo_v1"
CHECKPOINT = (
    ROOT
    / "checkpoints/rival2/unified_ground_selfplay_ppo_v1"
    / "rival2_unified_ground_selfplay_ppo_v1.pt"
)
SOURCE = (
    ROOT
    / "checkpoints/rival2/unified_capability_distillation_v5"
    / "rival2_unified_capability_v5.pt"
)
SOURCE_SHA256 = "955C93BF538BC913CC2E42F42E3B0EDC4CCDB1065DA9581FB88D84C363B7C216"
DEFAULT_RUN_DIR = Path("G:/dev/RivalSim-runs/unified-ground-selfplay-ppo-v1")
WORLD_COUNT = 32_768
TARGET_UPDATES = 300
SNAPSHOT_INTERVAL = 30
POLICY_LEARNING_RATE = 3.0e-5
CAMPAIGN_SEED = 2026090311

EXPLORATION_VERSION = "RIVAL2_UNIFIED_GROUND_EXPLORATION_V1"
EXPLORATION_CONTRACT = {
    "version": EXPLORATION_VERSION,
    "purpose": "bounded stochastic PPO exploration without overwhelming Unified V5 behavior",
    "analog": {
        "sigma_start": 0.02,
        "sigma_end": 0.04,
        "interpolation": "linear_in_log_sigma",
    },
    "buttons": {
        "temperature_start": 0.10,
        "temperature_end": 0.25,
        "effective_logits": "learned_logits/positive_temperature",
    },
    "schedule": {
        "ramp_start_accepted_update": 30,
        "ramp_end_accepted_update": 150,
        "interior": "smoothstep x*x*(3-2*x)",
    },
    "coherence": [
        "rollout sampling",
        "stored old log probability",
        "PPO recomputation",
        "ratio",
        "entropy telemetry",
        "KL telemetry",
    ],
}
EXPLORATION_CONTRACT_HASH = hashlib.sha256(
    json.dumps(EXPLORATION_CONTRACT, sort_keys=True, separators=(",", ":")).encode(
        "ascii"
    )
).hexdigest().upper()


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


def exploration_for_update(accepted_update: int) -> FreshHumanSeedExploration:
    update = int(accepted_update)
    if update <= 30:
        progress = 0.0
    elif update >= 150:
        progress = 1.0
    else:
        x = (update - 30) / 120.0
        progress = x * x * (3.0 - 2.0 * x)
    log_sigma = math.log(0.02) + (math.log(0.04) - math.log(0.02)) * progress
    sigma = 0.02 if progress == 0.0 else 0.04 if progress == 1.0 else math.exp(log_sigma)
    return FreshHumanSeedExploration(
        accepted_update=update,
        normalized_progress=progress,
        analog_sigma=sigma,
        analog_log_sigma=log_sigma,
        button_temperature=0.10 + 0.15 * progress,
        version=EXPLORATION_VERSION,
        contract_sha256=EXPLORATION_CONTRACT_HASH,
    )


def load_authority() -> dict[str, Any]:
    payload = json.loads(AUTHORITY.read_text(encoding="utf-8"))
    checks = {
        "format": payload.get("format") == f"{FORMAT}_AUTHORITY",
        "source": payload.get("source", {}).get("sha256") == SOURCE_SHA256,
        "reward": payload.get("reward", {}).get("contract_sha256")
        == REWARD_GAMEPLAY_120_V2_CONTRACT_HASH,
        "ppo": payload.get("ppo", {}).get("contract_sha256")
        == RIVAL2_PPO_120HZ_CONTRACT_HASH,
        "ppo_version": payload.get("ppo", {}).get("version")
        == RIVAL2_PPO_120HZ_V1,
        "worlds": payload.get("ppo", {}).get("worlds") == WORLD_COUNT,
        "target": payload.get("campaign", {}).get("accepted_updates")
        == TARGET_UPDATES,
        "exploration": payload.get("exploration", {}).get("contract_sha256")
        == EXPLORATION_CONTRACT_HASH,
        "pure_selfplay": payload.get("opponents", {}).get("current_selfplay") == 1.0,
        "kl_telemetry_only": payload.get("ppo", {}).get("kl_policy")
        == "telemetry_only_no_rejection_or_retry",
    }
    if not all(checks.values()):
        raise RuntimeError(f"unified ground PPO authority mismatch: {checks}")
    return payload


def make_env(collision_root: Path, worlds: int) -> Rival2Env:
    geometry = ArenaGeometry.load_soccar(collision_root)
    meshes = WarpArenaMeshes(geometry)
    return Rival2Env(
        worlds,
        str(collision_root),
        geometry=geometry,
        meshes=meshes,
        device="cuda:0",
        seed=CAMPAIGN_SEED,
        car_visitation_order="a_then_b",
        reward_version=RIVAL2_REWARD_GAMEPLAY_120_V2_VERSION,
        episode_version=RIVAL2_EPISODE_VERSION,
        observation_version=RIVAL2_OBS_V2_120HZ_VERSION,
        action_version=RIVAL2_ACTION_V2_120HZ_VERSION,
    )


def source_payload() -> dict[str, Any]:
    actual = sha256_file(SOURCE)
    if actual != SOURCE_SHA256:
        raise RuntimeError(f"Unified V5 source SHA mismatch: {actual}")
    payload = torch.load(SOURCE, map_location="cpu", weights_only=False)
    if payload.get("format") != "RIVAL2_UNIFIED_CAPABILITY_CHECKPOINT_V5":
        raise RuntimeError("Unified V5 source format mismatch")
    return payload


def initialize_trainer(
    collision_root: Path,
    worlds: int,
) -> tuple[Rival2RecurrentTrainer, dict[str, Any]]:
    authority = load_authority()
    source = source_payload()
    config = Rival2UnifiedPolicyConfig(**source["policy_config"])
    if source.get("policy_config_sha256") != config.content_hash:
        raise RuntimeError("Unified V5 policy configuration hash mismatch")
    model = Rival2UnifiedActorCritic(config)
    model.load_state_dict(source["model"], strict=True)
    source_model_hash = state_dict_sha256(source["model"])
    if state_dict_sha256(model.state_dict()) != source_model_hash:
        raise RuntimeError("Unified V5 model changed during load")
    model.requires_grad_(True)
    ppo = replace(
        rival2_ppo_120hz_config(),
        learning_rate=POLICY_LEARNING_RATE,
    )
    trainer = Rival2RecurrentTrainer(
        make_env(collision_root, worlds),
        policy_config=config,
        ppo_config=ppo,
        phase="unified_ground_selfplay_v1",
        source_identity=authority["source"],
        seed=CAMPAIGN_SEED,
        model=model,
        checkpoint_format=CHECKPOINT_FORMAT,
        lineage="Unified Capability V5 -> Ground Self-Play PPO V1",
    )
    trainer.set_exploration(exploration_for_update(1))
    trainer.phase_transition = {
        "format": f"{FORMAT}_TRANSITION",
        "source_model_tensor_sha256": source_model_hash,
        "loaded_model_tensor_sha256": state_dict_sha256(trainer.model.state_dict()),
        "model_exact_before_first_update": True,
        "source_optimizer_loaded": False,
        "source_rng_loaded": False,
        "fresh_ppo_optimizer": True,
        "fresh_rng_and_counters": True,
        "all_model_parameters_trainable": True,
        "authority_sha256": sha256_file(AUTHORITY),
    }
    return trainer, source


def preflight(
    trainer: Rival2RecurrentTrainer,
    source: dict[str, Any],
    worlds: int,
    *,
    resuming: bool,
) -> dict[str, Any]:
    inventory = trainer.env.world.gameplay_120.memory_inventory()
    checks = {
        "source_sha256_exact": sha256_file(SOURCE) == SOURCE_SHA256,
        "source_model_exact": trainer.phase_transition[
            "source_model_tensor_sha256"
        ]
        == trainer.phase_transition["loaded_model_tensor_sha256"],
        "distillation_optimizer_not_loaded": (
            source.get("optimizer") is not None
            and (
                len(trainer.optimizer.state) > 0
                if resuming
                else len(trainer.optimizer.state) == 0
            )
        ),
        "counter_state_valid": (
            trainer.accepted_updates_total
            == trainer.phase_accepted_updates
            == trainer.policy_version
            and (
                trainer.accepted_updates_total > 0
                if resuming
                else trainer.accepted_updates_total == 0
            )
        ),
        "resume_lineage_valid": (
            trainer.phase_transition is not None
            and trainer.phase_transition.get("source_model_tensor_sha256")
            == trainer.phase_transition.get("loaded_model_tensor_sha256")
            and trainer.phase_transition.get("source_optimizer_loaded") is False
            and trainer.phase_transition.get("fresh_ppo_optimizer") is True
        ),
        "all_parameters_trainable": all(
            parameter.requires_grad for parameter in trainer.model.parameters()
        ),
        "worlds_exact": worlds == WORLD_COUNT,
        "native_120hz": trainer.env.physics_hz == trainer.env.policy_hz == 120,
        "reward_contract_exact": trainer.env.contract_hashes
        == contract_hashes_for_reward(RIVAL2_REWARD_GAMEPLAY_120_V2_VERSION),
        "named_mechanics_hot_path_absent": trainer.env.world.gameplay_v3 is None,
        "named_mechanics_arrays_absent": inventory["named_mechanics_arrays"] == 0,
        "controlled_flick_arrays_absent": inventory["controlled_flick_arrays"] == 0,
        "sequence_ppo": True,
        "kl_telemetry_only": True,
        "nexto_wisp_historical_training_absent": True,
    }
    return {
        "format": f"{FORMAT}_PREFLIGHT",
        "created_utc": utc_now(),
        "resuming": resuming,
        "verdict": "PASS" if all(checks.values()) else "FAIL",
        "checks": checks,
        "source": authority_source(source),
        "transition": trainer.phase_transition,
        "contracts": dict(trainer.env.contract_hashes),
        "ppo_config": asdict(trainer.ppo_config),
        "ppo_config_sha256": trainer.ppo_config.content_hash,
        "exploration": EXPLORATION_CONTRACT,
        "exploration_contract_sha256": EXPLORATION_CONTRACT_HASH,
        "gameplay_120_memory_inventory": inventory,
    }


def authority_source(source: dict[str, Any]) -> dict[str, Any]:
    return {
        "path": SOURCE.relative_to(ROOT).as_posix(),
        "sha256": SOURCE_SHA256,
        "format": source.get("format"),
        "model_tensor_sha256": state_dict_sha256(source["model"]),
        "used_fields": ["model", "policy_config"],
    }


def checkpoint_record(
    trainer: Rival2RecurrentTrainer,
    path: Path,
    *,
    include_optimizer: bool,
) -> dict[str, Any]:
    trainer.save_checkpoint(path, include_optimizer=include_optimizer)
    return {
        "accepted_update": trainer.accepted_updates_total,
        "path": path.relative_to(ROOT).as_posix() if path.is_relative_to(ROOT) else str(path),
        "sha256": sha256_file(path),
        "bytes": path.stat().st_size,
        "optimizer_included": include_optimizer,
        "total_agent_samples": trainer.total_agent_samples,
        "exploration": trainer.exploration.as_dict(),
    }


def run(args: argparse.Namespace) -> int:
    run_dir = Path(args.run_dir).resolve()
    collision_root = Path(args.collision_root).resolve()
    results_curve = RESULTS / "training_curve.jsonl"
    manifest_path = RESULTS / "snapshot_manifest.json"
    run_dir.mkdir(parents=True, exist_ok=True)
    trainer, source = initialize_trainer(collision_root, args.worlds)
    if args.resume:
        trainer.load_checkpoint(Path(args.resume).resolve())
    preflight_payload = preflight(
        trainer,
        source,
        args.worlds,
        resuming=bool(args.resume),
    )
    write_json(
        RESULTS / ("resume_preflight.json" if args.resume else "preflight.json"),
        preflight_payload,
    )
    if preflight_payload["verdict"] != "PASS":
        raise RuntimeError(f"unified ground PPO preflight failed: {preflight_payload}")
    if args.preflight_only:
        print(json.dumps(preflight_payload, indent=2, sort_keys=True))
        return 0
    if results_curve.exists() and not args.resume:
        raise RuntimeError("training evidence exists; use --resume explicitly")

    manifest = (
        json.loads(manifest_path.read_text(encoding="utf-8"))
        if manifest_path.is_file()
        else {
            "format": f"{FORMAT}_SNAPSHOT_MANIFEST",
            "source_sha256": SOURCE_SHA256,
            "snapshots": [],
        }
    )
    rolling = run_dir / "rolling.pt"
    hard_failure: dict[str, Any] | None = None
    started = time.monotonic()
    first_rollout = trainer.accepted_updates_total == 0
    target = int(args.target_updates)
    if target != TARGET_UPDATES:
        raise ValueError(f"authority freezes exactly {TARGET_UPDATES} accepted updates")

    while trainer.accepted_updates_total < target:
        trainer.set_exploration(
            exploration_for_update(trainer.accepted_updates_total + 1)
        )
        rollout_started = time.monotonic()
        rollout = trainer.collect_rollout()
        rollout_seconds = time.monotonic() - rollout_started
        gameplay = dict(trainer.last_rollout_metrics)
        if first_rollout:
            write_json(
                RESULTS / "root_rollout_baseline.json",
                {
                    "format": f"{FORMAT}_ROOT_ROLLOUT_BASELINE",
                    "created_utc": utc_now(),
                    "source": authority_source(source),
                    "gameplay": gameplay,
                },
            )
            first_rollout = False
        update_started = time.monotonic()
        try:
            metrics = trainer.update(rollout)
        except Rival2RecurrentPPOCorruption as error:
            hard_failure = {
                "created_utc": utc_now(),
                "accepted_update": trainer.accepted_updates_total,
                "diagnostics": dict(error.diagnostics),
            }
            write_json(RESULTS / "hard_failure.json", hard_failure)
            checkpoint_record(trainer, rolling, include_optimizer=True)
            break
        ppo = scalar_metrics(metrics)
        row = {
            "accepted_update": trainer.accepted_updates_total,
            "created_utc": utc_now(),
            "elapsed_seconds": time.monotonic() - started,
            "rollout_seconds": rollout_seconds,
            "update_seconds": time.monotonic() - update_started,
            "total_agent_samples": trainer.total_agent_samples,
            "exploration": trainer.exploration.as_dict(),
            "ppo": ppo,
            "gameplay": gameplay,
        }
        append_jsonl(results_curve, row)
        checkpoint_record(trainer, rolling, include_optimizer=True)
        if trainer.accepted_updates_total % SNAPSHOT_INTERVAL == 0:
            record = checkpoint_record(
                trainer,
                run_dir / "snapshots" / f"unified_ground_u{trainer.accepted_updates_total:04d}.pt",
                include_optimizer=True,
            )
            manifest["snapshots"] = [
                prior
                for prior in manifest["snapshots"]
                if prior["accepted_update"] != trainer.accepted_updates_total
            ] + [record]
            manifest["snapshots"].sort(key=lambda item: item["accepted_update"])
            write_json(manifest_path, manifest)
        print(
            json.dumps(
                {
                    "update": trainer.accepted_updates_total,
                    "kl": ppo.get("completed_update_mean_kl"),
                    "touches_per_minute": gameplay["touches_per_minute"],
                    "goalward_touch_fraction": gameplay["goalward_touch_fraction"],
                    "control_mean": gameplay["control_score"]["mean"],
                    "control_ge_025": gameplay["control_score"]["ge_025_fraction"],
                    "goals": gameplay["goal_events"],
                    "no_touch": gameplay["no_touch_truncations"],
                },
                sort_keys=True,
            ),
            flush=True,
        )

    final_record = checkpoint_record(trainer, CHECKPOINT, include_optimizer=True)
    manifest["final"] = final_record
    write_json(manifest_path, manifest)
    summary = {
        "format": f"{FORMAT}_TRAINING_SUMMARY",
        "created_utc": utc_now(),
        "verdict": "BLOCKED" if hard_failure is not None else "PASS",
        "accepted_updates": trainer.accepted_updates_total,
        "target_updates": TARGET_UPDATES,
        "source": authority_source(source),
        "reward": {
            "version": RIVAL2_REWARD_GAMEPLAY_120_V2_VERSION,
            "contract_sha256": REWARD_GAMEPLAY_120_V2_CONTRACT_HASH,
        },
        "final_checkpoint": final_record,
        "hard_failure": hard_failure,
        "stop_reason": (
            "nonfinite_or_corruption_guard" if hard_failure else "accepted_update_target_reached"
        ),
    }
    write_json(RESULTS / "training_summary.json", summary)
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 2 if hard_failure else 0


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser()
    result.add_argument(
        "--collision-root",
        default=os.environ.get(
            "RIVALSIM_COLLISION_DIR", "G:/dev/RLBot-Rival/bot/collision_meshes"
        ),
    )
    result.add_argument("--worlds", type=int, default=WORLD_COUNT)
    result.add_argument("--target-updates", type=int, default=TARGET_UPDATES)
    result.add_argument("--run-dir", default=str(DEFAULT_RUN_DIR))
    result.add_argument("--resume")
    result.add_argument("--preflight-only", action="store_true")
    return result


if __name__ == "__main__":
    raise SystemExit(run(parser().parse_args()))
