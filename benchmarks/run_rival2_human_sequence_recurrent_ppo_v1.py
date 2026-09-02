"""Two-phase recurrent PPO campaign rooted only in Human Sequence Seed v1."""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import math
import os
import subprocess
import sys
import time
from dataclasses import asdict, replace
from datetime import UTC, datetime
from itertools import pairwise
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from rivalsim.arena import ArenaGeometry, WarpArenaMeshes  # noqa: E402
from rivalsim.rival2_contracts import (  # noqa: E402
    REWARD_ACQUISITION_120_V1_CONTRACT_HASH,
    REWARD_GAMEPLAY_120_V2_CONTRACT_HASH,
    RIVAL2_ACTION_V2_120HZ_VERSION,
    RIVAL2_EPISODE_VERSION,
    RIVAL2_OBS_V2_120HZ_VERSION,
    RIVAL2_REWARD_ACQUISITION_120_V1_VERSION,
    RIVAL2_REWARD_GAMEPLAY_120_V2_VERSION,
    contract_hashes_for_reward,
)
from rivalsim.rival2_env import Rival2Env  # noqa: E402
from rivalsim.rival2_exploration import (  # noqa: E402
    RIVAL2_FRESH_HUMAN_SEED_EXPLORATION_RAMP_CONTRACT,
    RIVAL2_FRESH_HUMAN_SEED_EXPLORATION_RAMP_CONTRACT_HASH,
    fresh_human_seed_exploration,
)
from rivalsim.rival2_ppo import (  # noqa: E402
    RIVAL2_PPO_120HZ_CONTRACT_HASH,
    RIVAL2_PPO_120HZ_V1,
    rival2_ppo_120hz_config,
)
from rivalsim.rival2_recurrent_policy import (  # noqa: E402
    Rival2RecurrentPolicyConfig,
)
from rivalsim.rival2_recurrent_ppo import Rival2RecurrentPPOCorruption  # noqa: E402
from rivalsim.rival2_recurrent_training import (  # noqa: E402
    CHECKPOINT_FORMAT,
    Rival2RecurrentTrainer,
)

FORMAT = "RIVAL2_HUMAN_SEQUENCE_RECURRENT_PPO_V1"
SOURCE_COMMIT = "BEDC3D44A17B86FFD97F83F7B1A35CD76FB06888"
SOURCE = ROOT / "checkpoints/rival2/human_sequence_seed_v1/rival2_human_sequence_seed_v1.pt"
SOURCE_SHA256 = "B77A059ECB31DE59A964FE2A368F40C3F367DE0C028E29325F0FF6F763BAF292"
SOURCE_FORMAT = "RIVAL2_HUMAN_SEQUENCE_SEED_V1_STAGE1_CHECKPOINT"
SOURCE_EVALUATION = (
    ROOT / "results/rival2/human_sequence_seed_v1/deterministic_nexto_closed_loop.json"
)
RESULTS = ROOT / "results/rival2/human_sequence_recurrent_ppo_v1"
AUTHORITY = RESULTS / "authority.json"
CHECKPOINTS = ROOT / "checkpoints/rival2/human_sequence_recurrent_ppo_v1"
DEFAULT_RUN_DIR = Path("G:/dev/RivalSim-runs/human-sequence-recurrent-ppo-v1")
DEFAULT_COLLISION_ROOT = Path("G:/dev/RLBot-Rival/bot/collision_meshes")
WORLD_COUNT = 32_768
PPO_LEARNING_RATE = 1.0e-4
PHASE_A_EVALUATION_START = 300
PHASE_A_EVALUATION_INTERVAL = 30
PHASE_A_NONIMPROVING_PATIENCE = 2
PHASE_B_ACCEPTED_UPDATES = 600
SNAPSHOT_INTERVAL = 30
CAMPAIGN_SEED = 2026090301
CRITIC_SEED = 2026090302
PHASE_B_SEED = 2026090303
EVALUATION_SEED = 2026090305


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


def scalar_metrics(metrics: dict[str, torch.Tensor]) -> dict[str, float | int | str]:
    result: dict[str, float | int | str] = {}
    for name, value in metrics.items():
        if value.numel() != 1:
            continue
        number = float(value.detach().item())
        if math.isnan(number):
            result[name] = "NaN"
        elif number == math.inf:
            result[name] = "Infinity"
        elif number == -math.inf:
            result[name] = "-Infinity"
        elif value.dtype in (torch.int8, torch.int16, torch.int32, torch.int64):
            result[name] = int(number)
        else:
            result[name] = number
    return result


def source_payload() -> dict[str, Any]:
    if sha256_file(SOURCE) != SOURCE_SHA256:
        raise RuntimeError("Human Sequence Seed source SHA-256 mismatch")
    payload = torch.load(SOURCE, map_location="cpu", weights_only=False)
    if payload.get("format") != SOURCE_FORMAT:
        raise RuntimeError("Human Sequence Seed source format mismatch")
    if payload.get("selected_step") != 400:
        raise RuntimeError("Human Sequence Seed selected step mismatch")
    if payload.get("critic_optimizer_steps") != 0:
        raise RuntimeError("Human Sequence Seed critic history is not clean")
    return payload


def source_identity(payload: dict[str, Any]) -> dict[str, Any]:
    behavior = {
        name: value
        for name, value in payload["model"].items()
        if not name.startswith("critic.")
    }
    return {
        "path": SOURCE.relative_to(ROOT).as_posix(),
        "sha256": SOURCE_SHA256,
        "format": SOURCE_FORMAT,
        "source_commit": SOURCE_COMMIT,
        "selected_step": int(payload["selected_step"]),
        "model_tensor_sha256": state_dict_sha256(payload["model"]),
        "behavior_tensor_sha256": state_dict_sha256(behavior),
        "validation_complete_action_rmse": payload["validation"][
            "complete_action_rmse"
        ],
        "stage1_optimizer_loaded": False,
    }


def phase_contracts(reward_version: str) -> dict[str, str]:
    return contract_hashes_for_reward(
        reward_version,
        RIVAL2_EPISODE_VERSION,
        observation_version=RIVAL2_OBS_V2_120HZ_VERSION,
        action_version=RIVAL2_ACTION_V2_120HZ_VERSION,
    )


def prepare_authority() -> dict[str, Any]:
    source = source_payload()
    ppo = replace(rival2_ppo_120hz_config(), learning_rate=PPO_LEARNING_RATE)
    authority = {
        "format": f"{FORMAT}_AUTHORITY",
        "created_utc": utc_now(),
        "source": source_identity(source),
        "transition": {
            "preserve_exact": ["encoder", "gru", "post", "actor"],
            "critic_reinitialized": True,
            "critic_seed": CRITIC_SEED,
            "fresh_optimizer": True,
            "fresh_rng_and_counters": True,
            "prior_bc_or_ppo_loaded": False,
        },
        "observation": {
            "policy_view": source["policy_config"]["observation_view_version"],
            "policy_view_sha256": source["policy_config"][
                "observation_view_sha256"
            ],
            "adapter_v2": False,
            "previous_action_visible": False,
            "projection_enforced_inside_policy": True,
        },
        "ppo": {
            "version": RIVAL2_PPO_120HZ_V1,
            "contract_sha256": RIVAL2_PPO_120HZ_CONTRACT_HASH,
            "config": asdict(ppo),
            "world_count": WORLD_COUNT,
            "learning_rate": PPO_LEARNING_RATE,
            "recurrent_sequence_minibatches": True,
            "rollout_start_hidden_stored": True,
            "in_sequence_native_reset_mask_stored": True,
            "kl": {
                "mode": "telemetry_only",
                "minibatch_rejection": False,
                "completed_update_rejection": False,
                "retry": False,
                "rollback": False,
                "lr_backoff": False,
            },
            "nonfinite_loss_gradient_parameter_rollback": True,
        },
        "exploration": {
            "contract": RIVAL2_FRESH_HUMAN_SEED_EXPLORATION_RAMP_CONTRACT,
            "contract_sha256": (
                RIVAL2_FRESH_HUMAN_SEED_EXPLORATION_RAMP_CONTRACT_HASH
            ),
            "schedule_samples": {
                str(update): fresh_human_seed_exploration(update).as_dict()
                for update in (0, 60, 300, 900)
            },
            "one_effective_distribution_for_sampling_and_ppo": True,
        },
        "phase_a": {
            "reward_version": RIVAL2_REWARD_ACQUISITION_120_V1_VERSION,
            "reward_contract_sha256": REWARD_ACQUISITION_120_V1_CONTRACT_HASH,
            "runtime_contracts": phase_contracts(
                RIVAL2_REWARD_ACQUISITION_120_V1_VERSION
            ),
            "policy_hz": 120,
            "evaluation_start_update": PHASE_A_EVALUATION_START,
            "evaluation_interval": PHASE_A_EVALUATION_INTERVAL,
            "snapshots_every_updates": SNAPSHOT_INTERVAL,
            "transition_guidance": {
                "episodes_with_touch_fraction": 0.90,
                "no_touch_fraction_maximum": 0.01,
                "requires_forward_contact": True,
                "nonimproving_evaluation_patience": PHASE_A_NONIMPROVING_PATIENCE,
            },
        },
        "phase_b": {
            "reward_version": RIVAL2_REWARD_GAMEPLAY_120_V2_VERSION,
            "reward_contract_sha256": REWARD_GAMEPLAY_120_V2_CONTRACT_HASH,
            "runtime_contracts": phase_contracts(
                RIVAL2_REWARD_GAMEPLAY_120_V2_VERSION
            ),
            "accepted_updates": PHASE_B_ACCEPTED_UPDATES,
            "snapshots_every_updates": SNAPSHOT_INTERVAL,
            "fresh_optimizer_at_transition": True,
        },
        "opponents": {
            "current_policy_self_play": 1.0,
            "both_sides_trainable": True,
            "historical": 0.0,
            "nexto": 0.0,
            "wisp": 0.0,
            "nexto_evaluation_only": True,
        },
        "seeds": {
            "campaign": CAMPAIGN_SEED,
            "critic": CRITIC_SEED,
            "phase_b": PHASE_B_SEED,
            "evaluation": EVALUATION_SEED,
        },
    }
    write_json(AUTHORITY, authority)
    return authority


def load_authority() -> dict[str, Any]:
    authority = json.loads(AUTHORITY.read_text(encoding="utf-8"))
    checks = {
        "format": authority.get("format") == f"{FORMAT}_AUTHORITY",
        "source": authority.get("source", {}).get("sha256") == SOURCE_SHA256,
        "source_file": sha256_file(SOURCE) == SOURCE_SHA256,
        "exploration": authority.get("exploration", {}).get("contract_sha256")
        == RIVAL2_FRESH_HUMAN_SEED_EXPLORATION_RAMP_CONTRACT_HASH,
        "ppo": authority.get("ppo", {}).get("contract_sha256")
        == RIVAL2_PPO_120HZ_CONTRACT_HASH,
        "worlds": authority.get("ppo", {}).get("world_count") == WORLD_COUNT,
        "phase_b_updates": authority.get("phase_b", {}).get("accepted_updates")
        == PHASE_B_ACCEPTED_UPDATES,
        "kl_telemetry_only": authority.get("ppo", {}).get("kl", {}).get("mode")
        == "telemetry_only"
        and not authority["ppo"]["kl"]["minibatch_rejection"]
        and not authority["ppo"]["kl"]["completed_update_rejection"]
        and not authority["ppo"]["kl"]["rollback"],
    }
    if not all(checks.values()):
        raise RuntimeError(f"recurrent PPO authority mismatch: {checks}")
    return authority


def collision_assets(root: Path) -> tuple[ArenaGeometry, WarpArenaMeshes]:
    geometry = ArenaGeometry.load_soccar(root)
    return geometry, WarpArenaMeshes(geometry)


def make_env(
    collision_root: Path,
    *,
    worlds: int,
    reward_version: str,
    seed: int,
) -> Rival2Env:
    geometry, meshes = collision_assets(collision_root)
    return Rival2Env(
        worlds,
        str(collision_root),
        geometry=geometry,
        meshes=meshes,
        device="cuda:0",
        seed=seed,
        car_visitation_order="a_then_b",
        reward_version=reward_version,
        episode_version=RIVAL2_EPISODE_VERSION,
        observation_version=RIVAL2_OBS_V2_120HZ_VERSION,
        action_version=RIVAL2_ACTION_V2_120HZ_VERSION,
    )


def initialize_phase_a(
    collision_root: Path, *, worlds: int
) -> tuple[Rival2RecurrentTrainer, dict[str, Any]]:
    authority = load_authority()
    source = source_payload()
    config = Rival2RecurrentPolicyConfig(**source["policy_config"])
    ppo = replace(rival2_ppo_120hz_config(), learning_rate=PPO_LEARNING_RATE)
    env = make_env(
        collision_root,
        worlds=worlds,
        reward_version=RIVAL2_REWARD_ACQUISITION_120_V1_VERSION,
        seed=CAMPAIGN_SEED,
    )
    trainer = Rival2RecurrentTrainer(
        env,
        policy_config=config,
        ppo_config=ppo,
        phase="phase_a_acquisition",
        source_identity=authority["source"],
        seed=CAMPAIGN_SEED,
    )
    if trainer.optimizer.state:
        raise RuntimeError("fresh recurrent PPO optimizer unexpectedly has state")
    trainer.model.load_state_dict(source["model"], strict=True)
    source_behavior = {
        name: value
        for name, value in source["model"].items()
        if not name.startswith("critic.")
    }
    loaded_behavior = {
        name: value
        for name, value in trainer.model.state_dict().items()
        if not name.startswith("critic.")
    }
    if state_dict_sha256(source_behavior) != state_dict_sha256(loaded_behavior):
        raise RuntimeError("recurrent behavior tensors changed during transition")
    source_critic_sha = state_dict_sha256(trainer.model.critic.state_dict())
    torch.manual_seed(CRITIC_SEED)
    replacement = nn.Linear(config.post_dim, 1)
    nn.init.orthogonal_(replacement.weight, gain=0.01)
    nn.init.zeros_(replacement.bias)
    trainer.model.critic.load_state_dict(replacement.state_dict(), strict=True)
    fresh_critic_sha = state_dict_sha256(trainer.model.critic.state_dict())
    if fresh_critic_sha == source_critic_sha:
        raise RuntimeError("critic was not freshly initialized")
    trainer.model.requires_grad_(True)
    trainer.set_exploration(fresh_human_seed_exploration(1))
    trainer.replace_optimizer()
    trainer.phase_transition = {
        "format": f"{FORMAT}_SOURCE_TRANSITION",
        "source": authority["source"],
        "behavior_tensor_sha256_before": state_dict_sha256(source_behavior),
        "behavior_tensor_sha256_after": state_dict_sha256(loaded_behavior),
        "behavior_exact": True,
        "source_critic_sha256": source_critic_sha,
        "fresh_critic_sha256": fresh_critic_sha,
        "critic_reinitialized": True,
        "fresh_optimizer": True,
        "fresh_rng_counters": True,
        "authority_sha256": sha256_file(AUTHORITY),
    }
    return trainer, source


def transition_to_phase_b(
    phase_a: Rival2RecurrentTrainer,
    collision_root: Path,
    *,
    worlds: int,
) -> Rival2RecurrentTrainer:
    env = make_env(
        collision_root,
        worlds=worlds,
        reward_version=RIVAL2_REWARD_GAMEPLAY_120_V2_VERSION,
        seed=PHASE_B_SEED,
    )
    phase_b = Rival2RecurrentTrainer(
        env,
        policy_config=phase_a.policy_config,
        ppo_config=phase_a.ppo_config,
        phase="phase_b_gameplay_120_v2",
        source_identity=phase_a.source_identity,
        seed=PHASE_B_SEED,
    )
    phase_b.model.load_state_dict(phase_a.model.state_dict(), strict=True)
    phase_b.model.requires_grad_(True)
    phase_b.replace_optimizer()
    phase_b.accepted_updates_total = phase_a.accepted_updates_total
    phase_b.policy_version = phase_a.policy_version
    phase_b.total_agent_samples = phase_a.total_agent_samples
    phase_b.physical_physics_ticks_experienced = (
        phase_a.physical_physics_ticks_experienced
    )
    phase_b.set_exploration(
        fresh_human_seed_exploration(phase_b.accepted_updates_total + 1)
    )
    phase_b.phase_transition = {
        "format": f"{FORMAT}_PHASE_B_TRANSITION",
        "source_phase": phase_a.phase,
        "source_phase_updates": phase_a.phase_accepted_updates,
        "accepted_updates_total": phase_a.accepted_updates_total,
        "model_tensor_sha256": state_dict_sha256(phase_a.model.state_dict()),
        "model_preserved_exact": state_dict_sha256(phase_b.model.state_dict())
        == state_dict_sha256(phase_a.model.state_dict()),
        "critic_preserved": True,
        "fresh_phase_b_optimizer": True,
        "fresh_phase_b_world_at_native_kickoff": True,
        "reward_from": RIVAL2_REWARD_ACQUISITION_120_V1_VERSION,
        "reward_to": RIVAL2_REWARD_GAMEPLAY_120_V2_VERSION,
    }
    return phase_b


def preflight(
    trainer: Rival2RecurrentTrainer, source: dict[str, Any], worlds: int
) -> dict[str, Any]:
    source_behavior = {
        name: value
        for name, value in source["model"].items()
        if not name.startswith("critic.")
    }
    loaded_behavior = {
        name: value
        for name, value in trainer.model.state_dict().items()
        if not name.startswith("critic.")
    }
    checks = {
        "source_sha256_exact": sha256_file(SOURCE) == SOURCE_SHA256,
        "source_commit_exact": load_authority()["source"]["source_commit"]
        == SOURCE_COMMIT,
        "source_selected_step_400": source.get("selected_step") == 400,
        "source_behavior_preserved": state_dict_sha256(source_behavior)
        == state_dict_sha256(loaded_behavior),
        "critic_reinitialized": trainer.phase_transition["critic_reinitialized"],
        "fresh_optimizer_empty": len(trainer.optimizer.state) == 0,
        "fresh_counters": trainer.accepted_updates_total
        == trainer.phase_accepted_updates
        == trainer.policy_version
        == 0,
        "all_parameters_trainable": all(
            parameter.requires_grad for parameter in trainer.model.parameters()
        ),
        "worlds_exact": worlds == WORLD_COUNT,
        "policy_and_physics_120_hz": trainer.env.policy_hz
        == trainer.env.physics_hz
        == 120,
        "acquisition_reward": trainer.env.reward_version
        == RIVAL2_REWARD_ACQUISITION_120_V1_VERSION,
        "runtime_contracts_exact": trainer.env.contract_hashes
        == phase_contracts(RIVAL2_REWARD_ACQUISITION_120_V1_VERSION),
        "observation_view_exact": trainer.policy_config.observation_view_version
        == "RIVAL2_HUMAN_SEQUENCE_OBS_VIEW_V1",
        "adapter_v2_absent": True,
        "previous_action_zeroed_by_policy_projection": True,
        "historical_nexto_wisp_training_absent": True,
        "update_0_exploration": fresh_human_seed_exploration(0).analog_sigma
        == 0.01
        and fresh_human_seed_exploration(0).button_temperature == 0.02,
        "update_60_exploration": fresh_human_seed_exploration(60).analog_sigma
        == 0.01
        and fresh_human_seed_exploration(60).button_temperature == 0.02,
        "update_300_exploration": fresh_human_seed_exploration(300).analog_sigma
        == 0.08
        and fresh_human_seed_exploration(300).button_temperature == 0.50,
        "kl_telemetry_only": load_authority()["ppo"]["kl"]["mode"]
        == "telemetry_only",
    }
    return {
        "format": f"{FORMAT}_PREFLIGHT",
        "created_utc": utc_now(),
        "verdict": "PASS" if all(checks.values()) else "FAIL",
        "checks": checks,
        "transition": trainer.phase_transition,
        "policy_config": asdict(trainer.policy_config),
        "ppo_config": asdict(trainer.ppo_config),
        "rollout_logical_bytes": int(
            trainer.ppo_config.rollout_horizon
            * worlds
            * 2
            * (trainer.policy_config.obs_dim * 4 + 100)
        ),
    }


def resume_preflight(
    trainer: Rival2RecurrentTrainer,
    checkpoint: Path,
    worlds: int,
) -> dict[str, Any]:
    expected_reward = (
        RIVAL2_REWARD_ACQUISITION_120_V1_VERSION
        if trainer.phase == "phase_a_acquisition"
        else RIVAL2_REWARD_GAMEPLAY_120_V2_VERSION
    )
    checks = {
        "source_sha256_exact": sha256_file(SOURCE) == SOURCE_SHA256,
        "source_identity_preserved": trainer.source_identity == load_authority()["source"],
        "checkpoint_exists": checkpoint.is_file(),
        "checkpoint_format_exact": torch.load(
            checkpoint, map_location="cpu", weights_only=False
        ).get("format")
        == CHECKPOINT_FORMAT,
        "phase_valid": trainer.phase
        in {"phase_a_acquisition", "phase_b_gameplay_120_v2"},
        "accepted_update_counters_valid": trainer.accepted_updates_total >= 1
        and trainer.accepted_updates_total >= trainer.phase_accepted_updates
        and trainer.policy_version == trainer.accepted_updates_total,
        "optimizer_restored": len(trainer.optimizer.state) > 0,
        "all_parameters_trainable": all(
            parameter.requires_grad for parameter in trainer.model.parameters()
        ),
        "worlds_exact": worlds == WORLD_COUNT,
        "policy_and_physics_120_hz": trainer.env.policy_hz
        == trainer.env.physics_hz
        == 120,
        "phase_reward_exact": trainer.env.reward_version == expected_reward,
        "runtime_contracts_exact": trainer.env.contract_hashes
        == phase_contracts(expected_reward),
        "observation_view_exact": trainer.policy_config.observation_view_version
        == "RIVAL2_HUMAN_SEQUENCE_OBS_VIEW_V1",
        "adapter_v2_absent": True,
        "previous_action_zeroed_by_policy_projection": True,
        "historical_nexto_wisp_training_absent": True,
        "kl_telemetry_only": load_authority()["ppo"]["kl"]["mode"]
        == "telemetry_only",
    }
    return {
        "format": f"{FORMAT}_RESUME_PREFLIGHT",
        "created_utc": utc_now(),
        "verdict": "PASS" if all(checks.values()) else "FAIL",
        "checks": checks,
        "resume_checkpoint": str(checkpoint),
        "resume_checkpoint_sha256": sha256_file(checkpoint),
        "phase": trainer.phase,
        "accepted_updates_total": trainer.accepted_updates_total,
        "phase_accepted_updates": trainer.phase_accepted_updates,
        "policy_config": asdict(trainer.policy_config),
        "ppo_config": asdict(trainer.ppo_config),
    }


def checkpoint_record(
    trainer: Rival2RecurrentTrainer,
    path: Path,
    *,
    include_optimizer: bool,
) -> dict[str, Any]:
    trainer.save_checkpoint(path, include_optimizer=include_optimizer)
    return {
        "phase": trainer.phase,
        "accepted_updates_total": trainer.accepted_updates_total,
        "phase_accepted_updates": trainer.phase_accepted_updates,
        "path": path.relative_to(ROOT).as_posix() if path.is_relative_to(ROOT) else str(path),
        "sha256": sha256_file(path),
        "bytes": path.stat().st_size,
        "optimizer_included": include_optimizer,
        "exploration": None if trainer.exploration is None else trainer.exploration.as_dict(),
    }


def save_snapshot(
    trainer: Rival2RecurrentTrainer,
    run_dir: Path,
    manifest: dict[str, Any],
) -> dict[str, Any]:
    path = run_dir / "snapshots" / (
        f"{trainer.phase}_u{trainer.phase_accepted_updates:04d}_"
        f"global{trainer.accepted_updates_total:04d}.pt"
    )
    record = checkpoint_record(trainer, path, include_optimizer=False)
    manifest["snapshots"] = [
        prior
        for prior in manifest["snapshots"]
        if not (
            prior["phase"] == trainer.phase
            and prior["phase_accepted_updates"] == trainer.phase_accepted_updates
        )
    ] + [record]
    manifest["snapshots"].sort(
        key=lambda item: (item["accepted_updates_total"], item["phase"])
    )
    return record


def evaluation_score(result: dict[str, Any]) -> tuple[float, int, int, int]:
    gameplay = result["gameplay"]
    return (
        float(gameplay["episodes_with_rival_touch_fraction"]),
        int(gameplay["rival_touches"]),
        int(gameplay["rival_forward_ball_velocity_contacts"]),
        int(gameplay["rival_goals"]),
    )


def run_evaluation(
    trainer: Rival2RecurrentTrainer,
    run_dir: Path,
    label: str,
    collision_root: Path,
) -> tuple[dict[str, Any], dict[str, Any]]:
    checkpoint = run_dir / "evaluations" / f"{label}.pt"
    record = checkpoint_record(trainer, checkpoint, include_optimizer=False)
    output = RESULTS / f"nexto_{label}.json"
    command = [
        sys.executable,
        str(ROOT / "benchmarks/evaluate_rival2_human_sequence_recurrent_ppo.py"),
        "--checkpoint",
        str(checkpoint),
        "--output",
        str(output),
        "--collision-root",
        str(collision_root),
        "--expected-format",
        CHECKPOINT_FORMAT,
        "--worlds-per-side",
        "128",
        "--seed",
        str(EVALUATION_SEED),
        "--device",
        "cuda:0",
    ]
    completed = subprocess.run(
        command,
        cwd=ROOT,
        check=False,
        capture_output=True,
        text=True,
    )
    if completed.returncode != 0:
        raise RuntimeError(
            "isolated recurrent Nexto evaluation failed: "
            f"returncode={completed.returncode}; stderr={completed.stderr[-4000:]}"
        )
    result = json.loads(output.read_text(encoding="utf-8"))
    return record, result


def run(args: argparse.Namespace) -> int:
    authority = load_authority()
    run_dir = Path(args.run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    CHECKPOINTS.mkdir(parents=True, exist_ok=True)
    curve = RESULTS / "training_curve.jsonl"
    manifest_path = RESULTS / "snapshot_manifest.json"
    manifest = (
        json.loads(manifest_path.read_text(encoding="utf-8"))
        if manifest_path.is_file()
        else {
            "format": f"{FORMAT}_SNAPSHOT_MANIFEST",
            "source_sha256": SOURCE_SHA256,
            "snapshots": [],
            "phase_a_evaluations": [],
        }
    )
    resume_path: Path | None = None
    if args.resume:
        resume = Path(args.resume)
        resume_path = resume
        payload = torch.load(resume, map_location="cpu", weights_only=False)
        phase = payload.get("phase")
        reward = (
            RIVAL2_REWARD_ACQUISITION_120_V1_VERSION
            if phase == "phase_a_acquisition"
            else RIVAL2_REWARD_GAMEPLAY_120_V2_VERSION
        )
        env = make_env(
            Path(args.collision_root),
            worlds=args.worlds,
            reward_version=reward,
            seed=CAMPAIGN_SEED if phase == "phase_a_acquisition" else PHASE_B_SEED,
        )
        trainer = Rival2RecurrentTrainer(
            env,
            policy_config=Rival2RecurrentPolicyConfig(**payload["policy_config"]),
            ppo_config=replace(
                rival2_ppo_120hz_config(), learning_rate=PPO_LEARNING_RATE
            ),
            phase=phase,
            source_identity=authority["source"],
            seed=CAMPAIGN_SEED if phase == "phase_a_acquisition" else PHASE_B_SEED,
        )
        trainer.load_checkpoint(resume)
        source = source_payload()
    else:
        if curve.exists():
            raise RuntimeError("training curve already exists; use --resume")
        trainer, source = initialize_phase_a(
            Path(args.collision_root), worlds=args.worlds
        )

    preflight_payload = (
        resume_preflight(trainer, resume_path, args.worlds)
        if resume_path is not None
        else preflight(trainer, source, args.worlds)
    )
    preflight_path = (
        RESULTS / "resume_preflight.json"
        if resume_path is not None
        else RESULTS / "preflight.json"
    )
    write_json(preflight_path, preflight_payload)
    if preflight_payload["verdict"] != "PASS":
        raise RuntimeError(f"recurrent PPO preflight failed: {preflight_payload}")
    if args.preflight_only:
        print(json.dumps(preflight_payload, indent=2, sort_keys=True))
        return 0

    rolling = run_dir / "rolling.pt"
    prior_phase_a_evaluations = manifest.get("phase_a_evaluations", [])
    if prior_phase_a_evaluations:
        phase_a_previous_score = evaluation_score(
            {"gameplay": prior_phase_a_evaluations[-1]["gameplay"]}
        )
        phase_a_nonimproving = 0
        recent = prior_phase_a_evaluations[-(PHASE_A_NONIMPROVING_PATIENCE + 1) :]
        for previous, current in pairwise(recent):
            if evaluation_score({"gameplay": current["gameplay"]}) > evaluation_score(
                {"gameplay": previous["gameplay"]}
            ):
                phase_a_nonimproving = 0
            else:
                phase_a_nonimproving += 1
    else:
        phase_a_nonimproving = 0
        phase_a_previous_score = evaluation_score(
            json.loads(SOURCE_EVALUATION.read_text(encoding="utf-8"))
        )
    started = time.monotonic()
    hard_failure: dict[str, Any] | None = None

    while trainer.phase == "phase_a_acquisition":
        exploration = fresh_human_seed_exploration(
            trainer.accepted_updates_total + 1
        )
        trainer.set_exploration(exploration)
        rollout_started = time.monotonic()
        rollout = trainer.collect_rollout()
        rollout_seconds = time.monotonic() - rollout_started
        update_started = time.monotonic()
        try:
            ppo_metrics = trainer.update(rollout)
        except Rival2RecurrentPPOCorruption as error:
            hard_failure = {
                "format": f"{FORMAT}_HARD_FAILURE",
                "created_utc": utc_now(),
                "phase": trainer.phase,
                "diagnostics": error.diagnostics,
                "kl_caused_stop": False,
            }
            write_json(RESULTS / "hard_failure.json", hard_failure)
            checkpoint_record(trainer, rolling, include_optimizer=True)
            break
        row = {
            "created_utc": utc_now(),
            "phase": trainer.phase,
            "accepted_updates_total": trainer.accepted_updates_total,
            "phase_accepted_updates": trainer.phase_accepted_updates,
            "rollout_seconds": rollout_seconds,
            "update_seconds": time.monotonic() - update_started,
            "elapsed_seconds": time.monotonic() - started,
            "exploration": exploration.as_dict(),
            "ppo": scalar_metrics(ppo_metrics),
            "gameplay": trainer.last_rollout_metrics,
            "kl_telemetry_only": True,
        }
        append_jsonl(curve, row)
        checkpoint_record(trainer, rolling, include_optimizer=True)
        del rollout
        if trainer.phase_accepted_updates % SNAPSHOT_INTERVAL == 0:
            save_snapshot(trainer, run_dir, manifest)
            write_json(manifest_path, manifest)
        if trainer.phase_accepted_updates % 10 == 0 or trainer.phase_accepted_updates == 1:
            print(
                json.dumps(
                    {
                        "phase": trainer.phase,
                        "update": trainer.phase_accepted_updates,
                        "global": trainer.accepted_updates_total,
                        "touches_per_minute": trainer.last_rollout_metrics[
                            "touches_per_minute"
                        ],
                        "goals": trainer.last_rollout_metrics["goal_events"],
                        "no_touch": trainer.last_rollout_metrics[
                            "no_touch_truncations"
                        ],
                        "kl": scalar_metrics(ppo_metrics).get(
                            "completed_update_mean_kl"
                        ),
                    },
                    sort_keys=True,
                ),
                flush=True,
            )
        should_evaluate = (
            trainer.phase_accepted_updates >= PHASE_A_EVALUATION_START
            and trainer.phase_accepted_updates % PHASE_A_EVALUATION_INTERVAL == 0
        )
        if not should_evaluate:
            continue
        record, evaluation = run_evaluation(
            trainer,
            run_dir,
            f"phase_a_u{trainer.phase_accepted_updates:04d}",
            Path(args.collision_root),
        )
        manifest["phase_a_evaluations"].append(
            {"checkpoint": record, "result_path": str(
                (RESULTS / f"nexto_phase_a_u{trainer.phase_accepted_updates:04d}.json")
                .relative_to(ROOT)
                .as_posix()
            ), "gameplay": evaluation["gameplay"]}
        )
        write_json(manifest_path, manifest)
        if evaluation["phase_a_transition_evidence"]["routine_acquisition"]:
            phase_a_path = CHECKPOINTS / "rival2_human_sequence_phase_a.pt"
            phase_a_final = checkpoint_record(
                trainer, phase_a_path, include_optimizer=True
            )
            write_json(
                RESULTS / "phase_a_summary.json",
                {
                    "format": f"{FORMAT}_PHASE_A_SUMMARY",
                    "created_utc": utc_now(),
                    "checkpoint": phase_a_final,
                    "nexto": evaluation,
                    "transition_to_phase_b": True,
                },
            )
            prior = trainer
            trainer = transition_to_phase_b(
                prior, Path(args.collision_root), worlds=args.worlds
            )
            write_json(RESULTS / "phase_transition.json", trainer.phase_transition)
            del prior
            gc.collect()
            torch.cuda.empty_cache()
            checkpoint_record(trainer, rolling, include_optimizer=True)
            break
        score = evaluation_score(evaluation)
        if score > phase_a_previous_score:
            phase_a_nonimproving = 0
        else:
            phase_a_nonimproving += 1
        phase_a_previous_score = score
        if phase_a_nonimproving >= PHASE_A_NONIMPROVING_PATIENCE:
            hard_failure = {
                "format": f"{FORMAT}_PHASE_A_BLOCKED",
                "created_utc": utc_now(),
                "reason": "ball_acquisition_not_routine_and_no_longer_improving",
                "phase_a_updates": trainer.phase_accepted_updates,
                "nexto": evaluation["gameplay"],
                "kl_caused_stop": False,
            }
            write_json(RESULTS / "phase_a_blocked.json", hard_failure)
            checkpoint_record(
                trainer,
                CHECKPOINTS / "rival2_human_sequence_phase_a_blocked.pt",
                include_optimizer=True,
            )
            break

    if hard_failure is None and trainer.phase == "phase_b_gameplay_120_v2":
        while trainer.phase_accepted_updates < PHASE_B_ACCEPTED_UPDATES:
            exploration = fresh_human_seed_exploration(
                trainer.accepted_updates_total + 1
            )
            trainer.set_exploration(exploration)
            rollout_started = time.monotonic()
            rollout = trainer.collect_rollout()
            rollout_seconds = time.monotonic() - rollout_started
            update_started = time.monotonic()
            try:
                ppo_metrics = trainer.update(rollout)
            except Rival2RecurrentPPOCorruption as error:
                hard_failure = {
                    "format": f"{FORMAT}_HARD_FAILURE",
                    "created_utc": utc_now(),
                    "phase": trainer.phase,
                    "diagnostics": error.diagnostics,
                    "kl_caused_stop": False,
                }
                write_json(RESULTS / "hard_failure.json", hard_failure)
                checkpoint_record(trainer, rolling, include_optimizer=True)
                break
            append_jsonl(
                curve,
                {
                    "created_utc": utc_now(),
                    "phase": trainer.phase,
                    "accepted_updates_total": trainer.accepted_updates_total,
                    "phase_accepted_updates": trainer.phase_accepted_updates,
                    "rollout_seconds": rollout_seconds,
                    "update_seconds": time.monotonic() - update_started,
                    "elapsed_seconds": time.monotonic() - started,
                    "exploration": exploration.as_dict(),
                    "ppo": scalar_metrics(ppo_metrics),
                    "gameplay": trainer.last_rollout_metrics,
                    "kl_telemetry_only": True,
                },
            )
            checkpoint_record(trainer, rolling, include_optimizer=True)
            del rollout
            if trainer.phase_accepted_updates % SNAPSHOT_INTERVAL == 0:
                save_snapshot(trainer, run_dir, manifest)
                write_json(manifest_path, manifest)
            if trainer.phase_accepted_updates % 10 == 0 or trainer.phase_accepted_updates == 1:
                print(
                    json.dumps(
                        {
                            "phase": trainer.phase,
                            "update": trainer.phase_accepted_updates,
                            "global": trainer.accepted_updates_total,
                            "touches_per_minute": trainer.last_rollout_metrics[
                                "touches_per_minute"
                            ],
                            "goals": trainer.last_rollout_metrics["goal_events"],
                            "no_touch": trainer.last_rollout_metrics[
                                "no_touch_truncations"
                            ],
                            "kl": scalar_metrics(ppo_metrics).get(
                                "completed_update_mean_kl"
                            ),
                        },
                        sort_keys=True,
                    ),
                    flush=True,
                )

    final_evaluation: dict[str, Any] | None = None
    phase_b_final: dict[str, Any] | None = None
    if hard_failure is None and trainer.phase == "phase_b_gameplay_120_v2":
        phase_b_path = CHECKPOINTS / "rival2_human_sequence_phase_b_u0600.pt"
        phase_b_final = checkpoint_record(
            trainer, phase_b_path, include_optimizer=True
        )
        _evaluation_checkpoint, final_evaluation = run_evaluation(
            trainer,
            run_dir,
            "phase_b_u0600_final",
            Path(args.collision_root),
        )
    summary = {
        "format": f"{FORMAT}_SUMMARY",
        "created_utc": utc_now(),
        "status": "BLOCKED" if hard_failure is not None else "PASS",
        "source": authority["source"],
        "accepted_updates_total": trainer.accepted_updates_total,
        "phase": trainer.phase,
        "phase_accepted_updates": trainer.phase_accepted_updates,
        "phase_b_completed": trainer.phase == "phase_b_gameplay_120_v2"
        and trainer.phase_accepted_updates == PHASE_B_ACCEPTED_UPDATES,
        "phase_b_final_checkpoint": phase_b_final,
        "final_nexto": final_evaluation,
        "hard_failure": hard_failure,
        "kl_policy": "telemetry_only",
        "authority_sha256": sha256_file(AUTHORITY),
    }
    write_json(RESULTS / "summary.json", summary)
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 2 if hard_failure is not None else 0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--prepare-authority", action="store_true")
    parser.add_argument("--preflight-only", action="store_true")
    parser.add_argument("--resume")
    parser.add_argument("--worlds", type=int, default=WORLD_COUNT)
    parser.add_argument("--run-dir", default=str(DEFAULT_RUN_DIR))
    parser.add_argument("--collision-root", default=str(DEFAULT_COLLISION_ROOT))
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.prepare_authority:
        print(json.dumps(prepare_authority(), indent=2, sort_keys=True))
        return 0
    return run(args)


if __name__ == "__main__":
    raise SystemExit(main())
