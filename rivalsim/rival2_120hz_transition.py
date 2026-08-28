"""Fail-closed iteration-479 to Rival 120 Hz bootstrap migration helpers."""

from __future__ import annotations

import copy
import hashlib
import json
from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np
import torch

from rivalsim.rival2_contracts import (
    RIVAL2_ACTION_V2_120HZ_VERSION,
    RIVAL2_EPISODE_VERSION,
    RIVAL2_OBS_V2_120HZ_VERSION,
    RIVAL2_REWARD_GAMEPLAY_120_V1_VERSION,
    RIVAL2_REWARD_GAMEPLAY_V2_VERSION,
    contract_hashes_for_reward,
)
from rivalsim.rival2_ppo import (
    RIVAL2_PPO_120HZ_CONTRACT_HASH,
    RIVAL2_PPO_120HZ_V1,
    rival2_ppo_120hz_config,
)

SOURCE_ITERATION = 479
SOURCE_POLICY_VERSION = 479
SOURCE_AGENT_DECISIONS = 3_655_854_038
SOURCE_SHA256 = "3B994E118A9498713DC6115D38F061958A900EA8F4D00CE568F916942E851D9A"
BOOTSTRAP_WORLD_COUNT = 32_768


def file_sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest().upper()


def tensor_tree_sha256(value: Any) -> str:
    """Canonical digest for nested tensor state independent of torch.save framing."""

    digest = hashlib.sha256()

    def update(item: Any, path: str) -> None:
        digest.update(path.encode("utf-8") + b"\0")
        if isinstance(item, torch.Tensor):
            tensor = item.detach().cpu().contiguous()
            digest.update(str(tensor.dtype).encode("ascii") + b"\0")
            digest.update(json.dumps(list(tensor.shape), separators=(",", ":")).encode("ascii"))
            digest.update(b"\0" + tensor.numpy().tobytes(order="C"))
        elif isinstance(item, np.ndarray):
            array = np.ascontiguousarray(item)
            digest.update(str(array.dtype).encode("ascii") + b"\0")
            digest.update(json.dumps(list(array.shape), separators=(",", ":")).encode("ascii"))
            digest.update(b"\0" + array.tobytes(order="C"))
        elif isinstance(item, dict):
            for key in sorted(item, key=lambda entry: str(entry)):
                update(item[key], f"{path}.{key}")
        elif isinstance(item, (list, tuple)):
            for index, child in enumerate(item):
                update(child, f"{path}[{index}]")
        else:
            digest.update(
                json.dumps(item, sort_keys=True, separators=(",", ":"), default=str).encode(
                    "utf-8"
                )
            )
    update(value, "root")
    return digest.hexdigest().upper()


def verify_source_checkpoint(path: str | Path) -> tuple[dict[str, Any], dict[str, Any]]:
    checkpoint = Path(path)
    digest = file_sha256(checkpoint)
    if digest != SOURCE_SHA256:
        raise ValueError(f"iteration-479 source SHA mismatch: {digest}")
    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    checks = {
        "sha256_exact": digest == SOURCE_SHA256,
        "iteration_exact": int(payload.get("iteration", -1)) == SOURCE_ITERATION,
        "policy_version_exact": int(payload.get("policy_version", -1))
        == SOURCE_POLICY_VERSION,
        "sample_counter_exact": int(payload.get("total_agent_samples", -1))
        == SOURCE_AGENT_DECISIONS,
        "reward_source_exact": payload.get("reward_version")
        == RIVAL2_REWARD_GAMEPLAY_V2_VERSION,
        "episode_source_exact": payload.get("episode_version") == RIVAL2_EPISODE_VERSION,
        "world_count_exact": int(payload["opponent_assignment"].shape[0]) == 131_072,
    }
    if not all(checks.values()):
        raise ValueError(f"iteration-479 source identity failed: {checks}")
    report = {
        "path": str(checkpoint.resolve()),
        "bytes": checkpoint.stat().st_size,
        "sha256": digest,
        "iteration": SOURCE_ITERATION,
        "policy_version": SOURCE_POLICY_VERSION,
        "source_30hz_agent_decisions": SOURCE_AGENT_DECISIONS,
        "model_tensor_sha256": tensor_tree_sha256(payload["model"]),
        "optimizer_tensor_sha256": tensor_tree_sha256(payload["optimizer"]),
        "historical_pool_tensor_sha256": tensor_tree_sha256(
            payload["historical_opponents"]
        ),
        "checks": checks,
    }
    return payload, report


def _fresh_curriculum_state(
    source: dict[str, Any],
    retention_observations: torch.Tensor,
    retention_summary: dict[str, Any],
    world_count: int,
) -> dict[str, Any]:
    curriculum = copy.deepcopy(source["opponent_curriculum"])
    curriculum["family"] = torch.full((world_count,), -1, dtype=torch.int64)
    curriculum["rival_side"] = torch.zeros(world_count, dtype=torch.int64)
    adaptive = curriculum["adaptive_ppo"]
    adaptive["retention_observations"] = retention_observations.detach().cpu().clone()
    adaptive["retention_corpus_summary"] = copy.deepcopy(retention_summary)
    adaptive["last_update_summary"] = None
    adaptive["next_update_policy_learning_rate"] = adaptive["config"][
        "initial_policy_learning_rate"
    ]
    adaptive["optimizer_learning_rates"] = {
        "policy": adaptive["config"]["initial_policy_learning_rate"],
        "critic": adaptive["config"]["critic_learning_rate"],
    }
    curriculum["nexto"] = {
        "player_index": torch.zeros(world_count, dtype=torch.int64),
        "previous_action": torch.zeros((world_count, 8), dtype=torch.float32),
        "neural_counter": torch.zeros(world_count, dtype=torch.int64),
        "kickoff_index": torch.full((world_count,), -1, dtype=torch.int64),
        "cadence_tick": 0,
    }
    source_wisp = source["opponent_curriculum"]["wisp"]
    curriculum["wisp"] = {
        "player_index": torch.zeros(world_count, dtype=torch.int64),
        "old_action": torch.zeros((world_count, 8), dtype=torch.float32),
        "new_action": torch.zeros((world_count, 8), dtype=torch.float32),
        "previous_action": torch.zeros((world_count, 8), dtype=torch.float32),
        "ticks": torch.full((world_count,), -1, dtype=torch.int64),
        "update_flag": torch.ones(world_count, dtype=torch.bool),
        "eta_cache": np.zeros((world_count, 2), dtype=np.float64),
        "observation_generator_state": source_wisp[
            "observation_generator_state"
        ].clone(),
        "opponent_slot": torch.zeros(world_count, dtype=torch.int64),
    }
    curriculum["historical_rival_cadence"] = {
        "cached_action": torch.zeros((world_count, 2, 8), dtype=torch.float32),
        "cache_valid": torch.zeros(world_count, dtype=torch.bool),
        "phase": torch.zeros(world_count, dtype=torch.int64),
        "policy_evaluation_calls": 0,
        "semantics": (
            "30 Hz snapshots evaluate on phase 0 and hold four 120 Hz ticks; "
            "120 Hz snapshots evaluate every tick"
        ),
    }
    return curriculum


def build_bootstrap_payload(
    source: dict[str, Any],
    source_report: dict[str, Any],
    retention_observations: torch.Tensor,
    retention_summary: dict[str, Any],
    *,
    world_count: int = BOOTSTRAP_WORLD_COUNT,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Create a loadable fresh-environment bootstrap without any optimizer step."""

    if retention_observations.shape != (512, 182):
        raise ValueError("120 Hz retention corpus must have shape [512, 182]")
    if not bool(torch.isfinite(retention_observations).all()):
        raise ValueError("120 Hz retention corpus contains nonfinite values")
    payload = copy.deepcopy(source)
    source_model_hash = tensor_tree_sha256(source["model"])
    source_optimizer_hash = tensor_tree_sha256(source["optimizer"])
    source_pool_hash = tensor_tree_sha256(source["historical_opponents"])

    payload["ppo_config"] = asdict(rival2_ppo_120hz_config())
    payload["ppo_config_hash"] = rival2_ppo_120hz_config().content_hash
    payload["contract_hashes"] = contract_hashes_for_reward(
        RIVAL2_REWARD_GAMEPLAY_120_V1_VERSION
    )
    payload["reward_version"] = RIVAL2_REWARD_GAMEPLAY_120_V1_VERSION
    payload["episode_version"] = RIVAL2_EPISODE_VERSION
    payload["observation_version"] = RIVAL2_OBS_V2_120HZ_VERSION
    payload["action_version"] = RIVAL2_ACTION_V2_120HZ_VERSION
    payload["physics_hz"] = 120
    payload["policy_hz"] = 120
    payload["total_agent_samples"] = 0
    payload["sample_accounting"] = {
        "source_30hz_agent_decisions": SOURCE_AGENT_DECISIONS,
        "agent_decisions_120hz": 0,
        "physical_physics_ticks_experienced": 0,
        "simulated_world_seconds": 0.0,
        "simulated_agent_seconds": 0.0,
        "decision_count_cross_cadence_comparable": False,
    }
    payload["opponent_assignment"] = torch.full(
        (world_count,), -1, dtype=torch.int64
    )
    payload["opponent_curriculum"] = _fresh_curriculum_state(
        source,
        retention_observations,
        retention_summary,
        world_count,
    )
    for entry in payload["historical_opponents"]:
        entry["policy_hz"] = 30
        entry["action_version"] = "RIVAL2_ACTION_V1"

    source_transition = copy.deepcopy(source.get("curriculum_transition"))
    payload["curriculum_transition"] = {
        "identity": "RIVAL2_120HZ_BOOTSTRAP_FROM_ITERATION_479_V1",
        "source_checkpoint": source_report,
        "source_curriculum_transition": source_transition,
        "source_iteration": SOURCE_ITERATION,
        "source_policy_version": SOURCE_POLICY_VERSION,
        "source_30hz_agent_decisions": SOURCE_AGENT_DECISIONS,
        "destination_world_count": world_count,
        "destination_contract_hashes": dict(payload["contract_hashes"]),
        "destination_ppo_identity": RIVAL2_PPO_120HZ_V1,
        "destination_ppo_contract_hash": RIVAL2_PPO_120HZ_CONTRACT_HASH,
        "preserved_checkpoint_state": [
            "model_parameter_tensors_exact",
            "optimizer_adam_moments_exact",
            "optimizer_parameter_step_counters_exact",
            "torch_cpu_rng_state_exact",
            "torch_cuda_rng_state_exact",
            "policy_generator_rng_state_exact",
            "opponent_generator_rng_state_exact",
            "opponent_curriculum_configuration_exact",
            "opponent_curriculum_rng_state_exact_before_fresh_assignments",
            "historical_policy_pool_model_tensors_exact",
            "historical_realized_family_assignment_totals",
            "wisp_observation_generator_rng_state_exact",
        ],
        "intentionally_reinitialized_state": [
            "32768_world_simulator_allocation",
            "short_episode_physical_state_and_timers",
            "per_world_opponent_family_assignments",
            "per_world_rival_side_assignments",
            "per_world_historical_snapshot_assignments",
            "nexto_temporal_and_kickoff_state",
            "wisp_action_delay_and_eta_state",
            "historical_rival_action_hold_cache",
            "120hz_agent_decision_and_physical_exposure_counters",
            "120hz_retention_observation_corpus",
        ],
        "training_performed": False,
        "behavior_cloning_performed": False,
        "optimizer_steps_performed": 0,
    }

    migrated_model_hash = tensor_tree_sha256(payload["model"])
    migrated_optimizer_hash = tensor_tree_sha256(payload["optimizer"])
    migrated_pool_hash = tensor_tree_sha256(payload["historical_opponents"])
    # Pool metadata is intentionally added, so compare only model tensors for
    # the historical snapshots as a separate invariant.
    source_pool_models_hash = tensor_tree_sha256(
        [entry["model"] for entry in source["historical_opponents"]]
    )
    migrated_pool_models_hash = tensor_tree_sha256(
        [entry["model"] for entry in payload["historical_opponents"]]
    )
    proof = {
        "source_model_tensor_sha256": source_model_hash,
        "migrated_model_tensor_sha256": migrated_model_hash,
        "source_optimizer_tensor_sha256": source_optimizer_hash,
        "migrated_optimizer_tensor_sha256": migrated_optimizer_hash,
        "source_historical_pool_tensor_sha256": source_pool_hash,
        "migrated_historical_pool_tensor_sha256_with_cadence_metadata": (
            migrated_pool_hash
        ),
        "source_historical_pool_models_sha256": source_pool_models_hash,
        "migrated_historical_pool_models_sha256": migrated_pool_models_hash,
        "optimizer_step_counters": sorted(
            {
                int(state["step"].item())
                for state in payload["optimizer"]["state"].values()
                if "step" in state
            }
        ),
        "checks": {
            "model_tensors_exact": migrated_model_hash == source_model_hash,
            "optimizer_state_exact": migrated_optimizer_hash == source_optimizer_hash,
            "historical_pool_model_tensors_exact": (
                migrated_pool_models_hash == source_pool_models_hash
            ),
            "no_optimizer_steps": True,
            "source_checkpoint_identity_exact": all(
                source_report["checks"].values()
            ),
        },
    }
    if not all(proof["checks"].values()):
        raise RuntimeError(f"120 Hz bootstrap preservation failed: {proof}")
    payload["curriculum_transition"]["tensor_preservation_proof"] = copy.deepcopy(
        proof
    )
    return payload, proof


__all__ = [
    "BOOTSTRAP_WORLD_COUNT",
    "SOURCE_AGENT_DECISIONS",
    "SOURCE_ITERATION",
    "SOURCE_POLICY_VERSION",
    "SOURCE_SHA256",
    "build_bootstrap_payload",
    "file_sha256",
    "tensor_tree_sha256",
    "verify_source_checkpoint",
]
