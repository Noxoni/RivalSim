"""Build and validate the no-learning Rival 2.0 120 Hz bootstrap line."""

from __future__ import annotations

import argparse
import gc
import json
import math
import sys
import time
from dataclasses import asdict
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import torch
import warp as wp

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from rivalsim.arena import ArenaGeometry, WarpArenaMeshes  # noqa: E402
from rivalsim.rival2_120hz_transition import (  # noqa: E402
    BOOTSTRAP_WORLD_COUNT,
    build_bootstrap_payload,
    file_sha256,
    tensor_tree_sha256,
    verify_source_checkpoint,
)
from rivalsim.rival2_contracts import (  # noqa: E402
    OBS_FIELD_NAMES,
    OBSERVATION_SCHEMA_V2_120HZ_HASH,
    POSITION_SCALE,
    RIVAL2_REWARD_GAMEPLAY_120_V1_VERSION,
)
from rivalsim.rival2_env import Rival2Env  # noqa: E402
from rivalsim.rival2_mixed_ppo import (  # noqa: E402
    Rival2MixedPPOSafetyConfig,
    build_retention_observation_corpus,
)
from rivalsim.rival2_opponent_curriculum import (  # noqa: E402
    OPPONENT_NAMES,
    Rival2OpponentCurriculumConfig,
    Rival2OpponentCurriculumTrainer,
)
from rivalsim.rival2_policy import Rival2PolicyConfig  # noqa: E402
from rivalsim.rival2_ppo import (  # noqa: E402
    RIVAL2_PPO_120HZ_CONTRACT_HASH,
    Rival2PPOConfig,
    rival2_ppo_120hz_config,
)
from rivalsim.rival2_training import Rival2SelfPlayConfig, Rival2Trainer  # noqa: E402
from rivalsim.viewer.spectator import RivalVisSession  # noqa: E402

SOURCE = Path(
    r"G:\dev\RivalSim-runs\opponent-curriculum-v1-safe-20260827-b2af03d"
    r"\checkpoints\rival2_opponent_curriculum_plus_120_resume.pt"
)
RESULT_ROOT = Path("results/rival2/120hz_transition_v1")
RETENTION_ARTIFACT = RESULT_ROOT / "retention_corpus_120hz.pt"
RETENTION_SUMMARY = RESULT_ROOT / "retention_corpus_120hz.json"
BOOTSTRAP = Path(
    "checkpoints/rival2/120hz_bootstrap/rival2_120hz_from_iteration_479.pt"
)
VALIDATION = RESULT_ROOT / "cuda_rollout_validation.json"
TRANSITION = RESULT_ROOT / "bootstrap_transition.json"
# Keep the collection buffer bounded while giving each trajectory enough
# physical time to leave kickoff, traverse field regions, become airborne, and
# recover.  The earlier 8192 x 240 candidate covered all behavior categories
# but correctly failed the field-position diversity gate after only two seconds.
RETENTION_WORLDS = 1_024
RETENTION_DECISIONS = 2_048
RETENTION_SEED = 2_026_082_812
VALIDATION_SEED = 2_026_082_813


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
        newline="\n",
    )


def _torch_artifact_hash(path: Path) -> dict[str, Any]:
    return {
        "path": path.as_posix(),
        "bytes": path.stat().st_size,
        "sha256": file_sha256(path),
    }


def _state_coverage(observations: torch.Tensor) -> dict[str, Any]:
    field = {name: OBS_FIELD_NAMES.index(name) for name in OBS_FIELD_NAMES}
    self_x = observations[:, field["self.position.x"]] * POSITION_SCALE[0]
    self_y = observations[:, field["self.position.y"]] * POSITION_SCALE[1]
    self_z = observations[:, field["self.position.z"]] * POSITION_SCALE[2]
    heading = torch.atan2(
        observations[:, field["self.forward.y"]],
        observations[:, field["self.forward.x"]],
    )
    x_region = torch.bucketize(
        self_x, torch.tensor([-1365.0, 1365.0], device=observations.device)
    )
    y_region = torch.bucketize(
        self_y, torch.tensor([-1706.0, 1706.0], device=observations.device)
    )
    heading_octant = (
        torch.floor((heading + math.pi) / (math.pi / 4.0))
        .clamp(0, 7)
        .to(torch.int64)
    )
    return {
        "self_position_x_min_max_uu": [float(self_x.min()), float(self_x.max())],
        "self_position_y_min_max_uu": [float(self_y.min()), float(self_y.max())],
        "self_position_z_min_max_uu": [float(self_z.min()), float(self_z.max())],
        "occupied_x_field_regions_of_3": int(torch.unique(x_region).numel()),
        "occupied_y_field_regions_of_3": int(torch.unique(y_region).numel()),
        "occupied_heading_octants_of_8": int(torch.unique(heading_octant).numel()),
        "heading_octant_counts": torch.bincount(
            heading_octant, minlength=8
        ).cpu().tolist(),
    }


def build_retention(
    source: dict[str, Any],
    source_report: dict[str, Any],
    collision_root: Path,
    geometry: ArenaGeometry,
    meshes: WarpArenaMeshes,
    device: str,
) -> tuple[torch.Tensor, dict[str, Any]]:
    base = rival2_ppo_120hz_config()
    collection_config = Rival2PPOConfig(
        **{**asdict(base), "rollout_horizon": RETENTION_DECISIONS}
    )
    kickoff_selector = (
        np.arange(RETENTION_WORLDS, dtype=np.int32) + RETENTION_SEED
    ) % 5
    env = Rival2Env(
        RETENTION_WORLDS,
        str(collision_root),
        device=device,
        seed=RETENTION_SEED,
        reward_version=RIVAL2_REWARD_GAMEPLAY_120_V1_VERSION,
        geometry=geometry,
        meshes=meshes,
        kickoff_selector=kickoff_selector,
        car_visitation_order="a_then_b",
    )
    trainer = Rival2Trainer(
        env,
        policy_config=Rival2PolicyConfig(**source["policy_config"]),
        ppo_config=collection_config,
        self_play_config=Rival2SelfPlayConfig(**source["self_play_config"]),
        seed=RETENTION_SEED,
    )
    trainer.model.load_state_dict(source["model"])
    trainer.policy_version = int(source["policy_version"])
    trainer.iteration = int(source["iteration"])
    model_before = tensor_tree_sha256(trainer.model.state_dict())
    optimizer_before = tensor_tree_sha256(trainer.optimizer.state_dict())
    rollout = trainer.collect_rollout()
    torch.cuda.synchronize(env.device)
    model_after = tensor_tree_sha256(trainer.model.state_dict())
    optimizer_after = tensor_tree_sha256(trainer.optimizer.state_dict())
    safety = Rival2MixedPPOSafetyConfig()
    observations, summary = build_retention_observation_corpus(
        rollout,
        corpus_size=safety.retention_corpus_size,
    )
    coverage = _state_coverage(observations)
    required_categories = (
        "ordinary_ground_play",
        "possession_ball_approach",
        "near_ball_interaction",
        "recovery",
        "airborne",
    )
    category_coverage = {
        name: int(summary["category_counts"][name]["selected"]) > 0
        for name in required_categories
    }
    summary.update(
        {
            "schema_version": 2,
            "identity": "RIVAL2_RETENTION_120HZ_FROM_ITERATION_479_V1",
            "observation_contract": "RIVAL2_OBS_V2_120HZ",
            "observation_contract_sha256": OBSERVATION_SCHEMA_V2_120HZ_HASH,
            "created_utc": datetime.now(UTC).isoformat(),
            "source_identity": source_report,
            "collection": {
                "worlds": RETENTION_WORLDS,
                "decisions_per_world": RETENTION_DECISIONS,
                "physics_ticks_per_decision": 1,
                "policy_hz": 120,
                "physics_hz": 120,
                "agent_observations_considered": int(rollout.train_mask.sum()),
                "seed": RETENTION_SEED,
                "all_five_kickoff_layouts": True,
                "migrated_iteration_479_actor_on_both_sides": True,
                "stochastic_actions": True,
                "training_performed": False,
                "optimizer_steps": 0,
            },
            "safety_config_hash": safety.content_hash,
            "state_coverage": coverage,
            "required_category_coverage": category_coverage,
        }
    )
    summary["checks"].update(
        {
            "observation_contract_v2_120hz": True,
            "required_category_coverage": all(category_coverage.values()),
            "field_position_orientation_coverage": (
                coverage["occupied_x_field_regions_of_3"] == 3
                and coverage["occupied_y_field_regions_of_3"] == 3
                and coverage["occupied_heading_octants_of_8"] == 8
            ),
            "source_model_unchanged": model_before == model_after,
            "collection_optimizer_unchanged": optimizer_before == optimizer_after,
            "no_optimizer_update": True,
        }
    )
    summary["verdict"] = (
        "PASS_GREEN" if all(summary["checks"].values()) else "FAIL_RED"
    )
    if summary["verdict"] != "PASS_GREEN":
        raise RuntimeError(f"120 Hz retention corpus failed: {summary['checks']}")
    host_observations = observations.detach().cpu().clone()
    RETENTION_ARTIFACT.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "format": "RIVAL2_RETENTION_OBSERVATIONS_120HZ_V1",
            "observations": host_observations,
            "summary": summary,
        },
        RETENTION_ARTIFACT,
    )
    summary["artifact"] = _torch_artifact_hash(RETENTION_ARTIFACT)
    _write_json(RETENTION_SUMMARY, summary)
    del rollout, trainer, env, observations
    gc.collect()
    torch.cuda.empty_cache()
    return host_observations, summary


def build_target_trainer(
    provisional: Path,
    collision_root: Path,
    geometry: ArenaGeometry,
    meshes: WarpArenaMeshes,
    device: str,
) -> Rival2OpponentCurriculumTrainer:
    kickoff_selector = (
        np.arange(BOOTSTRAP_WORLD_COUNT, dtype=np.int32) + VALIDATION_SEED
    ) % 5
    env = Rival2Env(
        BOOTSTRAP_WORLD_COUNT,
        str(collision_root),
        device=device,
        seed=VALIDATION_SEED,
        reward_version=RIVAL2_REWARD_GAMEPLAY_120_V1_VERSION,
        geometry=geometry,
        meshes=meshes,
        kickoff_selector=kickoff_selector,
        car_visitation_order="a_then_b",
    )
    payload = torch.load(provisional, map_location="cpu", weights_only=False)
    trainer = Rival2OpponentCurriculumTrainer(
        env,
        policy_config=Rival2PolicyConfig(**payload["policy_config"]),
        ppo_config=Rival2PPOConfig(**payload["ppo_config"]),
        self_play_config=Rival2SelfPlayConfig(**payload["self_play_config"]),
        opponent_curriculum=Rival2OpponentCurriculumConfig(
            **payload["opponent_curriculum"]["config"]
        ),
        seed=VALIDATION_SEED,
    )
    trainer.load_checkpoint(provisional)
    trainer.initialize_curriculum_assignments()
    return trainer


def validate_viewer(bootstrap: Path, collision_root: Path, device: str) -> dict[str, Any]:
    """Smoke the checkpoint-aware viewer without mutating the checkpoint."""

    digest_before = file_sha256(bootstrap)
    session = RivalVisSession(
        bootstrap,
        collision_root=collision_root,
        device=device,
        seed=VALIDATION_SEED,
        stochastic=False,
        opponent="self",
    )
    try:
        tick_before = session.env.world.tick_count
        decision_before = session.policy_decision
        session.advance_policy_decision()
        first_action = session.current_action.clone()
        session.advance_policy_decision()
        second_action = session.current_action.clone()
        checks = {
            "checkpoint_action_v2": (
                session.checkpoint_info.action_version == "RIVAL2_ACTION_V2_120HZ"
            ),
            "checkpoint_observation_v2": (
                session.checkpoint_info.observation_version == "RIVAL2_OBS_V2_120HZ"
            ),
            "viewer_policy_hz_120": session.env.policy_hz == 120,
            "viewer_one_tick_per_decision": session.env.physics_ticks_per_decision == 1,
            "two_decisions_advance_two_physics_ticks": (
                session.env.world.tick_count - tick_before == 2
                and session.policy_decision - decision_before == 2
            ),
            "finite_actions": bool(
                torch.isfinite(first_action).all() and torch.isfinite(second_action).all()
            ),
            "checkpoint_unchanged": session.checkpoint_unchanged(),
        }
        result = {
            "identity": "RIVALVIS_RIVAL2_ACTION_V2_120HZ_SMOKE_V1",
            "inference_contract_binding": session.env.inference_contract_binding,
            "checks": checks,
            "verdict": "PASS_GREEN" if all(checks.values()) else "FAIL_RED",
        }
    finally:
        session.close()
    if file_sha256(bootstrap) != digest_before:
        raise RuntimeError("viewer smoke mutated the 120 Hz bootstrap checkpoint")
    if result["verdict"] != "PASS_GREEN":
        raise RuntimeError(f"120 Hz RivalVis smoke failed: {result['checks']}")
    return result


def _profile_rollout(
    trainer: Rival2OpponentCurriculumTrainer,
    source_path: Path,
    bootstrap_path: Path,
) -> dict[str, Any]:
    source_hash_before = file_sha256(source_path)
    bootstrap_hash_before = file_sha256(bootstrap_path)
    model_before = tensor_tree_sha256(trainer.model.state_dict())
    optimizer_before = tensor_tree_sha256(trainer.optimizer.state_dict())
    iteration_before = trainer.iteration
    policy_version_before = trainer.policy_version
    optimizer_steps_before = sorted(
        {
            int(state["step"].item())
            for state in trainer.optimizer.state.values()
            if "step" in state
        }
    )
    policy_events: list[tuple[torch.cuda.Event, torch.cuda.Event]] = []
    physics_events: list[tuple[torch.cuda.Event, torch.cuda.Event]] = []
    original_policy_outputs = trainer._policy_outputs
    original_world_step = trainer.env.world.step

    def profiled_policy_outputs(observation: torch.Tensor):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        output = original_policy_outputs(observation)
        end.record()
        policy_events.append((start, end))
        return output

    def profiled_world_step(count: int) -> None:
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        original_world_step(count)
        end.record()
        physics_events.append((start, end))

    trainer._policy_outputs = profiled_policy_outputs  # type: ignore[method-assign]
    trainer.env.world.step = profiled_world_step  # type: ignore[method-assign]
    trainer.env.reset_transfer_counters()
    torch.cuda.synchronize(trainer.device)
    torch.cuda.reset_peak_memory_stats(trainer.device)
    baseline_allocated = torch.cuda.memory_allocated(trainer.device)
    baseline_reserved = torch.cuda.memory_reserved(trainer.device)
    wall_start = time.perf_counter()
    rollout = trainer.collect_rollout()
    torch.cuda.synchronize(trainer.device)
    wall_seconds = time.perf_counter() - wall_start
    policy_inference_ms = sum(start.elapsed_time(end) for start, end in policy_events)
    physics_ms = sum(start.elapsed_time(end) for start, end in physics_events)
    peak_allocated = torch.cuda.max_memory_allocated(trainer.device)
    peak_reserved = torch.cuda.max_memory_reserved(trainer.device)
    free_bytes, total_bytes = torch.cuda.mem_get_info(trainer.device)
    model_after = tensor_tree_sha256(trainer.model.state_dict())
    optimizer_after = tensor_tree_sha256(trainer.optimizer.state_dict())
    optimizer_steps_after = sorted(
        {
            int(state["step"].item())
            for state in trainer.optimizer.state.values()
            if "step" in state
        }
    )
    action_delta = torch.abs(rollout.actions[1:] - rollout.actions[:-1])
    consecutive_change = torch.any(action_delta > 1.0e-6, dim=-1)
    family_counts = torch.bincount(trainer.opponent_family, minlength=4).cpu().tolist()
    transfer = trainer.env.hot_path_transfer_bytes()
    checks = {
        "rollout_shape_exact": list(rollout.observations.shape)
        == [128, BOOTSTRAP_WORLD_COUNT, 2, 182],
        "finite_observations": bool(torch.isfinite(rollout.observations).all()),
        "finite_actions": bool(torch.isfinite(rollout.actions).all()),
        "finite_rewards": bool(torch.isfinite(rollout.rewards).all()),
        "one_tick_action_contract": trainer.env.physics_ticks_per_decision == 1,
        "consecutive_action_changes_observed": bool(consecutive_change.any()),
        "simulator_hot_path_h2d_zero": transfer["h2d"] == 0,
        "simulator_hot_path_d2h_zero": transfer["d2h"] == 0,
        "named_mechanics_hot_path_absent": trainer.env.world.gameplay_v3 is None,
        "physical_guard_present": trainer.env.world.gameplay_120 is not None,
        "model_unchanged": model_before == model_after,
        "optimizer_state_unchanged": optimizer_before == optimizer_after,
        "optimizer_step_counters_unchanged": (
            optimizer_steps_before == optimizer_steps_after
        ),
        "iteration_unchanged": trainer.iteration == iteration_before,
        "policy_version_unchanged": trainer.policy_version == policy_version_before,
        "bootstrap_file_unchanged": file_sha256(bootstrap_path)
        == bootstrap_hash_before,
        "source_file_unchanged": file_sha256(source_path) == source_hash_before,
        "no_ppo_update": True,
        "no_behavior_cloning": True,
    }
    result = {
        "identity": "RIVAL2_120HZ_32768X128_ROLLOUT_ONLY_VALIDATION_V1",
        "created_utc": datetime.now(UTC).isoformat(),
        "worlds": BOOTSTRAP_WORLD_COUNT,
        "rollout_horizon": 128,
        "physics_hz": 120,
        "policy_hz": 120,
        "physics_ticks_per_decision": 1,
        "logical_rollout_buffer_bytes": rollout.logical_bytes,
        "environment_logical_state_bytes": trainer.env.world.logical_state_bytes,
        "memory": {
            "device_total_bytes": total_bytes,
            "baseline_allocated_bytes": baseline_allocated,
            "baseline_reserved_bytes": baseline_reserved,
            "peak_allocated_bytes": peak_allocated,
            "peak_reserved_bytes": peak_reserved,
            "free_bytes_after_rollout": free_bytes,
            "allocated_headroom_at_peak_bytes": total_bytes - peak_allocated,
            "reserved_headroom_at_peak_bytes": total_bytes - peak_reserved,
            "ppo_memory_headroom_interpretation": (
                "free/headroom with the complete rollout resident; no PPO update was run"
            ),
        },
        "timing": {
            "rollout_wall_seconds": wall_seconds,
            "rival_and_historical_policy_inference_seconds": policy_inference_ms / 1000.0,
            "physics_reward_seconds": physics_ms / 1000.0,
            "other_adapter_sampling_observation_seconds": max(
                0.0, wall_seconds - (policy_inference_ms + physics_ms) / 1000.0
            ),
            "world_physics_ticks_per_wall_second": (
                BOOTSTRAP_WORLD_COUNT * 128 / wall_seconds
            ),
            "trainable_agent_decisions_per_wall_second": (
                int(rollout.train_mask.sum()) / wall_seconds
            ),
        },
        "opponent_family_worlds_after_rollout": {
            OPPONENT_NAMES[index]: int(family_counts[index]) for index in range(4)
        },
        "consecutive_action_change_fraction": float(consecutive_change.float().mean()),
        "terminated_events": int(rollout.terminated[:, :, 0].sum()),
        "truncated_events": int(rollout.truncated[:, :, 0].sum()),
        "simulator_transfer_bytes": transfer,
        "nexto_reported_timed_h2d_bytes": trainer.nexto.timed_h2d_bytes,
        "nexto_reported_timed_d2h_bytes": trainer.nexto.timed_d2h_bytes,
        "wisp_source_specific_host_eta_note": (
            "Pinned Wisp retains its accepted Windows CPU scalar ETA cache; this is not a "
            "RivalSim world-state or Rival-policy transfer and its cadence was not changed."
        ),
        "model_tensor_sha256_before_after": [model_before, model_after],
        "optimizer_tensor_sha256_before_after": [optimizer_before, optimizer_after],
        "optimizer_step_counters_before_after": [
            optimizer_steps_before,
            optimizer_steps_after,
        ],
        "checkpoint_sha256_before_after": [
            bootstrap_hash_before,
            file_sha256(bootstrap_path),
        ],
        "source_sha256_before_after": [source_hash_before, file_sha256(source_path)],
        "disposable_post_rollout_accounting_not_saved": trainer.checkpoint_payload()[
            "sample_accounting"
        ],
        "checks": checks,
        "verdict": "PASS_GREEN" if all(checks.values()) else "FAIL_RED",
    }
    if result["verdict"] != "PASS_GREEN":
        raise RuntimeError(f"120 Hz target rollout validation failed: {checks}")
    return result


def run(args: argparse.Namespace) -> dict[str, Any]:
    source, source_report = verify_source_checkpoint(args.source)
    geometry = ArenaGeometry.load_soccar(args.collision_dir)
    meshes = WarpArenaMeshes(geometry, args.device)
    retention, retention_summary = build_retention(
        source,
        source_report,
        args.collision_dir,
        geometry,
        meshes,
        args.device,
    )
    payload, preservation = build_bootstrap_payload(
        source,
        source_report,
        retention,
        retention_summary,
    )
    provisional = ROOT / ".tools" / "rival2_120hz_provisional.pt"
    provisional.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, provisional)
    trainer = build_target_trainer(
        provisional,
        args.collision_dir,
        geometry,
        meshes,
        args.device,
    )
    BOOTSTRAP.parent.mkdir(parents=True, exist_ok=True)
    trainer.save_checkpoint(BOOTSTRAP)
    bootstrap_hash_before = file_sha256(BOOTSTRAP)
    final_payload = torch.load(BOOTSTRAP, map_location="cpu", weights_only=False)
    bootstrap_checks = {
        "iteration_479": int(final_payload["iteration"]) == 479,
        "policy_version_479": int(final_payload["policy_version"]) == 479,
        "new_120hz_decisions_zero": int(final_payload["total_agent_samples"]) == 0,
        "source_30hz_decisions_preserved": int(
            final_payload["sample_accounting"]["source_30hz_agent_decisions"]
        )
        == 3_655_854_038,
        "model_tensors_exact": tensor_tree_sha256(final_payload["model"])
        == source_report["model_tensor_sha256"],
        "optimizer_state_exact": tensor_tree_sha256(final_payload["optimizer"])
        == source_report["optimizer_tensor_sha256"],
        "ppo_timing_hash_exact": final_payload["ppo_config_hash"]
        == rival2_ppo_120hz_config().content_hash,
        "policy_hz_120": int(final_payload["policy_hz"]) == 120,
        "physics_hz_120": int(final_payload["physics_hz"]) == 120,
        "retention_v2_present": final_payload["opponent_curriculum"]["adaptive_ppo"][
            "retention_corpus_summary"
        ]["observation_contract"]
        == "RIVAL2_OBS_V2_120HZ",
        "no_training": final_payload["curriculum_transition"]["training_performed"]
        is False,
        "no_behavior_cloning": final_payload["curriculum_transition"][
            "behavior_cloning_performed"
        ]
        is False,
    }
    if not all(bootstrap_checks.values()):
        raise RuntimeError(f"final bootstrap verification failed: {bootstrap_checks}")
    viewer = validate_viewer(BOOTSTRAP, args.collision_dir, args.device)
    transition = {
        "identity": "RIVAL2_120HZ_BOOTSTRAP_TRANSITION_EVIDENCE_V1",
        "created_utc": datetime.now(UTC).isoformat(),
        "source": source_report,
        "bootstrap": _torch_artifact_hash(BOOTSTRAP),
        "retention": retention_summary["artifact"],
        "tensor_preservation": preservation,
        "ppo_timing_contract_sha256": RIVAL2_PPO_120HZ_CONTRACT_HASH,
        "fresh_assignment_family_counts": {
            OPPONENT_NAMES[index]: int(value)
            for index, value in enumerate(
                torch.bincount(trainer.opponent_family, minlength=4).cpu().tolist()
            )
        },
        "viewer_smoke": viewer,
        "checks": bootstrap_checks,
        "verdict": "PASS_GREEN",
    }
    _write_json(TRANSITION, transition)
    validation = _profile_rollout(trainer, args.source, BOOTSTRAP)
    if file_sha256(BOOTSTRAP) != bootstrap_hash_before:
        raise RuntimeError("bootstrap changed during rollout-only validation")
    _write_json(VALIDATION, validation)
    return {
        "verdict": "PASS_GREEN",
        "source": source_report,
        "bootstrap": transition["bootstrap"],
        "retention": transition["retention"],
        "validation": {
            "path": VALIDATION.as_posix(),
            "verdict": validation["verdict"],
        },
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, default=SOURCE)
    parser.add_argument(
        "--collision-dir",
        type=Path,
        default=Path(r"G:\dev\RLBot-Rival\bot\collision_meshes"),
    )
    parser.add_argument("--device", default="cuda:0")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if not torch.cuda.is_available() or not wp.is_cuda_available():
        raise RuntimeError("CUDA PyTorch and Warp are required")
    torch.cuda.set_device(args.device)
    torch.set_float32_matmul_precision("highest")
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    result = run(args)
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
