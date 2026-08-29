"""Run the authorized Human BC V1 -> clean 120 Hz self-play PPO campaign.

The preflight mode is deliberately optimizer-free and must be committed and
pushed before training mode can cross the first PPO optimizer-step boundary.
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import subprocess
import sys
import time
from collections.abc import Mapping, Sequence
from dataclasses import asdict
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from rivalsim.arena import ArenaGeometry, WarpArenaMeshes  # noqa: E402
from rivalsim.rival2_120hz_transition import tensor_tree_sha256  # noqa: E402
from rivalsim.rival2_contracts import (  # noqa: E402
    ACTION_CONTRACT_V2_120HZ_HASH,
    OBSERVATION_SCHEMA_V2_120HZ_HASH,
    REWARD_GAMEPLAY_120_V1_CONTRACT,
    REWARD_GAMEPLAY_120_V1_CONTRACT_HASH,
    RIVAL2_ACTION_V2_120HZ_VERSION,
    RIVAL2_OBS_V2_120HZ_VERSION,
    RIVAL2_REWARD_GAMEPLAY_120_V1_VERSION,
    contract_hashes_for_reward,
)
from rivalsim.rival2_env import REWARD_MODE_GAMEPLAY_120_V1, Rival2Env  # noqa: E402
from rivalsim.rival2_mixed_ppo import (  # noqa: E402
    Rival2MixedPPOSafetyConfig,
    mixed_optimizer_learning_rates,
    probe_fresh_adam_first_minibatch,
    reset_policy_learning_rate_for_new_update,
)
from rivalsim.rival2_opponent_curriculum import (  # noqa: E402
    OPPONENT_NAMES,
    Rival2OpponentCurriculumConfig,
    Rival2OpponentCurriculumTrainer,
)
from rivalsim.rival2_policy import Rival2PolicyConfig  # noqa: E402
from rivalsim.rival2_ppo import (  # noqa: E402
    RIVAL2_PPO_120HZ_CONTRACT_HASH,
    Rival2KLGuardConfig,
    Rival2PolicyDisplacementRejected,
    Rival2PPOConfig,
    rival2_ppo_120hz_config,
)
from rivalsim.rival2_training import Rival2SelfPlayConfig  # noqa: E402

RESULTS_DIR = REPO_ROOT / "results/rival2/human_bc_ppo_v1"
CONFIG_PATH = RESULTS_DIR / "frozen_config.json"
PREFLIGHT_PATH = RESULTS_DIR / "mechanics_removal_preflight.json"
PRE_STEP_AUTHORITY_PATH = RESULTS_DIR / "pre_step_authority.json"
TRANSITION_SWEEP_PATH = RESULTS_DIR / "transition_lr_sweep.json"
WARMUP_AUTHORITY_PATH = RESULTS_DIR / "warmup_transition_authority.json"
TRANSITION_PATH = RESULTS_DIR / "transition.json"
RETENTION_PATH = RESULTS_DIR / "retention_corpus.json"
CURVE_PATH = RESULTS_DIR / "training_curve.jsonl"
MILESTONES_PATH = RESULTS_DIR / "checkpoint_milestones.json"
FINAL_EVIDENCE_PATH = RESULTS_DIR / "final_evidence.json"
FINAL_CHECKPOINT = (
    REPO_ROOT
    / "checkpoints/rival2/human_bc_ppo_v1/rival2_human_bc_ppo_10h.pt"
)

REQUIRED_PARENT = "90faba5918abefc032089c331c074a66b2391b9d"
BC_CHECKPOINT = REPO_ROOT / "checkpoints/rival2/human_bc_v1/rival2_human_bc_v1.pt"
BC_SHA256 = "560C2414C17039DC920126EA148BF73FE6CC4677EE440F043599A7E1C76D2874"
BOOTSTRAP_CHECKPOINT = (
    REPO_ROOT
    / "checkpoints/rival2/120hz_bootstrap/rival2_120hz_from_iteration_479.pt"
)
BOOTSTRAP_SHA256 = "ADAF8D015C340CAFAE857B7253FBBDE3A6C842C4EA0BB091B31F8B1C210ED350"
WORLD_COUNT = 32_768
TRAINING_DURATION_SECONDS = 10 * 60 * 60
CHECKPOINT_OFFSETS = (30,)
HISTORICAL_SNAPSHOT_INTERVAL = 30
CAMPAIGN_SEED = 2_026_082_907
CURRICULUM_SEED = 2_026_082_908
KL_GUARD = Rival2KLGuardConfig(0.10, 0.05)
SAFETY = Rival2MixedPPOSafetyConfig()
TRANSITION_LR_CANDIDATES = (
    2.5e-5,
    1.25e-5,
    6.25e-6,
    3.125e-6,
    1.5625e-6,
)
TRANSITION_LR_FLOOR = TRANSITION_LR_CANDIDATES[-1]
REJECTED_ANCHOR_LR = 1.0e-4
REJECTED_ANCHOR_MINIBATCH_KL = 0.414318710565567
REJECTED_ANCHOR_RETENTION_KL = 0.12192033976316452


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--collision-dir",
        type=Path,
        default=Path(r"G:\dev\RLBot-Rival\bot\collision_meshes\soccar"),
    )
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--preflight-only", action="store_true")
    parser.add_argument("--transition-sweep-only", action="store_true")
    parser.add_argument(
        "--work-dir",
        type=Path,
        help="external directory for rolling and milestone resumable checkpoints",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="resume the rolling checkpoint in --work-dir after validating its lineage",
    )
    return parser.parse_args()


def utc_now() -> str:
    return datetime.now(UTC).isoformat()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest().upper()


def git(*arguments: str) -> str:
    return subprocess.check_output(
        ["git", *arguments], cwd=REPO_ROOT, text=True
    ).strip()


def git_is_ancestor(ancestor: str, descendant: str) -> bool:
    return (
        subprocess.run(
            ["git", "merge-base", "--is-ancestor", ancestor, descendant],
            cwd=REPO_ROOT,
            check=False,
        ).returncode
        == 0
    )


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
        newline="\n",
    )


def append_jsonl(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8", newline="\n") as handle:
        handle.write(json.dumps(value, sort_keys=True, separators=(",", ":")) + "\n")


def json_safe(value: Any) -> Any:
    if isinstance(value, torch.Tensor):
        if value.numel() == 1:
            return value.item()
        return value.detach().cpu().tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Mapping):
        return {str(key): json_safe(item) for key, item in value.items()}
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        return [json_safe(item) for item in value]
    return value


def artifact(path: Path) -> dict[str, Any]:
    return {
        "path": path.relative_to(REPO_ROOT).as_posix()
        if path.is_relative_to(REPO_ROOT)
        else str(path),
        "bytes": path.stat().st_size,
        "sha256": sha256(path),
    }


def load_authority() -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    config = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
    if sha256(BC_CHECKPOINT) != BC_SHA256:
        raise RuntimeError("accepted Human BC V1 checkpoint SHA-256 mismatch")
    if sha256(BOOTSTRAP_CHECKPOINT) != BOOTSTRAP_SHA256:
        raise RuntimeError("120 Hz bootstrap checkpoint SHA-256 mismatch")
    bc = torch.load(BC_CHECKPOINT, map_location="cpu", weights_only=False)
    bootstrap = torch.load(BOOTSTRAP_CHECKPOINT, map_location="cpu", weights_only=False)
    checks = {
        "required_parent_is_ancestor": git_is_ancestor(REQUIRED_PARENT, "HEAD"),
        "human_bc_format": bc.get("format")
        == "RIVAL2_HUMAN_BEHAVIOR_CLONING_CHECKPOINT_V1",
        "human_bc_selected_step_160": int(bc["counters"]["accepted_optimizer_steps"])
        == 160,
        "human_bc_source_iteration_479": int(bc["counters"]["source_iteration"])
        == 479,
        "human_bc_ppo_requires_explicit_transition": (
            bc["resumability"]["ppo_resumable"] is False
            and bc["resumability"]["ppo_requires_explicit_new_transition_authority"]
            is True
        ),
        "human_bc_120hz_observation": bc["observation_version"]
        == RIVAL2_OBS_V2_120HZ_VERSION,
        "human_bc_120hz_action": bc["action_version"]
        == RIVAL2_ACTION_V2_120HZ_VERSION,
        "bootstrap_iteration_479": int(bootstrap["iteration"]) == 479,
        "bootstrap_policy_version_479": int(bootstrap["policy_version"]) == 479,
        "bootstrap_contracts_exact": bootstrap["contract_hashes"]
        == contract_hashes_for_reward(RIVAL2_REWARD_GAMEPLAY_120_V1_VERSION),
        "bootstrap_historical_pool_present": len(bootstrap["historical_opponents"])
        == 13,
        "bootstrap_legacy_pool_metadata_unchanged": all(
            int(entry["policy_hz"]) == 30
            and entry["action_version"] == "RIVAL2_ACTION_V1"
            for entry in bootstrap["historical_opponents"]
        ),
        "blocked_continuation_not_selected": (
            config["human_bc_parent"]["path"]
            == "checkpoints/rival2/human_bc_v1/rival2_human_bc_v1.pt"
        ),
    }
    if not all(checks.values()):
        raise RuntimeError(f"authority verification failed: {checks}")
    return config, bc, bootstrap


def ppo_config(*, horizon: int = 128) -> Rival2PPOConfig:
    base = rival2_ppo_120hz_config()
    return Rival2PPOConfig(**{**asdict(base), "rollout_horizon": horizon})


def load_geometry(collision_dir: Path, device: str) -> tuple[ArenaGeometry, WarpArenaMeshes]:
    geometry = ArenaGeometry.load_soccar(collision_dir)
    return geometry, WarpArenaMeshes(geometry, device)


def build_trainer(
    *,
    worlds: int,
    horizon: int,
    collision_dir: Path,
    geometry: ArenaGeometry,
    meshes: WarpArenaMeshes,
    device: str,
    bc: dict[str, Any],
    bootstrap: dict[str, Any],
) -> Rival2OpponentCurriculumTrainer:
    kickoff_selector = (np.arange(worlds, dtype=np.int32) + CAMPAIGN_SEED) % 5
    env = Rival2Env(
        worlds,
        str(collision_dir),
        device=device,
        seed=CAMPAIGN_SEED,
        reward_version=RIVAL2_REWARD_GAMEPLAY_120_V1_VERSION,
        geometry=geometry,
        meshes=meshes,
        kickoff_selector=kickoff_selector,
        car_visitation_order="a_then_b",
    )
    trainer = Rival2OpponentCurriculumTrainer(
        env,
        policy_config=Rival2PolicyConfig(**bc["policy_config"]),
        ppo_config=ppo_config(horizon=horizon),
        self_play_config=Rival2SelfPlayConfig(
            historical_chance=0.20,
            historical_pool_bound=16,
        ),
        opponent_curriculum=Rival2OpponentCurriculumConfig(
            nexto_probability=0.0,
            wisp_probability=0.0,
            current_probability=0.80,
            historical_probability=0.20,
            seed=CURRICULUM_SEED,
        ),
        seed=CAMPAIGN_SEED,
    )
    trainer.model.load_state_dict(bc["model"])
    trainer.policy_version = 480
    trainer.iteration = 479
    trainer.source_30hz_agent_decision_samples = int(
        bootstrap["sample_accounting"]["source_30hz_agent_decisions"]
    )
    trainer.opponent_pool.load_checkpoint_state(bootstrap["historical_opponents"])
    trainer.curriculum_transition = {
        "identity": "RIVAL2_HUMAN_BC_V1_TO_120HZ_PPO_TRANSITION_V1",
        "created_utc": utc_now(),
        "human_bc_parent": artifact(BC_CHECKPOINT),
        "human_bc_model_tensor_sha256": tensor_tree_sha256(bc["model"]),
        "human_bc_selected_step": 160,
        "human_bc_source_iteration": 479,
        "human_bc_source_policy_version": 479,
        "initial_ppo_policy_version": 480,
        "policy_version_480_semantics": (
            "accepted BC V1 step-160 actor/critic before its first PPO update; the new "
            "identity avoids collision with frozen historical policy version 479"
        ),
        "provenance_bootstrap": artifact(BOOTSTRAP_CHECKPOINT),
        "historical_iteration_479_optimizer": "provenance_only_not_restored",
        "historical_policy_pool_preserved": True,
        "historical_policy_pool_initial_versions": [
            int(entry["version"]) for entry in bootstrap["historical_opponents"]
        ],
        "fresh_ppo_optimizer": True,
        "human_bc_optimizer_restored": False,
        "historical_ppo_optimizer_restored": False,
        "human_demo_observation_adapter_used": False,
        "additional_behavior_cloning_performed": False,
        "named_mechanics_training_signal": False,
        "accepted_ppo_offset": 0,
    }
    trainer.initialize_curriculum_assignments()
    return trainer


def named_state_digest(trainer: Rival2OpponentCurriculumTrainer) -> dict[str, str]:
    return {
        "nexto": tensor_tree_sha256(
            {
                "previous_action": trainer.nexto.previous_action,
                "neural_counter": trainer.nexto.neural_counter,
                "kickoff_index": trainer.nexto.kickoff_index,
            }
        ),
        "wisp": tensor_tree_sha256(
            {
                "old_action": trainer.wisp.old_action,
                "new_action": trainer.wisp.new_action,
                "previous_action": trainer.wisp.previous_action,
                "ticks": trainer.wisp.ticks,
                "update_flag": trainer.wisp.update_flag,
            }
        ),
    }


def run_preflight(args: argparse.Namespace) -> dict[str, Any]:
    config, bc, bootstrap = load_authority()
    geometry, meshes = load_geometry(args.collision_dir, args.device)
    trainer = build_trainer(
        worlds=512,
        horizon=128,
        collision_dir=args.collision_dir,
        geometry=geometry,
        meshes=meshes,
        device=args.device,
        bc=bc,
        bootstrap=bootstrap,
    )
    source_model_hash = tensor_tree_sha256(bc["model"])
    model_before = tensor_tree_sha256(trainer.model.state_dict())
    optimizer_before_enable = trainer.optimizer.state_dict()
    opponent_state_before = named_state_digest(trainer)
    strict_before = trainer.env.bridge.views["rival2.strict_double_dash_count"].clone()
    proof = trainer.enable_safe_mixed_ppo(SAFETY)
    optimizer_after_enable = trainer.optimizer.state_dict()
    rollout = trainer.collect_rollout()
    torch.cuda.synchronize(trainer.device)
    metrics = trainer.last_rollout_gameplay_metrics
    curriculum = trainer.last_rollout_curriculum_metrics
    if metrics is None or curriculum is None:
        raise RuntimeError("clean 120 Hz preflight telemetry was not produced")
    strict_after = trainer.env.bridge.views["rival2.strict_double_dash_count"]
    family_counts = torch.bincount(trainer.opponent_family, minlength=4).cpu().tolist()
    component_abs = metrics["trusted_reward_component_absolute_sum"]
    inventory = trainer.env.world.gameplay_120.memory_inventory()
    contract = REWARD_GAMEPLAY_120_V1_CONTRACT
    checks = {
        "required_parent_is_ancestor": git_is_ancestor(REQUIRED_PARENT, "HEAD"),
        "bc_checkpoint_sha256_exact": sha256(BC_CHECKPOINT) == BC_SHA256,
        "bc_source_model_loaded_exactly": model_before == source_model_hash,
        "bootstrap_checkpoint_sha256_exact": sha256(BOOTSTRAP_CHECKPOINT)
        == BOOTSTRAP_SHA256,
        "active_reward_contract_exact": trainer.env.contract_hashes
        == contract_hashes_for_reward(RIVAL2_REWARD_GAMEPLAY_120_V1_VERSION),
        "active_observation_hash_exact": trainer.env.contract_hashes[
            RIVAL2_OBS_V2_120HZ_VERSION
        ]
        == OBSERVATION_SCHEMA_V2_120HZ_HASH,
        "active_action_hash_exact": trainer.env.contract_hashes[
            RIVAL2_ACTION_V2_120HZ_VERSION
        ]
        == ACTION_CONTRACT_V2_120HZ_HASH,
        "active_reward_hash_exact": trainer.env.contract_hashes[
            RIVAL2_REWARD_GAMEPLAY_120_V1_VERSION
        ]
        == REWARD_GAMEPLAY_120_V1_CONTRACT_HASH,
        "active_ppo_contract_exact": config["contracts"]["ppo_sha256"]
        == RIVAL2_PPO_120HZ_CONTRACT_HASH,
        "active_ppo_config_exact": trainer.ppo_config == rival2_ppo_120hz_config(),
        "physics_120_policy_120_hold_one": trainer.env.physics_hz == 120
        and trainer.env.policy_hz == 120
        and trainer.env.physics_ticks_per_decision == 1,
        "clean_reward_mode": trainer.env.world.reward_mode
        == REWARD_MODE_GAMEPLAY_120_V1,
        "gameplay_v3_not_allocated": trainer.env.world.gameplay_v3 is None,
        "gameplay_120_physical_guard_allocated": trainer.env.world.gameplay_120
        is not None,
        "named_mechanics_arrays_zero": inventory["named_mechanics_arrays"] == 0,
        "controlled_flick_arrays_zero": inventory["controlled_flick_arrays"] == 0,
        "unconditional_touch_reward_zero": contract["unconditional_unique_touch"]
        == 0.0,
        "named_mechanics_reward_zero": contract["named_mechanics_reward"] == 0.0,
        "named_mechanics_hot_path_false": contract["named_mechanics_hot_path"]
        is False,
        "recognized_mechanic_exemption_false": contract["bad_flip_guard"][
            "recognized_mechanic_exemption"
        ]
        is False,
        "controlled_flick_exemption_false": contract["bad_flip_guard"][
            "controlled_flick_exemption"
        ]
        is False,
        "only_two_physical_exemptions": contract["bad_flip_guard"][
            "active_exemptions_in_precedence_order"
        ]
        == ["EXEMPT_CONTESTED_50", "EXEMPT_POWER_CONTACT"],
        "generic_jump_penalty_zero": contract["bad_flip_guard"][
            "generic_jump_penalty"
        ]
        == 0.0,
        "generic_flip_penalty_zero": contract["bad_flip_guard"][
            "generic_flip_penalty"
        ]
        == 0.0,
        "historical_strict_dash_tracker_inactive": bool(
            torch.equal(strict_before, strict_after)
            and int(strict_after.abs().sum().item()) == 0
        ),
        "historical_strict_dash_reward_zero": component_abs[
            "strict_double_dash_component"
        ]
        == 0.0,
        "ordinary_touch_reward_component_zero": component_abs[
            "v1_touch_component"
        ]
        == 0.0,
        "nexto_probability_zero": trainer.opponent_curriculum.nexto_probability
        == 0.0,
        "wisp_probability_zero": trainer.opponent_curriculum.wisp_probability == 0.0,
        "nexto_assignments_zero": family_counts[2] == 0,
        "wisp_assignments_zero": family_counts[3] == 0,
        "nexto_trainable_samples_zero": curriculum["trainable_agent_samples"][
            "nexto"
        ]
        == 0,
        "wisp_trainable_samples_zero": curriculum["trainable_agent_samples"]["wisp"]
        == 0,
        "nexto_wisp_adapters_never_called": named_state_digest(trainer)
        == opponent_state_before,
        "current_current_majority_realized": family_counts[0] > family_counts[1],
        "legacy_pool_cadence_unchanged": all(
            hz == 30 and version == "RIVAL2_ACTION_V1"
            for hz, version in zip(
                trainer.opponent_pool.policy_hz,
                trainer.opponent_pool.action_versions,
                strict=True,
            )
        ),
        "fresh_optimizer_source_state_empty": len(optimizer_before_enable["state"])
        == 0,
        "fresh_split_optimizer_state_empty": len(optimizer_after_enable["state"])
        == 0,
        "fresh_optimizer_learning_rates_exact": mixed_optimizer_learning_rates(
            trainer.optimizer
        )
        == {"policy": 1.0e-4, "critic": 3.0e-4},
        "model_unchanged_by_preflight": tensor_tree_sha256(
            trainer.model.state_dict()
        )
        == model_before,
        "no_optimizer_steps": len(trainer.optimizer.state) == 0,
        "finite_observations_actions_rewards": bool(
            torch.isfinite(rollout.observations).all()
            and torch.isfinite(rollout.actions).all()
            and torch.isfinite(rollout.rewards).all()
        ),
        "human_demo_adapter_not_used": True,
        "no_additional_behavior_cloning": True,
    }
    report = {
        "format": "RIVAL2_HUMAN_BC_PPO_MECHANICS_REMOVAL_PREFLIGHT_V1",
        "created_utc": utc_now(),
        "git_head": git("rev-parse", "HEAD"),
        "origin_main": git("rev-parse", "origin/main"),
        "frozen_config": artifact(CONFIG_PATH),
        "human_bc_parent": artifact(BC_CHECKPOINT),
        "bootstrap_provenance": artifact(BOOTSTRAP_CHECKPOINT),
        "runtime_smoke": {
            "worlds": 512,
            "horizon": 128,
            "family_worlds_after_rollout": {
                name: int(family_counts[index])
                for index, name in enumerate(OPPONENT_NAMES)
            },
            "curriculum": curriculum,
            "gameplay": metrics,
            "reward_mode": int(trainer.env.world.reward_mode),
            "gameplay_120_memory_inventory": inventory,
        },
        "fresh_optimizer_transition": proof,
        "checks": checks,
        "optimizer_steps": 0,
        "verdict": "PASS_GREEN" if all(checks.values()) else "FAIL_RED",
    }
    write_json(PREFLIGHT_PATH, json_safe(report))
    authority = {
        "format": "RIVAL2_HUMAN_BC_PPO_PRE_STEP_AUTHORITY_V1",
        "created_utc": utc_now(),
        "required_parent": REQUIRED_PARENT,
        "required_parent_is_ancestor": git_is_ancestor(REQUIRED_PARENT, "HEAD"),
        "frozen_config": artifact(CONFIG_PATH),
        "mechanics_removal_preflight": artifact(PREFLIGHT_PATH),
        "human_bc_parent": artifact(BC_CHECKPOINT),
        "bootstrap_provenance": artifact(BOOTSTRAP_CHECKPOINT),
        "optimizer_steps": 0,
        "training_authorized_only_after_commit_and_push": True,
        "verdict": report["verdict"],
    }
    write_json(PRE_STEP_AUTHORITY_PATH, authority)
    if report["verdict"] != "PASS_GREEN":
        raise RuntimeError(f"mechanics-removal preflight failed: {checks}")
    return report


def require_committed_preflight() -> dict[str, Any]:
    if not PREFLIGHT_PATH.exists() or not PRE_STEP_AUTHORITY_PATH.exists():
        raise RuntimeError("mechanics-removal preflight evidence is missing")
    preflight = json.loads(PREFLIGHT_PATH.read_text(encoding="utf-8"))
    authority = json.loads(PRE_STEP_AUTHORITY_PATH.read_text(encoding="utf-8"))
    checks = {
        "preflight_pass": preflight.get("verdict") == "PASS_GREEN",
        "authority_pass": authority.get("verdict") == "PASS_GREEN",
        "preflight_hash_matches_authority": sha256(PREFLIGHT_PATH)
        == authority["mechanics_removal_preflight"]["sha256"],
        "config_hash_matches_authority": sha256(CONFIG_PATH)
        == authority["frozen_config"]["sha256"],
        "preflight_tracked": bool(
            git("ls-files", "--", str(PREFLIGHT_PATH.relative_to(REPO_ROOT)))
        ),
        "authority_tracked": bool(
            git("ls-files", "--", str(PRE_STEP_AUTHORITY_PATH.relative_to(REPO_ROOT)))
        ),
        "clean_worktree": git("status", "--porcelain") == "",
        "head_pushed_to_origin_main": git("rev-parse", "HEAD")
        == git("rev-parse", "origin/main"),
        "required_parent_is_ancestor": git_is_ancestor(REQUIRED_PARENT, "HEAD"),
    }
    if not all(checks.values()):
        raise RuntimeError(f"pre-step authority is not committed and pushed: {checks}")
    return {"preflight": preflight, "authority": authority, "checks": checks}


def warmup_safety_config(starting_policy_lr: float, *, active: bool) -> Rival2MixedPPOSafetyConfig:
    return Rival2MixedPPOSafetyConfig(
        initial_policy_learning_rate=starting_policy_lr,
        critic_learning_rate=3.0e-4,
        soft_minibatch_kl_target=0.02,
        retention_soft_mean_kl_target=0.02,
        policy_learning_rate_backoff=0.5,
        minimum_policy_learning_rate=(
            TRANSITION_LR_FLOOR if active else 2.5e-5
        ),
        retention_corpus_size=512,
    )


def advance_warmup_state(
    state: dict[str, Any], diagnostics: dict[str, Any], *, accepted_offset: int
) -> dict[str, Any]:
    result = dict(state)
    if not result["active"]:
        result["next_update_starting_policy_lr"] = 1.0e-4
        return result
    starting_lr = float(result["next_update_starting_policy_lr"])
    clean = (
        int(diagnostics["policy_learning_rate_backoffs"]) == 0
        and not bool(diagnostics["ppo_early_stop"])
    )
    result["warmup_updates_completed"] = int(result["warmup_updates_completed"]) + 1
    result["last_accepted_offset"] = accepted_offset
    result["last_update_starting_policy_lr"] = starting_lr
    result["last_update_policy_lr_end"] = float(
        diagnostics["policy_learning_rate_end"]
    )
    result["last_update_required_backoff"] = not clean
    result["last_update_retention_early_stop"] = bool(diagnostics["ppo_early_stop"])
    if clean and starting_lr == 1.0e-4:
        result["consecutive_clean_updates_at_1e-4"] = int(
            result["consecutive_clean_updates_at_1e-4"]
        ) + 1
    elif starting_lr != 1.0e-4 or not clean:
        result["consecutive_clean_updates_at_1e-4"] = 0

    if int(result["consecutive_clean_updates_at_1e-4"]) >= 2:
        result["active"] = False
        result["normal_production_operation_start_offset"] = accepted_offset + 1
        result["next_update_starting_policy_lr"] = 1.0e-4
    elif clean:
        result["next_update_starting_policy_lr"] = min(starting_lr * 2.0, 1.0e-4)
    else:
        result["next_update_starting_policy_lr"] = float(
            diagnostics["policy_learning_rate_end"]
        )
    return result


def run_transition_sweep(args: argparse.Namespace) -> dict[str, Any]:
    committed_preflight = require_committed_preflight()
    _config, bc, bootstrap = load_authority()
    geometry, meshes = load_geometry(args.collision_dir, args.device)
    trainer = build_trainer(
        worlds=WORLD_COUNT,
        horizon=128,
        collision_dir=args.collision_dir,
        geometry=geometry,
        meshes=meshes,
        device=args.device,
        bc=bc,
        bootstrap=bootstrap,
    )
    model_before = tensor_tree_sha256(trainer.model.state_dict())
    optimizer_proof = trainer.enable_safe_mixed_ppo(SAFETY)
    retention_rollout = trainer.collect_rollout()
    retention_summary = trainer.initialize_retention_corpus_from_rollout(
        retention_rollout,
        source_identity={
            "identity": "HUMAN_BC_V1_STEP_160_TRANSITION_SWEEP_PARENT",
            "checkpoint_sha256": BC_SHA256,
            "model_tensor_sha256": model_before,
        },
    )
    first_training_rollout = trainer.collect_rollout()
    first_training_rollout.compute_gae(trainer.ppo_config)
    torch.cuda.synchronize(trainer.device)
    model_after_rollouts = tensor_tree_sha256(trainer.model.state_dict())
    optimizer_before_sweep = tensor_tree_sha256(trainer.optimizer.state_dict())
    generator_before_sweep = tensor_tree_sha256(trainer.policy_generator.get_state())

    anchor = probe_fresh_adam_first_minibatch(
        trainer.model,
        first_training_rollout,
        trainer.ppo_config,
        retention_observations=trainer.retention_observations,
        family_names=OPPONENT_NAMES,
        generator=trainer.policy_generator,
        policy_learning_rate=REJECTED_ANCHOR_LR,
        critic_learning_rate=3.0e-4,
        policy_config=trainer.policy_config,
        gae_ready=True,
    )
    candidates: list[dict[str, Any]] = []
    selected: dict[str, Any] | None = None
    for learning_rate in TRANSITION_LR_CANDIDATES:
        candidate = probe_fresh_adam_first_minibatch(
            trainer.model,
            first_training_rollout,
            trainer.ppo_config,
            retention_observations=trainer.retention_observations,
            family_names=OPPONENT_NAMES,
            generator=trainer.policy_generator,
            policy_learning_rate=learning_rate,
            critic_learning_rate=3.0e-4,
            policy_config=trainer.policy_config,
            gae_ready=True,
        )
        candidates.append(candidate)
        if selected is None and candidate["passes_transition_gate"]:
            selected = candidate
    torch.cuda.synchronize(trainer.device)

    anchor_checks = {
        "minibatch_kl_reproduced": abs(
            anchor["post_step_minibatch_kl"] - REJECTED_ANCHOR_MINIBATCH_KL
        )
        <= 1.0e-7,
        "retention_kl_reproduced": abs(
            anchor["retention_mean_kl"] - REJECTED_ANCHOR_RETENTION_KL
        )
        <= 1.0e-7,
        "anchor_rejected": not anchor["passes_transition_gate"],
    }
    checks = {
        "required_parent_is_ancestor": git_is_ancestor(REQUIRED_PARENT, "HEAD"),
        "mechanics_preflight_preserved": all(committed_preflight["checks"].values()),
        "bc_parent_sha256_unchanged": sha256(BC_CHECKPOINT) == BC_SHA256,
        "bc_model_unchanged_by_rollouts": model_before == model_after_rollouts,
        "anchor_reproduced": all(anchor_checks.values()),
        "same_first_minibatch_for_every_candidate": len(
            {
                anchor["minibatch_index_sha256"],
                *(candidate["minibatch_index_sha256"] for candidate in candidates),
            }
        )
        == 1,
        "candidate_order_exact": [
            candidate["policy_learning_rate"] for candidate in candidates
        ]
        == list(TRANSITION_LR_CANDIDATES),
        "every_probe_rolled_back_completely": anchor["rollback_complete"]
        and all(candidate["rollback_complete"] for candidate in candidates),
        "model_restored_after_every_probe": tensor_tree_sha256(
            trainer.model.state_dict()
        )
        == model_before,
        "fresh_optimizer_unchanged_and_empty": tensor_tree_sha256(
            trainer.optimizer.state_dict()
        )
        == optimizer_before_sweep
        and len(trainer.optimizer.state) == 0,
        "policy_generator_restored": tensor_tree_sha256(
            trainer.policy_generator.get_state()
        )
        == generator_before_sweep,
        "no_ppo_update_accepted": trainer.iteration == 479
        and trainer.policy_version == 480,
        "highest_passing_lr_selected": selected is not None
        and selected["policy_learning_rate"]
        == next(
            (
                candidate["policy_learning_rate"]
                for candidate in candidates
                if candidate["passes_transition_gate"]
            ),
            None,
        ),
    }
    report = {
        "format": "RIVAL2_HUMAN_BC_FRESH_ADAM_LR_SWEEP_V1",
        "created_utc": utc_now(),
        "git_head": git("rev-parse", "HEAD"),
        "human_bc_parent": artifact(BC_CHECKPOINT),
        "frozen_config": artifact(CONFIG_PATH),
        "mechanics_removal_preflight": artifact(PREFLIGHT_PATH),
        "optimizer_transition": optimizer_proof,
        "retention": retention_summary,
        "exact_first_rollout": {
            "trainable_samples": int(first_training_rollout.train_mask.sum().item()),
            "curriculum": trainer.last_rollout_curriculum_metrics,
            "gameplay": trainer.last_rollout_gameplay_metrics,
            "policy_generator_state_sha256": generator_before_sweep,
        },
        "rejected_1e-4_anchor": anchor,
        "rejected_anchor_checks": anchor_checks,
        "candidates_descending": candidates,
        "selected_initial_policy_lr": (
            None if selected is None else selected["policy_learning_rate"]
        ),
        "accepted_optimizer_steps": 0,
        "checks": checks,
        "verdict": "PASS_GREEN" if all(checks.values()) else "BLOCKED",
    }
    write_json(TRANSITION_SWEEP_PATH, json_safe(report))
    authority = {
        "format": "RIVAL2_HUMAN_BC_FRESH_PPO_WARMUP_AUTHORITY_V1",
        "created_utc": utc_now(),
        "required_parent": REQUIRED_PARENT,
        "human_bc_parent": artifact(BC_CHECKPOINT),
        "mechanics_removal_preflight": artifact(PREFLIGHT_PATH),
        "transition_lr_sweep": artifact(TRANSITION_SWEEP_PATH),
        "selected_initial_policy_lr": report["selected_initial_policy_lr"],
        "critic_learning_rate": 3.0e-4,
        "warmup_schedule": {
            "begin_each_update_at_last_successfully_accepted_starting_lr": True,
            "clean_update_next_lr_multiplier_max": 2.0,
            "normal_policy_lr_cap": 1.0e-4,
            "transition_backoff_floor": TRANSITION_LR_FLOOR,
            "soft_minibatch_kl_target": 0.02,
            "soft_retention_mean_kl_target": 0.02,
            "hard_minibatch_kl_guard": 0.10,
            "hard_completed_update_mean_kl_guard": 0.05,
            "increase_forbidden_after_backoff_or_retention_early_stop": True,
            "normal_mode_after_consecutive_clean_1e-4_updates": 2,
            "normal_mode_update_start_lr": 1.0e-4,
            "normal_mode_backoff_floor": 2.5e-5,
        },
        "training_authorized_only_after_this_file_is_committed_and_pushed": True,
        "accepted_optimizer_steps": 0,
        "checks": checks,
        "verdict": report["verdict"],
    }
    write_json(WARMUP_AUTHORITY_PATH, authority)
    del first_training_rollout, retention_rollout, trainer
    gc.collect()
    torch.cuda.empty_cache()
    if report["verdict"] != "PASS_GREEN":
        raise RuntimeError(f"fresh-Adam transition sweep blocked: {checks}")
    return report


def require_committed_warmup_authority() -> dict[str, Any]:
    if not TRANSITION_SWEEP_PATH.exists() or not WARMUP_AUTHORITY_PATH.exists():
        raise RuntimeError("fresh-Adam transition sweep authority is missing")
    sweep = json.loads(TRANSITION_SWEEP_PATH.read_text(encoding="utf-8"))
    authority = json.loads(WARMUP_AUTHORITY_PATH.read_text(encoding="utf-8"))
    selected = authority.get("selected_initial_policy_lr")
    checks = {
        "sweep_pass": sweep.get("verdict") == "PASS_GREEN",
        "authority_pass": authority.get("verdict") == "PASS_GREEN",
        "sweep_hash_matches_authority": sha256(TRANSITION_SWEEP_PATH)
        == authority["transition_lr_sweep"]["sha256"],
        "selected_lr_is_authorized_candidate": selected in TRANSITION_LR_CANDIDATES,
        "selected_lr_matches_sweep": selected
        == sweep.get("selected_initial_policy_lr"),
        "sweep_tracked": bool(
            git("ls-files", "--", str(TRANSITION_SWEEP_PATH.relative_to(REPO_ROOT)))
        ),
        "authority_tracked": bool(
            git("ls-files", "--", str(WARMUP_AUTHORITY_PATH.relative_to(REPO_ROOT)))
        ),
        "clean_worktree": git("status", "--porcelain") == "",
        "head_pushed_to_origin_main": git("rev-parse", "HEAD")
        == git("rev-parse", "origin/main"),
        "required_parent_is_ancestor": git_is_ancestor(REQUIRED_PARENT, "HEAD"),
        "bc_checkpoint_unchanged": sha256(BC_CHECKPOINT) == BC_SHA256,
    }
    if not all(checks.values()):
        raise RuntimeError(f"warmup authority is not committed and pushed: {checks}")
    return {
        "sweep": sweep,
        "authority": authority,
        "selected_initial_policy_lr": float(selected),
        "checks": checks,
    }


def summarize_update(
    offset: int,
    trainer: Rival2OpponentCurriculumTrainer,
    wall_seconds: float,
) -> dict[str, Any]:
    diagnostics = trainer.last_adaptive_ppo_diagnostics
    if diagnostics is None:
        raise RuntimeError("accepted update has no adaptive PPO diagnostics")
    curriculum = trainer.last_rollout_curriculum_metrics
    gameplay = trainer.last_rollout_gameplay_metrics
    if curriculum is None or gameplay is None:
        raise RuntimeError("accepted update has no rollout monitoring")
    zero_external = all(
        curriculum[name][family] == 0
        for name in (
            "world_decisions",
            "trainable_agent_samples",
            "terminated_episodes",
            "truncated_episodes",
        )
        for family in ("nexto", "wisp")
    )
    component_abs = gameplay["trusted_reward_component_absolute_sum"]
    hard_checks = {
        "adaptive_ppo_pass": diagnostics["verdict"] == "PASS_GREEN",
        "nexto_wisp_zero": zero_external,
        "named_mechanics_hot_path_absent": gameplay[
            "named_mechanics_hot_path_absent"
        ],
        "named_mechanics_arrays_zero": gameplay["named_mechanics_arrays"] == 0,
        "ordinary_touch_reward_zero": component_abs["v1_touch_component"] == 0.0,
        "strict_dash_reward_zero": component_abs["strict_double_dash_component"]
        == 0.0,
        "completed_update_kl_within_hard_guard": diagnostics[
            "completed_update_mean_kl"
        ]
        <= 0.05,
        "maximum_minibatch_kl_within_hard_guard": diagnostics[
            "maximum_post_step_minibatch_kl"
        ]
        <= 0.10,
    }
    if not all(hard_checks.values()):
        raise RuntimeError(f"post-update campaign guard failed: {hard_checks}")
    goal_events = round(component_abs["v1_goal_component"] / 10.0)
    ppo_safety_keys = (
        "schema_version",
        "mode",
        "safety_config",
        "safety_config_hash",
        "trainable_sample_count",
        "expected_optimizer_steps",
        "accepted_optimizer_steps",
        "optimizer_step_proposals",
        "optimizer_step_retries",
        "policy_learning_rate_backoffs",
        "policy_learning_rate_scope",
        "policy_learning_rate_before_update_reset",
        "policy_learning_rate_update_start_reset_applied",
        "policy_learning_rate_start",
        "policy_learning_rate_end",
        "policy_learning_rate_after_update_rearm",
        "next_update_policy_learning_rate",
        "critic_learning_rate_start_end",
        "ppo_early_stop",
        "ppo_early_stop_reason",
        "maximum_post_step_minibatch_kl",
        "completed_update_mean_kl",
        "retention_corpus_mean_kl",
        "retention_reference_actor_sha256",
        "retention_kl_by_action_channel",
        "rollout_analytic_kl_by_action_channel",
        "family_empirical_kl",
        "maximum_gradient_norms",
        "family_statistics",
        "retry_log",
        "checks",
        "verdict",
    )
    ppo_safety = {key: diagnostics[key] for key in ppo_safety_keys}
    ppo_safety["per_minibatch_accepted_step_log_persisted"] = False
    ppo_safety["per_minibatch_log_omission_reason"] = (
        "bounded 10-hour evidence; aggregate counts, maxima, channel KL, retries, "
        "gradients, and all guard checks are retained"
    )
    return json_safe(
        {
            "accepted_ppo_offset": offset,
            "iteration": trainer.iteration,
            "policy_version": trainer.policy_version,
            "cumulative_agent_decisions_120hz": trainer.agent_decisions_120hz,
            "cumulative_physics_ticks": trainer.physical_physics_ticks_experienced,
            "rollout_wall_seconds": wall_seconds,
            "goal_events": goal_events,
            "curriculum": curriculum,
            "gameplay": gameplay,
            "ppo_safety": ppo_safety,
            "post_update_checks": hard_checks,
        }
    )


def trend(rows: list[dict[str, Any]]) -> dict[str, Any]:
    def mean(selected: list[dict[str, Any]], path: tuple[str, ...]) -> float:
        values: list[float] = []
        for row in selected:
            value: Any = row
            for key in path:
                value = value[key]
            values.append(float(value))
        return sum(values) / len(values)

    first = rows[:10]
    last = rows[-10:]
    fields = {
        "touches_per_minute": ("gameplay", "touches_per_minute"),
        "goals_per_update": ("goal_events",),
        "no_touch_truncations_per_update": ("gameplay", "no_touch_truncations"),
        "unnecessary_flip_contacts_per_minute": (
            "gameplay",
            "unnecessary_flip_contacts_per_minute",
        ),
        "unnecessary_fraction_of_flip_active_contacts": (
            "gameplay",
            "unnecessary_fraction_of_flip_active_contacts",
        ),
        "movement_speed_uu_per_second": (
            "gameplay",
            "mean_movement_speed_uu_per_second",
        ),
        "analog_action_saturation_fraction": (
            "gameplay",
            "analog_action_saturation_fraction",
        ),
    }
    return {
        name: {
            "first_10_mean": mean(first, path),
            "last_10_mean": mean(last, path),
            "delta": mean(last, path) - mean(first, path),
        }
        for name, path in fields.items()
    }


def save_external_checkpoint(
    trainer: Rival2OpponentCurriculumTrainer,
    path: Path,
    *,
    accepted_offset: int,
) -> dict[str, Any]:
    trainer.curriculum_transition["accepted_ppo_offset"] = accepted_offset
    trainer.curriculum_transition["final_or_latest_update_utc"] = utc_now()
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    trainer.save_checkpoint(temporary)
    temporary.replace(path)
    return artifact(path)


def run_campaign(args: argparse.Namespace) -> dict[str, Any]:
    if args.work_dir is None:
        raise ValueError("training requires an explicit external --work-dir")
    preflight_authority = require_committed_preflight()
    warmup_authority = require_committed_warmup_authority()
    selected_initial_lr = warmup_authority["selected_initial_policy_lr"]
    config, bc, bootstrap = load_authority()
    geometry, meshes = load_geometry(args.collision_dir, args.device)
    trainer = build_trainer(
        worlds=WORLD_COUNT,
        horizon=128,
        collision_dir=args.collision_dir,
        geometry=geometry,
        meshes=meshes,
        device=args.device,
        bc=bc,
        bootstrap=bootstrap,
    )
    rolling = args.work_dir.resolve() / "rolling/rival2_human_bc_ppo_latest.pt"
    milestone_dir = args.work_dir.resolve() / "milestones"
    milestone_records: list[dict[str, Any]] = []
    rows: list[dict[str, Any]] = []
    start_offset = 0
    warmup_state: dict[str, Any]

    if args.resume:
        if not rolling.exists():
            raise RuntimeError("--resume selected but rolling checkpoint does not exist")
        trainer.load_checkpoint(rolling)
        start_offset = int(trainer.curriculum_transition["accepted_ppo_offset"])
        warmup_state = dict(trainer.curriculum_transition["warmup_state"])
        if float(warmup_state["selected_initial_policy_lr"]) != selected_initial_lr:
            raise RuntimeError("rolling checkpoint warmup authority is incompatible")
        if CURVE_PATH.exists():
            rows = [json.loads(line) for line in CURVE_PATH.read_text().splitlines() if line]
        if len(rows) != start_offset:
            raise RuntimeError("training curve and rolling checkpoint offsets disagree")
        if MILESTONES_PATH.exists():
            milestone_records = json.loads(MILESTONES_PATH.read_text())["checkpoints"]
    else:
        if rolling.exists():
            raise RuntimeError(
                "rolling checkpoint already exists; use --resume or a fresh --work-dir"
            )
        model_hash_before = tensor_tree_sha256(trainer.model.state_dict())
        empty_optimizer_before = len(trainer.optimizer.state) == 0
        initial_safety = warmup_safety_config(selected_initial_lr, active=True)
        optimizer_proof = trainer.enable_safe_mixed_ppo(initial_safety)
        if not empty_optimizer_before or len(trainer.optimizer.state) != 0:
            raise RuntimeError("fresh PPO optimizer unexpectedly contains Adam moments")
        rollout = trainer.collect_rollout()
        model_hash_after = tensor_tree_sha256(trainer.model.state_dict())
        optimizer_state_after = tensor_tree_sha256(trainer.optimizer.state_dict())
        retention = trainer.initialize_retention_corpus_from_rollout(
            rollout,
            source_identity={
                "identity": "HUMAN_BC_V1_STEP_160_PPO_PARENT",
                "checkpoint_sha256": BC_SHA256,
                "model_tensor_sha256": model_hash_before,
                "optimizer": "fresh_empty_split_adam",
            },
        )
        torch.cuda.synchronize(trainer.device)
        retention_checks = {
            "model_unchanged": model_hash_before == model_hash_after,
            "fresh_optimizer_unchanged": optimizer_state_after
            == tensor_tree_sha256(trainer.optimizer.state_dict()),
            "optimizer_state_still_empty": len(trainer.optimizer.state) == 0,
            "retention_source_is_bc_parent": retention["source_identity"][
                "checkpoint_sha256"
            ]
            == BC_SHA256,
            "nexto_samples_zero": trainer.last_rollout_curriculum_metrics[
                "trainable_agent_samples"
            ]["nexto"]
            == 0,
            "wisp_samples_zero": trainer.last_rollout_curriculum_metrics[
                "trainable_agent_samples"
            ]["wisp"]
            == 0,
        }
        if not all(retention_checks.values()):
            raise RuntimeError(f"BC-parent retention refresh failed: {retention_checks}")
        retention_record = {
            "format": "RIVAL2_HUMAN_BC_PPO_RETENTION_INITIALIZATION_V1",
            "created_utc": utc_now(),
            "summary": retention,
            "optimizer_transition": optimizer_proof,
            "rollout_only_agent_decisions": int(rollout.train_mask.sum().item()),
            "checks": retention_checks,
            "optimizer_steps": 0,
            "verdict": "PASS_GREEN",
        }
        write_json(RETENTION_PATH, json_safe(retention_record))
        warmup_state = {
            "format": "RIVAL2_HUMAN_BC_FRESH_PPO_WARMUP_STATE_V1",
            "active": True,
            "selected_initial_policy_lr": selected_initial_lr,
            "next_update_starting_policy_lr": selected_initial_lr,
            "transition_backoff_floor": TRANSITION_LR_FLOOR,
            "warmup_updates_completed": 0,
            "consecutive_clean_updates_at_1e-4": 0,
            "normal_production_operation_start_offset": None,
            "last_accepted_offset": 0,
        }
        trainer.curriculum_transition["pre_step_authority"] = preflight_authority[
            "authority"
        ]
        trainer.curriculum_transition["warmup_transition_authority"] = (
            warmup_authority["authority"]
        )
        trainer.curriculum_transition["warmup_state"] = warmup_state
        trainer.curriculum_transition["retention_initialization"] = json_safe(
            retention_record
        )
        trainer.curriculum_transition["optimizer_migration_proof"] = optimizer_proof
        trainer.curriculum_transition["campaign"] = config["campaign"]
        trainer.curriculum_transition["opponent_regime"] = config["opponents"]
        trainer.curriculum_transition["contract_hashes"] = dict(
            trainer.env.contract_hashes
        )
        trainer.curriculum_transition["ppo_contract_hash"] = (
            RIVAL2_PPO_120HZ_CONTRACT_HASH
        )
        trainer.curriculum_transition["ppo_config_hash"] = trainer.ppo_config.content_hash
        transition = {
            "format": "RIVAL2_HUMAN_BC_V1_TO_120HZ_PPO_TRANSITION_EVIDENCE_V1",
            "created_utc": utc_now(),
            "git_head_before_first_step": git("rev-parse", "HEAD"),
            "human_bc_parent": artifact(BC_CHECKPOINT),
            "bootstrap_provenance": artifact(BOOTSTRAP_CHECKPOINT),
            "fresh_optimizer": optimizer_proof,
            "retention": retention_record,
            "initial_historical_pool": [
                {
                    "version": version,
                    "policy_hz": policy_hz,
                    "action_version": action_version,
                }
                for version, policy_hz, action_version in zip(
                    trainer.opponent_pool.versions,
                    trainer.opponent_pool.policy_hz,
                    trainer.opponent_pool.action_versions,
                    strict=True,
                )
            ],
            "checks": {
                "pre_step_authority_committed_and_pushed": all(
                    preflight_authority["checks"].values()
                ),
                "warmup_authority_committed_and_pushed": all(
                    warmup_authority["checks"].values()
                ),
                "fresh_optimizer": empty_optimizer_before
                and len(trainer.optimizer.state) == 0,
                "retention_rollout_only": True,
                "human_demo_adapter_not_used": True,
                "no_behavior_cloning": True,
                "nexto_wisp_zero": retention_checks["nexto_samples_zero"]
                and retention_checks["wisp_samples_zero"],
            },
            "optimizer_steps": 0,
            "verdict": "PASS_GREEN",
        }
        write_json(TRANSITION_PATH, json_safe(transition))
        if CURVE_PATH.exists():
            CURVE_PATH.unlink()
        del rollout
        gc.collect()

    campaign_state_path = args.work_dir.resolve() / "campaign_state.json"
    if args.resume:
        if not campaign_state_path.exists():
            raise RuntimeError("resume requires the original campaign wall-clock state")
        campaign_state = json.loads(campaign_state_path.read_text(encoding="utf-8"))
    else:
        campaign_state = {
            "format": "RIVAL2_HUMAN_BC_PPO_WALL_CLOCK_STATE_V1",
            "campaign_started_utc": utc_now(),
            "campaign_started_unix_seconds": time.time(),
            "campaign_deadline_unix_seconds": time.time() + TRAINING_DURATION_SECONDS,
            "training_duration_seconds": TRAINING_DURATION_SECONDS,
            "accepted_ppo_offset": 0,
        }
        write_json(campaign_state_path, campaign_state)

    hard_safety_status = "PASS_NO_HARD_GUARD_FIRED"
    rejection: dict[str, Any] | None = None
    offset = start_offset
    while time.time() < float(campaign_state["campaign_deadline_unix_seconds"]):
        offset += 1
        warmup_before_update = dict(warmup_state)
        starting_policy_lr = float(
            warmup_state["next_update_starting_policy_lr"]
        )
        update_safety = warmup_safety_config(
            starting_policy_lr, active=bool(warmup_state["active"])
        )
        trainer.mixed_ppo_safety = update_safety
        reset_policy_learning_rate_for_new_update(trainer.optimizer, update_safety)
        rollout_start = time.perf_counter()
        rollout = trainer.collect_rollout()
        try:
            trainer.update(rollout, kl_guard=KL_GUARD)
        except Rival2PolicyDisplacementRejected as error:
            hard_safety_status = "FAIL_HARD_GUARD_FIRED"
            rejection = json_safe(error.diagnostics)
            write_json(
                RESULTS_DIR / "hard_safety_rejection.json",
                {
                    "format": "RIVAL2_HUMAN_BC_PPO_HARD_SAFETY_REJECTION_V1",
                    "created_utc": utc_now(),
                    "attempted_offset": offset,
                    "last_accepted_offset": offset - 1,
                    "rollout_curriculum": trainer.last_rollout_curriculum_metrics,
                    "rollout_gameplay": trainer.last_rollout_gameplay_metrics,
                    "source_checkpoint_sha256_after_rejection": sha256(BC_CHECKPOINT),
                    "transactional_policy_state_restored": bool(
                        error.diagnostics.get("transactional_rollback_completed")
                    ),
                    "diagnostics": rejection,
                },
            )
            del rollout
            break
        torch.cuda.synchronize(trainer.device)
        wall_seconds = time.perf_counter() - rollout_start
        if trainer.last_adaptive_ppo_diagnostics is None:
            raise RuntimeError("accepted update omitted adaptive PPO diagnostics")
        warmup_state = advance_warmup_state(
            warmup_state,
            trainer.last_adaptive_ppo_diagnostics,
            accepted_offset=offset,
        )
        next_safety = warmup_safety_config(
            float(warmup_state["next_update_starting_policy_lr"]),
            active=bool(warmup_state["active"]),
        )
        trainer.mixed_ppo_safety = next_safety
        reset_policy_learning_rate_for_new_update(trainer.optimizer, next_safety)
        trainer.curriculum_transition["warmup_state"] = dict(warmup_state)
        row = summarize_update(offset, trainer, wall_seconds)
        row["transition_warmup"] = {
            "before_update": warmup_before_update,
            "after_update": dict(warmup_state),
        }
        rows.append(row)
        append_jsonl(CURVE_PATH, row)
        if offset % HISTORICAL_SNAPSHOT_INTERVAL == 0:
            trainer.add_historical_snapshot()
        rolling_artifact = save_external_checkpoint(
            trainer, rolling, accepted_offset=offset
        )
        if offset in CHECKPOINT_OFFSETS:
            milestone_path = milestone_dir / f"rival2_human_bc_ppo_plus_{offset:03d}.pt"
            milestone_artifact = save_external_checkpoint(
                trainer, milestone_path, accepted_offset=offset
            )
            milestone_artifact.update(
                {
                    "accepted_ppo_offset": offset,
                    "iteration": trainer.iteration,
                    "policy_version": trainer.policy_version,
                    "historical_pool": [
                        {
                            "version": version,
                            "policy_hz": policy_hz,
                            "action_version": action_version,
                        }
                        for version, policy_hz, action_version in zip(
                            trainer.opponent_pool.versions,
                            trainer.opponent_pool.policy_hz,
                            trainer.opponent_pool.action_versions,
                            strict=True,
                        )
                    ],
                }
            )
            milestone_records.append(milestone_artifact)
            write_json(
                MILESTONES_PATH,
                {
                    "format": "RIVAL2_HUMAN_BC_PPO_CHECKPOINT_MILESTONES_V1",
                    "rolling_checkpoint": rolling_artifact,
                    "checkpoints": milestone_records,
                },
            )
        campaign_state["accepted_ppo_offset"] = offset
        campaign_state["last_accepted_update_utc"] = utc_now()
        campaign_state["last_accepted_update_unix_seconds"] = time.time()
        campaign_state["remaining_wall_seconds"] = max(
            0.0,
            float(campaign_state["campaign_deadline_unix_seconds"]) - time.time(),
        )
        write_json(campaign_state_path, campaign_state)
        compact = {
            "offset": offset,
            "iteration": trainer.iteration,
            "policy_version": trainer.policy_version,
            "seconds": round(wall_seconds, 3),
            "touches_per_minute": round(
                row["gameplay"]["touches_per_minute"], 6
            ),
            "unnecessary_flip_contacts_per_minute": round(
                row["gameplay"]["unnecessary_flip_contacts_per_minute"], 6
            ),
            "goals": row["goal_events"],
            "no_touch": row["gameplay"]["no_touch_truncations"],
            "completed_mean_kl": row["ppo_safety"]["completed_update_mean_kl"],
            "max_minibatch_kl": row["ppo_safety"][
                "maximum_post_step_minibatch_kl"
            ],
            "early_stop": row["ppo_safety"]["ppo_early_stop"],
            "backoffs": row["ppo_safety"]["policy_learning_rate_backoffs"],
        }
        print(json.dumps(compact, sort_keys=True), flush=True)
        del rollout
        gc.collect()

    final_offset = len(rows)
    if final_offset == 0:
        evidence = {
            "format": "RIVAL2_HUMAN_BC_PPO_BLOCKED_BEFORE_FIRST_ACCEPTED_UPDATE_V1",
            "created_utc": utc_now(),
            "verdict": "BLOCKED",
            "human_bc_parent": artifact(BC_CHECKPOINT),
            "final_accepted_ppo_offset": 0,
            "final_checkpoint": None,
            "mechanics_removal_preflight": artifact(PREFLIGHT_PATH),
            "hard_safety_status": hard_safety_status,
            "hard_safety_rejection": rejection,
            "touches_goals_no_touch_trend": None,
            "unnecessary_flip_contact_trend": None,
            "ten_hour_campaign_completed": False,
            "recommendation": (
                "stop and establish prospective optimizer-transition authority; "
                "do not weaken or tune against the fired hard guard"
            ),
        }
        write_json(FINAL_EVIDENCE_PATH, evidence)
        return evidence
    campaign_elapsed_seconds = (
        time.time() - float(campaign_state["campaign_started_unix_seconds"])
    )
    duration_completed = campaign_elapsed_seconds >= TRAINING_DURATION_SECONDS
    campaign_success = (
        duration_completed and hard_safety_status == "PASS_NO_HARD_GUARD_FIRED"
    )
    if campaign_success:
        FINAL_CHECKPOINT.parent.mkdir(parents=True, exist_ok=True)
        trainer.curriculum_transition["campaign_completed"] = True
        trainer.curriculum_transition["campaign_duration_seconds"] = (
            TRAINING_DURATION_SECONDS
        )
        trainer.curriculum_transition["campaign_elapsed_seconds"] = (
            campaign_elapsed_seconds
        )
        trainer.curriculum_transition["hard_safety_status"] = hard_safety_status
        trainer.curriculum_transition["checkpoint_offsets"] = list(CHECKPOINT_OFFSETS)
        trainer.curriculum_transition["historical_snapshot_interval"] = (
            HISTORICAL_SNAPSHOT_INTERVAL
        )
        trainer.curriculum_transition["historical_policy_pool_final"] = [
            {
                "version": version,
                "policy_hz": policy_hz,
                "action_version": action_version,
            }
            for version, policy_hz, action_version in zip(
                trainer.opponent_pool.versions,
                trainer.opponent_pool.policy_hz,
                trainer.opponent_pool.action_versions,
                strict=True,
            )
        ]
        trainer.save_checkpoint(FINAL_CHECKPOINT)
        final_checkpoint_record = artifact(FINAL_CHECKPOINT)
    else:
        final_checkpoint_record = artifact(rolling)

    final_payload = torch.load(
        FINAL_CHECKPOINT if campaign_success else rolling,
        map_location="cpu",
        weights_only=False,
    )
    optimizer_steps = sorted(
        {
            int(state["step"].item())
            for state in final_payload["optimizer"]["state"].values()
            if "step" in state
        }
    )
    final_checks = {
        "ten_hour_wall_clock_completed": duration_completed,
        "original_plus_120_boundary_completed": final_offset >= 120,
        "bc_parent_identity_present": final_payload["curriculum_transition"][
            "human_bc_parent"
        ]["sha256"]
        == BC_SHA256,
        "fresh_ppo_optimizer_recorded": final_payload["curriculum_transition"][
            "fresh_ppo_optimizer"
        ],
        "not_historical_optimizer_continuation": final_payload[
            "curriculum_transition"
        ]["historical_ppo_optimizer_restored"]
        is False,
        "contracts_exact": final_payload["contract_hashes"]
        == contract_hashes_for_reward(RIVAL2_REWARD_GAMEPLAY_120_V1_VERSION),
        "policy_hz_120": int(final_payload["policy_hz"]) == 120,
        "physics_hz_120": int(final_payload["physics_hz"]) == 120,
        "fresh_optimizer_has_steps": bool(optimizer_steps),
        "adaptive_safety_state_present": final_payload["opponent_curriculum"][
            "adaptive_ppo"
        ]
        is not None,
        "rng_state_present": all(
            key in final_payload
            for key in (
                "torch_cpu_rng_state",
                "torch_cuda_rng_state",
                "policy_generator_state",
                "opponent_generator_state",
            )
        ),
        "historical_pool_bounded": len(final_payload["historical_opponents"]) <= 16,
        "all_new_snapshots_are_120hz": all(
            int(entry["policy_hz"]) == 120
            and entry["action_version"] == RIVAL2_ACTION_V2_120HZ_VERSION
            for entry in final_payload["historical_opponents"]
            if int(entry["version"]) >= 510
        ),
        "nexto_wisp_never_sampled": all(
            row["curriculum"]["trainable_agent_samples"][family] == 0
            for row in rows
            for family in ("nexto", "wisp")
        ),
        "named_mechanics_absent_every_update": all(
            row["gameplay"]["named_mechanics_hot_path_absent"]
            and row["gameplay"]["named_mechanics_arrays"] == 0
            and row["gameplay"]["trusted_reward_component_absolute_sum"][
                "strict_double_dash_component"
            ]
            == 0.0
            and row["gameplay"]["trusted_reward_component_absolute_sum"][
                "v1_touch_component"
            ]
            == 0.0
            for row in rows
        ),
        "no_hard_safety_guard": hard_safety_status
        == "PASS_NO_HARD_GUARD_FIRED",
    }
    evidence = {
        "format": "RIVAL2_HUMAN_BC_PPO_PLUS_120_FINAL_EVIDENCE_V1",
        "created_utc": utc_now(),
        "verdict": "PASS_GREEN" if all(final_checks.values()) else "BLOCKED",
        "git_head_before_final_evidence_commit": git("rev-parse", "HEAD"),
        "human_bc_parent": artifact(BC_CHECKPOINT),
        "final_checkpoint": final_checkpoint_record,
        "final_accepted_ppo_offset": final_offset,
        "campaign_duration_seconds": TRAINING_DURATION_SECONDS,
        "campaign_elapsed_seconds": campaign_elapsed_seconds,
        "final_iteration": int(final_payload["iteration"]),
        "final_policy_version": int(final_payload["policy_version"]),
        "sample_accounting": final_payload["sample_accounting"],
        "fresh_optimizer_step_counters": optimizer_steps,
        "mechanics_removal_preflight": artifact(PREFLIGHT_PATH),
        "hard_safety_status": hard_safety_status,
        "hard_safety_rejection": rejection,
        "trend_first_10_vs_last_10": trend(rows),
        "ppo_safety_summary": {
            "maximum_post_step_minibatch_kl": max(
                row["ppo_safety"]["maximum_post_step_minibatch_kl"]
                for row in rows
            ),
            "maximum_completed_update_mean_kl": max(
                row["ppo_safety"]["completed_update_mean_kl"] for row in rows
            ),
            "updates_with_soft_early_stop": sum(
                int(row["ppo_safety"]["ppo_early_stop"]) for row in rows
            ),
            "total_lr_backoffs": sum(
                row["ppo_safety"]["policy_learning_rate_backoffs"] for row in rows
            ),
            "total_transactional_retries": sum(
                row["ppo_safety"]["optimizer_step_retries"] for row in rows
            ),
        },
        "milestones": milestone_records,
        "small_sanity_scope": (
            "the final training rollout supplied finite-output, action-saturation, "
            "movement, touch, scoring, no-touch, and termination sanity telemetry; "
            "no large opponent evaluation was run"
        ),
        "checks": final_checks,
    }
    write_json(FINAL_EVIDENCE_PATH, json_safe(evidence))
    if evidence["verdict"] != "PASS_GREEN":
        raise RuntimeError(f"campaign did not satisfy final gates: {final_checks}")
    return evidence


def main() -> int:
    args = parse_args()
    if not torch.cuda.is_available() or not args.device.startswith("cuda"):
        raise RuntimeError("the authoritative campaign requires CUDA")
    if args.preflight_only:
        report = run_preflight(args)
    elif args.transition_sweep_only:
        report = run_transition_sweep(args)
    else:
        report = run_campaign(args)
    print(json.dumps({"verdict": report["verdict"]}, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
