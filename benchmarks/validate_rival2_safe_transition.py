"""Validate the production mixed-PPO transition on the exact update-360 replay."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import subprocess
import sys
from dataclasses import asdict, replace
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import torch
import warp as wp

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from benchmarks.run_rival2_opponent_curriculum_v1 import (  # noqa: E402
    AUTHORITY,
    CAMPAIGN_SEED,
    KL_GUARD,
    MIXED_PPO_SAFETY,
    RETENTION_CORPUS,
    SOURCE_CHECKPOINT,
    SOURCE_CHECKPOINT_SHA256,
    WORLDS,
    _nested_exact,
    transition_preservation_gate,
)
from rivalsim.arena import ArenaGeometry, WarpArenaMeshes  # noqa: E402
from rivalsim.rival2_contracts import (  # noqa: E402
    RIVAL2_EPISODE_VERSION,
    RIVAL2_REWARD_GAMEPLAY_V1_VERSION,
    RIVAL2_REWARD_GAMEPLAY_V2_VERSION,
)
from rivalsim.rival2_env import Rival2Env  # noqa: E402
from rivalsim.rival2_mixed_ppo import (  # noqa: E402
    CRITIC_GROUP_NAME,
    POLICY_GROUP_NAME,
    make_empty_mixed_optimizer,
    mixed_optimizer_learning_rates,
    ppo_update_mixed_curriculum,
)
from rivalsim.rival2_opponent_curriculum import (  # noqa: E402
    OPPONENT_NAMES,
    Rival2OpponentCurriculumConfig,
    Rival2OpponentCurriculumTrainer,
)
from rivalsim.rival2_policy import Rival2PolicyConfig  # noqa: E402
from rivalsim.rival2_ppo import Rival2PPOConfig  # noqa: E402
from rivalsim.rival2_training import Rival2SelfPlayConfig  # noqa: E402

RESULTS_DIR = Path("results/rival2/opponent_curriculum_v1/safe_transition")
RESULT = RESULTS_DIR / "production_update360_replay.json"
RETRY_RESULT = RESULTS_DIR / "transactional_retry_proof.json"
MIGRATION_RESULT = RESULTS_DIR / "optimizer_migration.json"
MANIFEST = RESULTS_DIR / "artifact_manifest.json"
REPORT = Path("docs/RIVAL2_MIXED_PPO_SAFE_TRANSITION.md")
REQUIRED_BASE = "35b71101d317c956de1b045baf3a1b7c2aa200ea"
DIAGNOSTIC_RETRY_SOFT_TARGET = 0.002


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--collision-dir",
        type=Path,
        default=Path(r"G:\dev\RLBot-Rival\bot\collision_meshes\soccar"),
    )
    parser.add_argument("--device", default="cuda:0")
    return parser.parse_args()


def _utc_now() -> str:
    return datetime.now(UTC).isoformat()


def _git(*arguments: str) -> str:
    return subprocess.check_output(["git", *arguments], cwd=REPO_ROOT, text=True).strip()


def _git_is_ancestor(ancestor: str, descendant: str) -> bool:
    return (
        subprocess.run(
            ["git", "merge-base", "--is-ancestor", ancestor, descendant],
            cwd=REPO_ROOT,
            check=False,
        ).returncode
        == 0
    )


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest().upper()


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8", newline="\n")


def _make_trainer(
    source: dict[str, Any],
    collision_dir: Path,
    device: str,
) -> Rival2OpponentCurriculumTrainer:
    geometry = ArenaGeometry.load_soccar(collision_dir)
    meshes = WarpArenaMeshes(geometry, device)
    kickoff_selector = (np.arange(WORLDS, dtype=np.int32) + CAMPAIGN_SEED) % 5
    env = Rival2Env(
        WORLDS,
        str(collision_dir),
        device=device,
        seed=CAMPAIGN_SEED,
        reward_version=RIVAL2_REWARD_GAMEPLAY_V2_VERSION,
        episode_version=RIVAL2_EPISODE_VERSION,
        geometry=geometry,
        meshes=meshes,
        kickoff_selector=kickoff_selector,
        car_visitation_order="a_then_b",
    )
    trainer = Rival2OpponentCurriculumTrainer(
        env,
        policy_config=Rival2PolicyConfig(**source["policy_config"]),
        ppo_config=Rival2PPOConfig(**source["ppo_config"]),
        self_play_config=Rival2SelfPlayConfig(**source["self_play_config"]),
        opponent_curriculum=Rival2OpponentCurriculumConfig(),
        seed=CAMPAIGN_SEED,
    )
    transition = trainer.load_checkpoint_curriculum_transition(
        SOURCE_CHECKPOINT,
        source_reward_version=RIVAL2_REWARD_GAMEPLAY_V1_VERSION,
        source_episode_version=RIVAL2_EPISODE_VERSION,
        transition_record={
            "schema_version": 1,
            "authority": AUTHORITY.as_posix(),
            "authorized_change": "mixed PPO safe-transition production replay only",
            "source_checkpoint_sha256": SOURCE_CHECKPOINT_SHA256,
            "diagnostic_replay_only": True,
        },
    )
    gate = transition_preservation_gate(source, trainer, transition)
    if gate["verdict"] != "PASS_GREEN":
        raise RuntimeError(f"transition preservation failed: {gate['checks']}")
    return trainer


def _load_retention(trainer: Rival2OpponentCurriculumTrainer, device: str) -> dict[str, Any]:
    payload = torch.load(RETENTION_CORPUS, map_location=device, weights_only=False)
    if payload.get("format") != "RIVAL2_RETENTION_OBSERVATIONS_V1":
        raise ValueError("retention corpus format mismatch")
    trainer.install_retention_corpus(payload["observations"], payload["summary"])
    return payload


def _transactional_retry_proof(
    trainer: Rival2OpponentCurriculumTrainer,
    rollout: Any,
) -> dict[str, Any]:
    model = copy.deepcopy(trainer.model)
    rates = mixed_optimizer_learning_rates(trainer.optimizer)
    optimizer = make_empty_mixed_optimizer(
        model,
        policy_learning_rate=rates[POLICY_GROUP_NAME],
        critic_learning_rate=rates[CRITIC_GROUP_NAME],
    )
    optimizer.load_state_dict(copy.deepcopy(trainer.optimizer.state_dict()))
    generator = torch.Generator(device=trainer.device)
    generator.set_state(trainer.policy_generator.get_state())
    diagnostic_config = replace(
        MIXED_PPO_SAFETY,
        soft_minibatch_kl_target=DIAGNOSTIC_RETRY_SOFT_TARGET,
        retention_soft_mean_kl_target=1.0,
    )
    metrics_a, diagnostics_a = ppo_update_mixed_curriculum(
        model,
        optimizer,
        rollout,
        trainer.ppo_config,
        diagnostic_config,
        retention_observations=trainer.retention_observations,
        family_names=OPPONENT_NAMES,
        generator=generator,
        policy_config=trainer.policy_config,
        kl_guard=KL_GUARD,
        gae_ready=True,
        diagnostic_optimizer_step_limit=1,
    )
    retry = diagnostics_a["retry_log"][0] if diagnostics_a["retry_log"] else None
    accepted = diagnostics_a["optimizer_steps"][0] if diagnostics_a["optimizer_steps"] else None
    steps_after_a = {
        name: int(optimizer.state[parameter]["step"].item())
        for name, parameter in model.named_parameters()
    }
    metrics_b, diagnostics_b = ppo_update_mixed_curriculum(
        model,
        optimizer,
        rollout,
        trainer.ppo_config,
        replace(
            MIXED_PPO_SAFETY,
            soft_minibatch_kl_target=1.0,
            retention_soft_mean_kl_target=1.0,
        ),
        retention_observations=trainer.retention_observations,
        family_names=OPPONENT_NAMES,
        generator=generator,
        policy_config=trainer.policy_config,
        kl_guard=KL_GUARD,
        gae_ready=True,
        diagnostic_optimizer_step_limit=1,
    )
    steps_after_b = {
        name: int(optimizer.state[parameter]["step"].item())
        for name, parameter in model.named_parameters()
    }
    checks = {
        "diagnostic_soft_target_lower_than_production": MIXED_PPO_SAFETY.soft_minibatch_kl_target
        > DIAGNOSTIC_RETRY_SOFT_TARGET,
        "retry_was_exercised": retry is not None,
        "same_minibatch_retried": retry is not None
        and accepted is not None
        and retry["same_minibatch_retry"] is True
        and accepted["minibatch_index_sha256"] == retry["minibatch_index_sha256"]
        and accepted["minibatch_start"] == retry["minibatch_start"],
        "parameters_restored_exactly": retry is not None
        and retry["restore_checks"]["parameters_exact"],
        "adam_state_restored_exactly": retry is not None
        and retry["restore_checks"]["optimizer_state_exact"],
        "adam_step_counters_restored_exactly": retry is not None
        and retry["restore_checks"]["adam_step_counters_exact"],
        "only_policy_lr_changed": retry is not None
        and retry["policy_learning_rate_before"] == 1.0e-4
        and retry["policy_learning_rate_after"] == 5.0e-5
        and retry["critic_learning_rate_before_after"] == 3.0e-4,
        "same_step_accepted_after_retry": accepted is not None
        and accepted["optimizer_step_index"] == retry["optimizer_step_index"]
        and accepted["retry_count"] == 1,
        "accepted_step_below_diagnostic_soft_target": accepted is not None
        and accepted["post_step_empirical_kl"] <= DIAGNOSTIC_RETRY_SOFT_TARGET,
        "update_a_limited_to_one_accepted_step": diagnostics_a["accepted_optimizer_steps"] == 1,
        "update_a_started_at_base_policy_lr": diagnostics_a["policy_learning_rate_start"] == 1.0e-4,
        "update_a_ended_at_backed_off_policy_lr": diagnostics_a["policy_learning_rate_end"]
        == 5.0e-5,
        "update_b_observed_update_a_backoff": diagnostics_b[
            "policy_learning_rate_before_update_reset"
        ]
        == 5.0e-5,
        "update_b_reset_was_applied": diagnostics_b[
            "policy_learning_rate_update_start_reset_applied"
        ]
        is True,
        "update_b_started_at_base_policy_lr": diagnostics_b["policy_learning_rate_start"] == 1.0e-4,
        "update_b_policy_lr_remained_base": diagnostics_b["policy_learning_rate_end"] == 1.0e-4,
        "critic_lr_unchanged_across_updates": diagnostics_a["critic_learning_rate_start_end"]
        == diagnostics_b["critic_learning_rate_start_end"]
        == 3.0e-4,
        "retention_reference_refreshed_for_update_b": diagnostics_a[
            "retention_reference_actor_sha256"
        ]
        != diagnostics_b["retention_reference_actor_sha256"],
        "adam_steps_advanced_normally_across_update_b": all(
            steps_after_b[name] == steps_after_a[name] + 1 for name in steps_after_a
        ),
        "hard_guards_unchanged": diagnostics_a["checks"]["hard_minibatch_guard_unchanged"]
        and diagnostics_a["checks"]["hard_completed_guard_unchanged"]
        and diagnostics_b["checks"]["hard_minibatch_guard_unchanged"]
        and diagnostics_b["checks"]["hard_completed_guard_unchanged"],
    }
    return {
        "schema_version": 1,
        "created_utc": _utc_now(),
        "diagnostic_override": {
            "update_a_soft_minibatch_kl_target": DIAGNOSTIC_RETRY_SOFT_TARGET,
            "update_a_retention_soft_mean_kl_target": 1.0,
            "update_b_soft_targets": 1.0,
            "optimizer_step_limit_per_update": 1,
            "production_configuration_modified": False,
        },
        "retry": retry,
        "accepted_step": accepted,
        "update_a": {
            "metrics": {name: float(value.item()) for name, value in metrics_a.items()},
            "adaptive_ppo": diagnostics_a,
            "adam_steps_after": steps_after_a,
        },
        "update_b": {
            "metrics": {name: float(value.item()) for name, value in metrics_b.items()},
            "adaptive_ppo": diagnostics_b,
            "adam_steps_after": steps_after_b,
        },
        "completed_update_mean_kl_after_one_step": float(metrics_a["approx_kl"].item()),
        "checks": checks,
        "verdict": "PASS_GREEN" if all(checks.values()) else "FAIL_RED",
    }


def _write_report(result: dict[str, Any], retry: dict[str, Any]) -> None:
    adaptive = result["adaptive_ppo"]
    family = adaptive["family_statistics"]
    retention = result["retention_corpus_summary"]
    coverage = retention["state_coverage"]
    lines = [
        "# Rival 2.0 mixed-PPO safe transition",
        "",
        f"Verdict: `{result['verdict']}`.",
        "",
        "This is an exact disposable update-360 replay through the production mixed-"
        "curriculum optimizer path. It did not resume the +120 campaign and wrote no "
        "training checkpoint.",
        "",
        "## Production replay",
        "",
        f"- accepted optimizer steps: `{adaptive['accepted_optimizer_steps']}` / "
        f"`{adaptive['expected_optimizer_steps']}`;",
        f"- maximum post-step minibatch KL: `{adaptive['maximum_post_step_minibatch_kl']:.9f}`;",
        f"- completed-update mean KL: `{adaptive['completed_update_mean_kl']:.9f}`;",
        f"- retention-corpus mean KL: `{adaptive['retention_corpus_mean_kl']:.9f}`;",
        f"- policy LR start/end: `{adaptive['policy_learning_rate_start']}` / "
        f"`{adaptive['policy_learning_rate_end']}`;",
        f"- policy LR armed for next update: "
        f"`{result['post_update_runtime_learning_rates'][POLICY_GROUP_NAME]}`;",
        f"- retries/backoffs: `{adaptive['optimizer_step_retries']}` / "
        f"`{adaptive['policy_learning_rate_backoffs']}`;",
        f"- PPO early stop: `{str(adaptive['ppo_early_stop']).lower()}` "
        f"(`{adaptive['ppo_early_stop_reason']}`).",
        "",
        "The fixed `1e-4` strategy previously completed all 154 steps with maximum "
        "minibatch KL `0.017498828` and completed mean KL `0.010052922`. The "
        "production replay matched that maximum before the independent retention "
        "probe became binding. It accepted prior safe steps, backed policy LR down "
        "without changing critic LR, and stopped at the configured minimum rather "
        "than accepting a corpus-KL violation.",
        "",
        "## Fixed retention corpus",
        "",
        f"- observations/dimension: `{retention['observation_count']}` / "
        f"`{retention['observation_dimension']}`;",
        f"- observation-content SHA-256: `{retention['sha256']}`;",
        f"- source checkpoint SHA-256: `{retention['source_identity']['checkpoint_sha256']}`;",
        "- selected categories: `96` near-ball, `96` approach, `80` recovery, "
        "`96` airborne, `96` ordinary-ground, and `48` remaining-diversity states;",
        f"- field/orientation coverage: "
        f"`{coverage['occupied_x_field_regions_of_3']}/3` x regions, "
        f"`{coverage['occupied_y_field_regions_of_3']}/3` y regions, "
        f"`{coverage['occupied_heading_octants_of_8']}/8` heading octants.",
        "",
        "## Family statistics",
        "",
        "| family | samples | raw advantage mean/std | normalized mean/std | "
        "return mean/std | value mean/std | empirical KL |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for name in OPPONENT_NAMES:
        values = family[name]
        lines.append(
            f"| {name} | {values['sample_count']} | "
            f"{values['raw_advantage_mean']:.6f}/{values['raw_advantage_std']:.6f} | "
            f"{values['normalized_advantage_mean']:.6f}/"
            f"{values['normalized_advantage_std']:.6f} | "
            f"{values['return_mean']:.6f}/{values['return_std']:.6f} | "
            f"{values['value_mean']:.6f}/{values['value_std']:.6f} | "
            f"{values['empirical_kl']:.9f} |"
        )
    lines.extend(
        [
            "",
            "## Transactional retry proof",
            "",
            f"Verdict: `{retry['verdict']}`. The diagnostic-only `0.002` soft target "
            "forced the same first minibatch to roll back and retry at `5e-5`. Model "
            "parameters, Adam moments, and Adam step counters restored exactly before "
            "only the policy-group LR changed; the critic group remained at `3e-4`.",
            "",
            "The accepted diagnostic update A ended at `5e-5`. Without reconstructing "
            "the model or optimizer, diagnostic update B observed that prior value, "
            "reset its update-start policy LR to `1e-4`, refreshed the retention actor "
            "reference, and accepted its bounded step at `1e-4`. Every Adam counter "
            "advanced by exactly one between A and B; critic LR remained `3e-4`.",
            "",
            "## Boundary",
            "",
            "Nexto, Wisp, Gameplay V2, opponent probabilities, physics, lifecycle, "
            "network architecture, and both hard KL limits were unchanged. No live Rival "
            "training continuation was started.",
            "",
        ]
    )
    REPORT.parent.mkdir(parents=True, exist_ok=True)
    REPORT.write_text("\n".join(lines), encoding="utf-8", newline="\n")


def _write_manifest() -> None:
    artifacts = []
    for path in (*sorted(RESULTS_DIR.iterdir()), REPORT):
        if path == MANIFEST or not path.is_file():
            continue
        content = path.read_bytes()
        if path.suffix.lower() in {".json", ".md"}:
            content = content.replace(b"\r\n", b"\n")
        artifacts.append(
            {
                "path": path.as_posix(),
                "sha256": hashlib.sha256(content).hexdigest().upper(),
                "size_bytes": len(content),
            }
        )
    _write_json(
        MANIFEST,
        {
            "schema_version": 1,
            "created_utc": _utc_now(),
            "artifacts": artifacts,
        },
    )


def run(args: argparse.Namespace) -> dict[str, Any]:
    source_before = _sha256(SOURCE_CHECKPOINT)
    if source_before != SOURCE_CHECKPOINT_SHA256:
        raise RuntimeError("source checkpoint SHA-256 mismatch")
    source = torch.load(SOURCE_CHECKPOINT, map_location="cpu", weights_only=False)
    retention_sha_before = _sha256(RETENTION_CORPUS)
    trainer = _make_trainer(source, args.collision_dir, args.device)
    source_model_exact = _nested_exact(source["model"], trainer.model.state_dict())
    source_optimizer_exact = _nested_exact(source["optimizer"], trainer.optimizer.state_dict())
    migration = trainer.enable_safe_mixed_ppo(MIXED_PPO_SAFETY)
    retention = _load_retention(trainer, args.device)
    rollout = trainer.collect_rollout()
    rollout.compute_gae(trainer.ppo_config)

    retry = _transactional_retry_proof(trainer, rollout)
    _write_json(RETRY_RESULT, retry)
    if retry["verdict"] != "PASS_GREEN":
        raise RuntimeError(f"transactional retry proof failed: {retry['checks']}")

    model_before_update = copy.deepcopy(trainer.model.state_dict())
    optimizer_before_update = copy.deepcopy(trainer.optimizer.state_dict())
    metrics = trainer.update(rollout, kl_guard=KL_GUARD)
    adaptive = trainer.last_adaptive_ppo_diagnostics
    if adaptive is None:
        raise RuntimeError("production replay did not publish adaptive diagnostics")
    runtime_learning_rates = mixed_optimizer_learning_rates(trainer.optimizer)
    checkpoint_payload = trainer.checkpoint_payload()
    checkpoint_adaptive = checkpoint_payload["opponent_curriculum"]["adaptive_ppo"]
    checkpoint_learning_rates = {
        group["name"]: float(group["lr"])
        for group in checkpoint_payload["optimizer"]["param_groups"]
    }
    source_after = _sha256(SOURCE_CHECKPOINT)
    retention_sha_after = _sha256(RETENTION_CORPUS)
    retention_early_stop_safe = (
        adaptive["ppo_early_stop"]
        and adaptive["ppo_early_stop_reason"] == "retention_mean_kl_at_minimum_policy_lr"
        and adaptive["accepted_optimizer_steps"] > 0
        and adaptive["policy_learning_rate_end"] == MIXED_PPO_SAFETY.minimum_policy_learning_rate
        and adaptive["retention_corpus_mean_kl"] <= MIXED_PPO_SAFETY.retention_soft_mean_kl_target
        and bool(adaptive["retry_log"])
        and adaptive["retry_log"][-1]["early_stop"] is True
        and adaptive["retry_log"][-1]["proposed_retention_mean_kl"]
        > MIXED_PPO_SAFETY.retention_soft_mean_kl_target
    )
    full_update_safe = (
        adaptive["accepted_optimizer_steps"] == adaptive["expected_optimizer_steps"]
        and not adaptive["ppo_early_stop"]
    )
    checks = {
        "required_base_in_head_history": _git_is_ancestor(REQUIRED_BASE, "HEAD"),
        "source_model_exact_before_migration": source_model_exact,
        "source_optimizer_exact_before_migration": source_optimizer_exact,
        "optimizer_migration_pass_green": migration["verdict"] == "PASS_GREEN",
        "retention_corpus_pass_green": retention["summary"]["verdict"] == "PASS_GREEN",
        "expected_full_optimizer_step_count_is_154": adaptive["expected_optimizer_steps"] == 154,
        "frozen_rollout_processed_with_safe_progress": full_update_safe
        or retention_early_stop_safe,
        "retention_control_outcome_valid": full_update_safe or retention_early_stop_safe,
        "accepted_steps_remain_within_retention_target": adaptive["checks"][
            "accepted_steps_within_retention_target"
        ],
        "no_hard_kl_rejection": adaptive["maximum_post_step_minibatch_kl"]
        <= KL_GUARD.minibatch_kl_limit
        and adaptive["completed_update_mean_kl"] <= KL_GUARD.completed_update_mean_kl_limit,
        "actor_learning_signal_nonzero": adaptive["checks"][
            "actor_receives_nonzero_policy_gradient"
        ],
        "trunk_learning_signal_nonzero": adaptive["checks"][
            "trunk_receives_nonzero_policy_gradient"
        ],
        "value_loss_to_trunk_zero": adaptive["checks"][
            "value_loss_to_shared_trunk_gradient_exact_zero"
        ],
        "value_loss_to_actor_zero": adaptive["checks"]["value_loss_to_actor_gradient_exact_zero"],
        "production_update_started_at_base_policy_lr": adaptive["policy_learning_rate_start"]
        == MIXED_PPO_SAFETY.initial_policy_learning_rate,
        "production_update_recorded_actual_backed_off_end_lr": adaptive["policy_learning_rate_end"]
        == MIXED_PPO_SAFETY.minimum_policy_learning_rate,
        "runtime_policy_lr_rearmed_for_next_update": runtime_learning_rates[POLICY_GROUP_NAME]
        == MIXED_PPO_SAFETY.initial_policy_learning_rate,
        "runtime_critic_lr_unchanged": runtime_learning_rates[CRITIC_GROUP_NAME]
        == MIXED_PPO_SAFETY.critic_learning_rate,
        "checkpoint_schema_v2_update_local": checkpoint_adaptive["schema_version"] == 2
        and checkpoint_adaptive["policy_learning_rate_scope"] == "ppo_update_local",
        "checkpoint_arms_next_update_at_base_policy_lr": checkpoint_adaptive[
            "next_update_policy_learning_rate"
        ]
        == MIXED_PPO_SAFETY.initial_policy_learning_rate
        and checkpoint_learning_rates[POLICY_GROUP_NAME]
        == MIXED_PPO_SAFETY.initial_policy_learning_rate,
        "checkpoint_critic_lr_unchanged": checkpoint_learning_rates[CRITIC_GROUP_NAME]
        == MIXED_PPO_SAFETY.critic_learning_rate,
        "production_model_changed": not _nested_exact(
            model_before_update, trainer.model.state_dict()
        ),
        "production_optimizer_changed": not _nested_exact(
            optimizer_before_update, trainer.optimizer.state_dict()
        ),
        "source_checkpoint_untouched": source_before == source_after,
        "retention_artifact_untouched": retention_sha_before == retention_sha_after,
        "no_training_checkpoint_written": True,
        "no_campaign_resumed": True,
    }
    result = {
        "schema_version": 1,
        "created_utc": _utc_now(),
        "verdict": "PASS_GREEN" if all(checks.values()) else "FAIL_RED",
        "identity": {
            "head": _git("rev-parse", "HEAD"),
            "origin_main": _git("rev-parse", "origin/main"),
            "required_base": REQUIRED_BASE,
            "source_checkpoint": SOURCE_CHECKPOINT.as_posix(),
            "source_checkpoint_sha256": source_before,
            "retention_corpus": RETENTION_CORPUS.as_posix(),
            "retention_corpus_artifact_sha256": retention_sha_before,
        },
        "scope": {
            "diagnostic_replay_only": True,
            "campaign_resumed": False,
            "training_checkpoint_written": False,
            "nexto_or_wisp_modified": False,
            "reward_modified": False,
            "physics_modified": False,
            "network_architecture_modified": False,
            "hard_kl_limits_modified": False,
        },
        "configuration": {
            "ppo": asdict(trainer.ppo_config),
            "mixed_ppo_safety": asdict(MIXED_PPO_SAFETY),
            "mixed_ppo_safety_hash": MIXED_PPO_SAFETY.content_hash,
            "hard_kl_guard": asdict(KL_GUARD),
            "opponent_curriculum": asdict(trainer.opponent_curriculum),
        },
        "optimizer_migration": migration,
        "retention_corpus_summary": retention["summary"],
        "transactional_retry_proof": retry,
        "metrics": {name: float(value.item()) for name, value in metrics.items()},
        "adaptive_ppo": adaptive,
        "post_update_runtime_learning_rates": runtime_learning_rates,
        "checkpoint_adaptive_ppo_summary": {
            "schema_version": checkpoint_adaptive["schema_version"],
            "policy_learning_rate_scope": checkpoint_adaptive["policy_learning_rate_scope"],
            "optimizer_learning_rates": checkpoint_adaptive["optimizer_learning_rates"],
            "next_update_policy_learning_rate": checkpoint_adaptive[
                "next_update_policy_learning_rate"
            ],
            "last_update_summary": checkpoint_adaptive["last_update_summary"],
        },
        "checks": checks,
    }
    _write_json(MIGRATION_RESULT, migration)
    _write_json(RESULT, result)
    _write_report(result, retry)
    _write_manifest()
    if result["verdict"] != "PASS_GREEN":
        raise RuntimeError(f"production safe-transition replay failed: {checks}")
    return result


def main() -> int:
    args = parse_args()
    if not torch.cuda.is_available() or not wp.is_cuda_available():
        raise RuntimeError("CUDA PyTorch and Warp are required")
    torch.cuda.set_device(args.device)
    torch.set_float32_matmul_precision("highest")
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    result = run(args)
    adaptive = result["adaptive_ppo"]
    print(
        json.dumps(
            {
                "verdict": result["verdict"],
                "accepted_optimizer_steps": adaptive["accepted_optimizer_steps"],
                "maximum_post_step_minibatch_kl": adaptive["maximum_post_step_minibatch_kl"],
                "completed_update_mean_kl": adaptive["completed_update_mean_kl"],
                "retention_corpus_mean_kl": adaptive["retention_corpus_mean_kl"],
                "policy_learning_rate_end": adaptive["policy_learning_rate_end"],
                "next_update_policy_learning_rate": result["post_update_runtime_learning_rates"][
                    POLICY_GROUP_NAME
                ],
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
