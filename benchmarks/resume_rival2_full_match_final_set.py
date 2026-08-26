"""Resume the completed Phase B checkpoint for one final full-match training set."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch

import benchmarks.run_rival2_campaign01 as campaign01
import benchmarks.run_rival2_campaign03 as campaign03
import benchmarks.run_rival2_full_match_curriculum as curriculum
from rivalsim.arena import ArenaGeometry, WarpArenaMeshes
from rivalsim.rival2_contracts import (
    RIVAL2_FULL_MATCH_EPISODE_VERSION,
    RIVAL2_REWARD_GOAL_ONLY_VERSION,
)
from rivalsim.rival2_full_match_env import Rival2FullMatchEnv
from rivalsim.rival2_training import Rival2Trainer


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--collision-dir", required=True)
    parser.add_argument("--work-dir", type=Path, required=True)
    parser.add_argument("--phase-b-checkpoint", type=Path, required=True)
    parser.add_argument(
        "--results-dir",
        type=Path,
        default=Path("results/rival2/full_match_curriculum"),
    )
    parser.add_argument("--device", default="cuda:0")
    return parser.parse_args()


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def main() -> int:
    args = parse_args()
    args.work_dir = args.work_dir.resolve()
    args.phase_b_checkpoint = args.phase_b_checkpoint.resolve()
    args.results_dir = args.results_dir.resolve()
    if not args.phase_b_checkpoint.is_file():
        raise FileNotFoundError(args.phase_b_checkpoint)
    if args.results_dir.exists() and any(args.results_dir.iterdir()):
        raise RuntimeError("results directory must be empty")

    configuration = curriculum.frozen_configuration()
    configuration["steering_adjustment"] = {
        "created_utc": campaign01._utc_now(),
        "superseded": "six real elapsed hours after Phase B",
        "replacement": "one fresh complete match per 131072 resident worlds",
        "phase_b_preserved": True,
    }
    campaign01._initialize_runtime(args.device)
    launch_gate = curriculum.verify_launch(configuration)
    curriculum._write_json(
        args.work_dir / "config_revised_before_final_match_set.json",
        configuration,
    )
    curriculum._write_json(
        args.work_dir / "final_match_set_launch_gate.json", launch_gate
    )

    geometry = ArenaGeometry.load_soccar(args.collision_dir)
    meshes = WarpArenaMeshes(geometry, args.device)
    kickoff_selector = (
        np.arange(curriculum.WORLDS, dtype=np.int32) + curriculum.CAMPAIGN_SEED
    ) % 5
    env = Rival2FullMatchEnv(
        curriculum.WORLDS,
        args.collision_dir,
        device=args.device,
        seed=curriculum.CAMPAIGN_SEED,
        reward_version=RIVAL2_REWARD_GOAL_ONLY_VERSION,
        geometry=geometry,
        meshes=meshes,
        kickoff_selector=kickoff_selector,
        car_visitation_order="a_then_b",
    )
    trainer = Rival2Trainer(
        env,
        ppo_config=campaign03.campaign03_ppo_config(),
        seed=curriculum.CAMPAIGN_SEED,
    )
    trainer.load_checkpoint(args.phase_b_checkpoint)
    if (
        trainer.env.reward_version != RIVAL2_REWARD_GOAL_ONLY_VERSION
        or trainer.env.episode_version != RIVAL2_FULL_MATCH_EPISODE_VERSION
    ):
        raise RuntimeError("Phase B checkpoint is not goal-only/full-match")

    trainer.env.start_fresh_matches()
    phase_started = time.perf_counter()
    phase_start_iteration = trainer.iteration
    phase_start_samples = trainer.total_agent_samples
    active = torch.ones(
        curriculum.WORLDS, dtype=torch.bool, device=trainer.device
    )
    ledger = args.work_dir / "final_match_training_curve.jsonl"
    if ledger.exists():
        raise RuntimeError("final match training ledger already exists")
    while bool(active.any().item()):
        curriculum._train_one_match_masked_update(
            trainer=trainer,
            active=active,
            phase_start_iteration=phase_start_iteration,
            device=args.device,
            ledger=ledger,
        )

    trainer.add_historical_snapshot()
    final_checkpoint = curriculum._save_checkpoint(
        "goal_only_final_match_set", trainer, args.work_dir
    )
    final_evaluation = curriculum._evaluate_and_write(
        trainer=trainer,
        collision_dir=args.collision_dir,
        geometry=geometry,
        meshes=meshes,
        device=args.device,
        phase="C_GOAL_ONLY_ONE_MATCH_SET",
        label="goal_only_final_match_set",
        work_dir=args.work_dir,
    )
    final_audit = curriculum._checkpoint_audit(final_checkpoint, trainer)
    if final_audit["verdict"] != "PASS_GREEN":
        raise RuntimeError("final one-match-set checkpoint audit failed")
    phase_c_summary = {
        "status": "COMPLETE",
        "bound": "one fresh complete standard match per resident world",
        "start_iteration": phase_start_iteration,
        "start_agent_decision_samples": phase_start_samples,
        "final_iteration": trainer.iteration,
        "final_agent_decision_samples": trainer.total_agent_samples,
        "additional_updates": trainer.iteration - phase_start_iteration,
        "additional_agent_decision_samples": trainer.total_agent_samples
        - phase_start_samples,
        "counted_complete_matches": curriculum.WORLDS,
        "remaining_active_worlds": int(active.sum().item()),
        "wall_seconds_including_final_evaluation": time.perf_counter()
        - phase_started,
        "training_curve": ledger.resolve().as_posix(),
        "final_checkpoint": final_checkpoint,
        "final_evaluation_label": final_evaluation["checkpoint_label"],
        "final_checkpoint_audit": final_audit,
    }
    curriculum._write_json(args.work_dir / "phase_c_summary.json", phase_c_summary)

    phase_a = _read_json(args.work_dir / "phase_a_summary.json")
    reward_transition = _read_json(args.work_dir / "reward_transition.json")
    phase_b = _read_json(args.work_dir / "phase_b_summary.json")
    summary = {
        "schema_version": curriculum.SCHEMA_VERSION,
        "created_utc": campaign01._utc_now(),
        "execution_status": "COMPLETE",
        "source_head": launch_gate["head"],
        "mechanics_fix_commit": curriculum.MECHANICS_FIX_COMMIT,
        "configuration": configuration,
        "phase_a": phase_a,
        "reward_transition": reward_transition,
        "reward_transition_checkpoint": reward_transition[
            "post_transition_checkpoint"
        ],
        "phase_b": phase_b,
        "phase_c": phase_c_summary,
        "final_iteration": trainer.iteration,
        "final_policy_version": trainer.policy_version,
        "final_agent_decision_samples": trainer.total_agent_samples,
        "final_reward_version": trainer.env.reward_version,
        "final_episode_version": trainer.env.episode_version,
        "final_contract_hashes": dict(trainer.env.contract_hashes),
        "final_historical_policy_versions": trainer.opponent_pool.versions,
        "all_training_used_complete_matches": True,
        "training_truncation_contract": None,
        "six_hour_phase_run": False,
        "viewer_built": False,
        "v06_begun": False,
    }
    curriculum._write_json(args.work_dir / "run_summary.json", summary)
    curriculum.publish_results(args, configuration, launch_gate, summary)
    print(
        f"one-match-set COMPLETE update={trainer.iteration} "
        f"samples={trainer.total_agent_samples}",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
