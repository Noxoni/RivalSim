"""On-policy DAgger bridge: query Nexto on the student's own native states."""

from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path
import shutil
import subprocess
import sys
from typing import Any

import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from benchmarks import run_rival2_codex_autonomous_v1 as base
from benchmarks import run_rival2_codex_autonomous_v14 as bridge
from rivalsim.full_match import FullMatchRunner
from rivalsim.human_demo.behavior_cloning import (
    MechanicHierarchySampler,
    human_behavior_cloning_objective,
)
from rivalsim.rival2_policy import Rival2ActorCritic, Rival2PolicyConfig
from third_party.nexto.adapter import NextoPolicyAdapter, NextoStateTensors

SOURCE = (
    ROOT
    / "checkpoints/rival2/codex_autonomous_match_v1/rival2_codex_autonomous_match_parent.pt"
)
SOURCE_SHA256 = "0B90C201A0E1A16E83CF5CCBDE3371434F78D455C2AED20E0DDA6414F3B84E39"
SOURCE_MODEL_SHA256 = "A8E6BCED160F9C6871B9D4DF8D91AD2FDF260AA76E64F5788FEACF0371564500"
AUTHORITY = ROOT / "results/rival2/codex_autonomous_v16/authority.json"
RESULTS = ROOT / "results/rival2/codex_autonomous_v16"
CHECKPOINT = ROOT / "checkpoints/rival2/codex_autonomous_v16/rival2_codex_autonomous_best.pt"
BASELINE = ROOT / "results/rival2/codex_autonomous_match_v1/window_sweep/v7_u0030.json"
BASELINE_FULL_MATCH = (
    ROOT / "results/rival2/codex_autonomous_match_v1/full_match_evaluation.json"
)
DEFAULT_RUN_DIR = Path("G:/dev/RivalSim-runs/codex-autonomous-v16")

SEED = 2_026_090_223
WORLDS = 8192
NATIVE_BATCH = 3072
HUMAN_GAMEPLAY_BATCH = 512
HUMAN_MECHANIC_BATCH = 512
OPTIMIZER_STEPS = 600
TICKS_PER_STEP = 4
VALIDATION_INTERVAL = 100
LEARNING_RATE = 2.0e-6
WEIGHT_DECAY = 1.0e-5
PREVIOUS_ACTION_DROPOUT = 0.25
GAMEPLAY_RMSE_CEILING = 0.60
MECHANIC_RMSE_CEILING = 0.62
AUTHORITY_FORMAT = "RIVAL2_CODEX_AUTONOMOUS_V16_AUTHORITY"
CAMPAIGN_IDENTITY = "RIVAL2_CODEX_AUTONOMOUS_V16_ON_POLICY_DAGGER"


def _configure_bridge() -> None:
    bridge.SOURCE = SOURCE
    bridge.SOURCE_SHA256 = SOURCE_SHA256
    bridge.SOURCE_MODEL_SHA256 = SOURCE_MODEL_SHA256
    bridge.AUTHORITY = AUTHORITY
    bridge.RESULTS = RESULTS
    bridge.CHECKPOINT = CHECKPOINT
    bridge.BASELINE = BASELINE
    bridge.SEED = SEED
    bridge.WORLDS = WORLDS
    bridge.NATIVE_BATCH = NATIVE_BATCH
    bridge.HUMAN_GAMEPLAY_BATCH = HUMAN_GAMEPLAY_BATCH
    bridge.HUMAN_MECHANIC_BATCH = HUMAN_MECHANIC_BATCH
    bridge.OPTIMIZER_STEPS = OPTIMIZER_STEPS
    bridge.VALIDATION_INTERVAL = VALIDATION_INTERVAL
    bridge.LEARNING_RATE = LEARNING_RATE
    bridge.WEIGHT_DECAY = WEIGHT_DECAY
    bridge.PREVIOUS_ACTION_DROPOUT = PREVIOUS_ACTION_DROPOUT
    bridge.GAMEPLAY_RMSE_CEILING = GAMEPLAY_RMSE_CEILING
    bridge.MECHANIC_RMSE_CEILING = MECHANIC_RMSE_CEILING
    bridge.CAMPAIGN_IDENTITY = CAMPAIGN_IDENTITY
    bridge.AUTHORITY_FORMAT = AUTHORITY_FORMAT


def _full_match_rank(evaluation: dict[str, Any]) -> tuple[int, int, int]:
    ledger = evaluation["canonical"]["canonical_match_ledger"]
    wins = sum(row["winner"] == "Rival" for row in ledger)
    goals_for = sum(int(row["rival_score"]) for row in ledger)
    goals_against = sum(int(row["nexto_score"]) for row in ledger)
    return int(wins), goals_for - goals_against, goals_for


def _full_match_evaluate(
    checkpoint: Path,
    checkpoint_sha256: str,
    output_dir: Path,
    collision_root: Path,
) -> dict[str, Any]:
    command = [
        str(ROOT / ".venv/Scripts/python.exe"),
        "-u",
        str(ROOT / "benchmarks/run_rival2_codex_autonomous_full_match_eval.py"),
        "--checkpoint",
        str(checkpoint),
        "--checkpoint-sha256",
        checkpoint_sha256,
        "--output-dir",
        str(output_dir),
        "--collision-root",
        str(collision_root),
        "--seed",
        str(bridge.EVALUATION_SEED),
    ]
    output_dir.mkdir(parents=True, exist_ok=True)
    with (output_dir / "stdout.txt").open(
        "w", encoding="utf-8", newline="\n"
    ) as stdout, (output_dir / "stderr.txt").open(
        "w", encoding="utf-8", newline="\n"
    ) as stderr:
        completed = subprocess.run(
            command, cwd=ROOT, stdout=stdout, stderr=stderr, check=False
        )
    if completed.returncode != 0:
        raise RuntimeError((output_dir / "stderr.txt").read_text(encoding="utf-8")[-4000:])
    return json.loads(
        (output_dir / "full_match_evaluation.json").read_text(encoding="utf-8")
    )


def _full_match_summary(evaluation: dict[str, Any]) -> dict[str, Any]:
    ledger = evaluation["canonical"]["canonical_match_ledger"]
    rank = _full_match_rank(evaluation)
    return {
        "wins": rank[0],
        "losses": len(ledger) - rank[0],
        "goals_for": sum(int(row["rival_score"]) for row in ledger),
        "goals_against": sum(int(row["nexto_score"]) for row in ledger),
        "rank": list(rank),
    }


def run(args: argparse.Namespace) -> int:
    _configure_bridge()
    torch.cuda.set_device(args.device)
    torch.set_float32_matmul_precision("highest")
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    source, preflight = bridge._preflight(args)
    preflight["format"] = "RIVAL2_CODEX_AUTONOMOUS_V16_PREFLIGHT"
    preflight["native_teacher_state_role"] = "current student car"
    preflight["dagger_on_policy"] = True
    RESULTS.mkdir(parents=True, exist_ok=True)
    base.write_json(RESULTS / "preflight.json", preflight)
    if args.preflight_only:
        print(json.dumps(preflight, indent=2, sort_keys=True))
        return 0

    run_dir = Path(args.run_dir)
    if run_dir.exists() and any(run_dir.iterdir()):
        raise RuntimeError("V16 run directory must be fresh")
    run_dir.mkdir(parents=True, exist_ok=True)
    base.SOURCE = SOURCE
    base.SOURCE_SHA256 = SOURCE_SHA256
    train, validation, teacher, human_identity = base.load_human_data(device=args.device)
    if human_identity["test_loaded"]:
        raise RuntimeError("V16 human test split must remain unopened")

    config = Rival2PolicyConfig(**source["policy_config"])
    student = Rival2ActorCritic(config).to(args.device)
    student.load_state_dict(source["model"])
    student.train()
    student.critic.requires_grad_(False)
    critic_hash = base.tensor_tree_sha256(
        {name: value.detach().cpu() for name, value in student.critic.state_dict().items()}
    )
    optimizer = torch.optim.AdamW(
        [*student.trunk.parameters(), *student.actor.parameters()],
        lr=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY,
    )
    teacher.eval().requires_grad_(False)

    cpu_generator = torch.Generator(device="cpu").manual_seed(SEED ^ 0x48554D41)
    cuda_generator = torch.Generator(device=args.device).manual_seed(SEED ^ 0x44414747)
    mechanic_sampler = MechanicHierarchySampler(
        train.mechanic_label,
        train.mechanic_attempt,
        uniform_label_fraction=0.10,
        maximum_oversampling_ratio=4.0,
        generator=cpu_generator,
    )
    layout_generator = torch.Generator(device="cpu").manual_seed(SEED ^ 0x4C41594F)
    starting_layout = torch.randint(5, (WORLDS,), generator=layout_generator).numpy()
    rival_side = torch.randint(2, (WORLDS,), generator=layout_generator).numpy()
    runner = FullMatchRunner(
        WORLDS,
        str(args.collision_root),
        SOURCE,
        starting_layout=starting_layout,
        rival_side=rival_side,
        stochastic_rival=True,
        evaluation_seed=SEED,
        device=args.device,
    )
    # The environment now follows the current student.  The frozen checkpoint
    # loaded by FullMatchRunner remains only the initialization and metadata.
    runner.rival_policy = student
    oracle = NextoPolicyAdapter(WORLDS, device=args.device)
    oracle.set_player_index(runner.rival_side)
    oracle_state = NextoStateTensors.from_bridge(runner.bridge)

    baseline_evaluation = json.loads(BASELINE.read_text(encoding="utf-8"))
    baseline_validation = bridge._validation(student, validation, device=args.device)
    best: dict[str, Any] = {
        "accepted_supervised_steps": 0,
        "checkpoint": {
            "path": SOURCE.relative_to(ROOT).as_posix(),
            "sha256": SOURCE_SHA256,
            "model_tensor_sha256": SOURCE_MODEL_SHA256,
        },
        "evaluation": baseline_evaluation,
        "human_validation": baseline_validation,
    }
    base.write_json(RESULTS / "baseline.json", best)

    accepted_steps = 0
    native_samples = 0
    human_samples = 0
    dropped_samples = 0
    for tick in range(1, OPTIMIZER_STEPS * TICKS_PER_STEP + 1):
        observation_all = runner.rival_observation[
            runner.batch_index, runner.rival_side
        ].detach().clone()
        kickoff_active = runner.match_views["kickoff_active"] != 0
        oracle_action, _ = oracle.tick_action(oracle_state, kickoff_active)
        oracle_action = oracle_action.detach().clone()
        runner.tick()
        if tick % TICKS_PER_STEP:
            continue

        native_index = torch.randint(
            WORLDS, (NATIVE_BATCH,), device=args.device, generator=cuda_generator
        )
        native_observation = observation_all.index_select(0, native_index)
        native_action = oracle_action.index_select(0, native_index)
        gameplay_index = torch.randint(
            train.gameplay_observation.shape[0],
            (HUMAN_GAMEPLAY_BATCH,),
            generator=cpu_generator,
        )
        mechanic_index = mechanic_sampler.sample(HUMAN_MECHANIC_BATCH)
        human_observation = torch.cat(
            (
                train.gameplay_observation.index_select(0, gameplay_index),
                train.mechanic_observation.index_select(0, mechanic_index),
            )
        ).to(args.device)
        human_action = torch.cat(
            (
                train.gameplay_action.index_select(0, gameplay_index),
                train.mechanic_action.index_select(0, mechanic_index),
            )
        ).to(args.device)
        native_observation, native_dropped = bridge._drop_previous_action(
            native_observation, generator=cuda_generator
        )
        human_observation, human_dropped = bridge._drop_previous_action(
            human_observation, generator=cuda_generator
        )
        observation = torch.cat((native_observation, human_observation))
        action = torch.cat((native_action, human_action))
        with torch.no_grad():
            teacher_actor, _ = teacher(observation)
        student_actor, _ = student(observation)
        objective = human_behavior_cloning_objective(
            student_actor,
            teacher_actor,
            action,
            smooth_l1_beta=0.1,
            analog_weight=1.0,
            button_weight=0.25,
            log_std_weight=0.05,
            policy_config=config,
        )
        if not bool(torch.isfinite(objective.loss).item()):
            raise RuntimeError("V16 nonfinite supervised objective")
        optimizer.zero_grad(set_to_none=True)
        objective.loss.backward()
        gradient_norm = torch.nn.utils.clip_grad_norm_(
            [*student.trunk.parameters(), *student.actor.parameters()], 0.5
        )
        if not bool(torch.isfinite(gradient_norm).item()):
            raise RuntimeError("V16 nonfinite gradient")
        optimizer.step()
        if not all(bool(torch.isfinite(parameter).all()) for parameter in student.parameters()):
            raise RuntimeError("V16 nonfinite model parameter")
        accepted_steps += 1
        native_samples += NATIVE_BATCH
        human_samples += HUMAN_GAMEPLAY_BATCH + HUMAN_MECHANIC_BATCH
        dropped_samples += native_dropped + human_dropped

        if accepted_steps % VALIDATION_INTERVAL:
            continue
        candidate_path = run_dir / f"candidate_s{accepted_steps:04d}.pt"
        checkpoint = bridge._save_candidate(
            source,
            student,
            optimizer,
            candidate_path,
            accepted_steps=accepted_steps,
            native_samples=native_samples,
            human_samples=human_samples,
        )
        validation_metrics = bridge._validation(student, validation, device=args.device)
        evaluation = bridge._evaluate(
            candidate_path,
            checkpoint["sha256"],
            run_dir / f"candidate_s{accepted_steps:04d}_window.json",
            Path(args.collision_root),
        )
        critic_after = base.tensor_tree_sha256(
            {
                name: value.detach().cpu()
                for name, value in student.critic.state_dict().items()
            }
        )
        row = {
            "accepted_supervised_steps": accepted_steps,
            "checkpoint": checkpoint,
            "evaluation": evaluation,
            "human_validation": validation_metrics,
            "critic_byte_identical": critic_after == critic_hash,
            "loss": {
                "total": float(objective.loss.detach().item()),
                "analog": float(objective.analog_smooth_l1.detach().item()),
                "buttons": float(objective.button_bce.detach().item()),
                "log_std": float(objective.log_std_retention.detach().item()),
            },
            "gradient_norm": float(gradient_norm.detach().item()),
            "native_oracle_samples": native_samples,
            "human_replay_samples": human_samples,
            "previous_action_dropped_samples": dropped_samples,
        }
        base.append_jsonl(RESULTS / "training_curve.jsonl", row)
        eligible = validation_metrics["eligible"] and row["critic_byte_identical"]
        if eligible and bridge._rank(evaluation) > bridge._rank(best["evaluation"]):
            best = copy.deepcopy(row)
            base.write_json(RESULTS / "best.json", best)
        print(
            json.dumps(
                {
                    "step": accepted_steps,
                    "goals": [
                        evaluation["overall"]["goals_for"],
                        evaluation["overall"]["goals_against"],
                    ],
                    "touches": [
                        evaluation["overall"]["rival_touches"],
                        evaluation["overall"]["opponent_touches"],
                    ],
                    "human_gameplay_rmse": validation_metrics["gameplay"][
                        "complete_action_rmse"
                    ],
                    "best_step": best["accepted_supervised_steps"],
                },
                sort_keys=True,
            ),
            flush=True,
        )
        student.train()

    if not (RESULTS / "best.json").exists():
        base.write_json(RESULTS / "best.json", best)
    baseline_full = json.loads(BASELINE_FULL_MATCH.read_text(encoding="utf-8"))
    full_match = {
        "required": best["accepted_supervised_steps"] != 0,
        "baseline": _full_match_summary(baseline_full),
        "candidate": None,
        "passed": False,
    }
    promoted_checkpoint: dict[str, Any] | None = None
    if full_match["required"]:
        selected_path = Path(best["checkpoint"]["path"])
        candidate_full = _full_match_evaluate(
            selected_path,
            best["checkpoint"]["sha256"],
            run_dir / "selected_full_match",
            Path(args.collision_root),
        )
        full_match["candidate"] = _full_match_summary(candidate_full)
        full_match["passed"] = _full_match_rank(candidate_full) > _full_match_rank(
            baseline_full
        )
        if full_match["passed"]:
            CHECKPOINT.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(selected_path, CHECKPOINT)
            promoted_checkpoint = {
                "path": CHECKPOINT.relative_to(ROOT).as_posix(),
                "sha256": base.sha256_file(CHECKPOINT),
                "bytes": CHECKPOINT.stat().st_size,
                "model_tensor_sha256": best["checkpoint"]["model_tensor_sha256"],
                "accepted_supervised_steps": best["accepted_supervised_steps"],
            }
    result = {
        "format": "RIVAL2_CODEX_AUTONOMOUS_V16_RESULT",
        "authority_sha256": base.sha256_file(AUTHORITY),
        "accepted_supervised_steps": accepted_steps,
        "native_oracle_samples": native_samples,
        "human_replay_samples": human_samples,
        "best": best,
        "source_rank": list(bridge._rank(baseline_evaluation)),
        "full_match": full_match,
        "promoted": promoted_checkpoint is not None,
        "promoted_checkpoint": promoted_checkpoint,
        "dagger_on_policy": True,
        "ppo_steps": 0,
        "reward_changes": 0,
        "human_test_not_loaded": True,
    }
    base.write_json(RESULTS / "result.json", result)
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--worlds", type=int, default=WORLDS)
    parser.add_argument(
        "--collision-root",
        type=Path,
        default=Path(r"G:\dev\RLBot-Rival\bot\collision_meshes"),
    )
    parser.add_argument("--run-dir", type=Path, default=DEFAULT_RUN_DIR)
    parser.add_argument("--preflight-only", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    raise SystemExit(run(parse_args()))
