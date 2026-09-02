"""Native-domain teacher distillation with bounded human demonstration replay.

This is a supervised bridge stage, not PPO.  A frozen competitive Rival policy
and frozen Nexto opponent generate authoritative simulator states.  The student
learns the exact Nexto action consumed by the simulator while every optimizer
batch retains equal reviewed human gameplay/mechanic representation.
"""

from __future__ import annotations

import argparse
import copy
import json
import subprocess
import sys
from pathlib import Path
from typing import Any

import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from benchmarks import run_rival2_codex_autonomous_v1 as base
from rivalsim.full_match import FullMatchRunner
from rivalsim.human_demo.behavior_cloning import (
    MechanicHierarchySampler,
    human_behavior_cloning_objective,
)
from rivalsim.rival2_policy import Rival2ActorCritic, Rival2PolicyConfig
from third_party.nexto.adapter import MODEL_SHA256 as NEXTO_MODEL_SHA256

SOURCE = (
    ROOT
    / "checkpoints/rival2/codex_autonomous_match_v1/rival2_codex_autonomous_match_parent.pt"
)
SOURCE_SHA256 = "0B90C201A0E1A16E83CF5CCBDE3371434F78D455C2AED20E0DDA6414F3B84E39"
SOURCE_MODEL_SHA256 = "A8E6BCED160F9C6871B9D4DF8D91AD2FDF260AA76E64F5788FEACF0371564500"
BASELINE = ROOT / "results/rival2/codex_autonomous_match_v1/window_sweep/v7_u0030.json"
AUTHORITY = ROOT / "results/rival2/codex_autonomous_v14/authority.json"
RESULTS = ROOT / "results/rival2/codex_autonomous_v14"
CHECKPOINT = ROOT / "checkpoints/rival2/codex_autonomous_v14/rival2_codex_autonomous_best.pt"
DEFAULT_RUN_DIR = Path("G:/dev/RivalSim-runs/codex-autonomous-v14")

SEED = 2_026_090_217
WORLDS = 8192
NATIVE_BATCH = 2048
HUMAN_GAMEPLAY_BATCH = 1024
HUMAN_MECHANIC_BATCH = 1024
OPTIMIZER_STEPS = 300
TICKS_PER_STEP = 4
LEARNING_RATE = 5.0e-6
WEIGHT_DECAY = 1.0e-5
PREVIOUS_ACTION_DROPOUT = 0.5
VALIDATION_INTERVAL = 50
WINDOW_SECONDS = 60
EVALUATION_SEED = 2_026_090_206
GAMEPLAY_RMSE_CEILING = 0.60
MECHANIC_RMSE_CEILING = 0.62
CAMPAIGN_IDENTITY = "RIVAL2_CODEX_AUTONOMOUS_V14_NATIVE_TEACHER_BRIDGE"
AUTHORITY_FORMAT = "RIVAL2_CODEX_AUTONOMOUS_V14_AUTHORITY"
PREVIOUS_ACTION_SLICE = slice(167, 175)


def _rank(evaluation: dict[str, Any]) -> tuple[int, int, int, int]:
    overall = evaluation["overall"]
    return (
        int(overall["goal_differential"]),
        int(overall["goals_for"]),
        int(overall["touch_differential"]),
        int(overall["rival_touches"]),
    )


def _drop_previous_action(
    observation: torch.Tensor,
    *,
    generator: torch.Generator,
) -> tuple[torch.Tensor, int]:
    mask = torch.rand(
        observation.shape[0], device=observation.device, generator=generator
    ) < PREVIOUS_ACTION_DROPOUT
    output = observation.clone()
    output[mask, PREVIOUS_ACTION_SLICE] = 0.0
    return output, int(mask.sum().item())


def _save_candidate(
    source: dict[str, Any],
    model: Rival2ActorCritic,
    optimizer: torch.optim.Optimizer,
    path: Path,
    *,
    accepted_steps: int,
    native_samples: int,
    human_samples: int,
) -> dict[str, Any]:
    payload = copy.deepcopy(source)
    payload["model"] = {
        name: tensor.detach().cpu().clone() for name, tensor in model.state_dict().items()
    }
    payload["optimizer"] = optimizer.state_dict()
    payload["iteration"] = int(source["iteration"]) + accepted_steps
    payload["policy_version"] = int(source["policy_version"]) + accepted_steps
    payload["curriculum_transition"] = {
        "identity": CAMPAIGN_IDENTITY,
        "authority": {
            "path": AUTHORITY.relative_to(ROOT).as_posix(),
            "sha256": base.sha256_file(AUTHORITY),
        },
        "source": {
            "path": SOURCE.relative_to(ROOT).as_posix(),
            "sha256": SOURCE_SHA256,
            "model_tensor_sha256": SOURCE_MODEL_SHA256,
        },
        "previous_transition": copy.deepcopy(source.get("curriculum_transition")),
        "training": "supervised native Nexto actions plus reviewed human actions",
        "accepted_supervised_steps": accepted_steps,
        "native_teacher_samples": native_samples,
        "human_replay_samples": human_samples,
        "ppo_steps": 0,
        "reward_changes": 0,
        "ppo_resumable": False,
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, path)
    return {
        "path": path.relative_to(ROOT).as_posix()
        if path.is_relative_to(ROOT)
        else path.as_posix(),
        "sha256": base.sha256_file(path),
        "bytes": path.stat().st_size,
        "model_tensor_sha256": base.tensor_tree_sha256(payload["model"]),
        "accepted_supervised_steps": accepted_steps,
    }


def _evaluate(
    checkpoint: Path,
    checkpoint_sha256: str,
    output: Path,
    collision_root: Path,
) -> dict[str, Any]:
    command = [
        str(ROOT / ".venv/Scripts/python.exe"),
        "-u",
        str(ROOT / "benchmarks/run_rival2_codex_autonomous_match_window_eval.py"),
        "--checkpoint",
        str(checkpoint),
        "--checkpoint-sha256",
        checkpoint_sha256,
        "--output",
        str(output),
        "--window-seconds",
        str(WINDOW_SECONDS),
        "--seed",
        str(EVALUATION_SEED),
        "--collision-root",
        str(collision_root),
    ]
    stdout = output.with_suffix(".stdout.txt")
    stderr = output.with_suffix(".stderr.txt")
    with stdout.open("w", encoding="utf-8", newline="\n") as out, stderr.open(
        "w", encoding="utf-8", newline="\n"
    ) as err:
        completed = subprocess.run(command, cwd=ROOT, stdout=out, stderr=err, check=False)
    if completed.returncode != 0:
        raise RuntimeError(stderr.read_text(encoding="utf-8")[-4000:])
    return json.loads(output.read_text(encoding="utf-8"))


@torch.no_grad()
def _validation(
    model: Rival2ActorCritic,
    validation: Any,
    *,
    device: str,
) -> dict[str, Any]:
    metrics = base.human_validation(model, validation, device=device)
    metrics["eligible"] = bool(
        metrics["gameplay"]["complete_action_rmse"] <= GAMEPLAY_RMSE_CEILING
        and metrics["mechanic"]["complete_action_rmse"] <= MECHANIC_RMSE_CEILING
        and metrics["gameplay"]["finite"]
        and metrics["mechanic"]["finite"]
    )
    return metrics


def _preflight(args: argparse.Namespace) -> tuple[dict[str, Any], dict[str, Any]]:
    authority = json.loads(AUTHORITY.read_text(encoding="utf-8"))
    if authority.get("format") != AUTHORITY_FORMAT:
        raise RuntimeError("V14 authority format mismatch")
    if base.sha256_file(SOURCE) != SOURCE_SHA256:
        raise RuntimeError("V14 source checkpoint changed")
    source = torch.load(SOURCE, map_location="cpu", weights_only=False)
    if base.tensor_tree_sha256(source["model"]) != SOURCE_MODEL_SHA256:
        raise RuntimeError("V14 source model changed")
    if authority["teacher"]["nexto_model_sha256"] != NEXTO_MODEL_SHA256:
        raise RuntimeError("V14 Nexto teacher identity mismatch")
    if int(args.worlds) != WORLDS:
        raise RuntimeError(f"V14 requires exactly {WORLDS} worlds")
    preflight = {
        "format": "RIVAL2_CODEX_AUTONOMOUS_V14_PREFLIGHT",
        "verdict": "PASS",
        "authority_sha256": base.sha256_file(AUTHORITY),
        "source_sha256": SOURCE_SHA256,
        "source_model_tensor_sha256": SOURCE_MODEL_SHA256,
        "nexto_model_sha256": NEXTO_MODEL_SHA256,
        "worlds": WORLDS,
        "optimizer_steps": OPTIMIZER_STEPS,
        "ppo_steps": 0,
        "reward_changes": 0,
        "human_test_not_loaded": True,
        "critic_trainable": False,
        "training_domains": ["native RivalSim/Nexto", "reviewed human demonstrations"],
        "previous_action_dropout": PREVIOUS_ACTION_DROPOUT,
    }
    return source, preflight


def run(args: argparse.Namespace) -> int:
    torch.cuda.set_device(args.device)
    torch.set_float32_matmul_precision("highest")
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    source, preflight = _preflight(args)
    RESULTS.mkdir(parents=True, exist_ok=True)
    base.write_json(RESULTS / "preflight.json", preflight)
    if args.preflight_only:
        print(json.dumps(preflight, indent=2, sort_keys=True))
        return 0

    run_dir = Path(args.run_dir)
    if run_dir.exists() and any(run_dir.iterdir()):
        raise RuntimeError("V14 run directory must be fresh")
    run_dir.mkdir(parents=True, exist_ok=True)
    base.SOURCE = SOURCE
    base.SOURCE_SHA256 = SOURCE_SHA256
    train, validation, teacher, human_identity = base.load_human_data(device=args.device)
    if human_identity["test_loaded"]:
        raise RuntimeError("V14 human test split must remain unopened")

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
    cuda_generator = torch.Generator(device=args.device).manual_seed(SEED ^ 0x4E415449)
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

    baseline_evaluation = json.loads(BASELINE.read_text(encoding="utf-8"))
    if baseline_evaluation.get("checkpoint_sha256") != SOURCE_SHA256:
        raise RuntimeError("V14 baseline checkpoint mismatch")
    baseline_validation = _validation(student, validation, device=args.device)
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

    native_samples = 0
    human_samples = 0
    dropout_samples = 0
    accepted_steps = 0
    for tick in range(1, OPTIMIZER_STEPS * TICKS_PER_STEP + 1):
        native_observation_all = runner.rival_observation[
            runner.batch_index, runner.nexto_side
        ].detach().clone()
        runner.tick()
        if tick % TICKS_PER_STEP:
            continue
        native_index = torch.randint(
            WORLDS, (NATIVE_BATCH,), device=args.device, generator=cuda_generator
        )
        native_observation = native_observation_all.index_select(0, native_index)
        native_action = runner.nexto.previous_action.index_select(0, native_index).detach()
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
        native_observation, native_dropped = _drop_previous_action(
            native_observation, generator=cuda_generator
        )
        human_observation, human_dropped = _drop_previous_action(
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
            raise RuntimeError("V14 nonfinite supervised objective")
        optimizer.zero_grad(set_to_none=True)
        objective.loss.backward()
        gradient_norm = torch.nn.utils.clip_grad_norm_(
            [*student.trunk.parameters(), *student.actor.parameters()], 0.5
        )
        if not bool(torch.isfinite(gradient_norm).item()):
            raise RuntimeError("V14 nonfinite gradient")
        optimizer.step()
        if not all(bool(torch.isfinite(parameter).all()) for parameter in student.parameters()):
            raise RuntimeError("V14 nonfinite model parameter")
        accepted_steps += 1
        native_samples += NATIVE_BATCH
        human_samples += HUMAN_GAMEPLAY_BATCH + HUMAN_MECHANIC_BATCH
        dropout_samples += native_dropped + human_dropped

        if accepted_steps % VALIDATION_INTERVAL:
            continue
        candidate_path = run_dir / f"candidate_s{accepted_steps:04d}.pt"
        checkpoint = _save_candidate(
            source,
            student,
            optimizer,
            candidate_path,
            accepted_steps=accepted_steps,
            native_samples=native_samples,
            human_samples=human_samples,
        )
        validation_metrics = _validation(student, validation, device=args.device)
        evaluation = _evaluate(
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
            "native_teacher_samples": native_samples,
            "human_replay_samples": human_samples,
            "previous_action_dropped_samples": dropout_samples,
        }
        base.append_jsonl(RESULTS / "training_curve.jsonl", row)
        eligible = validation_metrics["eligible"] and row["critic_byte_identical"]
        if eligible and _rank(evaluation) > _rank(best["evaluation"]):
            promoted = _save_candidate(
                source,
                student,
                optimizer,
                CHECKPOINT,
                accepted_steps=accepted_steps,
                native_samples=native_samples,
                human_samples=human_samples,
            )
            best = copy.deepcopy(row)
            best["checkpoint"] = promoted
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

    if not (RESULTS / "best.json").exists():
        base.write_json(RESULTS / "best.json", best)
    result = {
        "format": "RIVAL2_CODEX_AUTONOMOUS_V14_RESULT",
        "authority_sha256": base.sha256_file(AUTHORITY),
        "accepted_supervised_steps": accepted_steps,
        "native_teacher_samples": native_samples,
        "human_replay_samples": human_samples,
        "best": best,
        "source_rank": list(_rank(baseline_evaluation)),
        "promoted": best["accepted_supervised_steps"] != 0,
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
