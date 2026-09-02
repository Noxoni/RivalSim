"""Search the demonstrated update-25 to update-30 sustained-play direction."""

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
from rivalsim.rival2_policy import Rival2ActorCritic, Rival2PolicyConfig

SOURCE = (
    ROOT
    / "checkpoints/rival2/codex_autonomous_match_v1/rival2_codex_autonomous_match_parent.pt"
)
SOURCE_SHA256 = "0B90C201A0E1A16E83CF5CCBDE3371434F78D455C2AED20E0DDA6414F3B84E39"
SOURCE_MODEL_SHA256 = "A8E6BCED160F9C6871B9D4DF8D91AD2FDF260AA76E64F5788FEACF0371564500"
DIRECTION_FROM = Path("G:/dev/RivalSim-runs/codex-autonomous-v7/candidate_u0025.pt")
DIRECTION_FROM_SHA256 = "47932A97A17917AE88C5F9937EF62A9D1ABD14388FF59D71D4A2D50EFA9EBE34"
DIRECTION_FROM_MODEL_SHA256 = "D1847183F2235A07E900E715B8595B25A8A12F8ACB9D6EEFCF21F321ADB99089"
BASELINE = ROOT / "results/rival2/codex_autonomous_match_v1/window_sweep/v7_u0030.json"
AUTHORITY = ROOT / "results/rival2/codex_autonomous_v13/authority.json"
RESULTS = ROOT / "results/rival2/codex_autonomous_v13"
CHECKPOINT = ROOT / "checkpoints/rival2/codex_autonomous_v13/rival2_codex_autonomous_best.pt"
DEFAULT_RUN_DIR = Path("G:/dev/RivalSim-runs/codex-autonomous-v13")
LAMBDAS = (0.05, 0.10, 0.20, 0.35, 0.50, 0.75, 1.00, 1.50)
WINDOW_SECONDS = 60
EVALUATION_SEED = 2_026_090_206
CAMPAIGN_IDENTITY = "RIVAL2_CODEX_AUTONOMOUS_V13_SUSTAINED_DIRECTION_SEARCH"
AUTHORITY_FORMAT = "RIVAL2_CODEX_AUTONOMOUS_V13_AUTHORITY"


def _rank(evaluation: dict[str, Any]) -> tuple[int, int, int, int]:
    metrics = evaluation["overall"]
    return (
        int(metrics["goal_differential"]),
        int(metrics["goals_for"]),
        int(metrics["touch_differential"]),
        int(metrics["rival_touches"]),
    )


def _save_candidate(
    source: dict[str, Any],
    model_state: dict[str, torch.Tensor],
    path: Path,
    *,
    candidate_index: int,
    direction_lambda: float,
) -> dict[str, Any]:
    payload = copy.deepcopy(source)
    payload["model"] = {name: value.clone() for name, value in model_state.items()}
    payload["optimizer"] = {"state": {}, "param_groups": []}
    payload["policy_version"] = int(source["policy_version"]) + candidate_index
    payload["curriculum_transition"] = {
        "identity": CAMPAIGN_IDENTITY,
        "source": {
            "path": SOURCE.relative_to(ROOT).as_posix(),
            "sha256": SOURCE_SHA256,
        },
        "authority": {
            "path": AUTHORITY.relative_to(ROOT).as_posix(),
            "sha256": base.sha256_file(AUTHORITY),
        },
        "direction_from": {
            "path": DIRECTION_FROM.as_posix(),
            "sha256": DIRECTION_FROM_SHA256,
            "model_tensor_sha256": DIRECTION_FROM_MODEL_SHA256,
        },
        "direction_lambda": direction_lambda,
        "changed_parameters": ["actor.weight[0:5]", "actor.bias[0:5]"],
        "ppo_steps": 0,
        "optimizer_steps": 0,
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, path)
    return {
        "path": str(path),
        "sha256": base.sha256_file(path),
        "bytes": path.stat().st_size,
        "model_tensor_sha256": base.tensor_tree_sha256(payload["model"]),
        "direction_lambda": direction_lambda,
    }


def _evaluate(
    checkpoint: Path,
    *,
    checkpoint_sha256: str,
    output: Path,
    collision_root: Path,
) -> dict[str, Any]:
    stdout = output.with_suffix(".stdout.txt")
    stderr = output.with_suffix(".stderr.txt")
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
    with stdout.open("w", encoding="utf-8", newline="\n") as stdout_handle, stderr.open(
        "w", encoding="utf-8", newline="\n"
    ) as stderr_handle:
        completed = subprocess.run(
            command,
            cwd=ROOT,
            stdout=stdout_handle,
            stderr=stderr_handle,
            check=False,
        )
    if completed.returncode != 0:
        raise RuntimeError(
            f"match-window evaluation failed: {stderr.read_text(encoding='utf-8')[-4000:]}"
        )
    return json.loads(output.read_text(encoding="utf-8"))


def run(args: argparse.Namespace) -> int:
    authority = json.loads(AUTHORITY.read_text(encoding="utf-8"))
    if authority.get("format") != AUTHORITY_FORMAT:
        raise RuntimeError("V13 authority format mismatch")
    if authority.get("lambdas") != list(LAMBDAS):
        raise RuntimeError("V13 lambda authority mismatch")
    if base.sha256_file(SOURCE) != SOURCE_SHA256:
        raise RuntimeError("V13 source checkpoint changed")
    if base.sha256_file(DIRECTION_FROM) != DIRECTION_FROM_SHA256:
        raise RuntimeError("V13 direction checkpoint changed")
    run_dir = Path(args.run_dir)
    if run_dir.exists() and any(run_dir.iterdir()):
        raise RuntimeError("V13 run directory must be fresh")
    run_dir.mkdir(parents=True, exist_ok=True)
    RESULTS.mkdir(parents=True, exist_ok=True)

    source = torch.load(SOURCE, map_location="cpu", weights_only=False)
    direction_from = torch.load(DIRECTION_FROM, map_location="cpu", weights_only=False)
    if base.tensor_tree_sha256(source["model"]) != SOURCE_MODEL_SHA256:
        raise RuntimeError("V13 source model mismatch")
    if base.tensor_tree_sha256(direction_from["model"]) != DIRECTION_FROM_MODEL_SHA256:
        raise RuntimeError("V13 direction-from model mismatch")
    delta_weight = source["model"]["actor.weight"][:5] - direction_from["model"][
        "actor.weight"
    ][:5]
    delta_bias = source["model"]["actor.bias"][:5] - direction_from["model"][
        "actor.bias"
    ][:5]
    if not bool(torch.isfinite(delta_weight).all() and torch.isfinite(delta_bias).all()):
        raise RuntimeError("V13 direction is nonfinite")

    base.SOURCE = SOURCE
    base.SOURCE_SHA256 = SOURCE_SHA256
    train, validation, teacher, human_identity = base.load_human_data(device=args.device)
    del train, teacher
    baseline_evaluation = json.loads(BASELINE.read_text(encoding="utf-8"))
    if (
        baseline_evaluation.get("checkpoint_sha256") != SOURCE_SHA256
        or baseline_evaluation.get("window_seconds") != WINDOW_SECONDS
        or baseline_evaluation.get("seed") != EVALUATION_SEED
    ):
        raise RuntimeError("V13 committed baseline identity mismatch")
    policy_config = Rival2PolicyConfig(**source["policy_config"])
    model = Rival2ActorCritic(policy_config).to(args.device)
    model.load_state_dict(source["model"])
    baseline_human = base.human_validation(model, validation, device=args.device)
    best: dict[str, Any] = {
        "candidate_index": 0,
        "direction_lambda": 0.0,
        "checkpoint": {
            "path": SOURCE.relative_to(ROOT).as_posix(),
            "sha256": SOURCE_SHA256,
            "model_tensor_sha256": SOURCE_MODEL_SHA256,
        },
        "evaluation": baseline_evaluation,
        "human_validation": baseline_human,
    }
    preflight = {
        "format": "RIVAL2_CODEX_AUTONOMOUS_V13_PREFLIGHT",
        "verdict": "PASS",
        "authority_sha256": base.sha256_file(AUTHORITY),
        "source_sha256": base.sha256_file(SOURCE),
        "direction_from_sha256": base.sha256_file(DIRECTION_FROM),
        "direction_from_model_sha256": base.tensor_tree_sha256(
            direction_from["model"]
        ),
        "direction_to_model_sha256": base.tensor_tree_sha256(source["model"]),
        "direction_weight_rms": float(delta_weight.square().mean().sqrt().item()),
        "direction_weight_max_abs": float(delta_weight.abs().max().item()),
        "direction_bias_rms": float(delta_bias.square().mean().sqrt().item()),
        "human_test_not_loaded": human_identity["test_loaded"] is False,
        "changed_parameter_boundary": ["actor.weight[0:5]", "actor.bias[0:5]"],
        "baseline_rank": list(_rank(baseline_evaluation)),
        "ppo_steps": 0,
        "optimizer_steps": 0,
    }
    base.write_json(RESULTS / "preflight.json", preflight)
    base.write_json(RESULTS / "baseline.json", best)
    if args.preflight_only:
        print(json.dumps(preflight, indent=2, sort_keys=True))
        return 0

    rows: list[dict[str, Any]] = []
    for candidate_index, direction_lambda in enumerate(LAMBDAS, start=1):
        state = {name: value.clone() for name, value in source["model"].items()}
        state["actor.weight"][:5].add_(delta_weight, alpha=direction_lambda)
        state["actor.bias"][:5].add_(delta_bias, alpha=direction_lambda)
        candidate_path = run_dir / f"candidate_d{candidate_index:02d}.pt"
        checkpoint = _save_candidate(
            source,
            state,
            candidate_path,
            candidate_index=candidate_index,
            direction_lambda=direction_lambda,
        )
        model.load_state_dict(state)
        human = base.human_validation(model, validation, device=args.device)
        evaluation = _evaluate(
            candidate_path,
            checkpoint_sha256=checkpoint["sha256"],
            output=run_dir / f"candidate_d{candidate_index:02d}_match_window.json",
            collision_root=Path(args.collision_root),
        )
        eligible = bool(human["eligible"])
        row = {
            "candidate_index": candidate_index,
            "direction_lambda": direction_lambda,
            "checkpoint": checkpoint,
            "evaluation": evaluation,
            "human_validation": human,
            "eligible": eligible,
        }
        rows.append(row)
        base.append_jsonl(RESULTS / "candidates.jsonl", row)
        if eligible and _rank(evaluation) > _rank(best["evaluation"]):
            promoted = _save_candidate(
                source,
                state,
                CHECKPOINT,
                candidate_index=candidate_index,
                direction_lambda=direction_lambda,
            )
            best = copy.deepcopy(row)
            best["checkpoint"] = promoted
            base.write_json(RESULTS / "best.json", best)
        print(
            json.dumps(
                {
                    "lambda": direction_lambda,
                    "goals": [
                        evaluation["overall"]["goals_for"],
                        evaluation["overall"]["goals_against"],
                    ],
                    "touches": [
                        evaluation["overall"]["rival_touches"],
                        evaluation["overall"]["opponent_touches"],
                    ],
                    "human_gameplay_rmse": human["gameplay"][
                        "complete_action_rmse"
                    ],
                    "best_lambda": best["direction_lambda"],
                    "best_rank": list(_rank(best["evaluation"])),
                },
                sort_keys=True,
            ),
            flush=True,
        )
    if not (RESULTS / "best.json").exists():
        base.write_json(RESULTS / "best.json", best)
    result = {
        "format": "RIVAL2_CODEX_AUTONOMOUS_V13_RESULT",
        "authority_sha256": base.sha256_file(AUTHORITY),
        "candidates": len(rows),
        "best": best,
        "source_rank": list(_rank(baseline_evaluation)),
        "promoted": best["candidate_index"] != 0,
        "ppo_steps": 0,
        "optimizer_steps": 0,
    }
    base.write_json(RESULTS / "result.json", result)
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--collision-root", type=Path, default=Path(r"G:\dev\RLBot-Rival\bot\collision_meshes"))
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--run-dir", type=Path, default=DEFAULT_RUN_DIR)
    parser.add_argument("--preflight-only", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    raise SystemExit(run(parse_args()))
