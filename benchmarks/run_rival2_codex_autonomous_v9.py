"""Prospective line search along V4's only demonstrated winning PPO direction."""

from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path
import sys
from typing import Any

import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from benchmarks import run_rival2_codex_autonomous_v1 as base  # noqa: E402
from rivalsim.rival2_policy import Rival2ActorCritic, Rival2PolicyConfig  # noqa: E402


SOURCE = ROOT / "checkpoints/rival2/codex_autonomous_v4/rival2_codex_autonomous_best.pt"
SOURCE_SHA256 = "172BA59786A2E08EB6DC95CFE29F20C21826F7CB9429FF3C89F4D7C4F4BD9E10"
DIRECTION_FROM = Path("G:/dev/RivalSim-runs/codex-autonomous-v4/candidate_u0002.pt")
DIRECTION_FROM_SHA256 = "A4A153525BCD98DC38EBE4EA086C5CCBA46785C84F65C674ABD878DEEAA6711E"
DIRECTION_FROM_MODEL_SHA256 = "1E37E2942D67DAC978A52046DC9F5575A54C4A4C0DDEC425BF26FFAC771CACD7"
DIRECTION_TO_MODEL_SHA256 = "E7167577A87DB5E33E8A848C1B59770C92FAEAF39F0944662FB5F668EB5DD6A3"
AUTHORITY = ROOT / "results/rival2/codex_autonomous_v9/authority.json"
RESULTS = ROOT / "results/rival2/codex_autonomous_v9"
CHECKPOINT = ROOT / "checkpoints/rival2/codex_autonomous_v9/rival2_codex_autonomous_best.pt"
DEFAULT_RUN_DIR = Path("G:/dev/RivalSim-runs/codex-autonomous-v9")
LAMBDAS = (-0.25, 0.125, 0.25, 0.5, 0.75, 1.0, 1.5, 2.0, 3.0)


def save_candidate(
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
        "identity": "RIVAL2_CODEX_AUTONOMOUS_V9_DIRECTION_LINE_SEARCH",
        "source": {"path": SOURCE.relative_to(ROOT).as_posix(), "sha256": SOURCE_SHA256},
        "authority": {
            "path": AUTHORITY.relative_to(ROOT).as_posix(),
            "sha256": base.sha256_file(AUTHORITY),
        },
        "direction_from": {
            "path": str(DIRECTION_FROM),
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


def run(args: argparse.Namespace) -> int:
    authority = json.loads(AUTHORITY.read_text(encoding="utf-8"))
    if authority.get("lambdas") != list(LAMBDAS):
        raise RuntimeError("V9 lambda authority mismatch")
    if base.sha256_file(SOURCE) != SOURCE_SHA256:
        raise RuntimeError("V9 source checkpoint changed")
    if base.sha256_file(DIRECTION_FROM) != DIRECTION_FROM_SHA256:
        raise RuntimeError("V9 direction checkpoint changed")
    run_dir = Path(args.run_dir)
    if run_dir.exists() and any(run_dir.iterdir()):
        raise RuntimeError("V9 run directory must be fresh")
    run_dir.mkdir(parents=True, exist_ok=True)
    RESULTS.mkdir(parents=True, exist_ok=True)

    source = torch.load(SOURCE, map_location="cpu", weights_only=False)
    direction_from = torch.load(DIRECTION_FROM, map_location="cpu", weights_only=False)
    if base.tensor_tree_sha256(source["model"]) != DIRECTION_TO_MODEL_SHA256:
        raise RuntimeError("V9 direction-to model mismatch")
    if base.tensor_tree_sha256(direction_from["model"]) != DIRECTION_FROM_MODEL_SHA256:
        raise RuntimeError("V9 direction-from model mismatch")
    delta_weight = source["model"]["actor.weight"][:5] - direction_from["model"][
        "actor.weight"
    ][:5]
    delta_bias = source["model"]["actor.bias"][:5] - direction_from["model"][
        "actor.bias"
    ][:5]
    if not bool(torch.isfinite(delta_weight).all() and torch.isfinite(delta_bias).all()):
        raise RuntimeError("V9 direction is nonfinite")

    base.SOURCE = SOURCE
    base.SOURCE_SHA256 = SOURCE_SHA256
    train, validation, teacher, human_identity = base.load_human_data(device=args.device)
    del train, teacher
    baseline_evaluation = base.run_nexto_evaluation(
        SOURCE,
        campaign_step=0,
        run_dir=run_dir,
        device=args.device,
        collision_dir=Path(args.collision_dir),
    )
    policy_config = Rival2PolicyConfig(**source["policy_config"])
    model = Rival2ActorCritic(policy_config).to(args.device)
    model.load_state_dict(source["model"])
    baseline_human = base.human_validation(model, validation, device=args.device)
    best = {
        "candidate_index": 0,
        "direction_lambda": 0.0,
        "checkpoint": {
            "path": SOURCE.relative_to(ROOT).as_posix(),
            "sha256": SOURCE_SHA256,
            "model_tensor_sha256": DIRECTION_TO_MODEL_SHA256,
        },
        "evaluation": baseline_evaluation,
        "human_validation": baseline_human,
    }
    preflight = {
        "format": "RIVAL2_CODEX_AUTONOMOUS_V9_PREFLIGHT",
        "verdict": "PASS",
        "authority_sha256": base.sha256_file(AUTHORITY),
        "source_sha256": base.sha256_file(SOURCE),
        "direction_from_sha256": base.sha256_file(DIRECTION_FROM),
        "direction_from_model_sha256": base.tensor_tree_sha256(direction_from["model"]),
        "direction_to_model_sha256": base.tensor_tree_sha256(source["model"]),
        "direction_weight_rms": float(delta_weight.square().mean().sqrt().item()),
        "direction_weight_max_abs": float(delta_weight.abs().max().item()),
        "direction_bias_rms": float(delta_bias.square().mean().sqrt().item()),
        "human_test_not_loaded": human_identity["test_loaded"] is False,
        "changed_parameter_boundary": ["actor.weight[0:5]", "actor.bias[0:5]"],
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
        checkpoint = save_candidate(
            source,
            state,
            candidate_path,
            candidate_index=candidate_index,
            direction_lambda=direction_lambda,
        )
        model.load_state_dict(state)
        human = base.human_validation(model, validation, device=args.device)
        evaluation = base.run_nexto_evaluation(
            candidate_path,
            campaign_step=candidate_index,
            run_dir=run_dir,
            device=args.device,
            collision_dir=Path(args.collision_dir),
        )
        eligible = bool(human["eligible"] and evaluation["no_touch_episodes"] == 0)
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
        if eligible and base.candidate_rank(evaluation) > base.candidate_rank(best["evaluation"]):
            promoted = save_candidate(
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
                    "nexto": [evaluation["goals_for"], evaluation["goals_against"]],
                    "touches": evaluation["touches"],
                    "human_gameplay_rmse": human["gameplay"]["complete_action_rmse"],
                    "best_lambda": best["direction_lambda"],
                    "best_nexto": [best["evaluation"]["goals_for"], best["evaluation"]["goals_against"]],
                },
                sort_keys=True,
            ),
            flush=True,
        )
    if not (RESULTS / "best.json").exists():
        base.write_json(RESULTS / "best.json", best)
    result = {
        "format": "RIVAL2_CODEX_AUTONOMOUS_V9_RESULT",
        "authority_sha256": base.sha256_file(AUTHORITY),
        "candidates": len(rows),
        "best": best,
        "target_nexto_win_rate": 0.55,
        "target_reached": best["evaluation"]["win_rate"] >= 0.55,
        "ppo_steps": 0,
        "optimizer_steps": 0,
    }
    base.write_json(RESULTS / "result.json", result)
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--collision-dir", type=Path, default=base.DEFAULT_COLLISION_DIR)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--run-dir", type=Path, default=DEFAULT_RUN_DIR)
    parser.add_argument("--preflight-only", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    raise SystemExit(run(parse_args()))
