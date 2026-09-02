"""Tiny antithetic analog-head search around the V10 Nexto-competitive policy."""

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


SOURCE = ROOT / "checkpoints/rival2/codex_autonomous_v10/rival2_codex_autonomous_best.pt"
SOURCE_SHA256 = "2B1D082BD12AA5F1AE90206BB40C22AF24459B504B3429A62752795C35623AEB"
SOURCE_MODEL_SHA256 = "8DFDFE2C0D3CCEDA2CE0FA664017820E21122789C8E8EEF261CF3FF0081C75B9"
AUTHORITY = ROOT / "results/rival2/codex_autonomous_v11/authority.json"
RESULTS = ROOT / "results/rival2/codex_autonomous_v11"
CHECKPOINT = ROOT / "checkpoints/rival2/codex_autonomous_v11/rival2_codex_autonomous_best.pt"
DEFAULT_RUN_DIR = Path("G:/dev/RivalSim-runs/codex-autonomous-v11")
SEED = 2_026_090_229
DIRECTIONS = 16
PERTURBATION_RMS = 1.0e-7
TARGET_WIN_RATE = 0.55


def save_candidate(
    source: dict[str, Any],
    state: dict[str, torch.Tensor],
    path: Path,
    *,
    candidate_index: int,
    direction_index: int,
    sign: int,
) -> dict[str, Any]:
    payload = copy.deepcopy(source)
    payload["model"] = {name: value.clone() for name, value in state.items()}
    payload["optimizer"] = {"state": {}, "param_groups": []}
    payload["policy_version"] = int(source["policy_version"]) + candidate_index
    payload["curriculum_transition"] = {
        "identity": "RIVAL2_CODEX_AUTONOMOUS_V11_LOCAL_ANTITHETIC_SEARCH",
        "source": {"path": SOURCE.relative_to(ROOT).as_posix(), "sha256": SOURCE_SHA256},
        "authority": {
            "path": AUTHORITY.relative_to(ROOT).as_posix(),
            "sha256": base.sha256_file(AUTHORITY),
        },
        "direction_index": direction_index,
        "sign": sign,
        "perturbation_rms": PERTURBATION_RMS,
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
        "candidate_index": candidate_index,
        "direction_index": direction_index,
        "sign": sign,
    }


def run(args: argparse.Namespace) -> int:
    authority = json.loads(AUTHORITY.read_text(encoding="utf-8"))
    checks = {
        "format": authority.get("format") == "RIVAL2_CODEX_AUTONOMOUS_V11_AUTHORITY",
        "source": authority.get("source", {}).get("sha256") == SOURCE_SHA256,
        "seed": authority.get("search", {}).get("seed") == SEED,
        "directions": authority.get("search", {}).get("antithetic_directions")
        == DIRECTIONS,
        "rms": authority.get("search", {}).get("perturbation_rms")
        == PERTURBATION_RMS,
    }
    if not all(checks.values()):
        raise RuntimeError(f"V11 authority mismatch: {checks}")
    if base.sha256_file(SOURCE) != SOURCE_SHA256:
        raise RuntimeError("V11 source checkpoint changed")
    run_dir = Path(args.run_dir)
    if run_dir.exists() and any(run_dir.iterdir()):
        raise RuntimeError("V11 run directory must be fresh")
    run_dir.mkdir(parents=True, exist_ok=True)
    RESULTS.mkdir(parents=True, exist_ok=True)

    source = torch.load(SOURCE, map_location="cpu", weights_only=False)
    if base.tensor_tree_sha256(source["model"]) != SOURCE_MODEL_SHA256:
        raise RuntimeError("V11 source model tensor mismatch")
    base.SOURCE = SOURCE
    base.SOURCE_SHA256 = SOURCE_SHA256
    train, validation, teacher, human_identity = base.load_human_data(device=args.device)
    del train, teacher
    model = Rival2ActorCritic(Rival2PolicyConfig(**source["policy_config"])).to(args.device)
    model.load_state_dict(source["model"])
    baseline_human = base.human_validation(model, validation, device=args.device)
    baseline_evaluation = base.run_nexto_evaluation(
        SOURCE,
        campaign_step=0,
        run_dir=run_dir,
        device=args.device,
        collision_dir=Path(args.collision_dir),
    )
    best = {
        "candidate_index": 0,
        "checkpoint": {
            "path": SOURCE.relative_to(ROOT).as_posix(),
            "sha256": SOURCE_SHA256,
            "model_tensor_sha256": SOURCE_MODEL_SHA256,
        },
        "evaluation": baseline_evaluation,
        "human_validation": baseline_human,
    }
    preflight = {
        "format": "RIVAL2_CODEX_AUTONOMOUS_V11_PREFLIGHT",
        "verdict": "PASS",
        "authority_sha256": base.sha256_file(AUTHORITY),
        "source_sha256": base.sha256_file(SOURCE),
        "source_model_tensor_sha256": base.tensor_tree_sha256(source["model"]),
        "human_test_not_loaded": human_identity["test_loaded"] is False,
        "directions": DIRECTIONS,
        "candidate_count": DIRECTIONS * 2,
        "perturbation_rms": PERTURBATION_RMS,
        "changed_parameter_boundary": ["actor.weight[0:5]", "actor.bias[0:5]"],
        "optimizer_steps": 0,
        "ppo_steps": 0,
    }
    base.write_json(RESULTS / "preflight.json", preflight)
    base.write_json(RESULTS / "baseline.json", best)
    if args.preflight_only:
        print(json.dumps(preflight, indent=2, sort_keys=True))
        return 0

    generator = torch.Generator(device="cpu").manual_seed(SEED)
    candidate_index = 0
    target_reached = False
    for direction_index in range(DIRECTIONS):
        weight_noise = torch.randn(
            source["model"]["actor.weight"][:5].shape,
            generator=generator,
            dtype=torch.float32,
        )
        bias_noise = torch.randn(
            source["model"]["actor.bias"][:5].shape,
            generator=generator,
            dtype=torch.float32,
        )
        for sign in (-1, 1):
            candidate_index += 1
            state = {name: value.clone() for name, value in source["model"].items()}
            state["actor.weight"][:5].add_(
                weight_noise, alpha=float(sign) * PERTURBATION_RMS
            )
            state["actor.bias"][:5].add_(
                bias_noise, alpha=float(sign) * PERTURBATION_RMS
            )
            candidate_path = run_dir / f"candidate_p{candidate_index:03d}.pt"
            checkpoint = save_candidate(
                source,
                state,
                candidate_path,
                candidate_index=candidate_index,
                direction_index=direction_index,
                sign=sign,
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
                "direction_index": direction_index,
                "sign": sign,
                "checkpoint": checkpoint,
                "evaluation": evaluation,
                "human_validation": human,
                "eligible": eligible,
            }
            base.append_jsonl(RESULTS / "candidates.jsonl", row)
            if eligible and base.candidate_rank(evaluation) > base.candidate_rank(
                best["evaluation"]
            ):
                promoted = save_candidate(
                    source,
                    state,
                    CHECKPOINT,
                    candidate_index=candidate_index,
                    direction_index=direction_index,
                    sign=sign,
                )
                best = copy.deepcopy(row)
                best["checkpoint"] = promoted
                base.write_json(RESULTS / "best.json", best)
            target_reached = bool(best["evaluation"]["win_rate"] >= TARGET_WIN_RATE)
            print(
                json.dumps(
                    {
                        "candidate": candidate_index,
                        "direction": direction_index,
                        "sign": sign,
                        "nexto": [evaluation["goals_for"], evaluation["goals_against"]],
                        "touches": evaluation["touches"],
                        "best_nexto": [
                            best["evaluation"]["goals_for"],
                            best["evaluation"]["goals_against"],
                        ],
                        "target_reached": target_reached,
                    },
                    sort_keys=True,
                ),
                flush=True,
            )
            if target_reached:
                break
        if target_reached:
            break
    if not (RESULTS / "best.json").exists():
        base.write_json(RESULTS / "best.json", best)
    result = {
        "format": "RIVAL2_CODEX_AUTONOMOUS_V11_RESULT",
        "authority_sha256": base.sha256_file(AUTHORITY),
        "candidates_evaluated": candidate_index,
        "maximum_candidates": DIRECTIONS * 2,
        "best": best,
        "target_nexto_win_rate": TARGET_WIN_RATE,
        "target_reached": target_reached,
        "optimizer_steps": 0,
        "ppo_steps": 0,
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
