"""Antithetic search over deployment-aligned PPO directions rejected by V18."""

from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path
import shutil
import sys
from typing import Any

import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from benchmarks import run_rival2_codex_autonomous_v1 as base
from benchmarks import run_rival2_codex_autonomous_v13 as search
from benchmarks import run_rival2_codex_autonomous_v16 as regulation
from rivalsim.rival2_policy import Rival2ActorCritic, Rival2PolicyConfig


SOURCE = search.SOURCE
SOURCE_SHA256 = search.SOURCE_SHA256
SOURCE_MODEL_SHA256 = search.SOURCE_MODEL_SHA256
AUTHORITY = ROOT / "results/rival2/codex_autonomous_v19/authority.json"
RESULTS = ROOT / "results/rival2/codex_autonomous_v19"
CHECKPOINT = ROOT / "checkpoints/rival2/codex_autonomous_v19/rival2_codex_autonomous_best.pt"
DEFAULT_RUN_DIR = Path("G:/dev/RivalSim-runs/codex-autonomous-v19")
MAGNITUDES = (0.25, 0.50, 1.00)
DIRECTIONS = (
    (
        2_026_090_251,
        Path("G:/dev/RivalSim-runs/codex-autonomous-v18/branches/2026090251/candidate_u0001.pt"),
        "1A28D4BCE91E66466C688310214F8858AB3E9A8B1750FFF7DCFB13207A4905F5",
        "DB0B9DFFF054DF42495FFAEB48544952CD1BA3CDEFEB2FAA25745F7BC8027F4F",
    ),
    (
        2_026_090_253,
        Path("G:/dev/RivalSim-runs/codex-autonomous-v18/branches/2026090253/candidate_u0001.pt"),
        "B16DE99B447B40070456363EA5A0658F1D1ADE954AA9E00EFB57E1F16A207C6B",
        "EF49F5E4544716D11DFA7C41AA32A123D6408343F31FE55D6E8135BF1BBA78B5",
    ),
    (
        2_026_090_257,
        Path("G:/dev/RivalSim-runs/codex-autonomous-v18/branches/2026090257/candidate_u0001.pt"),
        "F1343E9009A76DF1E9BEB8298F62F6A3C07A129D36407CBF50E6C081DBF36EAE",
        "C132BDD15431CB28DF410572B19C2F18BB6F70DFA0948DD98B9168799655E735",
    ),
    (
        2_026_090_259,
        Path("G:/dev/RivalSim-runs/codex-autonomous-v18/branches/2026090259/candidate_u0001.pt"),
        "6921988159CFDC073ED2ECEDE2618712851B95A3B81DB462FB9263E8CFEC7F73",
        "3A0E14604AB79055F138AAD55400BBD1B6AB2338BE553F91095580B27A942E20",
    ),
)


def _save(
    source: dict[str, Any], state: dict[str, torch.Tensor], path: Path, metadata: dict[str, Any]
) -> dict[str, Any]:
    payload = copy.deepcopy(source)
    payload["model"] = {name: value.clone() for name, value in state.items()}
    payload["optimizer"] = {"state": {}, "param_groups": []}
    payload["policy_version"] = int(source["policy_version"]) + int(metadata["index"])
    payload["curriculum_transition"] = {
        "identity": "RIVAL2_CODEX_AUTONOMOUS_V19_ANTITHETIC_PPO_DIRECTION_SEARCH",
        "source": {"path": SOURCE.relative_to(ROOT).as_posix(), "sha256": SOURCE_SHA256},
        "authority": {
            "path": AUTHORITY.relative_to(ROOT).as_posix(),
            "sha256": base.sha256_file(AUTHORITY),
        },
        "candidate": metadata,
        "changed_parameters": ["actor.weight[0:5]", "actor.bias[0:5]"],
        "optimizer_steps": 0,
        "ppo_steps": 0,
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, path)
    return {
        "path": str(path),
        "sha256": base.sha256_file(path),
        "model_tensor_sha256": base.tensor_tree_sha256(payload["model"]),
        "bytes": path.stat().st_size,
    }


def run(args: argparse.Namespace) -> int:
    authority = json.loads(AUTHORITY.read_text(encoding="utf-8"))
    if authority.get("format") != "RIVAL2_CODEX_AUTONOMOUS_V19_AUTHORITY":
        raise RuntimeError("V19 authority format mismatch")
    if authority.get("magnitudes") != list(MAGNITUDES):
        raise RuntimeError("V19 magnitude authority mismatch")
    if base.sha256_file(SOURCE) != SOURCE_SHA256:
        raise RuntimeError("V19 source changed")
    source = torch.load(SOURCE, map_location="cpu", weights_only=False)
    if base.tensor_tree_sha256(source["model"]) != SOURCE_MODEL_SHA256:
        raise RuntimeError("V19 source model changed")

    deltas: list[tuple[int, torch.Tensor, torch.Tensor]] = []
    direction_identity: list[dict[str, Any]] = []
    for seed, path, digest, model_digest in DIRECTIONS:
        if base.sha256_file(path) != digest:
            raise RuntimeError(f"V19 direction {seed} checkpoint changed")
        payload = torch.load(path, map_location="cpu", weights_only=False)
        if base.tensor_tree_sha256(payload["model"]) != model_digest:
            raise RuntimeError(f"V19 direction {seed} model changed")
        weight = payload["model"]["actor.weight"][:5] - source["model"]["actor.weight"][:5]
        bias = payload["model"]["actor.bias"][:5] - source["model"]["actor.bias"][:5]
        deltas.append((seed, weight, bias))
        direction_identity.append(
            {
                "seed": seed,
                "path": str(path),
                "sha256": digest,
                "model_tensor_sha256": model_digest,
                "weight_rms": float(weight.square().mean().sqrt().item()),
                "bias_rms": float(bias.square().mean().sqrt().item()),
            }
        )
    mean_weight = torch.stack([row[1] for row in deltas]).mean(dim=0)
    mean_bias = torch.stack([row[2] for row in deltas]).mean(dim=0)
    deltas.append((0, mean_weight, mean_bias))

    base.SOURCE = SOURCE
    base.SOURCE_SHA256 = SOURCE_SHA256
    train, validation, teacher, human_identity = base.load_human_data(device=args.device)
    del train, teacher
    baseline = json.loads(search.BASELINE.read_text(encoding="utf-8"))
    config = Rival2PolicyConfig(**source["policy_config"])
    model = Rival2ActorCritic(config).to(args.device)
    model.load_state_dict(source["model"])
    baseline_human = base.human_validation(model, validation, device=args.device)
    best: dict[str, Any] = {
        "candidate_index": 0,
        "checkpoint": {
            "path": SOURCE.relative_to(ROOT).as_posix(),
            "sha256": SOURCE_SHA256,
            "model_tensor_sha256": SOURCE_MODEL_SHA256,
        },
        "evaluation": baseline,
        "human_validation": baseline_human,
    }
    preflight = {
        "format": "RIVAL2_CODEX_AUTONOMOUS_V19_PREFLIGHT",
        "verdict": "PASS",
        "authority_sha256": base.sha256_file(AUTHORITY),
        "source_sha256": SOURCE_SHA256,
        "source_model_tensor_sha256": SOURCE_MODEL_SHA256,
        "directions": direction_identity,
        "mean_weight_rms": float(mean_weight.square().mean().sqrt().item()),
        "mean_bias_rms": float(mean_bias.square().mean().sqrt().item()),
        "human_test_not_loaded": human_identity["test_loaded"] is False,
        "optimizer_steps": 0,
        "ppo_steps": 0,
    }
    RESULTS.mkdir(parents=True, exist_ok=True)
    base.write_json(RESULTS / "preflight.json", preflight)
    base.write_json(RESULTS / "baseline.json", best)
    if args.preflight_only:
        print(json.dumps(preflight, indent=2, sort_keys=True))
        return 0

    run_dir = Path(args.run_dir)
    if run_dir.exists() and any(run_dir.iterdir()):
        raise RuntimeError("V19 run directory must be fresh")
    run_dir.mkdir(parents=True, exist_ok=True)
    index = 0
    for seed, weight, bias in deltas:
        for magnitude in MAGNITUDES:
            index += 1
            state = {name: value.clone() for name, value in source["model"].items()}
            state["actor.weight"][:5].add_(weight, alpha=-magnitude)
            state["actor.bias"][:5].add_(bias, alpha=-magnitude)
            path = run_dir / f"candidate_{index:02d}.pt"
            metadata = {
                "index": index,
                "direction_seed": seed,
                "direction_role": "mean" if seed == 0 else "single_branch",
                "antithetic_magnitude": magnitude,
            }
            checkpoint = _save(source, state, path, metadata)
            model.load_state_dict(state)
            human = base.human_validation(model, validation, device=args.device)
            evaluation = search._evaluate(
                path,
                checkpoint_sha256=checkpoint["sha256"],
                output=run_dir / f"candidate_{index:02d}_window.json",
                collision_root=Path(args.collision_root),
            )
            row = {
                "candidate_index": index,
                "metadata": metadata,
                "checkpoint": checkpoint,
                "human_validation": human,
                "evaluation": evaluation,
                "eligible": bool(human["eligible"]),
            }
            base.append_jsonl(RESULTS / "candidates.jsonl", row)
            if row["eligible"] and search._rank(evaluation) > search._rank(best["evaluation"]):
                best = copy.deepcopy(row)
                base.write_json(RESULTS / "best.json", best)
            print(
                json.dumps(
                    {
                        "candidate": index,
                        "direction": seed,
                        "magnitude": magnitude,
                        "goals": [evaluation["overall"]["goals_for"], evaluation["overall"]["goals_against"]],
                        "best": best["candidate_index"],
                    },
                    sort_keys=True,
                ),
                flush=True,
            )
    if not (RESULTS / "best.json").exists():
        base.write_json(RESULTS / "best.json", best)

    baseline_full = json.loads(regulation.BASELINE_FULL_MATCH.read_text(encoding="utf-8"))
    full = {
        "required": best["candidate_index"] != 0,
        "baseline": regulation._full_match_summary(baseline_full),
        "candidate": None,
        "passed": False,
    }
    promoted = None
    if full["required"]:
        selected = Path(best["checkpoint"]["path"])
        full_evaluation = regulation._full_match_evaluate(
            selected,
            best["checkpoint"]["sha256"],
            run_dir / "selected_full_match",
            Path(args.collision_root),
        )
        full["candidate"] = regulation._full_match_summary(full_evaluation)
        full["passed"] = regulation._full_match_rank(full_evaluation) > regulation._full_match_rank(baseline_full)
        if full["passed"]:
            CHECKPOINT.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(selected, CHECKPOINT)
            promoted = {
                "path": CHECKPOINT.relative_to(ROOT).as_posix(),
                "sha256": base.sha256_file(CHECKPOINT),
                "model_tensor_sha256": best["checkpoint"]["model_tensor_sha256"],
            }
    result = {
        "format": "RIVAL2_CODEX_AUTONOMOUS_V19_RESULT",
        "authority_sha256": base.sha256_file(AUTHORITY),
        "candidates": index,
        "best": best,
        "full_match": full,
        "promoted_checkpoint": promoted,
        "optimizer_steps": 0,
        "ppo_steps": 0,
    }
    base.write_json(RESULTS / "result.json", result)
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--collision-root", type=Path, default=Path(r"G:\dev\RLBot-Rival\bot\collision_meshes"))
    parser.add_argument("--run-dir", type=Path, default=DEFAULT_RUN_DIR)
    parser.add_argument("--preflight-only", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    raise SystemExit(run(parse_args()))
