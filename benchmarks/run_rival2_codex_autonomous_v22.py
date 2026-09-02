"""Full-match antithetic search over fresh V21 competitive directions."""

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
from benchmarks import run_rival2_codex_autonomous_v16 as regulation
from rivalsim.rival2_policy import Rival2ActorCritic, Rival2PolicyConfig


SOURCE = ROOT / "checkpoints/rival2/codex_autonomous_v19/rival2_codex_autonomous_best.pt"
SOURCE_SHA256 = "0263546263285384D2D9A0CE55A471C41A41A8B7D4870DD9504D0ACCEA76723C"
SOURCE_MODEL_SHA256 = "C5B44186C1625A289F45F39459378EF353242CB842874F06E8E64A46E0B57EBA"
AUTHORITY = ROOT / "results/rival2/codex_autonomous_v22/authority.json"
RESULTS = ROOT / "results/rival2/codex_autonomous_v22"
CHECKPOINT = ROOT / "checkpoints/rival2/codex_autonomous_v22/rival2_codex_autonomous_best.pt"
DEFAULT_RUN_DIR = Path("G:/dev/RivalSim-runs/codex-autonomous-v22")
SINGLE_MAGNITUDES = (0.125, 0.25)
MEAN_MAGNITUDES = (0.125, 0.25, 0.50)
DIRECTIONS = (
    (2_026_090_261, Path("G:/dev/RivalSim-runs/codex-autonomous-v21/branches/2026090261/candidate_u0001.pt"), "7B5DCE9288C6C743750E89572C0A22D2EACD7D0E7FF33DFEBC03D2D8D8CAA238", "37BCA70310F27BAA2EC47A9E56786CDD511D0F773D37B7214C5FB3BBFB5F3049"),
    (2_026_090_263, Path("G:/dev/RivalSim-runs/codex-autonomous-v21/branches/2026090263/candidate_u0001.pt"), "4F05B0D40917F61ACAE97C08C9709A31D254C55B156AD4A091AFF86583A6BB91", "8CE451F71BA0A99803BBE546AAC853E25621A0C231CB2B50315DAF7F37849FBE"),
    (2_026_090_267, Path("G:/dev/RivalSim-runs/codex-autonomous-v21/branches/2026090267/candidate_u0001.pt"), "FF71B3DE5FF4F280510D009209B89F98F9DFEC60C261C9FD6AC3C0A2E1B88661", "BC281CAF1C9B2E28F31142A7396071FD507199A0131B1F5D54524C5B83062D31"),
    (2_026_090_269, Path("G:/dev/RivalSim-runs/codex-autonomous-v21/branches/2026090269/candidate_u0001.pt"), "E8EF55ADC69094924E3098AD315E81376792CD49A50A3CAE1C01868D91202598", "EA8EFE91F2A835C461A771518C1B37F7FC2ABFB4F43FB209247F5545FB794672"),
)


def _save(source: dict[str, Any], state: dict[str, torch.Tensor], path: Path, metadata: dict[str, Any]) -> dict[str, Any]:
    payload = copy.deepcopy(source)
    payload["model"] = {name: value.clone() for name, value in state.items()}
    payload["optimizer"] = {"state": {}, "param_groups": []}
    payload["policy_version"] = int(source["policy_version"]) + int(metadata["index"])
    payload["curriculum_transition"] = {
        "identity": "RIVAL2_CODEX_AUTONOMOUS_V22_FULL_MATCH_ANTITHETIC_SEARCH",
        "source": {"path": SOURCE.relative_to(ROOT).as_posix(), "sha256": SOURCE_SHA256},
        "authority": {"path": AUTHORITY.relative_to(ROOT).as_posix(), "sha256": base.sha256_file(AUTHORITY)},
        "candidate": metadata,
        "changed_parameters": ["actor.weight[0:5]", "actor.bias[0:5]"],
        "optimizer_steps": 0,
        "ppo_steps": 0,
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, path)
    return {"path": str(path), "sha256": base.sha256_file(path), "model_tensor_sha256": base.tensor_tree_sha256(payload["model"]), "bytes": path.stat().st_size}


def run(args: argparse.Namespace) -> int:
    authority = json.loads(AUTHORITY.read_text(encoding="utf-8"))
    if authority.get("format") != "RIVAL2_CODEX_AUTONOMOUS_V22_AUTHORITY":
        raise RuntimeError("V22 authority format mismatch")
    space = authority.get("candidate_space", {})
    if space.get("single_direction_antithetic_magnitudes") != list(SINGLE_MAGNITUDES) or space.get("mean_direction_antithetic_magnitudes") != list(MEAN_MAGNITUDES):
        raise RuntimeError("V22 candidate authority mismatch")
    if base.sha256_file(SOURCE) != SOURCE_SHA256:
        raise RuntimeError("V22 source changed")
    source = torch.load(SOURCE, map_location="cpu", weights_only=False)
    if base.tensor_tree_sha256(source["model"]) != SOURCE_MODEL_SHA256:
        raise RuntimeError("V22 source model changed")

    deltas: list[tuple[int, torch.Tensor, torch.Tensor]] = []
    identities: list[dict[str, Any]] = []
    for seed, path, digest, model_digest in DIRECTIONS:
        if base.sha256_file(path) != digest:
            raise RuntimeError(f"V22 direction checkpoint {seed} changed")
        payload = torch.load(path, map_location="cpu", weights_only=False)
        if base.tensor_tree_sha256(payload["model"]) != model_digest:
            raise RuntimeError(f"V22 direction model {seed} changed")
        weight = payload["model"]["actor.weight"][:5] - source["model"]["actor.weight"][:5]
        bias = payload["model"]["actor.bias"][:5] - source["model"]["actor.bias"][:5]
        if not bool(torch.isfinite(weight).all() and torch.isfinite(bias).all()):
            raise RuntimeError(f"V22 direction {seed} is nonfinite")
        deltas.append((seed, weight, bias))
        identities.append({"seed": seed, "path": str(path), "sha256": digest, "model_tensor_sha256": model_digest, "weight_rms": float(weight.square().mean().sqrt()), "bias_rms": float(bias.square().mean().sqrt())})
    mean_weight = torch.stack([item[1] for item in deltas]).mean(dim=0)
    mean_bias = torch.stack([item[2] for item in deltas]).mean(dim=0)

    base.SOURCE = SOURCE
    base.SOURCE_SHA256 = SOURCE_SHA256
    train, validation, teacher, human_identity = base.load_human_data(device=args.device)
    del train, teacher
    config = Rival2PolicyConfig(**source["policy_config"])
    model = Rival2ActorCritic(config).to(args.device)
    model.load_state_dict(source["model"])
    baseline_human = base.human_validation(model, validation, device=args.device)
    baseline_full = json.loads((ROOT / "results/rival2/codex_autonomous_v19/full_match_evaluation.json").read_text(encoding="utf-8"))
    baseline_rank = regulation._full_match_rank(baseline_full)
    preflight = {
        "format": "RIVAL2_CODEX_AUTONOMOUS_V22_PREFLIGHT",
        "verdict": "PASS",
        "authority_sha256": base.sha256_file(AUTHORITY),
        "source_sha256": SOURCE_SHA256,
        "source_model_tensor_sha256": SOURCE_MODEL_SHA256,
        "directions": identities,
        "mean_weight_rms": float(mean_weight.square().mean().sqrt()),
        "mean_bias_rms": float(mean_bias.square().mean().sqrt()),
        "source_full_match": regulation._full_match_summary(baseline_full),
        "source_human_validation": baseline_human,
        "human_test_not_loaded": human_identity["test_loaded"] is False,
        "candidate_count": len(DIRECTIONS) * len(SINGLE_MAGNITUDES) + len(MEAN_MAGNITUDES),
        "optimizer_steps": 0,
        "ppo_steps": 0,
    }
    RESULTS.mkdir(parents=True, exist_ok=True)
    base.write_json(RESULTS / "preflight.json", preflight)
    if args.preflight_only:
        print(json.dumps(preflight, indent=2, sort_keys=True))
        return 0

    run_dir = Path(args.run_dir)
    if run_dir.exists() and any(run_dir.iterdir()):
        raise RuntimeError("V22 run directory must be fresh")
    run_dir.mkdir(parents=True, exist_ok=True)
    specs: list[tuple[int, str, torch.Tensor, torch.Tensor, float]] = []
    for seed, weight, bias in deltas:
        for magnitude in SINGLE_MAGNITUDES:
            specs.append((seed, "single_branch", weight, bias, magnitude))
    for magnitude in MEAN_MAGNITUDES:
        specs.append((0, "four_branch_mean", mean_weight, mean_bias, magnitude))

    best_rank = baseline_rank
    best_row: dict[str, Any] | None = None
    rows: list[dict[str, Any]] = []
    for index, (seed, role, weight, bias, magnitude) in enumerate(specs, start=1):
        state = {name: value.clone() for name, value in source["model"].items()}
        state["actor.weight"][:5].add_(weight, alpha=-magnitude)
        state["actor.bias"][:5].add_(bias, alpha=-magnitude)
        metadata = {"index": index, "direction_seed": seed, "direction_role": role, "antithetic_magnitude": magnitude}
        path = run_dir / f"candidate_{index:02d}.pt"
        checkpoint = _save(source, state, path, metadata)
        model.load_state_dict(state)
        human = base.human_validation(model, validation, device=args.device)
        full = regulation._full_match_evaluate(path, checkpoint["sha256"], run_dir / f"candidate_{index:02d}_full_match", Path(args.collision_root))
        rank = regulation._full_match_rank(full)
        row = {"candidate_index": index, "metadata": metadata, "checkpoint": checkpoint, "human_validation": human, "full_match": regulation._full_match_summary(full), "rank": list(rank), "eligible": bool(human["eligible"])}
        rows.append(row)
        base.append_jsonl(RESULTS / "candidates.jsonl", row)
        if row["eligible"] and rank > best_rank:
            best_rank = rank
            best_row = copy.deepcopy(row)
            base.write_json(RESULTS / "best.json", best_row)
        print(json.dumps({"candidate": index, "direction": seed, "role": role, "magnitude": magnitude, "full_match": row["full_match"], "best_rank": list(best_rank)}, sort_keys=True), flush=True)

    promoted = None
    if best_row is not None:
        CHECKPOINT.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(Path(best_row["checkpoint"]["path"]), CHECKPOINT)
        promoted = {"path": CHECKPOINT.relative_to(ROOT).as_posix(), "sha256": base.sha256_file(CHECKPOINT), "model_tensor_sha256": best_row["checkpoint"]["model_tensor_sha256"]}
    if not (RESULTS / "best.json").exists():
        base.write_json(RESULTS / "best.json", {"source_retained": True, "full_match": regulation._full_match_summary(baseline_full), "rank": list(baseline_rank)})
    result = {
        "format": "RIVAL2_CODEX_AUTONOMOUS_V22_RESULT",
        "authority_sha256": base.sha256_file(AUTHORITY),
        "source_full_match": regulation._full_match_summary(baseline_full),
        "candidates_evaluated": len(rows),
        "best": best_row,
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
