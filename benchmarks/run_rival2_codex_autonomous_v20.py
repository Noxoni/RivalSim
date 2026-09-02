"""Robust tournament between independently improving V19 directions.

V19's strict full-match gate promoted a single-branch antithetic direction.
One separately generated four-branch-mean direction also improved the frozen
short window.  This prospective tournament tests blends of those two policy
states, selects two finalists on a longer paired window, and promotes only a
strict complete-match improvement over V19.
"""

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
from benchmarks import run_rival2_codex_autonomous_v16 as regulation
from rivalsim.rival2_policy import Rival2ActorCritic, Rival2PolicyConfig


SOURCE = ROOT / "checkpoints/rival2/codex_autonomous_v19/rival2_codex_autonomous_best.pt"
SOURCE_SHA256 = "0263546263285384D2D9A0CE55A471C41A41A8B7D4870DD9504D0ACCEA76723C"
SOURCE_MODEL_SHA256 = "C5B44186C1625A289F45F39459378EF353242CB842874F06E8E64A46E0B57EBA"
ALTERNATIVE = Path("G:/dev/RivalSim-runs/codex-autonomous-v19/candidate_14.pt")
ALTERNATIVE_SHA256 = "5C64506A548EA8C802B01F41EFA41D3656C58C022AB7A3F3CCEE9A195CF21396"
ALTERNATIVE_MODEL_SHA256 = "99E7BFC6D75D81612CE6AC17A986DE2C2EB39CB008AE5C918FE63EE450A76436"
AUTHORITY = ROOT / "results/rival2/codex_autonomous_v20/authority.json"
RESULTS = ROOT / "results/rival2/codex_autonomous_v20"
CHECKPOINT = ROOT / "checkpoints/rival2/codex_autonomous_v20/rival2_codex_autonomous_best.pt"
DEFAULT_RUN_DIR = Path("G:/dev/RivalSim-runs/codex-autonomous-v20")
BLEND_WEIGHTS = (0.25, 0.50, 0.75, 1.00)
WINDOW_SECONDS = 120
EVALUATION_SEED = 2_026_090_206
FULL_MATCH_FINALISTS = 2


def _save(
    source: dict[str, Any], state: dict[str, torch.Tensor], path: Path, *, index: int, blend: float
) -> dict[str, Any]:
    payload = copy.deepcopy(source)
    payload["model"] = {name: value.clone() for name, value in state.items()}
    payload["optimizer"] = {"state": {}, "param_groups": []}
    payload["policy_version"] = int(source["policy_version"]) + index
    payload["curriculum_transition"] = {
        "identity": "RIVAL2_CODEX_AUTONOMOUS_V20_ROBUST_DIRECTION_TOURNAMENT",
        "source": {"path": SOURCE.relative_to(ROOT).as_posix(), "sha256": SOURCE_SHA256},
        "alternative": {
            "path": ALTERNATIVE.as_posix(),
            "sha256": ALTERNATIVE_SHA256,
            "model_tensor_sha256": ALTERNATIVE_MODEL_SHA256,
        },
        "authority": {
            "path": AUTHORITY.relative_to(ROOT).as_posix(),
            "sha256": base.sha256_file(AUTHORITY),
        },
        "alternative_blend_weight": blend,
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


def _window(checkpoint: Path, digest: str, output: Path, collision_root: Path) -> dict[str, Any]:
    command = [
        str(ROOT / ".venv/Scripts/python.exe"),
        "-u",
        str(ROOT / "benchmarks/run_rival2_codex_autonomous_match_window_eval.py"),
        "--checkpoint",
        str(checkpoint),
        "--checkpoint-sha256",
        digest,
        "--output",
        str(output),
        "--window-seconds",
        str(WINDOW_SECONDS),
        "--seed",
        str(EVALUATION_SEED),
        "--collision-root",
        str(collision_root),
    ]
    with output.with_suffix(".stdout.txt").open("w", encoding="utf-8", newline="\n") as stdout, output.with_suffix(
        ".stderr.txt"
    ).open("w", encoding="utf-8", newline="\n") as stderr:
        completed = subprocess.run(command, cwd=ROOT, stdout=stdout, stderr=stderr, check=False)
    if completed.returncode != 0:
        raise RuntimeError(output.with_suffix(".stderr.txt").read_text(encoding="utf-8")[-4000:])
    return json.loads(output.read_text(encoding="utf-8"))


def _window_rank(evaluation: dict[str, Any]) -> tuple[int, int, int, int]:
    metrics = evaluation["overall"]
    return (
        int(metrics["goal_differential"]),
        int(metrics["goals_for"]),
        int(metrics["touch_differential"]),
        int(metrics["rival_touches"]),
    )


def run(args: argparse.Namespace) -> int:
    authority = json.loads(AUTHORITY.read_text(encoding="utf-8"))
    if authority.get("format") != "RIVAL2_CODEX_AUTONOMOUS_V20_AUTHORITY":
        raise RuntimeError("V20 authority format mismatch")
    if authority.get("candidate_space", {}).get("alternative_blend_weights") != list(BLEND_WEIGHTS):
        raise RuntimeError("V20 blend authority mismatch")
    if authority.get("selection", {}).get("full_match_finalists") != FULL_MATCH_FINALISTS:
        raise RuntimeError("V20 finalist authority mismatch")
    if base.sha256_file(SOURCE) != SOURCE_SHA256 or base.sha256_file(ALTERNATIVE) != ALTERNATIVE_SHA256:
        raise RuntimeError("V20 source identity changed")

    source = torch.load(SOURCE, map_location="cpu", weights_only=False)
    alternative = torch.load(ALTERNATIVE, map_location="cpu", weights_only=False)
    if base.tensor_tree_sha256(source["model"]) != SOURCE_MODEL_SHA256:
        raise RuntimeError("V20 source model changed")
    if base.tensor_tree_sha256(alternative["model"]) != ALTERNATIVE_MODEL_SHA256:
        raise RuntimeError("V20 alternative model changed")
    changed = []
    for name in source["model"]:
        if not torch.equal(source["model"][name], alternative["model"][name]):
            changed.append(name)
    if changed != ["actor.weight", "actor.bias"]:
        raise RuntimeError(f"V20 unexpected changed tensors: {changed}")
    if not torch.equal(source["model"]["actor.weight"][5:], alternative["model"]["actor.weight"][5:]):
        raise RuntimeError("V20 alternative changes button actor rows")
    if not torch.equal(source["model"]["actor.bias"][5:], alternative["model"]["actor.bias"][5:]):
        raise RuntimeError("V20 alternative changes button actor bias")

    base.SOURCE = SOURCE
    base.SOURCE_SHA256 = SOURCE_SHA256
    train, validation, teacher, human_identity = base.load_human_data(device=args.device)
    del train, teacher
    config = Rival2PolicyConfig(**source["policy_config"])
    model = Rival2ActorCritic(config).to(args.device)
    model.load_state_dict(source["model"])
    baseline_human = base.human_validation(model, validation, device=args.device)
    baseline_full = json.loads((ROOT / "results/rival2/codex_autonomous_v19/full_match_evaluation.json").read_text(encoding="utf-8"))
    preflight = {
        "format": "RIVAL2_CODEX_AUTONOMOUS_V20_PREFLIGHT",
        "verdict": "PASS",
        "authority_sha256": base.sha256_file(AUTHORITY),
        "source_sha256": SOURCE_SHA256,
        "source_model_tensor_sha256": SOURCE_MODEL_SHA256,
        "alternative_sha256": ALTERNATIVE_SHA256,
        "alternative_model_tensor_sha256": ALTERNATIVE_MODEL_SHA256,
        "source_full_match": regulation._full_match_summary(baseline_full),
        "source_human_validation": baseline_human,
        "human_test_not_loaded": human_identity["test_loaded"] is False,
        "changed_parameter_boundary": ["actor.weight[0:5]", "actor.bias[0:5]"],
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
        raise RuntimeError("V20 run directory must be fresh")
    run_dir.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, Any]] = []
    for index, blend in enumerate(BLEND_WEIGHTS, start=1):
        state = {name: value.clone() for name, value in source["model"].items()}
        for name in ("actor.weight", "actor.bias"):
            rows_slice = slice(0, 5)
            state[name][rows_slice].lerp_(alternative["model"][name][rows_slice], blend)
        path = run_dir / f"candidate_{index:02d}.pt"
        checkpoint = _save(source, state, path, index=index, blend=blend)
        model.load_state_dict(state)
        human = base.human_validation(model, validation, device=args.device)
        evaluation = _window(path, checkpoint["sha256"], run_dir / f"candidate_{index:02d}_window.json", Path(args.collision_root))
        row = {
            "candidate_index": index,
            "alternative_blend_weight": blend,
            "checkpoint": checkpoint,
            "human_validation": human,
            "evaluation": evaluation,
            "eligible": bool(human["eligible"]),
        }
        rows.append(row)
        base.append_jsonl(RESULTS / "candidates.jsonl", row)
        print(json.dumps({"candidate": index, "blend": blend, "goals": [evaluation["overall"]["goals_for"], evaluation["overall"]["goals_against"]]}, sort_keys=True), flush=True)

    finalists = sorted((row for row in rows if row["eligible"]), key=lambda row: _window_rank(row["evaluation"]), reverse=True)[:FULL_MATCH_FINALISTS]
    full_rows: list[dict[str, Any]] = []
    best_summary = regulation._full_match_summary(baseline_full)
    best_rank = regulation._full_match_rank(baseline_full)
    best_row: dict[str, Any] | None = None
    for place, row in enumerate(finalists, start=1):
        full = regulation._full_match_evaluate(Path(row["checkpoint"]["path"]), row["checkpoint"]["sha256"], run_dir / f"finalist_{place:02d}_full_match", Path(args.collision_root))
        summary = regulation._full_match_summary(full)
        full_row = {"candidate_index": row["candidate_index"], "alternative_blend_weight": row["alternative_blend_weight"], "checkpoint": row["checkpoint"], "summary": summary, "rank": list(regulation._full_match_rank(full))}
        full_rows.append(full_row)
        base.append_jsonl(RESULTS / "full_match_finalists.jsonl", full_row)
        if regulation._full_match_rank(full) > best_rank:
            best_rank = regulation._full_match_rank(full)
            best_summary = summary
            best_row = row
        print(json.dumps({"full_match_candidate": row["candidate_index"], "summary": summary}, sort_keys=True), flush=True)

    promoted = None
    if best_row is not None:
        CHECKPOINT.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(Path(best_row["checkpoint"]["path"]), CHECKPOINT)
        promoted = {
            "path": CHECKPOINT.relative_to(ROOT).as_posix(),
            "sha256": base.sha256_file(CHECKPOINT),
            "model_tensor_sha256": best_row["checkpoint"]["model_tensor_sha256"],
        }
    result = {
        "format": "RIVAL2_CODEX_AUTONOMOUS_V20_RESULT",
        "authority_sha256": base.sha256_file(AUTHORITY),
        "source_full_match": regulation._full_match_summary(baseline_full),
        "candidates": rows,
        "full_match_finalists": full_rows,
        "best_full_match": best_summary,
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
