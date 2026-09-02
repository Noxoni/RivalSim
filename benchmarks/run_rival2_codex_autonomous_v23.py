"""Validate and package a side-specialized Rival deployment bundle."""

from __future__ import annotations

import argparse
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import shutil
import sys
from typing import Any

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import benchmarks.run_rival2_nexto_matches as nexto_matches
from benchmarks import run_rival2_codex_autonomous_v1 as base
from benchmarks import run_rival2_codex_autonomous_v16 as regulation
from rivalsim.full_match import FullMatchRunner
from rivalsim.rival2_policy import Rival2ActorCritic, Rival2PolicyConfig, deterministic_hybrid_action


BLUE = ROOT / "checkpoints/rival2/codex_autonomous_v19/rival2_codex_autonomous_best.pt"
BLUE_SHA256 = "0263546263285384D2D9A0CE55A471C41A41A8B7D4870DD9504D0ACCEA76723C"
BLUE_MODEL_SHA256 = "C5B44186C1625A289F45F39459378EF353242CB842874F06E8E64A46E0B57EBA"
ORANGE = Path("G:/dev/RivalSim-runs/codex-autonomous-v22/candidate_08.pt")
ORANGE_SHA256 = "56E4ECA5075EB5748402BA3C5D8D51AC91FC1AFF55219E64EA5CE688DAD3491A"
ORANGE_MODEL_SHA256 = "2081095F83F3CCC964DCBA2B7F3BCA818D1B39DE1FE4B7CD97A95505D54FD514"
AUTHORITY = ROOT / "results/rival2/codex_autonomous_v23/authority.json"
RESULTS = ROOT / "results/rival2/codex_autonomous_v23"
BUNDLE = ROOT / "checkpoints/rival2/codex_autonomous_v23"


class SideSpecializedFullMatchRunner(FullMatchRunner):
    """Use one immutable checkpoint per physical team side."""

    orange_checkpoint = ORANGE

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        if self.stochastic_rival:
            raise RuntimeError("V23 requires deterministic deployment actions")
        path = Path(self.orange_checkpoint)
        if base.sha256_file(path) != ORANGE_SHA256:
            raise RuntimeError("V23 Orange checkpoint changed")
        payload = torch.load(path, map_location="cpu", weights_only=False)
        if base.tensor_tree_sha256(payload["model"]) != ORANGE_MODEL_SHA256:
            raise RuntimeError("V23 Orange model changed")
        config = Rival2PolicyConfig(**payload["policy_config"])
        if asdict(config) != self.checkpoint_identity["policy_config"]:
            raise RuntimeError("V23 side policy architecture mismatch")
        if int(payload.get("policy_hz", 0)) != self.rival_policy_hz:
            raise RuntimeError("V23 side policy cadence mismatch")
        if payload.get("contract_hashes") != self.checkpoint_identity["contract_hashes"]:
            raise RuntimeError("V23 side policy contract mismatch")
        self.orange_policy = Rival2ActorCritic(config).to(self.device)
        self.orange_policy.load_state_dict(payload["model"], strict=True)
        self.orange_policy.eval()
        self.checkpoint_identity = {
            "format": "RIVAL2_SIDE_SPECIALIZED_POLICY_BUNDLE_V1",
            "selector": "physical_team_side_before_match",
            "blue": self.checkpoint_identity,
            "orange": {
                "path": path.as_posix(),
                "sha256": ORANGE_SHA256,
                "model_tensor_sha256": ORANGE_MODEL_SHA256,
                "policy_config": asdict(config),
                "policy_config_hash": config.content_hash,
                "reward_version": payload.get("reward_version"),
                "contract_hashes": payload.get("contract_hashes"),
                "policy_hz": int(payload.get("policy_hz", 0)),
            },
        }

    def _update_rival_action(self) -> None:
        observation = self.rival_observation[self.batch_index, self.rival_side]
        with torch.inference_mode():
            blue_actor, _ = self.rival_policy(observation)
            orange_actor, _ = self.orange_policy(observation)
            orange_rows = (self.rival_side == 1).unsqueeze(1)
            actor = torch.where(orange_rows, orange_actor, blue_actor)
            self.rival_action.copy_(deterministic_hybrid_action(actor))


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest().upper()


def run(args: argparse.Namespace) -> int:
    authority = json.loads(AUTHORITY.read_text(encoding="utf-8"))
    if authority.get("format") != "RIVAL2_CODEX_AUTONOMOUS_V23_AUTHORITY":
        raise RuntimeError("V23 authority format mismatch")
    if _sha256(BLUE) != BLUE_SHA256 or _sha256(ORANGE) != ORANGE_SHA256:
        raise RuntimeError("V23 checkpoint identity changed")
    blue = torch.load(BLUE, map_location="cpu", weights_only=False)
    orange = torch.load(ORANGE, map_location="cpu", weights_only=False)
    if base.tensor_tree_sha256(blue["model"]) != BLUE_MODEL_SHA256 or base.tensor_tree_sha256(orange["model"]) != ORANGE_MODEL_SHA256:
        raise RuntimeError("V23 model identity changed")
    if blue["policy_config"] != orange["policy_config"] or blue["contract_hashes"] != orange["contract_hashes"]:
        raise RuntimeError("V23 policy contracts differ")
    v22_rows = [json.loads(line) for line in (ROOT / "results/rival2/codex_autonomous_v22/candidates.jsonl").read_text(encoding="utf-8").splitlines()]
    orange_row = next(row for row in v22_rows if row["candidate_index"] == 8)
    blue_result = json.loads((ROOT / "results/rival2/codex_autonomous_v19/result.json").read_text(encoding="utf-8"))
    blue_human = blue_result["best"]["human_validation"]
    orange_human = orange_row["human_validation"]
    preflight = {
        "format": "RIVAL2_CODEX_AUTONOMOUS_V23_PREFLIGHT",
        "verdict": "PASS",
        "authority_sha256": base.sha256_file(AUTHORITY),
        "blue_sha256": BLUE_SHA256,
        "blue_model_tensor_sha256": BLUE_MODEL_SHA256,
        "orange_sha256": ORANGE_SHA256,
        "orange_model_tensor_sha256": ORANGE_MODEL_SHA256,
        "contracts_identical": True,
        "policy_configs_identical": True,
        "blue_human_validation": blue_human,
        "orange_human_validation": orange_human,
        "both_human_eligible": bool(blue_human["eligible"] and orange_human["eligible"]),
        "human_test_not_loaded": True,
        "policy_mutation": False,
        "optimizer_steps": 0,
        "ppo_steps": 0,
    }
    RESULTS.mkdir(parents=True, exist_ok=True)
    base.write_json(RESULTS / "preflight.json", preflight)
    if args.preflight_only:
        print(json.dumps(preflight, indent=2, sort_keys=True))
        return 0

    nexto_matches.CHECKPOINT = BLUE
    nexto_matches.COLLISION_ROOT = Path(args.collision_root)
    nexto_matches.FullMatchRunner = SideSpecializedFullMatchRunner
    layout = np.repeat(np.arange(5, dtype=np.int32), 2)
    rival_side = np.tile(np.asarray([0, 1], dtype=np.int32), 5)
    canonical, _raw = nexto_matches._run_suite(name="canonical_side_specialized", layout=layout, rival_side=rival_side, stochastic_rival=False, seed=int(args.seed))
    evaluation = {
        "format": "RIVAL2_CODEX_AUTONOMOUS_V23_SIDE_SPECIALIZED_EVALUATION",
        "authority_sha256": base.sha256_file(AUTHORITY),
        "policies": {"blue": {"path": BLUE.as_posix(), "sha256": BLUE_SHA256}, "orange": {"path": ORANGE.as_posix(), "sha256": ORANGE_SHA256}},
        "selector": "physical_team_side_before_match",
        "opponent": "Nexto",
        "matrix": "five standard kickoff layouts x both Rival sides",
        "policy_mutation": False,
        "canonical": canonical,
    }
    base.write_json(RESULTS / "full_match_evaluation.json", evaluation)
    base.write_json(RESULTS / "full_match_ledger.json", canonical["canonical_match_ledger"])
    source_full = json.loads((ROOT / "results/rival2/codex_autonomous_v19/full_match_evaluation.json").read_text(encoding="utf-8"))
    candidate_summary = regulation._full_match_summary(evaluation)
    source_summary = regulation._full_match_summary(source_full)
    passed = regulation._full_match_rank(evaluation) > regulation._full_match_rank(source_full) and candidate_summary["wins"] > 5
    manifest = None
    if passed:
        BUNDLE.mkdir(parents=True, exist_ok=True)
        blue_out = BUNDLE / "rival2_blue.pt"
        orange_out = BUNDLE / "rival2_orange.pt"
        shutil.copy2(BLUE, blue_out)
        shutil.copy2(ORANGE, orange_out)
        manifest = {
            "format": "RIVAL2_SIDE_SPECIALIZED_POLICY_BUNDLE_V1",
            "authority": {"path": AUTHORITY.relative_to(ROOT).as_posix(), "sha256": base.sha256_file(AUTHORITY)},
            "selector": "physical_team_side_before_match",
            "blue": {"path": blue_out.name, "sha256": base.sha256_file(blue_out), "model_tensor_sha256": BLUE_MODEL_SHA256},
            "orange": {"path": orange_out.name, "sha256": base.sha256_file(orange_out), "model_tensor_sha256": ORANGE_MODEL_SHA256},
            "contracts": blue["contract_hashes"],
            "observation_version": blue["observation_version"],
            "action_version": blue["action_version"],
            "policy_hz": blue["policy_hz"],
            "full_match": candidate_summary,
        }
        base.write_json(BUNDLE / "bundle.json", manifest)
    result = {
        "format": "RIVAL2_CODEX_AUTONOMOUS_V23_RESULT",
        "verdict": "PASS" if passed else "FAIL",
        "authority_sha256": base.sha256_file(AUTHORITY),
        "source_full_match": source_summary,
        "bundle_full_match": candidate_summary,
        "bundle": manifest,
        "policy_mutation": False,
        "optimizer_steps": 0,
        "ppo_steps": 0,
    }
    base.write_json(RESULTS / "result.json", result)
    print(json.dumps(result, indent=2, sort_keys=True), flush=True)
    return 0 if passed else 2


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--collision-root", type=Path, default=Path(r"G:\dev\RLBot-Rival\bot\collision_meshes"))
    parser.add_argument("--seed", type=int, default=2_026_090_206)
    parser.add_argument("--preflight-only", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    raise SystemExit(run(parse_args()))
