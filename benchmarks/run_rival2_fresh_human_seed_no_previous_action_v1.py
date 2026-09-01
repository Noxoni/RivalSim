"""Fresh Human Seed Stage-1 with a permanently neutral previous-action input.

This is a new fresh-random gameplay-imitation lineage.  It deliberately does not
accept any checkpoint argument, does not run PPO, and does not use mechanic-practice
data.  The underlying reviewed trajectory, temporal split, frozen Observation Adapter
V2, supervised objective, and validation-only selection logic are inherited unchanged
from the clean Fresh Human Seed Stage-1 implementation.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from dataclasses import asdict
from pathlib import Path
from typing import Any

import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from benchmarks import run_rival2_fresh_human_seed_v1 as stage1  # noqa: E402
from rivalsim.human_demo.missing_feature_distillation import file_sha256  # noqa: E402
from rivalsim.rival2_policy import (  # noqa: E402
    PREVIOUS_ACTION_OBSERVATION_FIELDS,
    PREVIOUS_ACTION_OBSERVATION_INDICES,
    Rival2PolicyConfig,
)

AUTHORIZATION_PARENT = "38BDD501BECC0F6BA0867EABB205F25D05E0F31C"
FORMAT = "RIVAL2_FRESH_HUMAN_SEED_NO_PREVIOUS_ACTION_V1"
CHECKPOINT_FORMAT = f"{FORMAT}_STAGE1_CHECKPOINT"
RESULTS = ROOT / "results/rival2/fresh_human_seed_no_previous_action_v1"
AUTHORITY = RESULTS / "authority.json"
SPLIT_MANIFEST = RESULTS / "source_split_manifest.json"
CURVE = RESULTS / "stage1_curve.jsonl"
CHECKPOINT = (
    ROOT
    / "checkpoints/rival2/fresh_human_seed_no_previous_action_v1"
    / "rival2_fresh_human_seed_no_previous_action_v1.pt"
)
INITIALIZATION_SEED = 2026090106
TRAINING_SEED = 2026090107


def _configure_base() -> dict[str, Any]:
    overrides = {
        "PACKAGE_COMMIT": AUTHORIZATION_PARENT,
        "FORMAT": FORMAT,
        "CHECKPOINT_FORMAT": CHECKPOINT_FORMAT,
        "RESULTS": RESULTS,
        "AUTHORITY": AUTHORITY,
        "SPLIT_MANIFEST": SPLIT_MANIFEST,
        "CURVE": CURVE,
        "CHECKPOINT": CHECKPOINT,
        "INITIALIZATION_SEED": INITIALIZATION_SEED,
        "TRAINING_SEED": TRAINING_SEED,
        "AUTHORITY_PREPARATION_REQUIRES_EXACT_PACKAGE_COMMIT": False,
        "NEUTRALIZE_PREVIOUS_ACTION": True,
        "ZERO_PREVIOUS_ACTION_POLICY_INPUTS": True,
    }
    previous = {name: getattr(stage1, name) for name in overrides}
    for name, value in overrides.items():
        setattr(stage1, name, value)
    return previous


def _restore_base(previous: dict[str, Any]) -> None:
    for name, value in previous.items():
        setattr(stage1, name, value)


def prepare(args: argparse.Namespace) -> dict[str, Any]:
    previous = _configure_base()
    try:
        authority = stage1.prepare(args)
        authority.pop("stage2", None)
        authority["task_scope"] = {
            "stage1_fresh_gameplay_imitation_only": True,
            "ppo_authorized": False,
            "closed_loop_evaluation_after_selection": True,
        }
        authority["stage1"]["previous_action_input_contract"] = {
            "version": "RIVAL2_PREVIOUS_ACTION_ALWAYS_ZERO_V1",
            "fields_structurally_present": True,
            "field_names": list(PREVIOUS_ACTION_OBSERVATION_FIELDS),
            "indices": list(PREVIOUS_ACTION_OBSERVATION_INDICES),
            "human_before_observation_adapter_v2": {
                "value": 0.0,
                "quality": "unavailable",
            },
            "human_after_adapter_and_pad_overlay": 0.0,
            "rivalsim_immediately_before_policy_trunk": 0.0,
            "policy_config": {"zero_previous_action_inputs": True},
        }
        authority["forbidden"].update(
            {
                "ppo_or_reward_optimization": True,
                "existing_bc_or_ppo_checkpoint_load": True,
                "human_previous_action_as_policy_input": True,
            }
        )
        authority["policy_config"] = {
            "values": asdict(Rival2PolicyConfig(zero_previous_action_inputs=True)),
            "content_hash": Rival2PolicyConfig(
                zero_previous_action_inputs=True
            ).content_hash,
        }
        stage1.write_json(AUTHORITY, authority)
        return authority
    finally:
        _restore_base(previous)


def train(args: argparse.Namespace) -> dict[str, Any]:
    previous = _configure_base()
    try:
        selected = stage1.train(args)
        payload = torch.load(CHECKPOINT, map_location="cpu", weights_only=False)
        config = Rival2PolicyConfig(**payload["policy_config"])
        if payload.get("format") != CHECKPOINT_FORMAT:
            raise RuntimeError("selected checkpoint format mismatch")
        if not config.zero_previous_action_inputs:
            raise RuntimeError("selected policy does not enforce the previous-action mask")
        authority = json.loads(AUTHORITY.read_text(encoding="utf-8"))
        contract = authority["stage1"]["previous_action_input_contract"]
        if contract["indices"] != list(PREVIOUS_ACTION_OBSERVATION_INDICES):
            raise RuntimeError("frozen previous-action indices changed")
        selected["previous_action_input_contract"] = {
            "version": contract["version"],
            "policy_config_hash": config.content_hash,
            "eight_human_inputs_zero_before_adapter": True,
            "eight_human_inputs_unavailable_before_adapter": True,
            "eight_human_outputs_zero_after_adapter_overlay": True,
            "eight_rivalsim_inputs_zero_immediately_before_policy": True,
        }
        selected["checkpoint"]["sha256"] = file_sha256(CHECKPOINT)
        stage1.write_json(RESULTS / "stage1_selected.json", selected)
        return selected
    finally:
        _restore_base(previous)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--prepare", action="store_true")
    parser.add_argument("--train", action="store_true")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument(
        "--human-source-root",
        type=Path,
        default=(
            Path(os.environ["APPDATA"])
            / "bakkesmod/bakkesmod/data/rival2/human_demos"
        ),
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.prepare == args.train:
        raise SystemExit("choose exactly one of --prepare or --train")
    result = prepare(args) if args.prepare else train(args)
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
