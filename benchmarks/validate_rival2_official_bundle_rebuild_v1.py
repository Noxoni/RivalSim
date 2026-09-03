"""Prove a committed-source official bundle rebuild is behavior-identical."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from benchmarks.run_rival2_codex_autonomous_v1 import (  # noqa: E402
    sha256_file,
    tensor_tree_sha256,
)
from rivalsim.rival2_contracts import OBS_DIM  # noqa: E402
from rivalsim.rival2_official_bundle_v1 import (  # noqa: E402
    OFFICIAL_BUNDLE_V1_FORMAT,
    Rival2OfficialControllerV1,
)


def _load(path: Path) -> dict[str, Any]:
    payload = torch.load(path, map_location="cpu", weights_only=False)
    if payload.get("format") != OFFICIAL_BUNDLE_V1_FORMAT:
        raise RuntimeError(f"unsupported official bundle at {path}")
    return payload


def run(args: argparse.Namespace) -> int:
    reference_path = Path(args.reference).resolve()
    candidate_path = Path(args.candidate).resolve()
    reference = _load(reference_path)
    candidate = _load(candidate_path)
    if reference["router_config"] != candidate["router_config"]:
        raise RuntimeError("official router configuration changed across rebuild")
    if reference["contract_hashes"] != candidate["contract_hashes"]:
        raise RuntimeError("official contract hashes changed across rebuild")
    if set(reference["components"]) != set(candidate["components"]):
        raise RuntimeError("official component set changed across rebuild")

    components: dict[str, Any] = {}
    for name in sorted(reference["components"]):
        left = reference["components"][name]
        right = candidate["components"][name]
        left_tensor_hash = tensor_tree_sha256(left["model"])
        right_tensor_hash = tensor_tree_sha256(right["model"])
        exact = all(
            torch.equal(left["model"][key], right["model"][key])
            for key in left["model"]
        )
        if not exact or left_tensor_hash != right_tensor_hash:
            raise RuntimeError(f"official component changed across rebuild: {name}")
        components[name] = {
            "source_sha256": right["source_sha256"],
            "model_tensor_sha256": right_tensor_hash,
            "byte_exact": True,
        }

    lanes = int(args.lanes)
    reference_controller = Rival2OfficialControllerV1(reference, lanes, device="cpu")
    candidate_controller = Rival2OfficialControllerV1(candidate, lanes, device="cpu")
    generator = torch.Generator(device="cpu")
    generator.manual_seed(int(args.seed))
    compared_actions = 0
    for _tick in range(int(args.ticks)):
        observation = torch.randn(
            (lanes, OBS_DIM), generator=generator, dtype=torch.float32
        )
        side = torch.randint(0, 2, (lanes,), generator=generator)
        kickoff = torch.rand(lanes, generator=generator) < 0.01
        done = torch.rand(lanes, generator=generator) < 0.005
        reference_action, reference_selection = reference_controller.action(
            observation, side, kickoff_active=kickoff, match_done=done
        )
        candidate_action, candidate_selection = candidate_controller.action(
            observation, side, kickoff_active=kickoff, match_done=done
        )
        if not torch.equal(reference_action, candidate_action):
            raise RuntimeError("official deterministic action changed across rebuild")
        if not torch.equal(reference_selection.mode, candidate_selection.mode):
            raise RuntimeError("official route selection changed across rebuild")
        compared_actions += lanes

    report = {
        "format": "RIVAL2_OFFICIAL_BUNDLE_REBUILD_PARITY_V1",
        "verdict": "PASS",
        "reference": {
            "path": reference_path.as_posix(),
            "sha256": sha256_file(reference_path),
            "rivalsim_commit": reference["rivalsim_commit"],
        },
        "candidate": {
            "path": candidate_path.relative_to(ROOT).as_posix(),
            "sha256": sha256_file(candidate_path),
            "rivalsim_commit": candidate["rivalsim_commit"],
        },
        "router_config_exact": True,
        "contracts_exact": True,
        "components": components,
        "deterministic_action_parity": {
            "seed": int(args.seed),
            "lanes": lanes,
            "ticks": int(args.ticks),
            "actions_compared": compared_actions,
            "actions_byte_exact": True,
            "route_modes_byte_exact": True,
        },
        "physical_result_carried_forward": {
            "reference_report": (
                "results/rival2/official_v1/"
                "candidate_03_fail_closed_physical_validation.json"
            ),
            "basis": (
                "all component tensors, router configuration, contracts, "
                "deterministic actions, and route selections are exact"
            ),
        },
    }
    output = Path(args.output).resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument(
        "--candidate",
        type=Path,
        default=ROOT / "checkpoints/rival2/official_v1/rival2_official_v1.pt",
    )
    parser.add_argument("--lanes", type=int, default=64)
    parser.add_argument("--ticks", type=int, default=64)
    parser.add_argument("--seed", type=int, default=2_026_090_301)
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "results/rival2/official_v1/rebuild_parity.json",
    )
    return parser.parse_args()


if __name__ == "__main__":
    raise SystemExit(run(parse_args()))
