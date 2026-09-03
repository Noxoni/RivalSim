"""Read-only calibration of the passing aerial option on natural setup families."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import sys
from pathlib import Path
from typing import Any

import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from benchmarks import run_rival2_ground_to_air_goal_v3 as goal_v3  # noqa: E402
from rivalsim.arena import ArenaGeometry, WarpArenaMeshes  # noqa: E402
from rivalsim.rival2_ground_to_air_natural_v4 import (  # noqa: E402
    SETUP_NAMES,
    build_natural_ground_to_air_scenarios,
)
from rivalsim.rival2_policy import Rival2ActorCritic, Rival2PolicyConfig  # noqa: E402

OPTION = ROOT / "checkpoints/rival2/ground_to_air_goal_v3/rival2_ground_to_air_goal_v3.pt"
OPTION_SHA256 = "F7049F8EF6CC4D1EE3F7303D6D9CE1AA2207A10F6651A33BC71B7C344CC77154"
AUTHORITY = ROOT / "results/rival2/ground_to_air_goal_v3/authority.json"
OUTPUT = ROOT / "results/rival2/ground_to_air_natural_v4/calibration.json"
COLLISION_DIR = Path(r"G:\dev\RLBot-Rival\bot\collision_meshes\soccar")


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest().upper()


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
        newline="\n",
    )


def calibration_authority() -> dict[str, Any]:
    authority = json.loads(AUTHORITY.read_text(encoding="utf-8"))
    authority = copy.deepcopy(authority)
    authority["bootstrap_config"].update(
        {
            "minimum_boost_fraction": 0.20,
            "minimum_ball_height_uu": 90.0,
            "maximum_ball_height_uu": 205.0,
            "minimum_planar_distance_uu": 5.0,
            "maximum_planar_distance_uu": 180.0,
            "minimum_forward_alignment": 0.15,
            "release_ball_height_uu": 90.0,
        }
    )
    return authority


def _make_model(device: str) -> Rival2ActorCritic:
    payload = torch.load(OPTION, map_location="cpu", weights_only=False)
    model = Rival2ActorCritic(Rival2PolicyConfig(**payload["policy_config"])).to(device)
    model.load_state_dict(payload["model"], strict=True)
    model.eval().requires_grad_(False)
    return model


def run(args: argparse.Namespace) -> int:
    if sha256_file(OPTION) != OPTION_SHA256:
        raise RuntimeError("controlled ground-to-air option changed")
    authority = calibration_authority()
    model = _make_model(args.device)
    geometry = ArenaGeometry.load_soccar(args.collision_dir)
    meshes = WarpArenaMeshes(geometry, args.device)
    distribution = goal_v3.distribution_override(authority)
    generators = [
        torch.Generator(device=args.device).manual_seed(int(args.seed) ^ side)
        for side in (0, 1)
    ]
    original_builder = goal_v3.build_goal_directed_pop_scenarios
    rows: list[dict[str, Any]] = []
    try:
        for setup, setup_name in enumerate(SETUP_NAMES):
            for side in (0, 1):

                def builder(
                    worlds: int,
                    *,
                    seed: int,
                    attacker_side: int,
                    phase: int,
                    selected_setup: int = setup,
                ):
                    del phase
                    return build_natural_ground_to_air_scenarios(
                        worlds,
                        seed=seed,
                        attacker_side=attacker_side,
                        setup=selected_setup,
                    ).state

                goal_v3.build_goal_directed_pop_scenarios = builder
                _unused, metrics = goal_v3.collect_rollout(
                    model,
                    geometry,
                    meshes,
                    authority=authority,
                    side=side,
                    worlds=int(args.worlds_per_side),
                    horizon=int(authority["episode"]["horizon_ticks"]),
                    seed=int(args.seed) + setup * 10_000,
                    device=args.device,
                    generator=generators[side],
                    distribution=distribution,
                    deterministic=True,
                    collision_dir=args.collision_dir,
                )
                metrics["setup"] = setup_name
                rows.append(metrics)
                print(
                    json.dumps(
                        {
                            "setup": setup_name,
                            "side": side,
                            "fractions": metrics["fractions"],
                        },
                        sort_keys=True,
                    ),
                    flush=True,
                )
    finally:
        goal_v3.build_goal_directed_pop_scenarios = original_builder

    result = {
        "format": "RIVAL2_GROUND_TO_AIR_NATURAL_V4_CALIBRATION",
        "purpose": "read-only transfer measurement on realistic low-bounce setup families",
        "option": {
            "path": OPTION.relative_to(ROOT).as_posix(),
            "sha256": OPTION_SHA256,
        },
        "source_authority": {
            "path": AUTHORITY.relative_to(ROOT).as_posix(),
            "sha256": sha256_file(AUTHORITY),
        },
        "calibration_option_config": authority["bootstrap_config"],
        "worlds_per_side_per_setup": int(args.worlds_per_side),
        "seed": int(args.seed),
        "optimizer_steps": 0,
        "policy_mutation": False,
        "reward_contract_mutation": False,
        "rows": rows,
    }
    write_json(args.output, result)
    print(json.dumps(result, indent=2, sort_keys=True), flush=True)
    return 0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--worlds-per-side", type=int, default=512)
    parser.add_argument("--seed", type=int, default=2_026_092_201)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--collision-dir", type=Path, default=COLLISION_DIR)
    parser.add_argument("--output", type=Path, default=OUTPUT)
    return parser.parse_args()


if __name__ == "__main__":
    raise SystemExit(run(parse_args()))
