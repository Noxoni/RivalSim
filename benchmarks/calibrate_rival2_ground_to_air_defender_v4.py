"""Read-only natural-entry calibration with parked and live V23 defenders."""

from __future__ import annotations

import argparse
import copy
import json
import sys
from pathlib import Path
from typing import Any

import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from benchmarks import run_rival2_ground_to_air_goal_v3 as goal_v3  # noqa: E402
from benchmarks import run_rival2_ground_to_air_natural_v4 as natural  # noqa: E402
from rivalsim.arena import ArenaGeometry, WarpArenaMeshes  # noqa: E402

OUTPUT = ROOT / "results/rival2/ground_to_air_natural_v4/defender_baseline.json"


def provisional_authority() -> dict[str, Any]:
    authority = json.loads(goal_v3.AUTHORITY.read_text(encoding="utf-8"))
    authority = copy.deepcopy(authority)
    authority["option_config"] = copy.deepcopy(authority.pop("bootstrap_config"))
    authority["option_config"].update(
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
    authority["scenario"] = {"validation_boost_range": [20.0, 80.0]}
    return authority


def run(args: argparse.Namespace) -> int:
    if natural.capability.sha256_file(natural.PARENT) != natural.PARENT_SHA256:
        raise RuntimeError("controlled aerial scorer parent changed")
    authority = provisional_authority()
    payload = torch.load(natural.PARENT, map_location="cpu", weights_only=False)
    model = natural.make_model(payload, args.device).eval().requires_grad_(False)
    defenders = natural.load_defender_policies(args.device)
    geometry = ArenaGeometry.load_soccar(args.collision_dir)
    meshes = WarpArenaMeshes(geometry, args.device)
    generators = [
        torch.Generator(device=args.device).manual_seed(int(args.seed) ^ side)
        for side in (0, 1)
    ]
    distribution = natural.distribution_override(authority)
    rows = natural.validation_rows(
        model,
        defenders,
        geometry,
        meshes,
        authority=authority,
        worlds=int(args.worlds_per_row),
        seed=int(args.seed),
        device=args.device,
        generators=generators,
        distribution=distribution,
        collision_dir=args.collision_dir,
    )
    result = {
        "format": "RIVAL2_GROUND_TO_AIR_NATURAL_V4_DEFENDER_BASELINE",
        "purpose": (
            "read-only prospective calibration of natural entries against "
            "a frozen live V23 defender"
        ),
        "option": {
            "path": natural.PARENT.relative_to(ROOT).as_posix(),
            "sha256": natural.PARENT_SHA256,
        },
        "defenders": {
            "blue": {
                "path": natural.BLUE.relative_to(ROOT).as_posix(),
                "sha256": natural.BLUE_SHA256,
            },
            "orange": {
                "path": natural.ORANGE.relative_to(ROOT).as_posix(),
                "sha256": natural.ORANGE_SHA256,
            },
        },
        "scenario_config": authority["scenario"],
        "option_config": authority["option_config"],
        "worlds_per_row": int(args.worlds_per_row),
        "seed": int(args.seed),
        "rows": rows,
        "optimizer_steps": 0,
        "policy_mutation": False,
        "reward_contract_mutation": False,
    }
    natural.write_json(args.output, result)
    for row in rows:
        print(
            json.dumps(
                {
                    "setup": row["setup"],
                    "defender": row["defender_mode"],
                    "side": row["side"],
                    "fractions": row["fractions"],
                },
                sort_keys=True,
            ),
            flush=True,
        )
    return 0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--worlds-per-row", type=int, default=256)
    parser.add_argument("--seed", type=int, default=2_026_092_401)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--collision-dir", type=Path, default=natural.DEFAULT_COLLISION_DIR)
    parser.add_argument("--output", type=Path, default=OUTPUT)
    return parser.parse_args()


if __name__ == "__main__":
    raise SystemExit(run(parse_args()))
