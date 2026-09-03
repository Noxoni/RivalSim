"""Read-only boost sensitivity for the passing natural ground-to-air option."""

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

from benchmarks import calibrate_rival2_ground_to_air_natural_v4 as natural  # noqa: E402
from benchmarks import run_rival2_ground_to_air_goal_v3 as goal_v3  # noqa: E402
from rivalsim.arena import ArenaGeometry, WarpArenaMeshes  # noqa: E402
from rivalsim.rival2_ground_to_air_natural_v4 import (  # noqa: E402
    SETUP_NAMES,
    build_natural_ground_to_air_scenarios,
)

OUTPUT = ROOT / "results/rival2/ground_to_air_natural_v4/boost_sensitivity.json"


def run(args: argparse.Namespace) -> int:
    if natural.sha256_file(natural.OPTION) != natural.OPTION_SHA256:
        raise RuntimeError("controlled ground-to-air option changed")
    authority = natural.calibration_authority()
    authority = copy.deepcopy(authority)
    # The diagnostic must measure low-boost behavior rather than refusing to
    # activate below the deployment gate.  This changes only the read-only
    # controller latch used by this process; it does not alter the policy.
    authority["bootstrap_config"]["minimum_boost_fraction"] = 0.0
    model = natural._make_model(args.device)
    geometry = ArenaGeometry.load_soccar(args.collision_dir)
    meshes = WarpArenaMeshes(geometry, args.device)
    distribution = goal_v3.distribution_override(authority)
    generators = [
        torch.Generator(device=args.device).manual_seed(int(args.seed) ^ side)
        for side in (0, 1)
    ]
    rows: list[dict[str, Any]] = []
    original_builder = goal_v3.build_goal_directed_pop_scenarios
    try:
        for setup, setup_name in enumerate(SETUP_NAMES):
            for boost in args.boost:
                for side in (0, 1):

                    def builder(
                        worlds: int,
                        *,
                        seed: int,
                        attacker_side: int,
                        phase: int,
                        selected_setup: int = setup,
                        selected_boost: float = boost,
                    ):
                        del phase
                        state = build_natural_ground_to_air_scenarios(
                            worlds,
                            seed=seed,
                            attacker_side=attacker_side,
                            setup=selected_setup,
                        ).state
                        state.boost[:, attacker_side] = selected_boost
                        return state

                    goal_v3.build_goal_directed_pop_scenarios = builder
                    _unused, metrics = goal_v3.collect_rollout(
                        model,
                        geometry,
                        meshes,
                        authority=authority,
                        side=side,
                        worlds=int(args.worlds_per_side),
                        horizon=int(authority["episode"]["horizon_ticks"]),
                        # Keep the physical states identical at every boost
                        # value so this is a paired sensitivity measurement.
                        seed=int(args.seed) + setup * 100_000,
                        device=args.device,
                        generator=generators[side],
                        distribution=distribution,
                        deterministic=True,
                        collision_dir=args.collision_dir,
                    )
                    metrics["setup"] = setup_name
                    metrics["initial_boost"] = float(boost)
                    rows.append(metrics)
                    print(
                        json.dumps(
                            {
                                "setup": setup_name,
                                "boost": boost,
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
        "format": "RIVAL2_GROUND_TO_AIR_NATURAL_V4_BOOST_SENSITIVITY",
        "purpose": "read-only deterministic option performance by fixed initial boost",
        "option": {
            "path": natural.OPTION.relative_to(ROOT).as_posix(),
            "sha256": natural.OPTION_SHA256,
        },
        "source_authority": {
            "path": natural.AUTHORITY.relative_to(ROOT).as_posix(),
            "sha256": natural.sha256_file(natural.AUTHORITY),
        },
        "diagnostic_controller_config": authority["bootstrap_config"],
        "boost_values": [float(value) for value in args.boost],
        "worlds_per_side_per_setup_and_boost": int(args.worlds_per_side),
        "seed": int(args.seed),
        "optimizer_steps": 0,
        "policy_mutation": False,
        "reward_contract_mutation": False,
        "rows": rows,
    }
    natural.write_json(args.output, result)
    return 0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--worlds-per-side", type=int, default=128)
    parser.add_argument("--seed", type=int, default=2_026_092_301)
    parser.add_argument("--boost", type=float, nargs="+", default=(0, 5, 10, 20, 35, 50))
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--collision-dir", type=Path, default=natural.COLLISION_DIR)
    parser.add_argument("--output", type=Path, default=OUTPUT)
    return parser.parse_args()


if __name__ == "__main__":
    raise SystemExit(run(parse_args()))
