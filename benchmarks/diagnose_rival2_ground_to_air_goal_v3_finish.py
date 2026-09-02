"""Compare parent and rejected V3 descendant finish geometry without training."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from benchmarks import run_rival2_capability_curriculum_v1 as capability  # noqa: E402
from benchmarks import run_rival2_ground_to_air_goal_v3 as goal_v3  # noqa: E402
from rivalsim.arena import ArenaGeometry, WarpArenaMeshes  # noqa: E402
from rivalsim.rival2_ground_to_air_goal_v3 import PHASE_ATTACKING_HALF  # noqa: E402


def evaluate(
    checkpoint: Path,
    *,
    authority: dict,
    geometry: ArenaGeometry,
    meshes: WarpArenaMeshes,
    device: str,
    worlds_per_side: int,
    collision_dir: Path,
) -> dict:
    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    model = goal_v3.make_model(payload, device)
    distribution = goal_v3.distribution_override(authority)
    rows = []
    for side in (0, 1):
        generator = torch.Generator(device=device).manual_seed(
            int(authority["seeds"]["optimizer_and_exploration"]) ^ side
        )
        _, metrics = goal_v3.collect_rollout(
            model,
            geometry,
            meshes,
            authority=authority,
            side=side,
            worlds=worlds_per_side,
            horizon=int(authority["episode"]["horizon_ticks"]),
            seed=int(authority["seeds"]["validation"]),
            device=device,
            generator=generator,
            distribution=distribution,
            deterministic=True,
            collision_dir=collision_dir,
            phase=PHASE_ATTACKING_HALF,
        )
        rows.append(metrics)
    return {
        "checkpoint": checkpoint.as_posix(),
        "sha256": capability.sha256_file(checkpoint),
        "worlds_per_side": worlds_per_side,
        "sides": rows,
    }


def run(args: argparse.Namespace) -> int:
    authority = goal_v3.load_authority()
    collision_dir = Path(args.collision_dir)
    geometry = ArenaGeometry.load_soccar(collision_dir)
    meshes = WarpArenaMeshes(geometry, args.device)
    payload = {
        "format": "RIVAL2_GROUND_TO_AIR_GOAL_V3_FINISH_DIAGNOSTIC",
        "authority_sha256": goal_v3.AUTHORITY_SHA256,
        "training_or_optimizer_steps": 0,
        "parent": evaluate(
            Path(args.parent),
            authority=authority,
            geometry=geometry,
            meshes=meshes,
            device=args.device,
            worlds_per_side=args.worlds_per_side,
            collision_dir=collision_dir,
        ),
        "rejected_descendant": evaluate(
            Path(args.descendant),
            authority=authority,
            geometry=geometry,
            meshes=meshes,
            device=args.device,
            worlds_per_side=args.worlds_per_side,
            collision_dir=collision_dir,
        ),
    }
    goal_v3.write_json(Path(args.output), payload)
    print(json.dumps(payload, indent=2, sort_keys=True), flush=True)
    return 0


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("--device", default="cuda:0")
    result.add_argument("--worlds-per-side", type=int, default=4_096)
    result.add_argument(
        "--parent", default=str(goal_v3.PARENT)
    )
    result.add_argument(
        "--descendant",
        default="G:/dev/RivalSim-runs/ground-to-air-goal-v3/rolling.pt",
    )
    result.add_argument(
        "--collision-dir", default=str(goal_v3.DEFAULT_COLLISION_DIR)
    )
    result.add_argument(
        "--output",
        default=str(goal_v3.RESULTS / "finish_diagnostic.json"),
    )
    return result


if __name__ == "__main__":
    raise SystemExit(run(parser().parse_args()))
