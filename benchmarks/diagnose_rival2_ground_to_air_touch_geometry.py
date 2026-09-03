"""Measure native follow-touch geometry without changing policy or reward."""

from __future__ import annotations

import argparse
import json
import sys
from contextlib import contextmanager
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from benchmarks import run_rival2_ground_to_air_goal_v3 as goal_runner  # noqa: E402
from benchmarks import run_rival2_ground_to_air_natural_v7 as balanced  # noqa: E402
from benchmarks import run_rival2_ground_to_air_natural_v9 as v9  # noqa: E402
from rivalsim.arena import ArenaGeometry, WarpArenaMeshes  # noqa: E402
from rivalsim.rival2_ground_to_air_touch_geometry import (  # noqa: E402
    GROUND_TO_AIR_TOUCH_GEOMETRY_VERSION,
    NaturalAerialTouchGeometryProbe,
)

DEFAULT_OUTPUT = (
    ROOT / "results/rival2/ground_to_air_natural_v9/touch_geometry_probe.json"
)


class ProbedTrainingTracker(goal_runner.GoalDirectedTrainingTracker):
    """Add read-only geometry to the existing unchanged training tracker."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self.geometry_probe = NaturalAerialTouchGeometryProbe(
            self.worlds, attacker_side=self.side
        )

    def step(self, before: torch.Tensor, after: torch.Tensor, **kwargs: Any) -> Any:
        result = super().step(before, after, **kwargs)
        self.geometry_probe.step(
            before,
            after,
            tick=int(kwargs["tick"]),
            active=kwargs["active"],
        )
        return result

    def telemetry(self) -> dict[str, Any]:
        result = super().telemetry()
        result["touch_geometry_probe"] = self.geometry_probe.telemetry()
        return result


@contextmanager
def _install_probe() -> Any:
    original = goal_runner.GoalDirectedTrainingTracker
    try:
        goal_runner.GoalDirectedTrainingTracker = ProbedTrainingTracker
        yield
    finally:
        goal_runner.GoalDirectedTrainingTracker = original


def _aggregate(rows: list[dict[str, Any]]) -> dict[str, Any]:
    categories = (
        "first_distinct_follow",
        "first_airborne_follow",
        "first_prompt_airborne_follow",
        "first_strict_elevated_follow",
    )
    return {
        name: {
            "mean_attempt_fraction": sum(
                float(
                    row["telemetry"]["touch_geometry_probe"]["categories"][name][
                        "attempt_fraction"
                    ]
                )
                for row in rows
            )
            / len(rows),
            "nonzero_rows": sum(
                float(
                    row["telemetry"]["touch_geometry_probe"]["categories"][name][
                        "attempt_fraction"
                    ]
                )
                > 0.0
                for row in rows
            ),
        }
        for name in categories
    }


def run(args: argparse.Namespace) -> int:
    authority = v9.load_authority()
    checkpoint = Path(args.checkpoint)
    checkpoint_sha256 = balanced.capability.sha256_file(checkpoint)
    source = torch.load(checkpoint, map_location="cpu", weights_only=False)
    model = balanced.natural_v4.make_model(source, args.device)
    defenders = balanced.natural_v4.load_defender_policies(args.device)
    geometry = ArenaGeometry.load_soccar(args.collision_dir)
    meshes = WarpArenaMeshes(geometry, args.device)
    generators = [
        torch.Generator(device=args.device).manual_seed(
            int(authority["seeds"]["optimizer_and_exploration"]) ^ side
        )
        for side in (0, 1)
    ]
    distribution = balanced.natural_v4.distribution_override(authority)
    with _install_probe():
        rows = balanced.validation_rows(
            model,
            defenders,
            geometry,
            meshes,
            authority=authority,
            worlds=args.worlds_per_row,
            seed=args.seed,
            device=args.device,
            generators=generators,
            distribution=distribution,
            collision_dir=args.collision_dir,
            physical_probe=True,
        )
    payload = {
        "format": "RIVAL2_GROUND_TO_AIR_TOUCH_GEOMETRY_DIAGNOSTIC_V1",
        "created_utc": datetime.now(UTC).replace(microsecond=0).isoformat(),
        "probe_identity": GROUND_TO_AIR_TOUCH_GEOMETRY_VERSION,
        "authority": {
            "path": v9.AUTHORITY.relative_to(ROOT).as_posix(),
            "sha256": v9.AUTHORITY_SHA256,
        },
        "checkpoint": {
            "path": checkpoint.as_posix(),
            "sha256": checkpoint_sha256,
        },
        "deterministic": True,
        "worlds_per_row": args.worlds_per_row,
        "seed": args.seed,
        "policy_and_reward_unchanged": True,
        "aggregate": _aggregate(rows),
        "rows": rows,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(payload["aggregate"], indent=2, sort_keys=True))
    return 0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, default=balanced.PARENT)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument(
        "--collision-dir", type=Path, default=balanced.DEFAULT_COLLISION_DIR
    )
    parser.add_argument("--worlds-per-row", type=int, default=256)
    parser.add_argument("--seed", type=int, default=2026110912)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args()


if __name__ == "__main__":
    raise SystemExit(run(parse_args()))
