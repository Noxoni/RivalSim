"""Deterministic no-learning calibration for natural aerial continuation."""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Iterator
from contextlib import contextmanager
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from benchmarks import run_rival2_capability_curriculum_v1 as capability  # noqa: E402
from benchmarks import run_rival2_ground_to_air_goal_v3 as goal_runner  # noqa: E402
from benchmarks import run_rival2_ground_to_air_natural_v4 as natural_v4  # noqa: E402
from benchmarks import run_rival2_ground_to_air_natural_v7 as natural_v7  # noqa: E402
from benchmarks import run_rival2_ground_to_air_natural_v10 as natural_v10  # noqa: E402
from rivalsim.arena import ArenaGeometry, WarpArenaMeshes  # noqa: E402
from rivalsim.rival2_ground_to_air_natural_v4 import (  # noqa: E402
    DEFENDER_LIVE,
    DEFENDER_PARKED,
    SETUP_NAMES,
)
from rivalsim.rival2_ground_to_air_prompt_continuation_probe import (  # noqa: E402
    GROUND_TO_AIR_PROMPT_CONTINUATION_PROBE_VERSION,
    PromptContinuationDiagnosticTracker,
)
from rivalsim.rival2_policy import HybridDistributionOverride  # noqa: E402

VERSION = "RIVAL2_GROUND_TO_AIR_PROMPT_CONTINUATION_CALIBRATION_V1"
DEFAULT_OUTPUT = (
    ROOT
    / "results/rival2/ground_to_air_natural_v11/prompt_continuation_parent.json"
)


def utc_now() -> str:
    return datetime.now(UTC).replace(microsecond=0).isoformat().replace(
        "+00:00", "Z"
    )


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


@contextmanager
def diagnostic_tracker() -> Iterator[None]:
    original = goal_runner.GoalDirectedTrainingTracker
    try:
        goal_runner.GoalDirectedTrainingTracker = (
            PromptContinuationDiagnosticTracker
        )
        yield
    finally:
        goal_runner.GoalDirectedTrainingTracker = original


def collect_rows(
    model: Any,
    defenders: dict[int, Any],
    geometry: ArenaGeometry,
    meshes: WarpArenaMeshes,
    *,
    authority: dict[str, Any],
    worlds: int,
    device: str,
    collision_dir: Path,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    generators = [
        torch.Generator(device=device).manual_seed(
            int(authority["seeds"]["optimizer_and_exploration"]) ^ side
        )
        for side in (0, 1)
    ]
    exploration = authority["training"]["exploration"]
    distribution = HybridDistributionOverride(
        analog_log_std=float(
            torch.log(torch.tensor(exploration["analog_sigma"]))
        ),
        button_temperature=float(exploration["button_temperature"]),
    )
    horizon = int(authority["episode"]["horizon_ticks"])
    seed = int(authority["seeds"]["validation"])
    with diagnostic_tracker():
        for setup, setup_name in enumerate(SETUP_NAMES):
            for defender_mode in (DEFENDER_PARKED, DEFENDER_LIVE):
                for side in (0, 1):
                    _rollout, metrics = natural_v7.collect_rollout(
                        model,
                        defenders,
                        geometry,
                        meshes,
                        authority=authority,
                        side=side,
                        worlds=worlds,
                        horizon=horizon,
                        seed=(
                            seed
                            + setup * 100_000
                            + (
                                10_000
                                if defender_mode == DEFENDER_LIVE
                                else 0
                            )
                        ),
                        device=device,
                        generator=generators[side],
                        distribution=distribution,
                        deterministic=True,
                        collision_dir=collision_dir,
                        setup=setup,
                        defender_mode=defender_mode,
                        attacker_boost_range=tuple(
                            authority["scenario"]["validation_boost_range"]
                        ),
                        physical_probe=True,
                    )
                    if metrics["setup"] != setup_name:
                        raise RuntimeError("setup identity changed during probe")
                    rows.append(metrics)
    return rows


def summarize(rows: list[dict[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for setup in SETUP_NAMES:
        grouped = [row for row in rows if row["setup"] == setup]
        probes = [
            row["telemetry"]["prompt_continuation_probe"] for row in grouped
        ]
        result[setup] = {
            "rows": len(grouped),
            "prompt_fraction_minimum": min(
                float(probe["prompt_airborne_follow_fraction"])
                for probe in probes
            ),
            "prompt_fraction_mean": sum(
                float(probe["prompt_airborne_follow_fraction"])
                for probe in probes
            )
            / len(probes),
            "second_recontact_fraction_maximum": max(
                float(probe["second_recontact_fraction"])
                for probe in probes
            ),
            "bridge_elevated_fraction_maximum": max(
                float(probe["bridge_elevated_fraction"])
                for probe in probes
            ),
            "bridge_high_fraction_maximum": max(
                float(probe["bridge_high_fraction"])
                for probe in probes
            ),
        }
    return result


def run(args: argparse.Namespace) -> int:
    authority = natural_v10.load_authority()
    checkpoint = args.checkpoint.resolve()
    checkpoint_hash_before = capability.sha256_file(checkpoint)
    source = torch.load(checkpoint, map_location="cpu", weights_only=False)
    model = natural_v4.make_model(source, args.device)
    defenders = natural_v4.load_defender_policies(args.device)
    geometry = ArenaGeometry.load_soccar(args.collision_dir)
    meshes = WarpArenaMeshes(geometry, args.device)
    with torch.inference_mode():
        rows = collect_rows(
            model,
            defenders,
            geometry,
            meshes,
            authority=authority,
            worlds=args.worlds_per_row,
            device=args.device,
            collision_dir=args.collision_dir,
        )
    checkpoint_hash_after = capability.sha256_file(checkpoint)
    if checkpoint_hash_after != checkpoint_hash_before:
        raise RuntimeError("calibration mutated its checkpoint source")
    payload = {
        "format": VERSION,
        "created_utc": utc_now(),
        "probe_identity": GROUND_TO_AIR_PROMPT_CONTINUATION_PROBE_VERSION,
        "v10_authority_sha256": natural_v10.AUTHORITY_SHA256,
        "checkpoint": {
            "path": checkpoint.as_posix(),
            "sha256": checkpoint_hash_before,
            "unchanged": True,
        },
        "worlds_per_row": args.worlds_per_row,
        "rows": rows,
        "summary": summarize(rows),
        "optimizer_steps": 0,
        "state_mutation": False,
        "action_mutation": False,
        "reward_authority_created": False,
        "verdict": "PASS",
    }
    write_json(args.output, payload)
    print(json.dumps(payload["summary"], indent=2, sort_keys=True), flush=True)
    return 0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=natural_v7.PARENT,
    )
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument(
        "--collision-dir",
        type=Path,
        default=natural_v7.DEFAULT_COLLISION_DIR,
    )
    parser.add_argument("--worlds-per-row", type=int, default=256)
    return parser.parse_args()


if __name__ == "__main__":
    raise SystemExit(run(parse_args()))
