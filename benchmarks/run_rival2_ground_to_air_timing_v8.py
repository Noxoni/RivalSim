"""Run a no-learning matched-state takeoff-timing sweep."""

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

from benchmarks import run_rival2_capability_curriculum_v1 as capability  # noqa: E402
from benchmarks import run_rival2_ground_to_air_natural_v4 as natural_v4  # noqa: E402
from benchmarks import run_rival2_ground_to_air_natural_v6 as natural_v6  # noqa: E402
from benchmarks import run_rival2_ground_to_air_natural_v7 as natural_v7  # noqa: E402
from rivalsim.arena import ArenaGeometry, WarpArenaMeshes  # noqa: E402
from rivalsim.rival2_ground_to_air_timing_v8 import (  # noqa: E402
    GROUND_TO_AIR_TIMING_V8_VERSION,
    TIMING_CANDIDATES,
    TakeoffTimingCandidate,
    authority_with_timing,
)

VERSION = "RIVAL2_GROUND_TO_AIR_TIMING_V8"
AUTHORITY = ROOT / "results/rival2/ground_to_air_timing_v8/authority.json"
AUTHORITY_SHA256 = "D74DBA3DF55549FF3DDE8A9DA3C44714008BC8C2C3A0CEF7C3D8EB48ADC78592"
RESULTS = ROOT / "results/rival2/ground_to_air_timing_v8"
V7_AUTHORITY = ROOT / "results/rival2/ground_to_air_natural_v7/authority.json"
PARENT = natural_v7.PARENT
PARENT_SHA256 = natural_v7.PARENT_SHA256
BLUE = natural_v7.BLUE
ORANGE = natural_v7.ORANGE
BLUE_SHA256 = natural_v7.BLUE_SHA256
ORANGE_SHA256 = natural_v7.ORANGE_SHA256
DEFAULT_COLLISION_DIR = natural_v7.DEFAULT_COLLISION_DIR


def load_authority() -> dict[str, Any]:
    if capability.sha256_file(AUTHORITY) != AUTHORITY_SHA256:
        raise RuntimeError("timing V8 authority changed")
    authority = json.loads(AUTHORITY.read_text(encoding="utf-8"))
    if authority.get("format") != f"{VERSION}_AUTHORITY":
        raise RuntimeError("unexpected timing V8 authority format")
    for identity in authority["bound_inputs"].values():
        path = ROOT / identity["path"]
        if capability.sha256_file(path) != identity["sha256"]:
            raise RuntimeError(f"timing V8 bound input changed: {path}")
    if int(authority["integrity"]["optimizer_steps"]) != 0:
        raise RuntimeError("timing V8 must remain no-learning")
    frozen = authority["calibration"]["timing_candidates"]
    if frozen != [candidate.record() for candidate in TIMING_CANDIDATES]:
        raise RuntimeError("timing V8 candidate set changed")
    return authority


def timing_selection_key(
    rows: list[dict[str, Any]],
    base_authority: dict[str, Any],
) -> tuple[float, ...]:
    """Rank broad physical lift and reconnection before aggregate outcomes."""

    gate_key = natural_v6.selection_key(rows, base_authority)
    worst_elevated = min(
        float(row["fractions"]["elevated_follow_touch"]) for row in rows
    )
    worst_close = min(
        float(row["physical_probe"]["post_pop_within_160uu_fraction"])
        for row in rows
    )
    worst_takeoff = min(
        float(row["physical_probe"]["takeoff_after_pop_fraction"]) for row in rows
    )
    worst_median_height = min(
        float(row["physical_probe"]["maximum_self_height_uu"]["p50"])
        for row in rows
    )
    worst_median_vertical_speed = min(
        float(
            row["physical_probe"]["maximum_self_vertical_speed_uu_per_second"][
                "p50"
            ]
        )
        for row in rows
    )
    maximum_median_ball_distance = max(
        float(row["physical_probe"]["minimum_post_pop_ball_distance_uu"]["p50"])
        for row in rows
    )
    mean_elevated = sum(
        float(row["fractions"]["elevated_follow_touch"]) for row in rows
    ) / len(rows)
    return (
        float(gate_key[0]),
        worst_elevated,
        worst_close,
        worst_takeoff,
        worst_median_height,
        worst_median_vertical_speed,
        -maximum_median_ball_distance,
        mean_elevated,
        float(gate_key[1]),
    )


def summarize_candidate(
    candidate: TakeoffTimingCandidate,
    rows: list[dict[str, Any]],
    base_authority: dict[str, Any],
) -> dict[str, Any]:
    key = timing_selection_key(rows, base_authority)
    return {
        "candidate": candidate.record(),
        "selection_key": list(key),
        "existing_controlled_gate_passed": natural_v4.passes_gate(
            rows, base_authority
        ),
        "nonzero_elevated_rows": sum(
            float(row["fractions"]["elevated_follow_touch"]) > 0.0
            for row in rows
        ),
        "nonzero_high_rows": sum(
            float(row["fractions"]["high_follow_touch"]) > 0.0 for row in rows
        ),
        "mean_elevated_follow_fraction": sum(
            float(row["fractions"]["elevated_follow_touch"]) for row in rows
        )
        / len(rows),
        "minimum_takeoff_after_pop_fraction": min(
            float(row["physical_probe"]["takeoff_after_pop_fraction"])
            for row in rows
        ),
        "minimum_post_pop_within_160uu_fraction": min(
            float(row["physical_probe"]["post_pop_within_160uu_fraction"])
            for row in rows
        ),
        "minimum_row_median_self_height_uu": min(
            float(row["physical_probe"]["maximum_self_height_uu"]["p50"])
            for row in rows
        ),
        "minimum_row_median_vertical_speed_uu_per_second": min(
            float(
                row["physical_probe"]["maximum_self_vertical_speed_uu_per_second"][
                    "p50"
                ]
            )
            for row in rows
        ),
        "maximum_row_median_post_pop_ball_distance_uu": max(
            float(
                row["physical_probe"]["minimum_post_pop_ball_distance_uu"]["p50"]
            )
            for row in rows
        ),
        "rows": rows,
    }


def run(args: argparse.Namespace) -> int:
    authority = load_authority()
    if capability.sha256_file(PARENT) != PARENT_SHA256:
        raise RuntimeError("controlled scorer parent changed")
    if (
        capability.sha256_file(BLUE) != BLUE_SHA256
        or capability.sha256_file(ORANGE) != ORANGE_SHA256
    ):
        raise RuntimeError("protected V23 defender changed")
    base_authority = json.loads(V7_AUTHORITY.read_text(encoding="utf-8"))
    source = torch.load(PARENT, map_location="cpu", weights_only=False)
    model = natural_v4.make_model(source, args.device)
    defenders = natural_v4.load_defender_policies(args.device)
    geometry = ArenaGeometry.load_soccar(args.collision_dir)
    meshes = WarpArenaMeshes(geometry, args.device)
    generators = [
        torch.Generator(device=args.device).manual_seed(
            int(authority["calibration"]["seed"]) ^ side
        )
        for side in (0, 1)
    ]
    distribution = natural_v4.distribution_override(base_authority)
    candidates: list[dict[str, Any]] = []
    for candidate in TIMING_CANDIDATES:
        candidate_authority = authority_with_timing(base_authority, candidate)
        rows = natural_v7.validation_rows(
            model,
            defenders,
            geometry,
            meshes,
            authority=candidate_authority,
            worlds=args.worlds_per_row,
            seed=int(authority["calibration"]["seed"]),
            device=args.device,
            generators=generators,
            distribution=distribution,
            collision_dir=args.collision_dir,
            physical_probe=True,
        )
        summary = summarize_candidate(candidate, rows, base_authority)
        candidates.append(summary)
        print(
            json.dumps(
                {
                    "candidate": candidate.name,
                    "selection_key": summary["selection_key"],
                    "nonzero_elevated_rows": summary["nonzero_elevated_rows"],
                    "minimum_row_median_self_height_uu": summary[
                        "minimum_row_median_self_height_uu"
                    ],
                },
                sort_keys=True,
            ),
            flush=True,
        )
    selected = max(candidates, key=lambda item: tuple(item["selection_key"]))
    result = {
        "format": f"{VERSION}_RESULT",
        "timing_implementation": GROUND_TO_AIR_TIMING_V8_VERSION,
        "authority_sha256": AUTHORITY_SHA256,
        "optimizer_steps": 0,
        "matched_initial_states_across_candidates": True,
        "candidate_count": len(candidates),
        "candidates": candidates,
        "selected_candidate": selected["candidate"],
        "selected_key": selected["selection_key"],
        "selection_is_calibration_only": True,
        "untouched_test_opened": False,
        "production_reward_unchanged": True,
        "parent_unchanged": capability.sha256_file(PARENT) == PARENT_SHA256,
        "protected_v23_unchanged": (
            capability.sha256_file(BLUE) == BLUE_SHA256
            and capability.sha256_file(ORANGE) == ORANGE_SHA256
        ),
    }
    natural_v6.write_json(RESULTS / "result.json", result)
    print(json.dumps(result, indent=2, sort_keys=True), flush=True)
    return 0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--collision-dir", type=Path, default=DEFAULT_COLLISION_DIR)
    parser.add_argument("--worlds-per-row", type=int, default=256)
    return parser.parse_args()


if __name__ == "__main__":
    raise SystemExit(run(parse_args()))
