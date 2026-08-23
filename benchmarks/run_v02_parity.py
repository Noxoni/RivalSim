"""Measure and gate RivalSim v0.2 against pinned RocketSim Soccar trajectories."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import platform
import sys
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import warp as wp

from rivalsim.arena import ArenaGeometry, WarpArenaMeshes
from rivalsim.math import quat_to_matrix
from rivalsim.reference.rocketsim_oracle import (
    ROCKETSIM_BINDING_COMMIT,
    ROCKETSIM_BINDING_VERSION,
    ROCKETSIM_PRIMARY_COMMIT,
    RocketSimStaticWorldOracle,
)
from rivalsim.static_world import StaticWorldSim
from rivalsim.v02_scenarios import V02_HORIZONS, make_v02_scenarios


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--collision-dir", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--mode", choices=("measurement", "gate"), default="measurement")
    parser.add_argument("--device", default="cuda:0")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    wp.init()
    geometry = ArenaGeometry.load_soccar(args.collision_dir)
    meshes = WarpArenaMeshes(geometry, args.device)
    scenarios = make_v02_scenarios()
    tolerances = _load_tolerances() if args.mode == "gate" else None
    records: list[dict[str, object]] = []
    metric_values: dict[str, list[float]] = {}
    hard_failures: list[dict[str, object]] = []

    for scenario in scenarios:
        sim = StaticWorldSim(
            1,
            args.collision_dir,
            variant="B3",
            device=args.device,
            initial=scenario.initial.copy(),
            geometry=geometry,
            meshes=meshes,
        )
        oracle = RocketSimStaticWorldOracle(scenario.initial.copy(), args.collision_dir)
        current_controls = None
        horizon_records: list[dict[str, object]] = []
        for tick in range(1, max(V02_HORIZONS) + 1):
            controls = scenario.controls_at(tick - 1)
            if controls is not current_controls:
                sim.set_controls(controls)
                oracle.set_controls(controls)
                current_controls = controls
            sim.step(1)
            oracle.step()
            if tick not in V02_HORIZONS:
                continue
            state = sim.snapshot()
            vehicle = sim.vehicle_snapshot()
            reference = oracle.frame()
            errors = _numeric_errors(state, vehicle, reference)
            mismatches = _hard_mismatches(state, vehicle, reference)
            for metric, value in errors.items():
                metric_values.setdefault(metric, []).append(value)
            if mismatches:
                hard_failures.append(
                    {
                        "scenario": scenario.name,
                        "family": scenario.family,
                        "horizon_ticks": tick,
                        "mismatches": mismatches,
                    }
                )
            numeric_failures: list[str] = []
            if tolerances is not None:
                numeric_failures = [
                    metric for metric, value in errors.items() if value > tolerances[metric]
                ]
            horizon_records.append(
                {
                    "horizon_ticks": tick,
                    "errors": errors,
                    "hard_mismatches": mismatches,
                    "numeric_failures": numeric_failures,
                    "pass": not mismatches and not numeric_failures,
                }
            )
        records.append(
            {
                "name": scenario.name,
                "family": scenario.family,
                "horizons": horizon_records,
                "pass": all(bool(item["pass"]) for item in horizon_records),
            }
        )

    aggregates = {
        metric: {
            "max": float(np.max(values)),
            "p95": float(np.percentile(values, 95)),
            "median": float(np.median(values)),
        }
        for metric, values in sorted(metric_values.items())
    }
    numeric_failures_total = sum(
        len(item["numeric_failures"]) for scenario in records for item in scenario["horizons"]
    )
    result = {
        "milestone": "v0.2",
        "mode": args.mode,
        "generated_at_utc": datetime.now(UTC).isoformat(),
        "implementation": {
            "python": platform.python_version(),
            "warp": wp.__version__,
            "device": str(wp.get_device(args.device)),
            "rocketsim_package": ROCKETSIM_BINDING_VERSION,
            "rocketsim_primary_commit": ROCKETSIM_PRIMARY_COMMIT,
            "rocketsim_binding_commit": ROCKETSIM_BINDING_COMMIT,
        },
        "arena": {
            "source_argument": args.collision_dir,
            "combined_content_sha256": geometry.content_sha256,
            "file_count": len(geometry.meshes),
            "vertices": geometry.vertex_count,
            "triangles": geometry.triangle_count,
        },
        "horizons_ticks": list(V02_HORIZONS),
        "scenario_count": len(scenarios),
        "families": sorted({scenario.family for scenario in scenarios}),
        "measurement_aggregates": aggregates,
        "frozen_tolerances": tolerances,
        "scenarios": records,
        "summary": {
            "hard_mismatch_count": len(hard_failures),
            "numeric_failure_count": numeric_failures_total,
            "hard_mismatch_examples": hard_failures[:25],
            "parity_gate_pass": args.mode == "gate"
            and not hard_failures
            and numeric_failures_total == 0,
        },
        "invocation_sha256": hashlib.sha256(" ".join(sys.argv).encode()).hexdigest().upper(),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result["summary"], indent=2))
    return 0


def _numeric_errors(state, vehicle, reference) -> dict[str, float]:
    sim_matrix = quat_to_matrix(state.car_quat[0, 0])
    relative = np.asarray(sim_matrix, dtype=np.float64).T @ np.asarray(
        reference.car_matrix, dtype=np.float64
    )
    cosine = np.clip((np.trace(relative) - 1.0) * 0.5, -1.0, 1.0)
    sim_world_normal = np.zeros(3, dtype=np.float32)
    if vehicle.contact_count[0] > 0:
        sim_world_normal = vehicle.contact_normal[0, 0]
    normal_angle = 0.0
    sim_normal_length = float(np.linalg.norm(sim_world_normal))
    reference_normal_length = float(np.linalg.norm(reference.world_contact_normal))
    if sim_normal_length > 1e-7 and reference_normal_length > 1e-7:
        normal_cosine = np.clip(
            float(np.dot(sim_world_normal, reference.world_contact_normal))
            / (sim_normal_length * reference_normal_length),
            -1.0,
            1.0,
        )
        normal_angle = math.acos(normal_cosine)
    return {
        "position_uu": float(np.linalg.norm(state.car_pos[0, 0] - reference.car_pos)),
        "linear_velocity_uu_per_s": float(np.linalg.norm(state.car_vel[0, 0] - reference.car_vel)),
        "orientation_rad": float(math.acos(float(cosine))),
        "angular_velocity_rad_per_s": float(
            np.linalg.norm(state.car_ang_vel[0, 0] - reference.car_ang_vel)
        ),
        "boost": abs(float(state.boost[0, 0]) - reference.boost),
        "handbrake_value": abs(float(vehicle.handbrake_value[0]) - reference.handbrake_value),
        "world_contact_normal_rad": normal_angle,
    }


def _hard_mismatches(state, vehicle, reference) -> list[str]:
    mismatches: list[str] = []
    if bool(state.on_ground[0, 0]) != reference.on_ground:
        mismatches.append("on_ground")
    for index, reference_contact in enumerate(reference.wheel_contacts):
        if bool(vehicle.wheel_contact[0, index]) != reference_contact:
            mismatches.append(f"wheel_contact_{index}")
    sim_world_contact = bool(vehicle.contact_count[0])
    if sim_world_contact != reference.has_world_contact:
        mismatches.append("world_contact")
    sim_velocity = state.car_vel[0, 0].astype(np.float64)
    reference_velocity = reference.car_vel.astype(np.float64)
    if (
        np.linalg.norm(sim_velocity) > 25.0
        and np.linalg.norm(reference_velocity) > 25.0
        and float(np.dot(sim_velocity, reference_velocity)) < 0.0
    ):
        mismatches.append("linear_velocity_direction")
    if sim_world_contact and reference.has_world_contact:
        sim_normal = vehicle.contact_normal[0, 0].astype(np.float64)
        reference_normal = reference.world_contact_normal.astype(np.float64)
        if (
            np.linalg.norm(sim_normal) > 1e-7
            and np.linalg.norm(reference_normal) > 1e-7
            and float(np.dot(sim_normal, reference_normal)) <= 0.0
        ):
            mismatches.append("world_contact_normal_direction")
    return mismatches


def _load_tolerances() -> dict[str, float]:
    from rivalsim.v02_tolerances import V02_PARITY_TOLERANCES

    return dict(V02_PARITY_TOLERANCES)


if __name__ == "__main__":
    raise SystemExit(main())
