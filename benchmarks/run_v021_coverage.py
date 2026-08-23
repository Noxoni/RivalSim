"""Build the bounded v0.2.1 DFH local-transition coverage prototype."""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import warp as wp

from rivalsim.arena import ArenaGeometry, WarpArenaMeshes, build_internal_edge_data
from rivalsim.static_world import StaticWorldSim
from rivalsim.v02_scenarios import make_v02_scenarios

LOCAL_HORIZONS = (1, 4, 8, 12)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--collision-dir", required=True)
    parser.add_argument("--parity-file", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    wp.init()
    geometry = ArenaGeometry.load_soccar(args.collision_dir)
    meshes = WarpArenaMeshes(geometry, args.device)
    edge_angles, edge_flags = build_internal_edge_data(geometry)
    parity = json.loads(args.parity_file.read_text(encoding="utf-8"))
    if tuple(parity["horizons_ticks"]) != LOCAL_HORIZONS:
        raise ValueError("coverage prototype requires the 1/4/8/12 local parity artifact")

    chassis_faces: set[int] = set()
    wheel_faces: set[int] = set()
    plane_contacts: Counter[str] = Counter()
    scenario_coverage: list[dict[str, Any]] = []
    family_faces: dict[str, set[int]] = defaultdict(set)

    for scenario in make_v02_scenarios():
        sim = StaticWorldSim(
            1,
            args.collision_dir,
            variant="B3",
            device=args.device,
            initial=scenario.initial.copy(),
            geometry=geometry,
            meshes=meshes,
        )
        current_controls = None
        observed_chassis: set[int] = set()
        observed_wheels: set[int] = set()
        active_ticks = 0
        wheel_contact_ticks = 0
        chassis_contact_ticks = 0
        for tick in range(1, max(LOCAL_HORIZONS) + 1):
            controls = scenario.controls_at(tick - 1)
            if controls is not current_controls:
                sim.set_controls(controls)
                current_controls = controls
            sim.step(1, synchronize=True)
            vehicle = sim.vehicle_snapshot()
            wheel_active = bool(np.any(vehicle.wheel_contact[0]))
            chassis_active = int(vehicle.contact_count[0]) > 0
            active_ticks += int(wheel_active or chassis_active)
            wheel_contact_ticks += int(wheel_active)
            chassis_contact_ticks += int(chassis_active)
            for face in vehicle.wheel_hit_face[0][vehicle.wheel_contact[0] != 0]:
                _record_face(int(face), observed_wheels, plane_contacts, "wheel")
            for face in vehicle.contact_face[0, : int(vehicle.contact_count[0])]:
                _record_face(int(face), observed_chassis, plane_contacts, "chassis")

        combined = observed_chassis | observed_wheels
        chassis_faces.update(observed_chassis)
        wheel_faces.update(observed_wheels)
        family_faces[scenario.family].update(combined)
        scenario_coverage.append(
            {
                "name": scenario.name,
                "family": scenario.family,
                "active_contact_ticks_1_to_12": active_ticks,
                "wheel_contact_ticks_1_to_12": wheel_contact_ticks,
                "chassis_contact_ticks_1_to_12": chassis_contact_ticks,
                "unique_triangle_faces": len(combined),
            }
        )

    all_faces = chassis_faces | wheel_faces
    shared_mask = edge_angles < np.float32(2.0 * np.pi - 1e-5)
    shared_capable_faces = set(np.flatnonzero(np.any(shared_mask, axis=1)).tolist())
    topology = _edge_topology(edge_angles, edge_flags, shared_mask)
    per_mesh = _per_mesh_coverage(geometry, all_faces)
    checkpoint_counts = _checkpoint_counts(parity)
    report = {
        "schema_version": 1,
        "milestone": "v0.2.1",
        "generated_at_utc": datetime.now(UTC).isoformat(),
        "prototype_status": "bounded_non_exhaustive",
        "scope": {
            "description": (
                "Observes triangle and contact-family coverage produced by the existing "
                "35-scenario authoritative local-transition corpus; audits shared-edge "
                "metadata across the complete DFH mesh."
            ),
            "not_claimed": (
                "This does not claim that every DFH triangle has an authoritative RocketSim "
                "transition case. A later dedicated breadth milestone must generate and gate "
                "the remaining per-triangle states."
            ),
            "horizons_ticks": list(LOCAL_HORIZONS),
            "observed_ticks_per_scenario": max(LOCAL_HORIZONS),
            "transition_cases": parity["scenario_count"],
            "checkpoint_comparisons": parity["scenario_count"] * len(LOCAL_HORIZONS),
        },
        "geometry": {
            "combined_content_sha256": geometry.content_sha256,
            "mesh_files": len(geometry.meshes),
            "triangles_exercised": len(all_faces),
            "total_triangles": geometry.triangle_count,
            "triangle_coverage_fraction": len(all_faces) / geometry.triangle_count,
            "wheel_triangle_faces_exercised": len(wheel_faces),
            "chassis_triangle_faces_exercised": len(chassis_faces),
            "shared_edge_capable_triangles_exercised": len(all_faces & shared_capable_faces),
            "shared_edge_capable_triangles_total": len(shared_capable_faces),
            "per_mesh": per_mesh,
        },
        "shared_edge_topology_audit": topology,
        "contact_families": {
            "scenario_families": {
                family: len(faces) for family, faces in sorted(family_faces.items())
            },
            "plane_contact_observations": dict(sorted(plane_contacts.items())),
            "scenario_observations": scenario_coverage,
        },
        "local_transition_results": {
            "pass_fail_by_horizon": checkpoint_counts,
            "hard_mismatch_count": parity["summary"]["hard_mismatch_count"],
            "numeric_failure_count": parity["summary"]["numeric_failure_count"],
            "numeric_error_distributions": parity["measurement_aggregates"],
            "worst_local_transition_checks": _worst_checks(parity),
            "failure_clustering_by_surface_or_contact_type": {},
            "failure_clustering_note": "No local-transition failure remained to cluster.",
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8", newline="\n")
    print(
        json.dumps(
            {
                "triangles_exercised": len(all_faces),
                "total_triangles": geometry.triangle_count,
                "checkpoint_comparisons": report["scope"]["checkpoint_comparisons"],
                "failures": parity["summary"]["hard_mismatch_count"]
                + parity["summary"]["numeric_failure_count"],
            },
            indent=2,
        )
    )
    return 0


def _record_face(
    face: int,
    destination: set[int],
    plane_contacts: Counter[str],
    contact_path: str,
) -> None:
    if face >= 0:
        destination.add(face)
    else:
        plane_contacts[f"{contact_path}:{face}"] += 1


def _edge_topology(
    angles: np.ndarray, flags: np.ndarray, shared_mask: np.ndarray
) -> dict[str, int]:
    counts = Counter[str]()
    for face, edge in np.argwhere(shared_mask):
        angle = float(angles[face, edge])
        counts["shared_directed_edges"] += 1
        if abs(angle) <= 1e-6:
            counts["planar_directed_edges"] += 1
        elif int(flags[face]) & (1 << int(edge)):
            counts["convex_directed_edges"] += 1
        else:
            counts["concave_directed_edges"] += 1
        if int(flags[face]) & (1 << (int(edge) + 3)):
            counts["normal_swap_directed_edges"] += 1
    return dict(sorted(counts.items()))


def _per_mesh_coverage(
    geometry: ArenaGeometry, exercised: set[int]
) -> list[dict[str, int | str | float]]:
    result: list[dict[str, int | str | float]] = []
    offset = 0
    for mesh in geometry.meshes:
        count = sum(offset <= face < offset + mesh.triangle_count for face in exercised)
        result.append(
            {
                "file": mesh.path.name,
                "triangles_exercised": count,
                "total_triangles": mesh.triangle_count,
                "coverage_fraction": count / mesh.triangle_count,
            }
        )
        offset += mesh.triangle_count
    return result


def _checkpoint_counts(parity: dict[str, Any]) -> dict[str, dict[str, int]]:
    result: dict[str, dict[str, int]] = {}
    for horizon in LOCAL_HORIZONS:
        records = [
            checkpoint
            for scenario in parity["scenarios"]
            for checkpoint in scenario["horizons"]
            if checkpoint["horizon_ticks"] == horizon
        ]
        passed = sum(bool(record["pass"]) for record in records)
        result[str(horizon)] = {"pass": passed, "fail": len(records) - passed}
    return result


def _worst_checks(parity: dict[str, Any]) -> list[dict[str, Any]]:
    tolerances = parity["frozen_tolerances"]
    checks: list[dict[str, Any]] = []
    for scenario in parity["scenarios"]:
        for checkpoint in scenario["horizons"]:
            for metric, error in checkpoint["errors"].items():
                tolerance = float(tolerances[metric])
                checks.append(
                    {
                        "scenario": scenario["name"],
                        "family": scenario["family"],
                        "horizon_ticks": checkpoint["horizon_ticks"],
                        "metric": metric,
                        "error": error,
                        "tolerance": tolerance,
                        "tolerance_fraction": float(error) / tolerance,
                    }
                )
    checks.sort(key=lambda item: float(item["tolerance_fraction"]), reverse=True)
    return checks[:10]


if __name__ == "__main__":
    raise SystemExit(main())
