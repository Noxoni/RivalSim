"""Run resumable RocketSim-authoritative DFH breadth validation for v0.2.2."""

from __future__ import annotations

import argparse
import gc
import gzip
import json
import math
import platform
import time
from collections import Counter, defaultdict
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import warp as wp

from rivalsim.arena import ArenaGeometry, WarpArenaMeshes
from rivalsim.dfh_breadth import (
    GENERATOR_SCHEMA_VERSION,
    GENERATOR_SEED,
    LOCAL_HORIZONS,
    BreadthCase,
    BreadthCatalog,
    build_breadth_catalog,
    cases_to_controls,
    generate_breadth_cases,
)
from rivalsim.dfh_breadth import (
    corpus_sha256 as frozen_corpus_sha256,
)
from rivalsim.dfh_breadth import (
    selection_sha256 as frozen_selection_sha256,
)
from rivalsim.math import quat_to_matrix
from rivalsim.reference.rocketsim_oracle import (
    ROCKETSIM_BINDING_COMMIT,
    ROCKETSIM_BINDING_VERSION,
    ROCKETSIM_PRIMARY_COMMIT,
    StaticWorldBatchOracleFrame,
)
from rivalsim.static_world import StaticWorldSim
from rivalsim.v02_tolerances import V02_PARITY_TOLERANCES
from rivalsim.v022_oracle_cache import (
    RocketSimAuthorityCache,
    build_authority_identity,
)

SCHEMA_VERSION = 1
RESIDUAL_TOLERANCE_FRACTION = 0.25
EDGE_OBSERVATION_DISTANCE_UU = 5.01
EXPECTED_GEOMETRY_SHA256 = "2239556BDC74D205CAA6E46A0F6E91FA2C6E4257E84D4F608BA775958B0A5538"
EXPECTED_TOPOLOGY = {
    "shared_directed_edges": 23176,
    "planar_directed_edges": 12024,
    "convex_directed_edges": 856,
    "concave_directed_edges": 10296,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--collision-dir", required=True)
    parser.add_argument(
        "--oracle-cache-root",
        type=Path,
        required=True,
        help="content-addressed cache root produced by build_v022_oracle_cache.py",
    )
    parser.add_argument("--work-dir", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--chunk-size", type=int, default=256)
    parser.add_argument(
        "--sample-count",
        type=int,
        help="deterministic pilot size; omit for the complete corpus",
    )
    parser.add_argument(
        "--overwrite-chunks",
        action="store_true",
        help="replace completed chunks in this exact work directory",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.chunk_size <= 0:
        raise ValueError("chunk size must be positive")
    if args.sample_count is not None and args.sample_count <= 0:
        raise ValueError("sample count must be positive")

    wp.init()
    geometry = ArenaGeometry.load_soccar(args.collision_dir)
    if geometry.content_sha256 != EXPECTED_GEOMETRY_SHA256:
        raise RuntimeError(
            f"unexpected DFH geometry: {geometry.content_sha256}, "
            f"expected {EXPECTED_GEOMETRY_SHA256}"
        )
    catalog = build_breadth_catalog(geometry)
    if catalog.topology_counts != EXPECTED_TOPOLOGY:
        raise RuntimeError(
            f"unexpected shared-edge topology: {catalog.topology_counts}, "
            f"expected {EXPECTED_TOPOLOGY}"
        )
    all_cases = generate_breadth_cases(catalog)
    selected_indices = _selected_indices(len(all_cases), args.sample_count, all_cases)
    selected_cases = tuple(all_cases[index] for index in selected_indices)
    corpus_sha256 = frozen_corpus_sha256(all_cases)
    selection_sha256 = frozen_selection_sha256(selected_indices, corpus_sha256)
    authority_identity = build_authority_identity(geometry, all_cases)
    authority_cache = RocketSimAuthorityCache(
        args.oracle_cache_root,
        authority_identity,
        all_cases,
    )
    run_metadata = {
        "schema_version": SCHEMA_VERSION,
        "milestone": "v0.2.2",
        "generator_schema_version": GENERATOR_SCHEMA_VERSION,
        "generator_seed": GENERATOR_SEED,
        "geometry_sha256": geometry.content_sha256,
        "topology_counts": catalog.topology_counts,
        "full_case_count": len(all_cases),
        "selected_case_count": len(selected_cases),
        "selection_kind": "complete" if args.sample_count is None else "deterministic_pilot",
        "corpus_sha256": corpus_sha256,
        "selection_sha256": selection_sha256,
        "oracle_authority_identity_sha256": authority_cache.identity_sha256,
        "oracle_cache_dir": str(authority_cache.cache_dir.resolve()),
        "oracle_execution": "cached_native_authority_only",
        "horizons_ticks": list(LOCAL_HORIZONS),
        "residual_tolerance_fraction": RESIDUAL_TOLERANCE_FRACTION,
        "edge_observation_distance_uu": EDGE_OBSERVATION_DISTANCE_UU,
        "chunk_size": args.chunk_size,
        "device": str(wp.get_device(args.device)),
    }
    args.work_dir.mkdir(parents=True, exist_ok=True)
    _validate_or_write_run_metadata(args.work_dir / "run.json", run_metadata)

    meshes = WarpArenaMeshes(geometry, args.device)
    chunk_paths: list[Path] = []
    started = time.perf_counter()
    for chunk_number, start in enumerate(range(0, len(selected_cases), args.chunk_size)):
        stop = min(start + args.chunk_size, len(selected_cases))
        chunk_cases = selected_cases[start:stop]
        chunk_indices = selected_indices[start:stop]
        chunk_path = args.work_dir / f"chunk-{chunk_number:05d}-{start:06d}-{stop:06d}.json.gz"
        chunk_paths.append(chunk_path)
        if chunk_path.exists() and not args.overwrite_chunks:
            _validate_chunk(chunk_path, selection_sha256, chunk_cases)
            print(
                json.dumps(
                    {
                        "chunk": chunk_number,
                        "range": [start, stop],
                        "status": "resumed",
                    }
                ),
                flush=True,
            )
            continue
        chunk_started = time.perf_counter()
        result = _run_chunk(
            chunk_number,
            chunk_cases,
            chunk_indices,
            selection_sha256,
            authority_cache,
            args.collision_dir,
            geometry,
            meshes,
            args.device,
        )
        _write_gzip_json(chunk_path, result)
        print(
            json.dumps(
                {
                    "chunk": chunk_number,
                    "range": [start, stop],
                    "status": "measured",
                    "seconds": round(time.perf_counter() - chunk_started, 3),
                    "hard_mismatch_events": result["summary"]["hard_mismatch_events"],
                    "numeric_failure_events": result["summary"]["numeric_failure_events"],
                    "paired_target_contacts": result["summary"]["paired_target_contacts"],
                }
            ),
            flush=True,
        )
        gc.collect()

    aggregate = _aggregate(
        geometry,
        catalog,
        all_cases,
        selected_indices,
        chunk_paths,
        run_metadata,
    )
    aggregate["run_seconds"] = time.perf_counter() - started
    aggregate_path = args.work_dir / "aggregate.json"
    aggregate_path.write_text(
        json.dumps(aggregate, indent=2) + "\n", encoding="utf-8", newline="\n"
    )
    print(json.dumps(aggregate["gate"], indent=2), flush=True)
    print(f"aggregate={aggregate_path.resolve()}", flush=True)
    return 0 if bool(aggregate["gate"]["selected_run_pass"]) else 1


def _selected_indices(
    total: int, sample_count: int | None, cases: tuple[BreadthCase, ...]
) -> tuple[int, ...]:
    if sample_count is None or sample_count >= total:
        return tuple(range(total))
    base_count = min(sample_count, total)
    selected = set(np.linspace(0, total - 1, base_count, dtype=np.int64).tolist())
    # Always include every analytic plane mode in pilots. They are only 20
    # states and form a distinct RocketSim collision path.
    selected.update(index for index, case in enumerate(cases) if case.case_kind == "analytic_plane")
    return tuple(sorted(int(index) for index in selected))


def _run_chunk(
    chunk_number: int,
    cases: tuple[BreadthCase, ...],
    corpus_indices: tuple[int, ...],
    selection_sha256: str,
    authority_cache: RocketSimAuthorityCache,
    collision_root: str,
    geometry: ArenaGeometry,
    meshes: WarpArenaMeshes,
    device: str,
) -> dict[str, Any]:
    controls = cases_to_controls(cases)
    cached = authority_cache.load(corpus_indices)
    authoritative = cached.authoritative_snapshot
    sim = StaticWorldSim(
        len(cases),
        collision_root,
        variant="B3",
        device=device,
        initial=authoritative.copy(),
        geometry=geometry,
        meshes=meshes,
    )
    sim.set_controls(controls)
    observations = [
        {
            "oracle_contact_tick_mask": 0,
            "rival_target_tick_mask": 0,
            "paired_target_tick_mask": 0,
            "observed_chassis_faces": set(),
            "observed_wheel_faces": set(),
            "checkpoints": [],
        }
        for _case in cases
    ]

    for tick in range(1, max(LOCAL_HORIZONS) + 1):
        sim.step(1)
        state = sim.snapshot()
        vehicle = sim.vehicle_snapshot()
        reference = cached.frame(tick)
        _observe_contacts(cases, observations, vehicle, reference, tick)
        if tick not in LOCAL_HORIZONS:
            continue
        numeric = _numeric_error_arrays(state, vehicle, reference)
        hard = _hard_mismatch_arrays(state, vehicle, reference)
        for index, _case in enumerate(cases):
            errors = {metric: float(values[index]) for metric, values in numeric.items()}
            mismatches = [metric for metric, values in hard.items() if bool(values[index])]
            numeric_failures = [
                metric for metric, value in errors.items() if value > V02_PARITY_TOLERANCES[metric]
            ]
            residual_metrics = []
            if not mismatches and not numeric_failures:
                residual_metrics = [
                    metric
                    for metric, value in errors.items()
                    if value >= V02_PARITY_TOLERANCES[metric] * RESIDUAL_TOLERANCE_FRACTION
                ]
            observations[index]["checkpoints"].append(
                {
                    "tick": tick,
                    "errors": errors,
                    "hard_mismatches": mismatches,
                    "numeric_failures": numeric_failures,
                    "residual_metrics": residual_metrics,
                    "pass": not mismatches and not numeric_failures,
                }
            )

    records: list[dict[str, Any]] = []
    for case, corpus_index, observation in zip(cases, corpus_indices, observations, strict=True):
        records.append(
            {
                **_case_metadata(case, corpus_index),
                "oracle_contact_tick_mask": observation["oracle_contact_tick_mask"],
                "rival_target_tick_mask": observation["rival_target_tick_mask"],
                "paired_target_tick_mask": observation["paired_target_tick_mask"],
                "observed_chassis_faces": sorted(observation["observed_chassis_faces"]),
                "observed_wheel_faces": sorted(observation["observed_wheel_faces"]),
                "checkpoints": observation["checkpoints"],
                "pass": all(item["pass"] for item in observation["checkpoints"]),
            }
        )

    hard_events = sum(
        len(checkpoint["hard_mismatches"])
        for record in records
        for checkpoint in record["checkpoints"]
    )
    numeric_events = sum(
        len(checkpoint["numeric_failures"])
        for record in records
        for checkpoint in record["checkpoints"]
    )
    return {
        "schema_version": SCHEMA_VERSION,
        "milestone": "v0.2.2",
        "chunk_number": chunk_number,
        "selection_sha256": selection_sha256,
        "case_count": len(cases),
        "case_ids": [case.case_id for case in cases],
        "generated_at_utc": datetime.now(UTC).isoformat(),
        "records": records,
        "summary": {
            "checkpoint_comparisons": len(cases) * len(LOCAL_HORIZONS),
            "hard_mismatch_events": hard_events,
            "numeric_failure_events": numeric_events,
            "paired_target_contacts": sum(
                bool(record["paired_target_tick_mask"]) for record in records
            ),
        },
    }


def _observe_contacts(
    cases: tuple[BreadthCase, ...],
    observations: list[dict[str, Any]],
    vehicle: Any,
    reference: StaticWorldBatchOracleFrame,
    tick: int,
) -> None:
    bit = 1 << (tick - 1)
    for env_index, case in enumerate(cases):
        car_index = env_index * 2
        contact_count = int(vehicle.contact_count[car_index])
        chassis_faces = vehicle.contact_face[car_index, :contact_count]
        chassis_points = vehicle.contact_point[car_index, :contact_count]
        wheel_mask = vehicle.wheel_contact[car_index] != 0
        wheel_faces = vehicle.wheel_hit_face[car_index][wheel_mask]
        wheel_points = vehicle.wheel_hit_point[car_index][wheel_mask]
        observations[env_index]["observed_chassis_faces"].update(
            int(face) for face in chassis_faces if int(face) >= 0
        )
        observations[env_index]["observed_wheel_faces"].update(
            int(face) for face in wheel_faces if int(face) >= 0
        )
        oracle_contact = (
            bool(reference.has_world_contact[env_index])
            if case.contact_path == "chassis"
            else bool(np.any(reference.wheel_contacts[env_index]))
        )
        if oracle_contact:
            observations[env_index]["oracle_contact_tick_mask"] |= bit
        target_observed = _target_contact_observed(
            case,
            chassis_faces,
            chassis_points,
            wheel_faces,
            wheel_points,
        )
        if target_observed:
            observations[env_index]["rival_target_tick_mask"] |= bit
        if oracle_contact and target_observed:
            observations[env_index]["paired_target_tick_mask"] |= bit


def _target_contact_observed(
    case: BreadthCase,
    chassis_faces: np.ndarray,
    chassis_points: np.ndarray,
    wheel_faces: np.ndarray,
    wheel_points: np.ndarray,
) -> bool:
    faces = chassis_faces if case.contact_path == "chassis" else wheel_faces
    points = chassis_points if case.contact_path == "chassis" else wheel_points
    if case.case_kind == "analytic_plane":
        return any(int(face) == case.expected_plane_face for face in faces)
    accepted_faces = {case.target_face}
    if case.target_neighbor_face is not None:
        accepted_faces.add(case.target_neighbor_face)
    for face, point in zip(faces, points, strict=True):
        if int(face) not in accepted_faces:
            continue
        if case.case_kind != "shared_directed_edge":
            return True
        if _point_segment_distance(point, case.edge_start, case.edge_end) <= (
            EDGE_OBSERVATION_DISTANCE_UU
        ):
            return True
    return False


def _numeric_error_arrays(
    state: Any, vehicle: Any, reference: StaticWorldBatchOracleFrame
) -> dict[str, np.ndarray]:
    sim_matrix = quat_to_matrix(state.car_quat[:, 0])
    trace = np.einsum("nij,nij->n", sim_matrix.astype(np.float64), reference.car_matrix)
    orientation = np.arccos(np.clip((trace - 1.0) * 0.5, -1.0, 1.0))
    car_indices = np.arange(state.num_envs) * 2
    sim_normal = vehicle.world_contact_normal[car_indices].astype(np.float64)
    reference_normal = reference.world_contact_normal.astype(np.float64)
    sim_length = np.linalg.norm(sim_normal, axis=1)
    reference_length = np.linalg.norm(reference_normal, axis=1)
    normal_angle = np.zeros(state.num_envs, dtype=np.float64)
    valid = (sim_length > 1e-7) & (reference_length > 1e-7)
    if np.any(valid):
        cosine = np.sum(sim_normal[valid] * reference_normal[valid], axis=1) / (
            sim_length[valid] * reference_length[valid]
        )
        normal_angle[valid] = np.arccos(np.clip(cosine, -1.0, 1.0))
    return {
        "position_uu": np.linalg.norm(
            state.car_pos[:, 0].astype(np.float64) - reference.car_pos, axis=1
        ),
        "linear_velocity_uu_per_s": np.linalg.norm(
            state.car_vel[:, 0].astype(np.float64) - reference.car_vel, axis=1
        ),
        "orientation_rad": orientation,
        "angular_velocity_rad_per_s": np.linalg.norm(
            state.car_ang_vel[:, 0].astype(np.float64) - reference.car_ang_vel, axis=1
        ),
        "boost": np.abs(state.boost[:, 0].astype(np.float64) - reference.boost),
        "handbrake_value": np.abs(
            vehicle.handbrake_value[car_indices].astype(np.float64) - reference.handbrake_value
        ),
        "world_contact_normal_rad": normal_angle,
    }


def _hard_mismatch_arrays(
    state: Any, vehicle: Any, reference: StaticWorldBatchOracleFrame
) -> dict[str, np.ndarray]:
    car_indices = np.arange(state.num_envs) * 2
    sim_normal = vehicle.world_contact_normal[car_indices].astype(np.float64)
    sim_world_contact = np.linalg.norm(sim_normal, axis=1) > 0.5
    result: dict[str, np.ndarray] = {
        "on_ground": state.on_ground[:, 0].astype(bool) != reference.on_ground,
        "world_contact": sim_world_contact != reference.has_world_contact,
    }
    for wheel in range(4):
        result[f"wheel_contact_{wheel}"] = (
            vehicle.wheel_contact[car_indices, wheel].astype(bool)
            != reference.wheel_contacts[:, wheel]
        )
    sim_velocity = state.car_vel[:, 0].astype(np.float64)
    reference_velocity = reference.car_vel.astype(np.float64)
    result["linear_velocity_direction"] = (
        (np.linalg.norm(sim_velocity, axis=1) > 25.0)
        & (np.linalg.norm(reference_velocity, axis=1) > 25.0)
        & (np.sum(sim_velocity * reference_velocity, axis=1) < 0.0)
    )
    reference_normal = reference.world_contact_normal.astype(np.float64)
    result["world_contact_normal_direction"] = (
        sim_world_contact
        & reference.has_world_contact
        & (np.linalg.norm(sim_normal, axis=1) > 1e-7)
        & (np.linalg.norm(reference_normal, axis=1) > 1e-7)
        & (np.sum(sim_normal * reference_normal, axis=1) <= 0.0)
    )
    return result


def _aggregate(
    geometry: ArenaGeometry,
    catalog: BreadthCatalog,
    all_cases: tuple[BreadthCase, ...],
    selected_indices: tuple[int, ...],
    chunk_paths: list[Path],
    run_metadata: dict[str, Any],
) -> dict[str, Any]:
    records = [record for path in chunk_paths for record in _read_gzip_json(path)["records"]]
    if len(records) != len(selected_indices):
        raise RuntimeError("aggregate record count does not match selected corpus")
    if [record["corpus_index"] for record in records] != list(selected_indices):
        raise RuntimeError("aggregate record order does not match selected corpus")

    metric_values: dict[str, list[float]] = defaultdict(list)
    pass_by_horizon = {str(tick): {"pass": 0, "fail": 0} for tick in LOCAL_HORIZONS}
    hard_events = 0
    numeric_events = 0
    residual_events = 0
    hard_checkpoints = 0
    numeric_checkpoints = 0
    residual_checkpoints = 0
    failed_case_ids: set[str] = set()
    hard_case_ids: set[str] = set()
    numeric_case_ids: set[str] = set()
    residual_case_ids: set[str] = set()
    failure_events: list[dict[str, Any]] = []
    worst: list[dict[str, Any]] = []
    for record in records:
        for checkpoint in record["checkpoints"]:
            passed = bool(checkpoint["pass"])
            pass_by_horizon[str(checkpoint["tick"])]["pass" if passed else "fail"] += 1
            if not passed:
                failed_case_ids.add(record["case_id"])
            hard = checkpoint["hard_mismatches"]
            numeric = checkpoint["numeric_failures"]
            residual = checkpoint["residual_metrics"]
            hard_events += len(hard)
            numeric_events += len(numeric)
            residual_events += len(residual)
            hard_checkpoints += bool(hard)
            numeric_checkpoints += bool(numeric)
            residual_checkpoints += bool(residual)
            if hard:
                hard_case_ids.add(record["case_id"])
            if numeric:
                numeric_case_ids.add(record["case_id"])
            if residual:
                residual_case_ids.add(record["case_id"])
            for kind, names in (
                ("hard", hard),
                ("numeric", numeric),
                ("residual", residual),
            ):
                for metric in names:
                    failure_events.append(
                        {
                            "kind": kind,
                            "metric": metric,
                            "tick": checkpoint["tick"],
                            **_cluster_dimensions(record),
                        }
                    )
            for metric, value in checkpoint["errors"].items():
                value = float(value)
                metric_values[metric].append(value)
                tolerance = V02_PARITY_TOLERANCES[metric]
                worst.append(
                    {
                        "case_id": record["case_id"],
                        "family": record["family"],
                        "mode": record["mode"],
                        "mesh_file": record["mesh_file"],
                        "edge_class": record["edge_class"],
                        "tick": checkpoint["tick"],
                        "metric": metric,
                        "error": value,
                        "tolerance": tolerance,
                        "tolerance_fraction": value / tolerance,
                    }
                )
    worst.sort(key=lambda item: item["tolerance_fraction"], reverse=True)

    selected_cases = tuple(all_cases[index] for index in selected_indices)
    full_selection = len(selected_indices) == len(all_cases)
    coverage = _coverage_summary(geometry, catalog, selected_cases, records)
    distributions = {
        metric: {
            "p50": float(np.percentile(values, 50)),
            "p95": float(np.percentile(values, 95)),
            "p99": float(np.percentile(values, 99)),
            "max": float(np.max(values)),
            "tolerance": V02_PARITY_TOLERANCES[metric],
            "max_tolerance_fraction": float(np.max(values)) / V02_PARITY_TOLERANCES[metric],
        }
        for metric, values in sorted(metric_values.items())
    }
    blocking_events = [event for event in failure_events if event["kind"] != "residual"]
    residual_only_events = [event for event in failure_events if event["kind"] == "residual"]
    selected_pass = hard_events == 0 and numeric_events == 0
    complete_gate_pass = full_selection and selected_pass
    return {
        "schema_version": SCHEMA_VERSION,
        "milestone": "v0.2.2",
        "generated_at_utc": datetime.now(UTC).isoformat(),
        "run": run_metadata,
        "implementation": {
            "python": platform.python_version(),
            "warp": wp.__version__,
            "device": run_metadata["device"],
            "rocketsim_package": ROCKETSIM_BINDING_VERSION,
            "rocketsim_primary_commit": ROCKETSIM_PRIMARY_COMMIT,
            "rocketsim_binding_commit": ROCKETSIM_BINDING_COMMIT,
        },
        "frozen_protocol": {
            "hard_horizons_ticks": list(LOCAL_HORIZONS),
            "long_horizons_status": "diagnostic_only_not_run",
            "tolerances": V02_PARITY_TOLERANCES,
            "residual_definition": (
                "A checkpoint metric at or above 25% of its frozen tolerance, "
                "while that checkpoint has no hard or numeric failure."
            ),
            "residual_tolerance_fraction": RESIDUAL_TOLERANCE_FRACTION,
        },
        "counts": {
            "selected_starting_states": len(records),
            "full_corpus_starting_states": len(all_cases),
            "checkpoint_comparisons": len(records) * len(LOCAL_HORIZONS),
            "metric_comparisons": len(records) * len(LOCAL_HORIZONS) * len(V02_PARITY_TOLERANCES),
            "hard_mismatch_events": hard_events,
            "hard_mismatch_checkpoints": hard_checkpoints,
            "hard_mismatch_cases": len(hard_case_ids),
            "numeric_failure_events": numeric_events,
            "numeric_failure_checkpoints": numeric_checkpoints,
            "numeric_failure_cases": len(numeric_case_ids),
            "residual_nonblocking_deviation_events": residual_events,
            "residual_nonblocking_checkpoints": residual_checkpoints,
            "residual_nonblocking_cases": len(residual_case_ids),
            "failed_cases": len(failed_case_ids),
        },
        "pass_fail_by_horizon": pass_by_horizon,
        "numeric_error_distributions": distributions,
        "worst_local_errors": worst[:50],
        "coverage": coverage,
        "failure_clusters": {
            "blocking": _cluster_events(blocking_events),
            "residual_nonblocking": _cluster_events(residual_only_events),
        },
        "gate": {
            "selection_complete": full_selection,
            "selected_run_pass": selected_pass,
            "complete_v022_gate_pass": complete_gate_pass,
            "classification": (
                "PASS_GREEN"
                if complete_gate_pass
                else "PILOT_PASS"
                if selected_pass
                else "BLOCKING_FAILURE"
            ),
            "unresolved_hard_or_numeric_failures": hard_events + numeric_events,
        },
    }


def _coverage_summary(
    geometry: ArenaGeometry,
    catalog: BreadthCatalog,
    selected_cases: tuple[BreadthCase, ...],
    records: list[dict[str, Any]],
) -> dict[str, Any]:
    generated_chassis_faces = {
        case.target_face
        for case in selected_cases
        if case.case_kind == "triangle_face" and case.contact_path == "chassis"
    }
    generated_wheel_faces = {
        case.target_face
        for case in selected_cases
        if case.case_kind == "triangle_face" and case.contact_path == "wheel"
    }
    actual_target_chassis: set[int] = set()
    actual_target_wheel: set[int] = set()
    observed_chassis_faces: set[int] = set()
    observed_wheel_faces: set[int] = set()
    generated_edges: dict[str, set[int]] = defaultdict(set)
    actual_edges: dict[str, set[int]] = defaultdict(set)
    missing_triangle_reasons: Counter[str] = Counter()
    missing_edge_reasons: Counter[str] = Counter()
    missing_examples: list[dict[str, Any]] = []
    analytic = defaultdict(lambda: Counter[str]())
    region_counts: dict[str, Counter[str]] = defaultdict(Counter)
    for case, record in zip(selected_cases, records, strict=True):
        paired = bool(record["paired_target_tick_mask"])
        oracle_contact = bool(record["oracle_contact_tick_mask"])
        rival_target = bool(record["rival_target_tick_mask"])
        observed_chassis_faces.update(record["observed_chassis_faces"])
        observed_wheel_faces.update(record["observed_wheel_faces"])
        for region in case.region_labels:
            region_counts[region]["generated_states"] += 1
            region_counts[region]["paired_target_contacts"] += paired
            region_counts[region]["parity_passed_states"] += bool(record["pass"])
        if case.case_kind == "triangle_face":
            if paired:
                destination = (
                    actual_target_chassis if case.contact_path == "chassis" else actual_target_wheel
                )
                destination.add(int(case.target_face))
            else:
                reason = _missing_reason(oracle_contact, rival_target)
                missing_triangle_reasons[f"{case.contact_path}:{reason}"] += 1
                if len(missing_examples) < 50:
                    missing_examples.append(
                        {
                            "case_id": case.case_id,
                            "kind": "triangle_face",
                            "contact_path": case.contact_path,
                            "mesh_file": case.mesh_file,
                            "target_face": case.target_face,
                            "reason": reason,
                        }
                    )
        elif case.case_kind == "shared_directed_edge":
            generated_edges[str(case.edge_class)].add(
                int(case.target_edge) + 3 * int(case.target_face)
            )
            if paired:
                actual_edges[str(case.edge_class)].add(
                    int(case.target_edge) + 3 * int(case.target_face)
                )
            else:
                reason = _missing_reason(oracle_contact, rival_target)
                missing_edge_reasons[f"{case.edge_class}:{case.contact_path}:{reason}"] += 1
                if len(missing_examples) < 50:
                    missing_examples.append(
                        {
                            "case_id": case.case_id,
                            "kind": "shared_directed_edge",
                            "contact_path": case.contact_path,
                            "mesh_file": case.mesh_file,
                            "target_face": case.target_face,
                            "edge_class": case.edge_class,
                            "reason": reason,
                        }
                    )
        else:
            counts = analytic[str(case.analytic_plane)]
            counts["generated_states"] += 1
            counts["paired_target_contacts"] += paired
            counts["parity_passed_states"] += bool(record["pass"])

    per_mesh: list[dict[str, Any]] = []
    offset = 0
    record_by_id = {record["case_id"]: record for record in records}
    for mesh_index, mesh in enumerate(geometry.meshes):
        mesh_faces = set(range(offset, offset + mesh.triangle_count))
        mesh_cases = [case for case in selected_cases if case.mesh_index == mesh_index]
        mesh_edges = [case for case in mesh_cases if case.case_kind == "shared_directed_edge"]
        edge_generated = Counter(case.edge_class for case in mesh_edges)
        edge_actual = Counter(
            case.edge_class
            for case in mesh_edges
            if record_by_id[case.case_id]["paired_target_tick_mask"]
        )
        per_mesh.append(
            {
                "file": mesh.path.name,
                "total_triangles": mesh.triangle_count,
                "generated_chassis_triangle_states": len(generated_chassis_faces & mesh_faces),
                "generated_wheel_triangle_states": len(generated_wheel_faces & mesh_faces),
                "actual_target_chassis_triangles": len(actual_target_chassis & mesh_faces),
                "actual_target_wheel_triangles": len(actual_target_wheel & mesh_faces),
                "observed_chassis_triangles_any_case": len(observed_chassis_faces & mesh_faces),
                "observed_wheel_triangles_any_case": len(observed_wheel_faces & mesh_faces),
                "directed_edges": {
                    edge_class: {
                        "generated": edge_generated[edge_class],
                        "actual_paired_target_contact": edge_actual[edge_class],
                    }
                    for edge_class in ("planar", "convex", "concave")
                },
            }
        )
        offset += mesh.triangle_count

    return {
        "definitions": {
            "topology_audited": "source CMF connectivity enumerated without implying contact",
            "state_generated": "a deterministic starting state targets the topology element",
            "actual_paired_target_contact": (
                "RivalSim reported the targeted face/edge while RocketSim simultaneously "
                "reported the corresponding chassis or wheel semantic contact"
            ),
            "parity_passed": "all 1/4/8/12 checkpoints passed hard and numeric gates",
            "edge_proximity_uu": EDGE_OBSERVATION_DISTANCE_UU,
        },
        "topology_audited": {
            "mesh_files": len(geometry.meshes),
            "vertices": geometry.vertex_count,
            "triangles": geometry.triangle_count,
            **catalog.topology_counts,
        },
        "generated": {
            "triangle_chassis_states": len(generated_chassis_faces),
            "triangle_wheel_states": len(generated_wheel_faces),
            "unique_triangles_with_any_state": len(generated_chassis_faces | generated_wheel_faces),
            "shared_directed_edges": sum(len(values) for values in generated_edges.values()),
            "planar_directed_edges": len(generated_edges["planar"]),
            "convex_directed_edges": len(generated_edges["convex"]),
            "concave_directed_edges": len(generated_edges["concave"]),
            "analytic_plane_states": sum(
                counts["generated_states"] for counts in analytic.values()
            ),
        },
        "actual_paired_target_contact": {
            "chassis_triangles": len(actual_target_chassis),
            "wheel_triangles": len(actual_target_wheel),
            "unique_triangles": len(actual_target_chassis | actual_target_wheel),
            "shared_directed_edges": sum(len(values) for values in actual_edges.values()),
            "planar_directed_edges": len(actual_edges["planar"]),
            "convex_directed_edges": len(actual_edges["convex"]),
            "concave_directed_edges": len(actual_edges["concave"]),
            "observed_chassis_triangles_any_case": len(observed_chassis_faces),
            "observed_wheel_triangles_any_case": len(observed_wheel_faces),
        },
        "analytic_planes": {
            name: dict(sorted(counts.items())) for name, counts in sorted(analytic.items())
        },
        "important_regions": {
            name: dict(sorted(counts.items())) for name, counts in sorted(region_counts.items())
        },
        "per_mesh": per_mesh,
        "missing_or_unexercised": {
            "triangle_reasons": dict(sorted(missing_triangle_reasons.items())),
            "directed_edge_reasons": dict(sorted(missing_edge_reasons.items())),
            "bounded_examples": missing_examples,
        },
    }


def _missing_reason(oracle_contact: bool, rival_target: bool) -> str:
    if not oracle_contact and not rival_target:
        return "no_semantic_or_target_contact_within_12_ticks"
    if not oracle_contact:
        return "rocketsim_semantic_contact_not_observed"
    if not rival_target:
        return "target_occluded_or_adjacent_contact_selected"
    return "not_observed_on_the_same_tick"


def _cluster_dimensions(record: dict[str, Any]) -> dict[str, Any]:
    return {
        "case_id": record["case_id"],
        "case_kind": record["case_kind"],
        "family": record["family"],
        "mode": record["mode"],
        "mesh_file": record["mesh_file"],
        "edge_class": record["edge_class"],
        "primary_region": record["region_labels"][0],
    }


def _cluster_events(events: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[Any, ...], list[dict[str, Any]]] = defaultdict(list)
    for event in events:
        key = (
            event["kind"],
            event["metric"],
            event["case_kind"],
            event["family"],
            event["mode"],
            event["mesh_file"],
            event["edge_class"],
            event["primary_region"],
        )
        grouped[key].append(event)
    result = []
    for key, members in grouped.items():
        result.append(
            {
                "kind": key[0],
                "metric": key[1],
                "case_kind": key[2],
                "family": key[3],
                "mode": key[4],
                "mesh_file": key[5],
                "edge_class": key[6],
                "primary_region": key[7],
                "event_count": len(members),
                "case_count": len({member["case_id"] for member in members}),
                "ticks": dict(sorted(Counter(member["tick"] for member in members).items())),
                "example_case_ids": sorted({member["case_id"] for member in members})[:10],
            }
        )
    result.sort(key=lambda item: (-item["event_count"], str(item)))
    return result


def _case_metadata(case: BreadthCase, corpus_index: int) -> dict[str, Any]:
    return {
        "corpus_index": corpus_index,
        "case_id": case.case_id,
        "case_kind": case.case_kind,
        "family": case.family,
        "mode": case.mode,
        "contact_path": case.contact_path,
        "mesh_index": case.mesh_index,
        "mesh_file": case.mesh_file,
        "target_face": case.target_face,
        "target_neighbor_face": case.target_neighbor_face,
        "target_edge": case.target_edge,
        "edge_class": case.edge_class,
        "analytic_plane": case.analytic_plane,
        "expected_plane_face": case.expected_plane_face,
        "region_labels": list(case.region_labels),
    }


def _point_segment_distance(
    point: np.ndarray, start: np.ndarray | None, end: np.ndarray | None
) -> float:
    if start is None or end is None:
        return math.inf
    segment = np.asarray(end - start, dtype=np.float64)
    denominator = float(np.dot(segment, segment))
    if denominator <= 0.0:
        return float(np.linalg.norm(np.asarray(point, dtype=np.float64) - start))
    parameter = float(
        np.clip(
            np.dot(np.asarray(point, dtype=np.float64) - start, segment) / denominator,
            0.0,
            1.0,
        )
    )
    closest = np.asarray(start, dtype=np.float64) + segment * parameter
    return float(np.linalg.norm(np.asarray(point, dtype=np.float64) - closest))


def _validate_or_write_run_metadata(path: Path, metadata: dict[str, Any]) -> None:
    if path.exists():
        existing = json.loads(path.read_text(encoding="utf-8"))
        if existing != metadata:
            raise RuntimeError(f"work directory metadata does not match this run: {path.resolve()}")
        return
    path.write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8", newline="\n")


def _validate_chunk(path: Path, selection_sha256: str, cases: tuple[BreadthCase, ...]) -> None:
    payload = _read_gzip_json(path)
    expected_ids = [case.case_id for case in cases]
    if (
        payload.get("schema_version") != SCHEMA_VERSION
        or payload.get("selection_sha256") != selection_sha256
        or payload.get("case_ids") != expected_ids
        or len(payload.get("records", ())) != len(cases)
    ):
        raise RuntimeError(f"invalid or stale completed chunk: {path.resolve()}")


def _write_gzip_json(path: Path, payload: dict[str, Any]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    with gzip.open(temporary, "wt", encoding="utf-8", newline="\n", compresslevel=6) as stream:
        json.dump(payload, stream, separators=(",", ":"))
        stream.write("\n")
    temporary.replace(path)


def _read_gzip_json(path: Path) -> dict[str, Any]:
    with gzip.open(path, "rt", encoding="utf-8") as stream:
        return json.load(stream)


if __name__ == "__main__":
    raise SystemExit(main())
