"""Run cached-native v0.3 Phase A ball/world parity acceptance."""

from __future__ import annotations

import argparse
import json
import platform
from collections import Counter, defaultdict
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import warp as wp

from rivalsim.arena import ArenaGeometry, WarpArenaMeshes
from rivalsim.ball_world_state import MAX_BALL_CONTACTS
from rivalsim.math import quat_to_matrix
from rivalsim.static_world import DynamicWorldSim
from rivalsim.v03_corpus import (
    V03_HARD_HORIZONS,
    BallWorldCase,
    generate_phase_a_cases,
    phase_a_cases_to_state,
    phase_a_corpus_sha256,
    phase_a_representative_indices,
    phase_a_selection_sha256,
)
from rivalsim.v03_oracle_cache import (
    EXPECTED_SOCCAR_CMF_SHA256,
    build_phase_a_identity,
    load_phase_a_frames,
)

SCHEMA_VERSION = 1
TOLERANCES = {
    "position_uu": 10.0,
    "linear_velocity_uu_per_s": 25.0,
    "orientation_rad": 0.025,
    "angular_velocity_rad_per_s": 0.1,
}
PLANE_FACE = {
    "floor": -10,
    "ceiling": -11,
    "negative_x_wall": -12,
    "positive_x_wall": -13,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--collision-dir", required=True)
    parser.add_argument("--oracle-cache-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument(
        "--full",
        action="store_true",
        help="run all frozen cases instead of the deterministic representative gate",
    )
    parser.add_argument("--representative-base-count", type=int, default=1024)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.representative_base_count <= 0:
        raise ValueError("representative base count must be positive")
    wp.init()
    geometry = ArenaGeometry.load_soccar(args.collision_dir)
    if geometry.content_sha256 != EXPECTED_SOCCAR_CMF_SHA256:
        raise RuntimeError(
            f"unexpected Soccar geometry {geometry.content_sha256}; "
            f"expected {EXPECTED_SOCCAR_CMF_SHA256}"
        )
    catalog, all_cases = generate_phase_a_cases(geometry)
    identity = build_phase_a_identity(geometry, all_cases)
    indices = (
        tuple(range(len(all_cases)))
        if args.full
        else phase_a_representative_indices(all_cases, args.representative_base_count)
    )
    cases = tuple(all_cases[index] for index in indices)
    selection_sha256 = phase_a_selection_sha256(all_cases, indices)
    cached = load_phase_a_frames(
        args.oracle_cache_root, identity, all_cases, indices
    )

    source_state = phase_a_cases_to_state(cases)
    meshes = WarpArenaMeshes(geometry, args.device)
    sim = DynamicWorldSim(
        len(cases),
        args.collision_dir,
        variant="B3",
        device=args.device,
        initial=source_state,
        geometry=geometry,
        meshes=meshes,
    )

    observed_target = np.zeros(len(cases), dtype=bool)
    observed_any_contact = np.zeros(len(cases), dtype=bool)
    max_contacts = np.zeros(len(cases), dtype=np.int32)
    metric_values: dict[str, list[np.ndarray]] = defaultdict(list)
    failure_records: list[dict[str, Any]] = []
    pass_by_horizon: dict[str, dict[str, int]] = {
        str(tick): {"pass": 0, "fail": 0} for tick in V03_HARD_HORIZONS
    }
    hard_events = 0
    numeric_events = 0
    failed_cases: set[str] = set()

    for tick in range(1, max(V03_HARD_HORIZONS) + 1):
        sim.step(1)
        contact_count = sim.ball_world.contact_count.numpy().astype(np.int32)
        contact_face = sim.ball_world.contact_face.numpy().reshape(
            len(cases), MAX_BALL_CONTACTS
        )
        candidate_overflow = sim.ball_world.candidate_overflow.numpy() != 0
        contact_overflow = sim.ball_world.contact_overflow.numpy() != 0
        max_contacts = np.maximum(max_contacts, contact_count)
        observed_any_contact |= contact_count > 0
        for case_index, case in enumerate(cases):
            if _target_contact_observed(
                case, contact_face[case_index, : contact_count[case_index]]
            ):
                observed_target[case_index] = True

        if tick not in V03_HARD_HORIZONS:
            continue
        state = sim.snapshot()
        authority_index = tick - 1
        errors = _numeric_errors(state, cached, authority_index)
        for metric, values in errors.items():
            metric_values[metric].append(values)
        finite = _finite_state(state)
        sim_velocity = state.ball_vel.astype(np.float64)
        native_velocity = cached["ball_vel"][:, authority_index].astype(np.float64)
        wrong_direction = (
            (np.linalg.norm(sim_velocity, axis=1) > 25.0)
            & (np.linalg.norm(native_velocity, axis=1) > 25.0)
            & (np.sum(sim_velocity * native_velocity, axis=1) < 0.0)
        )

        for case_index, case in enumerate(cases):
            hard = []
            if not finite[case_index]:
                hard.append("non_finite_state")
            if candidate_overflow[case_index]:
                hard.append("candidate_overflow")
            if contact_overflow[case_index]:
                hard.append("contact_overflow")
            if wrong_direction[case_index]:
                hard.append("linear_velocity_direction")
            numeric = [
                metric
                for metric, values in errors.items()
                if float(values[case_index]) > TOLERANCES[metric]
            ]
            passed = not hard and not numeric
            pass_by_horizon[str(tick)]["pass" if passed else "fail"] += 1
            hard_events += len(hard)
            numeric_events += len(numeric)
            if not passed:
                failed_cases.add(case.case_id)
                if len(failure_records) < 200:
                    failure_records.append(
                        {
                            **_case_metadata(case, indices[case_index]),
                            "tick": tick,
                            "hard_mismatches": hard,
                            "numeric_failures": numeric,
                            "errors": {
                                metric: float(values[case_index])
                                for metric, values in errors.items()
                            },
                        }
                    )

    selected_pass = hard_events == 0 and numeric_events == 0
    full_selection = len(indices) == len(all_cases)
    report = {
        "schema_version": SCHEMA_VERSION,
        "milestone": "v0.3",
        "phase": "A_ball_world",
        "generated_at_utc": datetime.now(UTC).isoformat(),
        "classification": (
            "PASS_GREEN"
            if full_selection and selected_pass
            else "REPRESENTATIVE_PASS"
            if selected_pass
            else "BLOCKING_FAILURE"
        ),
        "selection": {
            "kind": "complete" if full_selection else "deterministic_representative",
            "selected_case_count": len(indices),
            "full_case_count": len(all_cases),
            "representative_base_count": (
                None if full_selection else args.representative_base_count
            ),
            "corpus_sha256": phase_a_corpus_sha256(all_cases),
            "selection_sha256": selection_sha256,
        },
        "authority": {
            "execution": "cached_native_authority_only_no_live_fallback",
            "identity_sha256": identity["authority_identity_sha256"],
            "cache_format_version": identity["cache_format_version"],
            "source_state": "exact frozen corpus record",
            "initial_native_readback": "cached immediately after SetState",
        },
        "implementation": {
            "python": platform.python_version(),
            "warp": wp.__version__,
            "device": str(wp.get_device(args.device)),
        },
        "frozen_protocol": {
            "hard_horizons_ticks": list(V03_HARD_HORIZONS),
            "cached_ticks": list(range(1, 13)),
            "tolerances": TOLERANCES,
            "long_horizons": "diagnostic_only_not_run",
        },
        "initial_state_custody": _initial_state_custody(source_state, cached),
        "counts": {
            "checkpoint_comparisons": len(cases) * len(V03_HARD_HORIZONS),
            "metric_comparisons": (
                len(cases) * len(V03_HARD_HORIZONS) * len(TOLERANCES)
            ),
            "hard_mismatch_events": hard_events,
            "numeric_failure_events": numeric_events,
            "failed_cases": len(failed_cases),
            "actual_target_contact_cases": int(np.count_nonzero(observed_target)),
            "any_retained_contact_cases": int(np.count_nonzero(observed_any_contact)),
        },
        "pass_fail_by_horizon": pass_by_horizon,
        "numeric_error_distributions": _distributions(metric_values),
        "coverage": _coverage(
            geometry, catalog, cases, observed_target, observed_any_contact, max_contacts
        ),
        "failures": failure_records,
        "gate": {
            "selected_run_pass": selected_pass,
            "selection_complete": full_selection,
            "phase_a_complete_gate_pass": full_selection and selected_pass,
            "unresolved_hard_or_numeric_failures": hard_events + numeric_events,
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))
    return 0 if selected_pass else 1


def _numeric_errors(
    state: Any, cached: dict[str, np.ndarray], tick_index: int
) -> dict[str, np.ndarray]:
    sim_matrix = quat_to_matrix(state.ball_quat)
    native_matrix = cached["ball_matrix"][:, tick_index]
    trace = np.einsum(
        "nij,nij->n", sim_matrix.astype(np.float64), native_matrix.astype(np.float64)
    )
    return {
        "position_uu": np.linalg.norm(
            state.ball_pos.astype(np.float64)
            - cached["ball_pos"][:, tick_index].astype(np.float64),
            axis=1,
        ),
        "linear_velocity_uu_per_s": np.linalg.norm(
            state.ball_vel.astype(np.float64)
            - cached["ball_vel"][:, tick_index].astype(np.float64),
            axis=1,
        ),
        "orientation_rad": np.arccos(np.clip((trace - 1.0) * 0.5, -1.0, 1.0)),
        "angular_velocity_rad_per_s": np.linalg.norm(
            state.ball_ang_vel.astype(np.float64)
            - cached["ball_ang_vel"][:, tick_index].astype(np.float64),
            axis=1,
        ),
    }


def _finite_state(state: Any) -> np.ndarray:
    values = np.column_stack(
        (
            state.ball_pos,
            state.ball_vel,
            state.ball_quat,
            state.ball_ang_vel,
        )
    )
    return np.isfinite(values).all(axis=1)


def _target_contact_observed(case: BallWorldCase, faces: np.ndarray) -> bool:
    if case.case_kind == "analytic_plane":
        return PLANE_FACE[case.analytic_plane] in set(int(face) for face in faces)
    accepted = {case.target_face}
    if case.target_neighbor_face is not None:
        accepted.add(case.target_neighbor_face)
    return any(int(face) in accepted for face in faces)


def _initial_state_custody(
    source: Any, cached: dict[str, np.ndarray]
) -> dict[str, Any]:
    source_matrix = quat_to_matrix(source.ball_quat)
    native_matrix = cached["initial_ball_matrix"]
    trace = np.einsum(
        "nij,nij->n", source_matrix.astype(np.float64), native_matrix.astype(np.float64)
    )
    return {
        "source_state_count": source.num_envs,
        "native_readback_count": len(native_matrix),
        "all_native_readback_finite": bool(
            all(
                np.isfinite(cached[field]).all()
                for field in (
                    "initial_ball_pos",
                    "initial_ball_vel",
                    "initial_ball_matrix",
                    "initial_ball_ang_vel",
                )
            )
        ),
        "max_source_to_native_readback": {
            "position_uu": float(
                np.linalg.norm(source.ball_pos - cached["initial_ball_pos"], axis=1).max()
            ),
            "linear_velocity_uu_per_s": float(
                np.linalg.norm(source.ball_vel - cached["initial_ball_vel"], axis=1).max()
            ),
            "orientation_rad": float(
                np.arccos(np.clip((trace - 1.0) * 0.5, -1.0, 1.0)).max()
            ),
            "angular_velocity_rad_per_s": float(
                np.linalg.norm(
                    source.ball_ang_vel - cached["initial_ball_ang_vel"], axis=1
                ).max()
            ),
        },
    }


def _distributions(values: dict[str, list[np.ndarray]]) -> dict[str, Any]:
    result = {}
    for metric, chunks in sorted(values.items()):
        array = np.concatenate(chunks).astype(np.float64)
        result[metric] = {
            "p50": float(np.percentile(array, 50)),
            "p95": float(np.percentile(array, 95)),
            "p99": float(np.percentile(array, 99)),
            "max": float(array.max()),
            "tolerance": TOLERANCES[metric],
            "max_tolerance_fraction": float(array.max() / TOLERANCES[metric]),
        }
    return result


def _coverage(
    geometry: ArenaGeometry,
    catalog: Any,
    cases: tuple[BallWorldCase, ...],
    target: np.ndarray,
    any_contact: np.ndarray,
    max_contacts: np.ndarray,
) -> dict[str, Any]:
    per_mesh: dict[str, Counter[str]] = defaultdict(Counter)
    edge_class: dict[str, Counter[str]] = defaultdict(Counter)
    analytic: dict[str, Counter[str]] = defaultdict(Counter)
    modes: dict[str, Counter[str]] = defaultdict(Counter)
    generated_faces: set[int] = set()
    contacted_faces: set[int] = set()
    generated_edges: dict[str, set[str]] = defaultdict(set)
    contacted_edges: dict[str, set[str]] = defaultdict(set)
    missing_examples = []
    for index, case in enumerate(cases):
        contacted = bool(target[index])
        modes[case.mode]["generated"] += 1
        modes[case.mode]["target_contact"] += contacted
        if case.mesh_file is not None:
            per_mesh[case.mesh_file]["generated"] += 1
            per_mesh[case.mesh_file]["target_contact"] += contacted
        if case.case_kind == "triangle_face":
            generated_faces.add(int(case.target_face))
            if contacted:
                contacted_faces.add(int(case.target_face))
        elif case.case_kind == "shared_directed_edge":
            generated_edges[case.edge_class].add(case.case_id)
            if contacted:
                contacted_edges[case.edge_class].add(case.case_id)
            edge_class[case.edge_class]["generated"] += 1
            edge_class[case.edge_class]["target_contact"] += contacted
        else:
            analytic[case.analytic_plane]["generated"] += 1
            analytic[case.analytic_plane]["target_contact"] += contacted
        if not contacted and len(missing_examples) < 50:
            missing_examples.append(
                {
                    "case_id": case.case_id,
                    "kind": case.case_kind,
                    "mode": case.mode,
                    "reason": (
                        "other_contact_observed"
                        if any_contact[index]
                        else "no_retained_target_contact"
                    ),
                }
            )
    return {
        "geometry_topology_audited": {
            "cmf_files": len(geometry.meshes),
            "triangles": len(catalog.triangles),
            **catalog.topology_counts,
        },
        "starting_states_generated": len(cases),
        "triangle_coverage": {
            "generated_unique": len(generated_faces),
            "target_contact_unique": len(contacted_faces),
            "arena_total": len(catalog.triangles),
        },
        "shared_edge_coverage": {
            edge: {
                **dict(edge_class[edge]),
                "generated_directed_edges": len(generated_edges[edge]),
                "contacted_directed_edges": len(contacted_edges[edge]),
            }
            for edge in sorted(edge_class)
        },
        "analytic_plane_coverage": {
            key: dict(value) for key, value in sorted(analytic.items())
        },
        "per_cmf": {key: dict(value) for key, value in sorted(per_mesh.items())},
        "motion_families": {key: dict(value) for key, value in sorted(modes.items())},
        "multi_face_or_corner_contact_cases": int(np.count_nonzero(max_contacts >= 2)),
        "unexercised_target_cases": int(np.count_nonzero(~target)),
        "unexercised_target_examples": missing_examples,
    }


def _case_metadata(case: BallWorldCase, corpus_index: int) -> dict[str, Any]:
    return {
        "corpus_index": corpus_index,
        "case_id": case.case_id,
        "kind": case.case_kind,
        "family": case.family,
        "mode": case.mode,
        "mesh_file": case.mesh_file,
        "target_face": case.target_face,
        "target_neighbor_face": case.target_neighbor_face,
        "edge_class": case.edge_class,
        "analytic_plane": case.analytic_plane,
    }


if __name__ == "__main__":
    raise SystemExit(main())
