"""Run cached-native v0.3 Phase B Octane/ball parity acceptance."""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
from collections import Counter, defaultdict
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import warp as wp

from rivalsim.arena import ArenaGeometry, WarpArenaMeshes
from rivalsim.math import quat_to_matrix
from rivalsim.static_world import DynamicWorldSim
from rivalsim.v03_corpus import (
    V03_HARD_HORIZONS,
    CarBallCase,
    generate_phase_b_cases,
    phase_b_cases_to_state,
    phase_b_corpus_sha256,
    phase_b_representative_indices,
)
from rivalsim.v03_oracle_cache import EXPECTED_SOCCAR_CMF_SHA256
from rivalsim.v03_phase_b_cache import build_phase_b_identity, load_phase_b_frames

SCHEMA_VERSION = 1
TOLERANCES = {
    "position_uu": 10.0,
    "linear_velocity_uu_per_s": 25.0,
    "orientation_rad": 0.025,
    "angular_velocity_rad_per_s": 0.1,
    "boost": 0.01,
    "handbrake": 0.0001,
    "world_contact_normal_rad": 0.05,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--collision-dir", required=True)
    parser.add_argument("--oracle-cache-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--full", action="store_true")
    parser.add_argument("--representative-base-count", type=int, default=1024)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    wp.init()
    geometry = ArenaGeometry.load_soccar(args.collision_dir)
    if geometry.content_sha256 != EXPECTED_SOCCAR_CMF_SHA256:
        raise RuntimeError(
            f"unexpected Soccar geometry {geometry.content_sha256}; "
            f"expected {EXPECTED_SOCCAR_CMF_SHA256}"
        )
    all_cases = generate_phase_b_cases()
    identity = build_phase_b_identity(geometry, all_cases)
    indices = (
        tuple(range(len(all_cases)))
        if args.full
        else phase_b_representative_indices(
            all_cases, args.representative_base_count
        )
    )
    cases = tuple(all_cases[index] for index in indices)
    cached = load_phase_b_frames(
        args.oracle_cache_root, identity, all_cases, indices
    )
    source_state = phase_b_cases_to_state(cases)
    sim = DynamicWorldSim(
        len(cases),
        args.collision_dir,
        variant="B3",
        device=args.device,
        initial=source_state,
        geometry=geometry,
        meshes=WarpArenaMeshes(geometry, args.device),
    )

    metric_values: dict[str, list[np.ndarray]] = defaultdict(list)
    callback_extra_errors: list[np.ndarray] = []
    callback_position_errors: list[np.ndarray] = []
    failures: list[dict[str, Any]] = []
    failed_cases: set[str] = set()
    pass_by_horizon = {
        str(tick): {"pass": 0, "fail": 0} for tick in V03_HARD_HORIZONS
    }
    context_by_horizon: dict[str, dict[str, Counter[str]]] = {
        str(tick): defaultdict(Counter) for tick in V03_HARD_HORIZONS
    }
    hard_events = 0
    numeric_events = 0
    gpu_hit_ever = np.zeros(len(cases), dtype=bool)
    native_hit_current_count = 0
    gpu_hit_current_count = 0
    matched_hit_current_count = 0
    car_indices = np.arange(len(cases), dtype=np.int64) * 2

    for tick in range(1, max(V03_HARD_HORIZONS) + 1):
        sim.step(1)
        pair = sim.car_ball.snapshot()
        gpu_hit_current = pair.hit_this_tick != 0
        gpu_hit_ever |= gpu_hit_current
        authority_index = tick - 1
        native_hit_current = cached["pair_hit_valid"][:, authority_index] & (
            cached["pair_hit_tick"][:, authority_index]
            == np.uint64(tick - 1)
        )
        native_hit_current_count += int(np.count_nonzero(native_hit_current))
        gpu_hit_current_count += int(np.count_nonzero(gpu_hit_current))
        matched_hit_current_count += int(
            np.count_nonzero(native_hit_current & gpu_hit_current)
        )
        active_hits = native_hit_current & gpu_hit_current
        extra_error = np.zeros(len(cases), dtype=np.float64)
        position_error = np.zeros(len(cases), dtype=np.float64)
        if np.any(active_hits):
            extra_error[active_hits] = np.linalg.norm(
                pair.extra_hit_velocity_uu[active_hits].astype(np.float64)
                - cached["pair_extra_hit_vel"][
                    active_hits, authority_index
                ].astype(np.float64),
                axis=1,
            )
            position_error[active_hits] = np.linalg.norm(
                pair.relative_pos_on_ball_uu[active_hits].astype(np.float64)
                - cached["pair_relative_pos_on_ball"][
                    active_hits, authority_index
                ].astype(np.float64),
                axis=1,
            )
        callback_extra_errors.append(extra_error[active_hits])
        callback_position_errors.append(position_error[active_hits])

        if tick not in V03_HARD_HORIZONS:
            continue
        state = sim.snapshot()
        vehicle = sim.vehicle_snapshot()
        errors = _numeric_errors(state, vehicle, car_indices, cached, authority_index)
        for metric, values in errors.items():
            metric_values[metric].append(values)
        finite = _finite_state(state)
        candidate_overflow = sim.ball_world.candidate_overflow.numpy() != 0
        contact_overflow = sim.ball_world.contact_overflow.numpy() != 0
        car_candidate_overflow = vehicle.mesh_candidate_overflow[car_indices] != 0
        car_contact_overflow = vehicle.contact_overflow[car_indices] != 0
        gpu_last_hit = gpu_hit_ever
        native_last_hit = cached["pair_hit_valid"][:, authority_index]
        car_wrong_direction = _wrong_direction(
            state.car_vel[:, 0], cached["car_vel"][:, authority_index]
        )
        ball_wrong_direction = _wrong_direction(
            state.ball_vel, cached["ball_vel"][:, authority_index]
        )

        for case_index, case in enumerate(cases):
            hard: list[str] = []
            if not finite[case_index]:
                hard.append("non_finite_state")
            if candidate_overflow[case_index]:
                hard.append("ball_world_candidate_overflow")
            if contact_overflow[case_index]:
                hard.append("ball_world_contact_overflow")
            if car_candidate_overflow[case_index]:
                hard.append("car_world_candidate_overflow")
            if car_contact_overflow[case_index]:
                hard.append("car_world_contact_overflow")
            if gpu_hit_current[case_index] != native_hit_current[case_index]:
                hard.append("car_ball_callback_presence")
            if gpu_last_hit[case_index] != native_last_hit[case_index]:
                hard.append("car_ball_last_hit_presence")
            if car_wrong_direction[case_index]:
                hard.append("car_linear_velocity_direction")
            if ball_wrong_direction[case_index]:
                hard.append("ball_linear_velocity_direction")
            hard.extend(
                _car_semantic_mismatches(
                    state,
                    vehicle,
                    car_indices,
                    cached,
                    authority_index,
                    case_index,
                )
            )
            numeric = [
                metric
                for metric, values in errors.items()
                if float(values[case_index])
                > TOLERANCES[_metric_tolerance(metric)]
            ]
            passed = not hard and not numeric
            pass_by_horizon[str(tick)]["pass" if passed else "fail"] += 1
            context_by_horizon[str(tick)][case.static_context][
                "pass" if passed else "fail"
            ] += 1
            hard_events += len(hard)
            numeric_events += len(numeric)
            if not passed:
                failed_cases.add(case.case_id)
                if len(failures) < 300:
                    failures.append(
                        {
                            **_case_metadata(case, indices[case_index]),
                            "tick": tick,
                            "hard_mismatches": hard,
                            "numeric_failures": numeric,
                            "errors": {
                                metric: float(values[case_index])
                                for metric, values in errors.items()
                            },
                            "native_hit_current": bool(
                                native_hit_current[case_index]
                            ),
                            "gpu_hit_current": bool(gpu_hit_current[case_index]),
                            "callback_extra_velocity_error_uu_per_s": float(
                                extra_error[case_index]
                            ),
                            "callback_relative_position_error_uu": float(
                                position_error[case_index]
                            ),
                        }
                    )

    selected_pass = hard_events == 0 and numeric_events == 0
    full_selection = len(indices) == len(all_cases)
    report = {
        "schema_version": SCHEMA_VERSION,
        "milestone": "v0.3",
        "phase": "B_car_ball",
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
            "corpus_sha256": phase_b_corpus_sha256(all_cases),
            "selection_sha256": _selection_sha256(all_cases, indices),
        },
        "authority": {
            "execution": "cached_native_authority_only_no_live_fallback",
            "identity_sha256": identity["authority_identity_sha256"],
            "cache_format_version": identity["cache_format_version"],
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
        },
        "initial_state_custody": _initial_state_custody(source_state, cached),
        "counts": {
            "checkpoint_comparisons": len(cases) * len(V03_HARD_HORIZONS),
            "hard_mismatch_events": hard_events,
            "numeric_failure_events": numeric_events,
            "failed_cases": len(failed_cases),
            "native_callback_frames": native_hit_current_count,
            "gpu_callback_frames": gpu_hit_current_count,
            "matched_callback_frames": matched_hit_current_count,
            "native_actual_contact_cases": int(
                np.count_nonzero(cached["pair_hit_valid"][:, -1])
            ),
            "gpu_actual_contact_cases": int(np.count_nonzero(gpu_hit_ever)),
        },
        "pass_fail_by_horizon": pass_by_horizon,
        "pass_fail_by_static_context_horizon": {
            tick: {
                context: dict(counts)
                for context, counts in sorted(contexts.items())
            }
            for tick, contexts in context_by_horizon.items()
        },
        "numeric_error_distributions": _distributions(metric_values),
        "callback_diagnostics": {
            "extra_hit_velocity_uu_per_s": _array_distribution(
                callback_extra_errors
            ),
            "relative_position_on_ball_uu": _array_distribution(
                callback_position_errors
            ),
        },
        "coverage": _coverage(cases, cached, gpu_hit_ever),
        "failures": failures,
        "gate": {
            "selected_run_pass": selected_pass,
            "selection_complete": full_selection,
            "phase_b_complete_gate_pass": full_selection and selected_pass,
            "unresolved_hard_or_numeric_failures": hard_events + numeric_events,
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))
    return 0 if selected_pass else 1


def _numeric_errors(
    state: Any,
    vehicle: Any,
    car_indices: np.ndarray,
    cached: dict[str, np.ndarray],
    tick_index: int,
) -> dict[str, np.ndarray]:
    result: dict[str, np.ndarray] = {}
    for body, position, velocity, quaternion, angular in (
        (
            "car",
            state.car_pos[:, 0],
            state.car_vel[:, 0],
            state.car_quat[:, 0],
            state.car_ang_vel[:, 0],
        ),
        (
            "ball",
            state.ball_pos,
            state.ball_vel,
            state.ball_quat,
            state.ball_ang_vel,
        ),
    ):
        sim_matrix = quat_to_matrix(quaternion)
        native_matrix = cached[f"{body}_matrix"][:, tick_index]
        trace = np.einsum(
            "nij,nij->n",
            sim_matrix.astype(np.float64),
            native_matrix.astype(np.float64),
        )
        result[f"{body}_position_uu"] = np.linalg.norm(
            position.astype(np.float64)
            - cached[f"{body}_pos"][:, tick_index].astype(np.float64),
            axis=1,
        )
        result[f"{body}_linear_velocity_uu_per_s"] = np.linalg.norm(
            velocity.astype(np.float64)
            - cached[f"{body}_vel"][:, tick_index].astype(np.float64),
            axis=1,
        )
        result[f"{body}_orientation_rad"] = np.arccos(
            np.clip((trace - 1.0) * 0.5, -1.0, 1.0)
        )
        result[f"{body}_angular_velocity_rad_per_s"] = np.linalg.norm(
            angular.astype(np.float64)
            - cached[f"{body}_ang_vel"][:, tick_index].astype(np.float64),
            axis=1,
        )
    result["car_boost"] = np.abs(
        state.boost[:, 0].astype(np.float64)
        - cached["car_boost"][:, tick_index].astype(np.float64)
    )
    result["car_handbrake"] = np.abs(
        vehicle.handbrake_value[car_indices].astype(np.float64)
        - cached["car_handbrake"][:, tick_index].astype(np.float64)
    )
    return result


def _metric_tolerance(metric: str) -> str:
    for suffix in (
        "linear_velocity_uu_per_s",
        "angular_velocity_rad_per_s",
        "position_uu",
        "orientation_rad",
        "boost",
        "handbrake",
    ):
        if metric.endswith(suffix):
            return suffix
    raise KeyError(metric)


def _car_semantic_mismatches(
    state: Any,
    vehicle: Any,
    car_indices: np.ndarray,
    cached: dict[str, np.ndarray],
    tick_index: int,
    case_index: int,
) -> list[str]:
    mismatches = []
    if bool(state.on_ground[case_index, 0]) != bool(
        cached["car_on_ground"][case_index, tick_index]
    ):
        mismatches.append("car_on_ground")
    car_index = int(car_indices[case_index])
    if not np.array_equal(
        vehicle.wheel_contact[car_index].astype(bool),
        cached["car_wheel_contacts"][case_index, tick_index],
    ):
        mismatches.append("car_wheel_contacts")
    gpu_normal = vehicle.world_contact_normal[car_index].astype(np.float64)
    gpu_world = bool(np.linalg.norm(gpu_normal) > 0.5)
    native_world = bool(cached["car_world_contact"][case_index, tick_index])
    if gpu_world != native_world:
        mismatches.append("car_world_contact")
    elif gpu_world:
        a = gpu_normal
        b = cached["car_world_contact_normal"][case_index, tick_index].astype(
            np.float64
        )
        denom = np.linalg.norm(a) * np.linalg.norm(b)
        angle = np.pi if denom == 0.0 else np.arccos(
            np.clip(float(np.dot(a, b) / denom), -1.0, 1.0)
        )
        if angle > TOLERANCES["world_contact_normal_rad"]:
            mismatches.append("car_world_contact_normal")
    return mismatches


def _wrong_direction(sim: np.ndarray, native: np.ndarray) -> np.ndarray:
    sim64 = sim.astype(np.float64)
    native64 = native.astype(np.float64)
    return (
        (np.linalg.norm(sim64, axis=1) > 25.0)
        & (np.linalg.norm(native64, axis=1) > 25.0)
        & (np.sum(sim64 * native64, axis=1) < 0.0)
    )


def _finite_state(state: Any) -> np.ndarray:
    values = np.column_stack(
        (
            state.car_pos[:, 0],
            state.car_vel[:, 0],
            state.car_quat[:, 0],
            state.car_ang_vel[:, 0],
            state.ball_pos,
            state.ball_vel,
            state.ball_quat,
            state.ball_ang_vel,
        )
    )
    return np.isfinite(values).all(axis=1)


def _initial_state_custody(
    source: Any, cached: dict[str, np.ndarray]
) -> dict[str, Any]:
    result: dict[str, Any] = {"source_state_count": source.num_envs}
    for body, position, velocity, quaternion, angular in (
        (
            "car",
            source.car_pos[:, 0],
            source.car_vel[:, 0],
            source.car_quat[:, 0],
            source.car_ang_vel[:, 0],
        ),
        (
            "ball",
            source.ball_pos,
            source.ball_vel,
            source.ball_quat,
            source.ball_ang_vel,
        ),
    ):
        matrix = quat_to_matrix(quaternion)
        native_matrix = cached[f"initial_{body}_matrix"]
        trace = np.einsum(
            "nij,nij->n", matrix.astype(np.float64), native_matrix.astype(np.float64)
        )
        result[body] = {
            "max_position_uu": float(
                np.linalg.norm(position - cached[f"initial_{body}_pos"], axis=1).max()
            ),
            "max_linear_velocity_uu_per_s": float(
                np.linalg.norm(velocity - cached[f"initial_{body}_vel"], axis=1).max()
            ),
            "max_orientation_rad": float(
                np.arccos(np.clip((trace - 1.0) * 0.5, -1.0, 1.0)).max()
            ),
            "max_angular_velocity_rad_per_s": float(
                np.linalg.norm(
                    angular - cached[f"initial_{body}_ang_vel"], axis=1
                ).max()
            ),
        }
    return result


def _distributions(values: dict[str, list[np.ndarray]]) -> dict[str, Any]:
    result = {}
    for metric, chunks in sorted(values.items()):
        array = np.concatenate(chunks).astype(np.float64)
        tolerance = TOLERANCES[_metric_tolerance(metric)]
        result[metric] = {
            **_distribution(array),
            "tolerance": tolerance,
            "max_tolerance_fraction": float(array.max() / tolerance),
        }
    return result


def _array_distribution(chunks: list[np.ndarray]) -> dict[str, Any]:
    nonempty = [chunk for chunk in chunks if len(chunk)]
    if not nonempty:
        return {"count": 0}
    array = np.concatenate(nonempty).astype(np.float64)
    return {"count": len(array), **_distribution(array)}


def _distribution(array: np.ndarray) -> dict[str, float]:
    return {
        "p50": float(np.percentile(array, 50)),
        "p95": float(np.percentile(array, 95)),
        "p99": float(np.percentile(array, 99)),
        "max": float(array.max()),
    }


def _coverage(
    cases: tuple[CarBallCase, ...],
    cached: dict[str, np.ndarray],
    gpu_hit_ever: np.ndarray,
) -> dict[str, Any]:
    native_hit = cached["pair_hit_valid"][:, -1]
    strata: dict[str, dict[str, Counter[str]]] = {
        name: defaultdict(Counter)
        for name in ("contact_region", "motion_mode", "orientation_mode", "static_context")
    }
    for index, case in enumerate(cases):
        for name in strata:
            value = str(getattr(case, name))
            strata[name][value]["generated"] += 1
            strata[name][value]["native_contact"] += bool(native_hit[index])
            strata[name][value]["gpu_contact"] += bool(gpu_hit_ever[index])
    return {
        name: {key: dict(value) for key, value in sorted(groups.items())}
        for name, groups in strata.items()
    }


def _case_metadata(case: CarBallCase, corpus_index: int) -> dict[str, Any]:
    return {
        "corpus_index": corpus_index,
        "case_id": case.case_id,
        "contact_region": case.contact_region,
        "feature_index": case.feature_index,
        "motion_mode": case.motion_mode,
        "orientation_mode": case.orientation_mode,
        "static_context": case.static_context,
        "overlap_uu": case.overlap_uu,
    }


def _selection_sha256(
    cases: tuple[CarBallCase, ...], indices: tuple[int, ...]
) -> str:
    digest = hashlib.sha256()
    for index in indices:
        digest.update(index.to_bytes(4, "little"))
        digest.update(cases[index].case_id.encode())
    return digest.hexdigest().upper()


if __name__ == "__main__":
    raise SystemExit(main())
