"""Run cached-native v0.3 Phase C Octane/Octane parity acceptance."""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import warp as wp

from rivalsim.arena import ArenaGeometry, WarpArenaMeshes
from rivalsim.math import quat_to_matrix
from rivalsim.static_world import CarCarWorldSim
from rivalsim.v03_oracle_cache import EXPECTED_SOCCAR_CMF_SHA256
from rivalsim.v03_phase_c_cache import (
    PHASE_C_NATIVE_BRANCHES,
    build_phase_c_identity,
    load_frozen_phase_c_identity,
    load_phase_c_frames,
)
from rivalsim.v03_phase_c_corpus import (
    PHASE_C_HARD_HORIZONS,
    generate_phase_c_cases,
    phase_c_cases_to_state,
    phase_c_corpus_sha256,
    phase_c_representative_indices,
    phase_c_selection_sha256,
)

TOLERANCES = {
    "position_uu": 10.0,
    "linear_velocity_uu_per_s": 25.0,
    "orientation_rad": 0.025,
    "angular_velocity_rad_per_s": 0.1,
    "boost": 0.01,
    "handbrake": 0.0001,
    "world_contact_normal_rad": 0.05,
    "contact_cooldown_s": 0.00001,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--collision-dir", required=True)
    parser.add_argument("--oracle-cache-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument(
        "--authority-identity",
        help="select an immutable cached authority by its SHA-256 identity",
    )
    parser.add_argument("--full", action="store_true")
    parser.add_argument("--representative-base-count", type=int, default=1024)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    wp.init()
    geometry = ArenaGeometry.load_soccar(args.collision_dir)
    if geometry.content_sha256 != EXPECTED_SOCCAR_CMF_SHA256:
        raise RuntimeError("unexpected Soccar collision geometry")
    all_cases = generate_phase_c_cases()
    identity = (
        build_phase_c_identity(geometry, all_cases)
        if args.authority_identity is None
        else load_frozen_phase_c_identity(
            args.oracle_cache_root,
            args.authority_identity,
            geometry,
            all_cases,
        )
    )
    indices = (
        tuple(range(len(all_cases)))
        if args.full
        else phase_c_representative_indices(
            all_cases, args.representative_base_count
        )
    )
    cases = tuple(all_cases[index] for index in indices)
    cached = load_phase_c_frames(
        args.oracle_cache_root, identity, all_cases, indices
    )
    branch_count = len(PHASE_C_NATIVE_BRANCHES)
    runtime_cases = tuple(case for case in cases for _branch in PHASE_C_NATIVE_BRANCHES)
    runtime_orders = np.tile(
        np.arange(branch_count, dtype=np.int32), len(cases)
    )
    sim = CarCarWorldSim(
        len(runtime_cases),
        args.collision_dir,
        variant="B3",
        device=args.device,
        initial=phase_c_cases_to_state(runtime_cases),
        geometry=geometry,
        meshes=WarpArenaMeshes(geometry, args.device),
        car_visitation_order=runtime_orders,
    )
    metric_maxima: dict[str, dict[str, float]] = {
        branch: defaultdict(float) for branch in PHASE_C_NATIVE_BRANCHES
    }
    pass_by_horizon = {
        str(tick): {"pass": 0, "fail": 0} for tick in PHASE_C_HARD_HORIZONS
    }
    branch_valid = np.ones((len(cases), branch_count), dtype=bool)
    first_branch_failures: dict[tuple[int, int], dict[str, Any]] = {}
    raw_hard_events = np.zeros(branch_count, dtype=np.int64)
    raw_numeric_events = np.zeros(branch_count, dtype=np.int64)
    exact_native_events = np.zeros(branch_count, dtype=np.int64)
    exact_gpu_events = np.zeros(branch_count, dtype=np.int64)
    exact_matched_events = np.zeros(branch_count, dtype=np.int64)
    demo_masked_car_frames = 0

    for tick in range(1, max(PHASE_C_HARD_HORIZONS) + 1):
        sim.step(1)
        pair = sim.car_car.snapshot()
        authority_index = tick - 1
        native_counts = cached["bump_event_count"][:, :, authority_index]
        gpu_counts = pair.event_count.reshape(len(cases), branch_count)
        exact_native_events += native_counts.sum(axis=0, dtype=np.int64)
        exact_gpu_events += gpu_counts.sum(axis=0, dtype=np.int64)
        event_match = _event_matches(pair, cached, authority_index)
        for branch_index in range(branch_count):
            exact_matched_events[branch_index] += int(
                native_counts[:, branch_index][event_match[:, branch_index]].sum()
            )
        event_mismatch = ~event_match
        raw_hard_events += event_mismatch.sum(axis=0, dtype=np.int64)
        for case_index, branch_index in np.argwhere(event_mismatch):
            _remember_branch_failure(
                first_branch_failures,
                int(case_index),
                int(branch_index),
                tick,
                ["ordered_bump_demo_event_stream"],
                [],
                {},
            )
        branch_valid &= event_match

        if tick not in PHASE_C_HARD_HORIZONS:
            continue

        state = sim.snapshot()
        vehicle = sim.vehicle_snapshot()
        numeric = _numeric_errors(state, vehicle, cached, authority_index)
        # Demo removal is a v0.4 game-rule consequence.  Once either car has
        # been demolished, RocketSim removes that body and the surviving car's
        # later physical world is no longer the bounded v0.3 two-body system.
        # Keep the ordered demo predicate/event and demo flags as hard gates,
        # but stop gating numeric/vehicle state for the whole environment from
        # that frame onward.
        pre_demo_environment = ~np.any(
            cached["car_is_demoed"][:, :, authority_index],
            axis=2,
            keepdims=True,
        )
        active_numeric = np.broadcast_to(
            pre_demo_environment,
            cached["car_is_demoed"][:, :, authority_index].shape,
        )
        demo_masked_car_frames += int(np.count_nonzero(~active_numeric))
        for metric, values in numeric.items():
            for branch_index, branch in enumerate(PHASE_C_NATIVE_BRANCHES):
                active = values[:, branch_index][active_numeric[:, branch_index]]
                if active.size:
                    metric_maxima[branch][metric] = max(
                        metric_maxima[branch][metric], float(np.max(active))
                    )
        finite = np.isfinite(
            np.concatenate((state.car_pos, state.car_vel, state.car_ang_vel), axis=2)
        ).all(axis=(1, 2)).reshape(len(cases), branch_count)
        finite &= np.isfinite(state.car_quat).all(axis=(1, 2)).reshape(
            len(cases), branch_count
        )
        overflow = (
            vehicle.mesh_candidate_overflow.reshape(len(cases), branch_count, 2) != 0
        ).any(axis=2) | (
            vehicle.contact_overflow.reshape(len(cases), branch_count, 2) != 0
        ).any(axis=2)
        resident_order = pair.pre_tick_first_car.reshape(len(cases), branch_count)

        for case_index in range(len(cases)):
            for branch_index in range(branch_count):
                hard: list[str] = []
                if event_mismatch[case_index, branch_index]:
                    hard.append("ordered_bump_demo_event_stream")
                if not finite[case_index, branch_index]:
                    hard.append("non_finite_state")
                if overflow[case_index, branch_index]:
                    hard.append("car_world_contact_overflow")
                if resident_order[case_index, branch_index] != branch_index:
                    hard.append("car_pre_tick_lifecycle_order")
                hard.extend(
                    _semantic_mismatches(
                        state,
                        vehicle,
                        pair,
                        cached,
                        authority_index,
                        case_index,
                        branch_index,
                        branch_count,
                    )
                )
                numeric_failures = []
                errors: dict[str, float] = {}
                for metric, values in numeric.items():
                    car_values = values[case_index, branch_index]
                    errors[metric] = float(np.max(car_values))
                    active = active_numeric[case_index, branch_index]
                    if np.any(active) and float(np.max(car_values[active])) > TOLERANCES[
                        _tolerance_key(metric)
                    ]:
                        numeric_failures.append(metric)
                raw_hard_events[branch_index] += len(hard) - int(
                    event_mismatch[case_index, branch_index]
                )
                raw_numeric_events[branch_index] += len(numeric_failures)
                passed = not hard and not numeric_failures
                branch_valid[case_index, branch_index] &= passed
                if not passed:
                    _remember_branch_failure(
                        first_branch_failures,
                        case_index,
                        branch_index,
                        tick,
                        hard,
                        numeric_failures,
                        errors,
                    )
        relation_at_horizon = np.any(branch_valid, axis=1)
        pass_by_horizon[str(tick)]["pass"] = int(relation_at_horizon.sum())
        pass_by_horizon[str(tick)]["fail"] = int((~relation_at_horizon).sum())

    relation_pass = np.any(branch_valid, axis=1)
    selected_pass = bool(np.all(relation_pass))
    full_selection = len(indices) == len(all_cases)
    failures = _relation_failures(
        cases,
        indices,
        relation_pass,
        first_branch_failures,
    )
    branch_diagnostics = {
        branch: {
            "complete_trajectory_pass_count": int(branch_valid[:, branch_index].sum()),
            "complete_trajectory_fail_count": int((~branch_valid[:, branch_index]).sum()),
            "raw_hard_mismatch_count": int(raw_hard_events[branch_index]),
            "raw_numeric_failure_count": int(raw_numeric_events[branch_index]),
        }
        for branch_index, branch in enumerate(PHASE_C_NATIVE_BRANCHES)
    }
    report = {
        "schema_version": 2,
        "milestone": "v0.3",
        "phase": "C_car_car",
        "generated_at_utc": datetime.now(UTC).isoformat(),
        "classification": (
            "PASS_GREEN"
            if full_selection and selected_pass
            else "REPRESENTATIVE_PASS"
            if selected_pass
            else "BLOCKING_FAILURE"
        ),
        "authority_identity_sha256": identity["authority_identity_sha256"],
        "corpus_sha256": phase_c_corpus_sha256(all_cases),
        "selection": {
            "kind": "complete" if full_selection else "deterministic_representative",
            "selected_case_count": len(indices),
            "full_case_count": len(all_cases),
            "selection_sha256": phase_c_selection_sha256(indices),
        },
        "tolerances": TOLERANCES,
        "native_multi_outcome_relation": {
            "branches": list(PHASE_C_NATIVE_BRANCHES),
            "comparison": "same-labeled complete trajectory only",
            "metric_branch_mixing": False,
            "runtime_best_match_selection": False,
            "case_acceptance": "at least one complete native-valid branch",
            "both_branches_complete_count": int(np.all(branch_valid, axis=1).sum()),
            "one_branch_complete_count": int(
                np.count_nonzero(np.sum(branch_valid, axis=1) == 1)
            ),
            "no_branch_complete_count": int((~relation_pass).sum()),
        },
        "pass_by_horizon": pass_by_horizon,
        "blocking": {
            "failed_case_count": int((~relation_pass).sum()),
            "relation_failure_count": int((~relation_pass).sum()),
        },
        "branch_diagnostics_non_blocking_when_alternate_branch_is_complete": (
            branch_diagnostics
        ),
        "ordered_bump_demo_events_by_branch": {
            branch: {
                "native": int(exact_native_events[branch_index]),
                "gpu": int(exact_gpu_events[branch_index]),
                "matched_native_events_in_exact_frames": int(
                    exact_matched_events[branch_index]
                ),
            }
            for branch_index, branch in enumerate(PHASE_C_NATIVE_BRANCHES)
        },
        "v0_4_boundary": {
            "demo_removal_respawn_not_implemented": True,
            "numeric_car_frames_masked_at_or_after_native_demo": demo_masked_car_frames,
            "demo_predicate_and_ordered_event_remain_hard_gates": True,
        },
        "metric_maxima_non_demo_by_branch": {
            branch: dict(sorted(values.items()))
            for branch, values in metric_maxima.items()
        },
        "failures": failures,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))
    return 0 if selected_pass else 1


def _event_matches(pair: Any, cached: dict[str, np.ndarray], tick: int) -> np.ndarray:
    case_count, branch_count = cached["bump_event_count"].shape[:2]
    count = pair.event_count.reshape(case_count, branch_count)
    native_count = cached["bump_event_count"][:, :, tick]
    result = count == native_count
    bumper = pair.event_bumper.reshape(case_count, branch_count, -1)
    victim = pair.event_victim.reshape(case_count, branch_count, -1)
    is_demo = pair.event_is_demo.reshape(case_count, branch_count, -1)
    for index in range(bumper.shape[2]):
        active = index < native_count
        result &= ~active | (
            (bumper[:, :, index] == cached["bump_event_bumper"][:, :, tick, index])
            & (victim[:, :, index] == cached["bump_event_victim"][:, :, tick, index])
            & (
                is_demo[:, :, index].astype(bool)
                == cached["bump_event_is_demo"][:, :, tick, index]
            )
        )
    return result


def _numeric_errors(
    state: Any, vehicle: Any, cached: dict[str, np.ndarray], tick: int
) -> dict[str, np.ndarray]:
    case_count, branch_count = cached["car_pos"].shape[:2]
    matrix = quat_to_matrix(state.car_quat.reshape(-1, 4)).reshape(
        case_count, branch_count, 2, 3, 3
    )
    native_matrix = cached["car_matrix"][:, :, tick]
    relative_trace = np.einsum(
        "nbcij,nbcij->nbc",
        matrix.astype(np.float64),
        native_matrix.astype(np.float64),
    )
    position = state.car_pos.reshape(case_count, branch_count, 2, 3)
    velocity = state.car_vel.reshape(case_count, branch_count, 2, 3)
    angular = state.car_ang_vel.reshape(case_count, branch_count, 2, 3)
    boost = state.boost.reshape(case_count, branch_count, 2)
    return {
        "car_position_uu": np.linalg.norm(
            position.astype(np.float64)
            - cached["car_pos"][:, :, tick].astype(np.float64),
            axis=3,
        ),
        "car_linear_velocity_uu_per_s": np.linalg.norm(
            velocity.astype(np.float64)
            - cached["car_vel"][:, :, tick].astype(np.float64),
            axis=3,
        ),
        "car_orientation_rad": np.arccos(
            np.clip((relative_trace - 1.0) * 0.5, -1.0, 1.0)
        ),
        "car_angular_velocity_rad_per_s": np.linalg.norm(
            angular.astype(np.float64)
            - cached["car_ang_vel"][:, :, tick].astype(np.float64),
            axis=3,
        ),
        "car_boost": np.abs(
            boost.astype(np.float64)
            - cached["car_boost"][:, :, tick].astype(np.float64)
        ),
        "car_handbrake": np.abs(
            vehicle.handbrake_value.reshape(case_count, branch_count, 2).astype(
                np.float64
            )
            - cached["car_handbrake"][:, :, tick].astype(np.float64)
        ),
    }


def _semantic_mismatches(
    state: Any,
    vehicle: Any,
    pair: Any,
    cached: dict[str, np.ndarray],
    tick: int,
    case: int,
    branch: int,
    branch_count: int,
) -> list[str]:
    mismatches: list[str] = []
    active = np.full(
        2,
        not np.any(cached["car_is_demoed"][case, branch, tick]),
        dtype=bool,
    )
    runtime_world = case * branch_count + branch
    flat = slice(runtime_world * 2, runtime_world * 2 + 2)
    if not np.array_equal(
        state.on_ground[runtime_world].astype(bool)[active],
        cached["car_on_ground"][case, branch, tick][active],
    ):
        mismatches.append("car_on_ground")
    if not np.array_equal(
        vehicle.wheel_contact[flat].astype(bool)[active],
        cached["car_wheel_contacts"][case, branch, tick][active],
    ):
        mismatches.append("car_wheel_contacts")
    gpu_world = np.linalg.norm(vehicle.world_contact_normal[flat], axis=1) > 0.5
    if not np.array_equal(
        gpu_world[active], cached["car_world_contact"][case, branch, tick][active]
    ):
        mismatches.append("car_world_contact")
    if not np.array_equal(
        state.is_supersonic[runtime_world].astype(bool)[active],
        cached["car_is_supersonic"][case, branch, tick][active],
    ):
        mismatches.append("car_is_supersonic")
    if not np.array_equal(
        pair.car_is_demoed[runtime_world].astype(bool),
        cached["car_is_demoed"][case, branch, tick],
    ):
        mismatches.append("car_is_demoed")
    native_cooldown = cached["car_contact_cooldown"][case, branch, tick]
    cooldown_active = native_cooldown > 0.0
    if not np.array_equal(
        pair.car_contact_id[runtime_world][cooldown_active],
        cached["car_contact_id"][case, branch, tick][cooldown_active].astype(np.int32),
    ):
        mismatches.append("car_contact_id")
    if np.any(
        np.abs(pair.car_contact_cooldown[runtime_world] - native_cooldown)
        > TOLERANCES["contact_cooldown_s"]
    ):
        mismatches.append("car_contact_cooldown")
    return mismatches


def _tolerance_key(metric: str) -> str:
    for key in TOLERANCES:
        if metric.endswith(key):
            return key
    raise KeyError(metric)


def _remember_branch_failure(
    failures: dict[tuple[int, int], dict[str, Any]],
    case_index: int,
    branch_index: int,
    tick: int,
    hard: list[str],
    numeric: list[str],
    errors: dict[str, float],
) -> None:
    key = (case_index, branch_index)
    if key in failures:
        return
    failures[key] = {
        "branch": PHASE_C_NATIVE_BRANCHES[branch_index],
        "first_failed_tick": tick,
        "hard_mismatches": hard,
        "numeric_failures": numeric,
        "errors": errors,
    }


def _relation_failures(
    cases: tuple[Any, ...],
    indices: tuple[int, ...],
    relation_pass: np.ndarray,
    branch_failures: dict[tuple[int, int], dict[str, Any]],
) -> list[dict[str, Any]]:
    failures: list[dict[str, Any]] = []
    for case_index in np.flatnonzero(~relation_pass):
        if len(failures) >= 300:
            break
        case = cases[int(case_index)]
        failures.append(
            {
                "case_id": case.case_id,
                "corpus_index": indices[int(case_index)],
                "contact_feature": case.contact_feature,
                "orientation_mode": case.orientation_mode,
                "motion_mode": case.motion_mode,
                "static_context": case.static_context,
                "complete_branch_failures": [
                    branch_failures[(int(case_index), branch_index)]
                    for branch_index in range(len(PHASE_C_NATIVE_BRANCHES))
                ],
            }
        )
    return failures


if __name__ == "__main__":
    raise SystemExit(main())
