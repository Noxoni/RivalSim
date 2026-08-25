"""Run cached-native v0.3 Phase D integrated relational acceptance."""

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
from rivalsim.static_world import IntegratedWorldSim
from rivalsim.v03_oracle_cache import EXPECTED_SOCCAR_CMF_SHA256
from rivalsim.v03_phase_d_cache import (
    PHASE_D_NATIVE_BRANCHES,
    build_phase_d_identity,
    load_frozen_phase_d_identity,
    load_phase_d_frames,
)
from rivalsim.v03_phase_d_corpus import (
    PHASE_D_HARD_HORIZONS,
    generate_phase_d_cases,
    phase_d_cases_to_state,
    phase_d_controls_at,
    phase_d_corpus_sha256,
    phase_d_representative_indices,
    phase_d_selection_sha256,
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
    "callback_extra_velocity_uu_per_s": 25.0,
    "callback_relative_position_uu": 10.0,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--collision-dir", required=True)
    parser.add_argument("--oracle-cache-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--authority-identity")
    parser.add_argument("--full", action="store_true")
    parser.add_argument("--representative-base-count", type=int, default=128)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    wp.init()
    geometry = ArenaGeometry.load_soccar(args.collision_dir)
    if geometry.content_sha256 != EXPECTED_SOCCAR_CMF_SHA256:
        raise RuntimeError("unexpected Soccar collision geometry")
    all_cases = generate_phase_d_cases()
    identity = (
        build_phase_d_identity(geometry, all_cases)
        if args.authority_identity is None
        else load_frozen_phase_d_identity(
            args.oracle_cache_root,
            args.authority_identity,
            geometry,
            all_cases,
        )
    )
    indices = (
        tuple(range(len(all_cases)))
        if args.full
        else phase_d_representative_indices(all_cases, args.representative_base_count)
    )
    cases = tuple(all_cases[index] for index in indices)
    cached = load_phase_d_frames(
        args.oracle_cache_root, identity, all_cases, indices
    )
    branch_count = len(PHASE_D_NATIVE_BRANCHES)
    runtime_cases = tuple(case for case in cases for _branch in PHASE_D_NATIVE_BRANCHES)
    runtime_orders = np.tile(np.arange(branch_count, dtype=np.int32), len(cases))
    sim = IntegratedWorldSim(
        len(runtime_cases),
        args.collision_dir,
        variant="B3",
        device=args.device,
        initial=phase_d_cases_to_state(runtime_cases),
        geometry=geometry,
        meshes=WarpArenaMeshes(geometry, args.device),
        car_visitation_order=runtime_orders,
    )

    valid = np.ones((len(cases), branch_count), dtype=bool)
    first_failures: dict[tuple[int, int], dict[str, Any]] = {}
    metric_maxima: dict[str, dict[str, float]] = {
        branch: defaultdict(float) for branch in PHASE_D_NATIVE_BRANCHES
    }
    pass_by_horizon = {
        str(tick): {"pass": 0, "fail": 0} for tick in PHASE_D_HARD_HORIZONS
    }
    raw_hard = np.zeros(branch_count, dtype=np.int64)
    raw_numeric = np.zeros(branch_count, dtype=np.int64)
    native_car_hits = np.zeros((branch_count, 2), dtype=np.int64)
    gpu_car_hits = np.zeros((branch_count, 2), dtype=np.int64)
    native_bumps = np.zeros(branch_count, dtype=np.int64)
    gpu_bumps = np.zeros(branch_count, dtype=np.int64)
    demo_masked_car_frames = 0

    for tick in range(1, max(PHASE_D_HARD_HORIZONS) + 1):
        sim.set_controls(phase_d_controls_at(runtime_cases, tick - 1))
        sim.step(1)
        state = sim.snapshot()
        vehicle = sim.vehicle_snapshot()
        pair_a = sim.car_ball.snapshot()
        pair_b = sim.car_ball_b.snapshot()
        car_pair = sim.car_car.snapshot()
        authority_index = tick - 1

        native_hit_current = cached["car_ball_hit_valid"][:, :, authority_index] & (
            cached["car_ball_hit_tick"][:, :, authority_index] == np.uint64(tick - 1)
        )
        gpu_hit_current = np.stack(
            (
                pair_a.hit_this_tick.reshape(len(cases), branch_count) != 0,
                pair_b.hit_this_tick.reshape(len(cases), branch_count) != 0,
            ),
            axis=2,
        )
        native_car_hits += native_hit_current.sum(axis=0, dtype=np.int64)
        gpu_car_hits += gpu_hit_current.sum(axis=0, dtype=np.int64)
        hit_match = np.array_equal(native_hit_current, gpu_hit_current)
        del hit_match

        native_event_count = cached["bump_event_count"][:, :, authority_index]
        gpu_event_count = car_pair.event_count.reshape(len(cases), branch_count)
        native_bumps += native_event_count.sum(axis=0, dtype=np.int64)
        gpu_bumps += gpu_event_count.sum(axis=0, dtype=np.int64)
        event_match = _event_matches(car_pair, cached, authority_index, len(cases), branch_count)

        numeric = _numeric_errors(state, vehicle, cached, authority_index, len(cases), branch_count)
        callback_numeric = _callback_errors(
            pair_a,
            pair_b,
            cached,
            authority_index,
            native_hit_current,
            gpu_hit_current,
            len(cases),
            branch_count,
        )
        numeric.update(callback_numeric)
        # Phase C defines demolition removal/respawn as a v0.4 game-rule
        # consequence. Keep event order and demo flags hard, but once either
        # car is removed, no later physical metric in that environment is a
        # bounded v0.3 comparison.
        pre_demo_environment = ~np.any(
            cached["car_is_demoed"][:, :, authority_index],
            axis=2,
        )
        active_numeric = np.broadcast_to(
            pre_demo_environment[:, :, None],
            cached["car_is_demoed"][:, :, authority_index].shape,
        )
        demo_masked_car_frames += int(np.count_nonzero(~active_numeric))
        for metric, values in numeric.items():
            for branch_index, branch in enumerate(PHASE_D_NATIVE_BRANCHES):
                branch_values = values[:, branch_index]
                metric_active = np.broadcast_to(
                    pre_demo_environment[:, branch_index, None],
                    branch_values.shape,
                )
                if branch_values.shape[1:] == (2,):
                    metric_active = active_numeric[:, branch_index]
                active_values = branch_values[metric_active]
                if active_values.size:
                    metric_maxima[branch][metric] = max(
                        metric_maxima[branch][metric],
                        float(np.max(active_values, initial=0.0)),
                    )

        finite = _finite_state(state).reshape(len(cases), branch_count)
        overflow = (
            vehicle.mesh_candidate_overflow.reshape(len(cases), branch_count, 2) != 0
        ).any(axis=2) | (
            vehicle.contact_overflow.reshape(len(cases), branch_count, 2) != 0
        ).any(axis=2) | (
            sim.ball_world.candidate_overflow.numpy().reshape(len(cases), branch_count) != 0
        ) | (
            sim.ball_world.contact_overflow.numpy().reshape(len(cases), branch_count) != 0
        )
        resident_order = car_pair.pre_tick_first_car.reshape(len(cases), branch_count)

        for case_index in range(len(cases)):
            for branch_index in range(branch_count):
                hard: list[str] = []
                if not np.array_equal(
                    native_hit_current[case_index, branch_index],
                    gpu_hit_current[case_index, branch_index],
                ):
                    hard.append("car_ball_hit_callback_stream")
                if not event_match[case_index, branch_index]:
                    hard.append("ordered_bump_demo_event_stream")
                if not finite[case_index, branch_index]:
                    hard.append("non_finite_state")
                if overflow[case_index, branch_index]:
                    hard.append("contact_capacity_overflow")
                if resident_order[case_index, branch_index] != branch_index:
                    hard.append("car_pre_tick_lifecycle_order")
                hard.extend(
                    _semantic_mismatches(
                        state,
                        vehicle,
                        car_pair,
                        cached,
                        authority_index,
                        case_index,
                        branch_index,
                        branch_count,
                    )
                )
                numeric_failures: list[str] = []
                errors: dict[str, float] = {}
                for metric, values in numeric.items():
                    body_values = np.atleast_1d(values[case_index, branch_index])
                    errors[metric] = float(np.max(body_values, initial=0.0))
                    active = np.full(
                        body_values.shape,
                        pre_demo_environment[case_index, branch_index],
                        dtype=bool,
                    )
                    if body_values.shape == (2,):
                        active = active_numeric[case_index, branch_index]
                    maximum = float(np.max(body_values[active], initial=0.0))
                    if np.any(active) and maximum > TOLERANCES[_tolerance_key(metric)]:
                        numeric_failures.append(metric)
                raw_hard[branch_index] += len(hard)
                raw_numeric[branch_index] += len(numeric_failures)
                passed = not hard and not numeric_failures
                valid[case_index, branch_index] &= passed
                if not passed:
                    first_failures.setdefault(
                        (case_index, branch_index),
                        {
                            "tick": tick,
                            "hard": hard,
                            "numeric": numeric_failures,
                            "errors": errors,
                        },
                    )
        if tick in PHASE_D_HARD_HORIZONS:
            relation = np.any(valid, axis=1)
            pass_by_horizon[str(tick)] = {
                "pass": int(relation.sum()),
                "fail": int((~relation).sum()),
            }

    relation_pass = np.any(valid, axis=1)
    selected_pass = bool(np.all(relation_pass))
    family_summary = {}
    for family in sorted({case.family for case in cases}):
        family_mask = np.asarray([case.family == family for case in cases])
        family_summary[family] = {
            "case_count": int(family_mask.sum()),
            "pass": int(np.count_nonzero(relation_pass & family_mask)),
            "fail": int(np.count_nonzero((~relation_pass) & family_mask)),
        }
    failures = []
    for case_index in np.flatnonzero(~relation_pass)[:64]:
        failures.append(
            {
                "case_id": cases[case_index].case_id,
                "corpus_index": indices[case_index],
                "family": cases[case_index].family,
                "mode": cases[case_index].mode,
                "branches": {
                    branch: first_failures.get((case_index, branch_index), {})
                    for branch_index, branch in enumerate(PHASE_D_NATIVE_BRANCHES)
                },
            }
        )
    full = len(indices) == len(all_cases)
    report = {
        "schema_version": 1,
        "milestone": "v0.3",
        "phase": "D_integrated",
        "generated_utc": datetime.now(UTC).isoformat(),
        "status": "PASS_GREEN" if full and selected_pass else (
            "REPRESENTATIVE_PASS" if selected_pass else "FAIL"
        ),
        "authority_identity_sha256": identity["authority_identity_sha256"],
        "corpus_sha256": phase_d_corpus_sha256(all_cases),
        "selection_sha256": phase_d_selection_sha256(indices),
        "case_count": len(indices),
        "complete_corpus_case_count": len(all_cases),
        "native_branches": list(PHASE_D_NATIVE_BRANCHES),
        "relation": {
            "unit": "one complete trajectory from one labeled native-valid branch",
            "metric_mixing": False,
            "best_match_runtime_selection": False,
        },
        "tolerances": TOLERANCES,
        "pass_by_horizon": pass_by_horizon,
        "family_summary": family_summary,
        "branch_diagnostics": {
            branch: {
                "complete_trajectory_pass_count": int(valid[:, branch_index].sum()),
                "raw_hard_mismatch_count": int(raw_hard[branch_index]),
                "raw_numeric_failure_count": int(raw_numeric[branch_index]),
                "native_car_ball_callback_counts": native_car_hits[branch_index].tolist(),
                "gpu_car_ball_callback_counts": gpu_car_hits[branch_index].tolist(),
                "native_bump_event_count": int(native_bumps[branch_index]),
                "gpu_bump_event_count": int(gpu_bumps[branch_index]),
                "metric_maxima": dict(metric_maxima[branch]),
            }
            for branch_index, branch in enumerate(PHASE_D_NATIVE_BRANCHES)
        },
        "demo_masked_car_frames": demo_masked_car_frames,
        "failures": failures,
        "gate": {
            "selected_run_pass": selected_pass,
            "selection_complete": full,
            "phase_d_complete_gate_pass": full and selected_pass,
            "relation_failure_count": int((~relation_pass).sum()),
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))
    return 0 if selected_pass else 1


def _numeric_errors(
    state: Any,
    vehicle: Any,
    cached: dict[str, np.ndarray],
    tick: int,
    cases: int,
    branches: int,
) -> dict[str, np.ndarray]:
    result: dict[str, np.ndarray] = {}
    for body, position, velocity, quaternion, angular in (
        ("car", state.car_pos, state.car_vel, state.car_quat, state.car_ang_vel),
        (
            "ball",
            state.ball_pos[:, None, :],
            state.ball_vel[:, None, :],
            state.ball_quat[:, None, :],
            state.ball_ang_vel[:, None, :],
        ),
    ):
        tail_count = position.shape[1]
        pos = position.reshape(cases, branches, tail_count, 3)
        vel = velocity.reshape(cases, branches, tail_count, 3)
        quat = quaternion.reshape(cases * branches * tail_count, 4)
        matrix = quat_to_matrix(quat).reshape(cases, branches, tail_count, 3, 3)
        angular_value = angular.reshape(cases, branches, tail_count, 3)
        native_pos = cached[f"{body}_pos"][:, :, tick]
        native_vel = cached[f"{body}_vel"][:, :, tick]
        native_matrix = cached[f"{body}_matrix"][:, :, tick]
        native_angular = cached[f"{body}_ang_vel"][:, :, tick]
        if body == "ball":
            native_pos = native_pos[:, :, None, :]
            native_vel = native_vel[:, :, None, :]
            native_matrix = native_matrix[:, :, None, :, :]
            native_angular = native_angular[:, :, None, :]
        trace = np.einsum(
            "...ij,...ij->...", matrix.astype(np.float64), native_matrix.astype(np.float64)
        )
        result[f"{body}_position_uu"] = np.linalg.norm(
            pos.astype(np.float64) - native_pos.astype(np.float64), axis=-1
        )
        result[f"{body}_linear_velocity_uu_per_s"] = np.linalg.norm(
            vel.astype(np.float64) - native_vel.astype(np.float64), axis=-1
        )
        result[f"{body}_orientation_rad"] = np.arccos(
            np.clip((trace - 1.0) * 0.5, -1.0, 1.0)
        )
        result[f"{body}_angular_velocity_rad_per_s"] = np.linalg.norm(
            angular_value.astype(np.float64) - native_angular.astype(np.float64), axis=-1
        )
    result["car_boost"] = np.abs(
        state.boost.reshape(cases, branches, 2).astype(np.float64)
        - cached["car_boost"][:, :, tick].astype(np.float64)
    )
    result["car_handbrake"] = np.abs(
        vehicle.handbrake_value.reshape(cases, branches, 2).astype(np.float64)
        - cached["car_handbrake"][:, :, tick].astype(np.float64)
    )
    return result


def _callback_errors(
    pair_a: Any,
    pair_b: Any,
    cached: dict[str, np.ndarray],
    tick: int,
    native_current: np.ndarray,
    gpu_current: np.ndarray,
    cases: int,
    branches: int,
) -> dict[str, np.ndarray]:
    extra = np.stack(
        (
            pair_a.extra_hit_velocity_uu.reshape(cases, branches, 3),
            pair_b.extra_hit_velocity_uu.reshape(cases, branches, 3),
        ),
        axis=2,
    )
    relative = np.stack(
        (
            pair_a.relative_pos_on_ball_uu.reshape(cases, branches, 3),
            pair_b.relative_pos_on_ball_uu.reshape(cases, branches, 3),
        ),
        axis=2,
    )
    active = native_current & gpu_current
    extra_error = np.zeros(active.shape, dtype=np.float64)
    relative_error = np.zeros(active.shape, dtype=np.float64)
    extra_error[active] = np.linalg.norm(
        extra[active].astype(np.float64)
        - cached["car_ball_extra_hit_vel"][:, :, tick][active].astype(np.float64),
        axis=1,
    )
    relative_error[active] = np.linalg.norm(
        relative[active].astype(np.float64)
        - cached["car_ball_relative_pos"][:, :, tick][active].astype(np.float64),
        axis=1,
    )
    return {
        "callback_extra_velocity_uu_per_s": extra_error,
        "callback_relative_position_uu": relative_error,
    }


def _event_matches(
    pair: Any,
    cached: dict[str, np.ndarray],
    tick: int,
    cases: int,
    branches: int,
) -> np.ndarray:
    counts = pair.event_count.reshape(cases, branches)
    native = cached["bump_event_count"][:, :, tick]
    result = counts == native
    bumper = pair.event_bumper.reshape(cases, branches, -1)
    victim = pair.event_victim.reshape(cases, branches, -1)
    demo = pair.event_is_demo.reshape(cases, branches, -1).astype(bool)
    for case, branch in np.argwhere(result):
        count = int(counts[case, branch])
        expected_bumper = cached["bump_event_bumper"][case, branch, tick, :count]
        expected_victim = cached["bump_event_victim"][case, branch, tick, :count]
        expected_demo = cached["bump_event_is_demo"][case, branch, tick, :count]
        result[case, branch] = (
            np.array_equal(bumper[case, branch, :count], expected_bumper)
            and np.array_equal(victim[case, branch, :count], expected_victim)
            and np.array_equal(demo[case, branch, :count], expected_demo)
        )
    return result


def _semantic_mismatches(
    state: Any,
    vehicle: Any,
    pair: Any,
    cached: dict[str, np.ndarray],
    tick: int,
    case: int,
    branch: int,
    branches: int,
) -> list[str]:
    runtime = case * branches + branch
    mismatches: list[str] = []
    active = np.full(
        2,
        not np.any(cached["car_is_demoed"][case, branch, tick]),
        dtype=bool,
    )
    if not np.array_equal(
        state.on_ground[runtime].astype(bool)[active],
        cached["car_on_ground"][case, branch, tick][active],
    ):
        mismatches.append("car_on_ground")
    if not np.array_equal(
        vehicle.wheel_contact[runtime * 2 : runtime * 2 + 2].astype(bool)[active],
        cached["car_wheel_contacts"][case, branch, tick][active],
    ):
        mismatches.append("car_wheel_contacts")
    gpu_world_normal = vehicle.world_contact_normal[runtime * 2 : runtime * 2 + 2]
    gpu_world = np.linalg.norm(gpu_world_normal, axis=1) > 0.5
    native_world = cached["car_world_contact"][case, branch, tick]
    if not np.array_equal(gpu_world[active], native_world[active]):
        mismatches.append("car_world_contact")
    for car in range(2):
        if active[car] and gpu_world[car] and native_world[car]:
            left = gpu_world_normal[car].astype(np.float64)
            right = cached["car_world_contact_normal"][case, branch, tick, car].astype(np.float64)
            denom = np.linalg.norm(left) * np.linalg.norm(right)
            angle = (
                np.pi
                if denom == 0
                else np.arccos(np.clip(np.dot(left, right) / denom, -1.0, 1.0))
            )
            if angle > TOLERANCES["world_contact_normal_rad"]:
                mismatches.append("car_world_contact_normal")
                break
    if not np.array_equal(
        state.is_supersonic[runtime].astype(bool)[active],
        cached["car_is_supersonic"][case, branch, tick][active],
    ):
        mismatches.append("car_is_supersonic")
    gpu_contact_id = pair.car_contact_id[runtime].astype(np.uint32)
    native_cooldown = cached["car_contact_cooldown"][case, branch, tick]
    cooldown_active = native_cooldown > 0.0
    if not np.array_equal(
        gpu_contact_id[cooldown_active],
        cached["car_contact_id"][case, branch, tick][cooldown_active],
    ):
        mismatches.append("car_contact_id")
    gpu_cooldown = pair.car_contact_cooldown[runtime]
    if np.max(np.abs(gpu_cooldown - native_cooldown)) > TOLERANCES["contact_cooldown_s"]:
        mismatches.append("car_contact_cooldown")
    if not np.array_equal(
        pair.car_is_demoed[runtime].astype(bool),
        cached["car_is_demoed"][case, branch, tick],
    ):
        mismatches.append("car_is_demoed")
    return mismatches


def _finite_state(state: Any) -> np.ndarray:
    values = np.column_stack(
        (
            state.car_pos.reshape(state.num_envs, -1),
            state.car_vel.reshape(state.num_envs, -1),
            state.car_quat.reshape(state.num_envs, -1),
            state.car_ang_vel.reshape(state.num_envs, -1),
            state.ball_pos,
            state.ball_vel,
            state.ball_quat,
            state.ball_ang_vel,
        )
    )
    return np.isfinite(values).all(axis=1)


def _tolerance_key(metric: str) -> str:
    for suffix in TOLERANCES:
        if metric.endswith(suffix):
            return suffix
    raise KeyError(metric)


if __name__ == "__main__":
    raise SystemExit(main())
