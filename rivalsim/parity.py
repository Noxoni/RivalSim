"""Parity error calculations and measured-tolerance evaluation."""

from __future__ import annotations

import math

import numpy as np

from rivalsim.math import orientation_error_radians, quat_to_matrix
from rivalsim.reference.rocketsim_oracle import OracleFrame
from rivalsim.state import StateSnapshot

CONTINUOUS_CAR_METRICS = (
    "position_uu",
    "linear_velocity_uu_per_s",
    "orientation_rad",
    "angular_velocity_rad_per_s",
    "boost",
    "jump_time_s",
    "air_time_s",
    "air_time_since_jump_s",
    "flip_time_s",
    "flip_rel_torque",
)
DISCRETE_FIELDS = (
    "has_jumped",
    "is_jumping",
    "has_double_jumped",
    "has_flipped",
    "is_flipping",
)


def same_equation_errors(left: StateSnapshot, right: StateSnapshot) -> dict[str, float | int]:
    car_left = _car_zero(left)
    car_right = _car_zero(right)
    errors: dict[str, float | int] = {
        "position_uu": _norm(car_left["pos"] - car_right["pos"]),
        "linear_velocity_uu_per_s": _norm(car_left["vel"] - car_right["vel"]),
        "orientation_rad": float(
            orientation_error_radians(car_left["quat"][None], car_right["quat"][None])[0]
        ),
        "angular_velocity_rad_per_s": _norm(car_left["ang_vel"] - car_right["ang_vel"]),
        "boost": abs(float(car_left["boost"] - car_right["boost"])),
        "jump_time_s": abs(float(car_left["jump_time"] - car_right["jump_time"])),
        "air_time_s": abs(float(car_left["air_time"] - car_right["air_time"])),
        "air_time_since_jump_s": abs(
            float(car_left["air_time_since_jump"] - car_right["air_time_since_jump"])
        ),
        "flip_time_s": abs(float(car_left["flip_time"] - car_right["flip_time"])),
        "flip_rel_torque": _norm(car_left["flip_rel_torque"] - car_right["flip_rel_torque"]),
    }
    for field in DISCRETE_FIELDS:
        errors[f"{field}_mismatch"] = int(car_left[field] != car_right[field])
    return errors


def rocketsim_errors(sim: StateSnapshot, oracle: OracleFrame) -> dict[str, float | int]:
    car = _car_zero(sim)
    errors: dict[str, float | int] = {
        "position_uu": _norm(car["pos"] - oracle.car_pos),
        "linear_velocity_uu_per_s": _norm(car["vel"] - oracle.car_vel),
        "orientation_rad": matrix_orientation_error(quat_to_matrix(car["quat"]), oracle.car_matrix),
        "angular_velocity_rad_per_s": _norm(car["ang_vel"] - oracle.car_ang_vel),
        "boost": abs(float(car["boost"] - oracle.boost)),
        "jump_time_s": abs(float(car["jump_time"] - oracle.jump_time)),
        "air_time_s": abs(float(car["air_time"] - oracle.air_time)),
        "air_time_since_jump_s": abs(
            float(car["air_time_since_jump"] - oracle.air_time_since_jump)
        ),
        "flip_time_s": abs(float(car["flip_time"] - oracle.flip_time)),
        "flip_rel_torque": _norm(car["flip_rel_torque"] - oracle.flip_rel_torque),
    }
    for field in DISCRETE_FIELDS:
        errors[f"{field}_mismatch"] = int(bool(car[field]) != bool(getattr(oracle, field)))
    return errors


def ball_same_equation_errors(left: StateSnapshot, right: StateSnapshot) -> dict[str, float]:
    return {
        "ball_position_uu": _norm(left.ball_pos[0] - right.ball_pos[0]),
        "ball_linear_velocity_uu_per_s": _norm(left.ball_vel[0] - right.ball_vel[0]),
        "ball_orientation_rad": float(
            orientation_error_radians(left.ball_quat[:1], right.ball_quat[:1])[0]
        ),
        "ball_angular_velocity_rad_per_s": _norm(left.ball_ang_vel[0] - right.ball_ang_vel[0]),
    }


def ball_rocketsim_errors(sim: StateSnapshot, oracle: OracleFrame) -> dict[str, float]:
    return {
        "ball_position_uu": _norm(sim.ball_pos[0] - oracle.ball_pos),
        "ball_linear_velocity_uu_per_s": _norm(sim.ball_vel[0] - oracle.ball_vel),
        "ball_orientation_rad": matrix_orientation_error(
            quat_to_matrix(sim.ball_quat[0]), oracle.ball_matrix
        ),
        "ball_angular_velocity_rad_per_s": _norm(sim.ball_ang_vel[0] - oracle.ball_ang_vel),
    }


def matrix_orientation_error(left: np.ndarray, right: np.ndarray) -> float:
    relative = np.asarray(left, dtype=np.float64).T @ np.asarray(right, dtype=np.float64)
    cosine = np.clip((np.trace(relative) - 1.0) * 0.5, -1.0, 1.0)
    return float(math.acos(float(cosine)))


def evaluate_errors(
    errors: dict[str, float | int], tolerances: dict[str, float]
) -> tuple[bool, list[str]]:
    failures: list[str] = []
    for metric, value in errors.items():
        if metric.endswith("_mismatch"):
            if int(value) != 0:
                failures.append(metric)
            continue
        tolerance = tolerances.get(metric)
        if tolerance is not None and float(value) > tolerance:
            failures.append(metric)
    return not failures, failures


def axis_sign_check(scenario_name: str, sim: StateSnapshot, oracle: OracleFrame) -> bool:
    if scenario_name.startswith("isolated_"):
        sim_vector = sim.car_ang_vel[0, 0].astype(np.float64)
        oracle_vector = oracle.car_ang_vel.astype(np.float64)
    elif "dodge" in scenario_name:
        sim_vector = sim.car_vel[0, 0, :2].astype(np.float64)
        oracle_vector = oracle.car_vel[:2].astype(np.float64)
    else:
        return True
    sim_norm = np.linalg.norm(sim_vector)
    oracle_norm = np.linalg.norm(oracle_vector)
    if sim_norm < 1e-7 or oracle_norm < 1e-7:
        return sim_norm < 1e-7 and oracle_norm < 1e-7
    cosine = float(np.dot(sim_vector, oracle_vector) / (sim_norm * oracle_norm))
    return cosine > 0.999


def _norm(value: np.ndarray) -> float:
    return float(np.linalg.norm(np.asarray(value, dtype=np.float64)))


def _car_zero(state: StateSnapshot) -> dict[str, np.ndarray | float | int]:
    return {
        "pos": state.car_pos[0, 0],
        "vel": state.car_vel[0, 0],
        "quat": state.car_quat[0, 0],
        "ang_vel": state.car_ang_vel[0, 0],
        "boost": state.boost[0, 0],
        "jump_time": state.jump_time[0, 0],
        "air_time": state.air_time[0, 0],
        "air_time_since_jump": state.air_time_since_jump[0, 0],
        "flip_time": state.flip_time[0, 0],
        "flip_rel_torque": state.flip_rel_torque[0, 0],
        **{field: getattr(state, field)[0, 0] for field in DISCRETE_FIELDS},
    }
