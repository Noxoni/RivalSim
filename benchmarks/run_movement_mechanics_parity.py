"""Source-pinned RocketSim parity gate for the complete Octane control surface.

This is an open-loop mechanics gate: both simulators receive the same controls
at 120 Hz.  Policy feedback is deliberately excluded so the first physical
operation, rather than accumulated policy sensitivity, owns every residual.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from rivalsim import CompleteWorldSim, make_standard_kickoff_state
from rivalsim.arena import ArenaGeometry, WarpArenaMeshes
from rivalsim.controls import ControlBatch
from rivalsim.math import quat_to_matrix
from rivalsim.reference.rocketsim_oracle import (
    ROCKETSIM_BINDING_COMMIT,
    ROCKETSIM_PRIMARY_COMMIT,
    RocketSimStaticWorldBatchOracle,
)

CAPTURE_TICKS = (1, 2, 4, 8, 16, 24, 32, 48, 64, 96, 120)


@dataclass(frozen=True, slots=True)
class MechanicsCase:
    name: str
    family: str
    mode: str


CASES = (
    MechanicsCase("ground_neutral", "ground_drive", "neutral"),
    MechanicsCase("ground_throttle", "ground_drive", "throttle"),
    MechanicsCase("ground_reverse", "ground_drive", "reverse"),
    MechanicsCase("ground_brake", "ground_drive", "brake"),
    MechanicsCase("ground_coast", "ground_drive", "coast"),
    MechanicsCase("ground_steer", "steering", "steer"),
    MechanicsCase("ground_powerslide", "powerslide", "powerslide"),
    MechanicsCase("ground_boost", "boost", "ground_boost"),
    MechanicsCase("air_boost", "boost", "air_boost"),
    MechanicsCase("jump_tap", "jump", "jump_tap"),
    MechanicsCase("jump_hold", "jump", "jump_hold"),
    MechanicsCase("double_jump", "double_jump", "double_jump"),
    MechanicsCase("front_flip", "dodge", "front_flip"),
    MechanicsCase("back_flip", "dodge", "back_flip"),
    MechanicsCase("left_flip", "dodge", "left_flip"),
    MechanicsCase("right_flip", "dodge", "right_flip"),
    MechanicsCase("diagonal_flip", "dodge", "diagonal_flip"),
    MechanicsCase("flip_cancel", "flip_cancel", "flip_cancel"),
    MechanicsCase("stall", "dodge", "stall"),
    MechanicsCase("air_pitch", "air_control", "air_pitch"),
    MechanicsCase("air_yaw", "air_control", "air_yaw"),
    MechanicsCase("air_roll", "air_control", "air_roll"),
    MechanicsCase("supersonic_coast", "supersonic", "supersonic"),
    MechanicsCase("auto_flip", "auto_flip", "auto_flip"),
)


def _controls_for_tick(tick: int) -> ControlBatch:
    count = len(CASES)
    values = {
        name: np.zeros((count, 2), dtype=np.float32)
        for name in ("throttle", "steer", "pitch", "yaw", "roll")
    }
    buttons = {
        name: np.zeros((count, 2), dtype=np.int32)
        for name in ("jump", "boost", "handbrake")
    }
    for index, case in enumerate(CASES):
        mode = case.mode
        if mode in {"throttle", "steer", "powerslide", "ground_boost"}:
            values["throttle"][index, 0] = 1.0
        if mode == "reverse":
            values["throttle"][index, 0] = -1.0
        if mode == "brake":
            values["throttle"][index, 0] = -1.0
        if mode == "steer":
            values["steer"][index, 0] = 1.0
        if mode == "powerslide":
            values["steer"][index, 0] = 1.0
            buttons["handbrake"][index, 0] = 1
        if mode in {"ground_boost", "air_boost"}:
            buttons["boost"][index, 0] = 1
        if mode == "air_pitch":
            values["pitch"][index, 0] = 1.0
        elif mode == "air_yaw":
            values["yaw"][index, 0] = 1.0
        elif mode == "air_roll":
            values["roll"][index, 0] = 1.0

        mechanic = mode in {
            "jump_tap", "jump_hold", "double_jump", "front_flip", "back_flip",
            "left_flip", "right_flip", "diagonal_flip", "flip_cancel", "stall",
        }
        if mechanic and (tick == 0 or (mode == "jump_hold" and tick < 24)):
            buttons["jump"][index, 0] = 1
        if mode in {
            "double_jump", "front_flip", "back_flip", "left_flip", "right_flip",
            "diagonal_flip", "flip_cancel", "stall",
        } and tick == 21:
            buttons["jump"][index, 0] = 1
        if mode in {"front_flip", "flip_cancel"} and tick == 21:
            values["pitch"][index, 0] = -1.0
        elif mode == "back_flip" and tick == 21:
            values["pitch"][index, 0] = 1.0
        elif mode == "left_flip" and tick == 21:
            values["yaw"][index, 0] = -1.0
        elif mode == "right_flip" and tick == 21:
            values["yaw"][index, 0] = 1.0
        elif mode == "diagonal_flip" and tick == 21:
            values["pitch"][index, 0] = -1.0
            values["yaw"][index, 0] = 1.0
        elif mode == "stall" and tick == 21:
            values["yaw"][index, 0] = 1.0
            values["roll"][index, 0] = -1.0
        if mode == "flip_cancel" and 22 <= tick < 48:
            values["pitch"][index, 0] = 1.0
        if mode == "auto_flip" and tick == 4:
            buttons["jump"][index, 0] = 1
    return ControlBatch(
        values["throttle"], values["steer"], values["pitch"], values["yaw"],
        values["roll"], buttons["jump"], buttons["boost"], buttons["handbrake"],
    )


def _initial_state():
    count = len(CASES)
    state = make_standard_kickoff_state(count, np.full(count, 4, dtype=np.int32))
    state.car_pos[:, 0, :2] = 0.0
    state.car_pos[:, 1] = (3000.0, 3000.0, 1400.0)
    state.ball_pos[:] = (-3000.0, 3000.0, 1700.0)
    for index, case in enumerate(CASES):
        if case.mode in {"brake", "coast"}:
            state.car_vel[index, 0] = (0.0, 1000.0, 0.0)
        if case.mode in {"air_boost", "air_pitch", "air_yaw", "air_roll"}:
            state.car_pos[index, 0] = (0.0, 0.0, 1000.0)
            state.on_ground[index, 0] = 0
        if case.mode == "supersonic":
            state.car_vel[index, 0] = (0.0, 2200.0, 0.0)
            state.is_supersonic[index, 0] = 1
        if case.mode == "auto_flip":
            state.car_pos[index, 0] = (0.0, 0.0, 25.0)
            state.car_quat[index, 0] = (1.0, 0.0, 0.0, 0.0)
            state.on_ground[index, 0] = 0
    state.validate()
    return state


def _max_abs(left: np.ndarray, right: np.ndarray) -> float:
    return float(np.max(np.abs(left.astype(np.float64) - right.astype(np.float64))))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--collision-dir", required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--output", type=Path)
    parser.add_argument(
        "--include-captures",
        action="store_true",
        help="include every per-tick vector in JSON instead of compact aggregate evidence",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    initial = _initial_state()
    native = RocketSimStaticWorldBatchOracle(initial, args.collision_dir)
    authority_initial = native.authoritative_snapshot()
    # The batch authority owns one native car per arena. Park RivalSim's
    # otherwise-unused second car and ball away from the isolated subject.
    authority_initial.car_pos[:, 1] = (3000.0, 3000.0, 1400.0)
    authority_initial.ball_pos[:] = (-3000.0, 3000.0, 1700.0)
    geometry = ArenaGeometry.load_soccar(args.collision_dir)
    gpu = CompleteWorldSim(
        len(CASES), args.collision_dir, device=args.device,
        initial=authority_initial, geometry=geometry, meshes=WarpArenaMeshes(geometry),
        car_visitation_order="a_then_b", auto_kickoff=False,
    )
    captures: list[dict[str, object]] = []
    native_coverage = {
        case.name: {
            "has_jumped": False,
            "has_double_jumped": False,
            "has_flipped": False,
            "is_auto_flipping": False,
        }
        for case in CASES
    }
    worst_by_case: dict[str, dict[str, float]] = {
        case.name: {name: 0.0 for name in (
            "position_uu", "velocity_uu_s", "orientation_component",
            "angular_velocity_rad_s", "boost", "mechanic_float",
            "mechanic_discrete",
        )}
        for case in CASES
    }
    for tick in range(120):
        controls = _controls_for_tick(tick)
        native.set_controls(controls)
        gpu.set_controls(controls)
        native.step()
        gpu.step(1, synchronize=tick + 1 in CAPTURE_TICKS)
        if tick + 1 not in CAPTURE_TICKS:
            continue
        rs = gpu.snapshot()
        rs_vehicle = gpu.vehicle.snapshot()
        rk = native.frame()
        rs_matrix = quat_to_matrix(rs.car_quat[:, 0])
        for index, case in enumerate(CASES):
            native_coverage[case.name]["has_jumped"] |= bool(rk.has_jumped[index])
            native_coverage[case.name]["has_double_jumped"] |= bool(
                rk.has_double_jumped[index]
            )
            native_coverage[case.name]["has_flipped"] |= bool(rk.has_flipped[index])
            native_coverage[case.name]["is_auto_flipping"] |= bool(
                rk.is_auto_flipping[index]
            )
            mechanic_float = max(
                abs(float(rs.jump_time[index, 0]) - float(rk.jump_time[index])),
                abs(float(rs.air_time[index, 0]) - float(rk.air_time[index])),
                abs(float(rs.air_time_since_jump[index, 0]) - float(rk.air_time_since_jump[index])),
                abs(float(rs.flip_time[index, 0]) - float(rk.flip_time[index])),
                _max_abs(rs.flip_rel_torque[index, 0], rk.flip_rel_torque[index]),
                abs(float(rs.auto_flip_timer[index, 0]) - float(rk.auto_flip_timer[index])),
                abs(
                    float(rs.auto_flip_torque_scale[index, 0])
                    - float(rk.auto_flip_torque_scale[index])
                ),
                abs(float(rs.boosting_time[index, 0]) - float(rk.boosting_time[index])),
                abs(float(rs.supersonic_time[index, 0]) - float(rk.supersonic_time[index])),
                abs(
                    float(rs_vehicle.handbrake_value[index * 2])
                    - float(rk.handbrake_value[index])
                ),
            )
            mechanic_discrete = float(max(
                abs(int(rs.on_ground[index, 0]) - int(rk.on_ground[index])),
                abs(int(rs.has_jumped[index, 0]) - int(rk.has_jumped[index])),
                abs(int(rs.is_jumping[index, 0]) - int(rk.is_jumping[index])),
                abs(int(rs.has_double_jumped[index, 0]) - int(rk.has_double_jumped[index])),
                abs(int(rs.has_flipped[index, 0]) - int(rk.has_flipped[index])),
                abs(int(rs.is_flipping[index, 0]) - int(rk.is_flipping[index])),
                abs(int(rs.is_auto_flipping[index, 0]) - int(rk.is_auto_flipping[index])),
                abs(int(rs.is_supersonic[index, 0]) - int(rk.is_supersonic[index])),
                int(np.any(
                    rs_vehicle.wheel_contact[index * 2].astype(np.int32)
                    != rk.wheel_contacts[index].astype(np.int32)
                )),
            ))
            errors = {
                "position_uu": _max_abs(rs.car_pos[index, 0], rk.car_pos[index]),
                "velocity_uu_s": _max_abs(rs.car_vel[index, 0], rk.car_vel[index]),
                "orientation_component": _max_abs(rs_matrix[index], rk.car_matrix[index]),
                "angular_velocity_rad_s": _max_abs(rs.car_ang_vel[index, 0], rk.car_ang_vel[index]),
                "boost": abs(float(rs.boost[index, 0]) - float(rk.boost[index])),
                "mechanic_float": mechanic_float,
                "mechanic_discrete": mechanic_discrete,
            }
            for name, value in errors.items():
                worst_by_case[case.name][name] = max(worst_by_case[case.name][name], value)
            captures.append({
                "tick": tick + 1,
                "case": case.name,
                "errors": errors,
                "rivalsim": {
                    "position": rs.car_pos[index, 0].tolist(),
                    "velocity": rs.car_vel[index, 0].tolist(),
                    "angular_velocity": rs.car_ang_vel[index, 0].tolist(),
                    "supersonic_time": float(rs.supersonic_time[index, 0]),
                    "is_supersonic": int(rs.is_supersonic[index, 0]),
                },
                "rocketsim": {
                    "position": rk.car_pos[index].tolist(),
                    "velocity": rk.car_vel[index].tolist(),
                    "angular_velocity": rk.car_ang_vel[index].tolist(),
                    "supersonic_time": float(rk.supersonic_time[index]),
                    "is_supersonic": int(rk.is_supersonic[index]),
                },
            })

    thresholds = {
        "position_uu": 2.0,
        "velocity_uu_s": 5.0,
        "orientation_component": 0.01,
        "angular_velocity_rad_s": 0.05,
        "boost": 1.0e-4,
        "mechanic_float": 1.0e-5,
        "mechanic_discrete": 0.0,
    }
    failures = []
    for case in CASES:
        for metric, value in worst_by_case[case.name].items():
            if value > thresholds[metric]:
                failures.append(
                    {
                        "case": case.name,
                        "family": case.family,
                        "metric": metric,
                        "value": value,
                        "limit": thresholds[metric],
                    }
                )
    required_coverage = {
        "jump_tap": "has_jumped",
        "jump_hold": "has_jumped",
        "double_jump": "has_double_jumped",
        "front_flip": "has_flipped",
        "back_flip": "has_flipped",
        "left_flip": "has_flipped",
        "right_flip": "has_flipped",
        "diagonal_flip": "has_flipped",
        "flip_cancel": "has_flipped",
        "stall": "has_flipped",
        "auto_flip": "is_auto_flipping",
    }
    for case_name, field in required_coverage.items():
        if not native_coverage[case_name][field]:
            failures.append({
                "case": case_name,
                "family": next(case.family for case in CASES if case.name == case_name),
                "metric": f"native_coverage.{field}",
                "value": False,
                "limit": True,
            })
    result = {
        "status": "PASS" if not failures else "FAIL",
        "authority": {
            "rocketsim_commit": ROCKETSIM_PRIMARY_COMMIT,
            "rocketsim_binding_commit": ROCKETSIM_BINDING_COMMIT,
            "tick_rate_hz": 120,
            "car": "Octane",
            "game_mode": "Soccar",
        },
        "case_count": len(CASES),
        "capture_ticks": list(CAPTURE_TICKS),
        "thresholds": thresholds,
        "failures": failures,
        "native_coverage": native_coverage,
        "worst_by_case": worst_by_case,
        "capture_count": len(captures),
    }
    if args.include_captures:
        result["captures"] = captures
    text = json.dumps(result, indent=2)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text + "\n", encoding="utf-8")
    print(text)
    return 0 if not failures else 1


if __name__ == "__main__":
    raise SystemExit(main())
