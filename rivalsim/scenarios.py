"""Deterministic v0.1 parity corpus from BENCHMARK_AND_PARITY.md."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from rivalsim import constants as c
from rivalsim.controls import ControlBatch
from rivalsim.state import StateSnapshot

HORIZONS = (1, 4, 8, 30, 60, 120)


@dataclass(frozen=True, slots=True)
class Scenario:
    name: str
    family: str
    initial: StateSnapshot
    controls: tuple[ControlBatch, ...]
    compare_ball: bool = False
    note: str = ""


def parity_scenarios(ticks: int = 120) -> tuple[Scenario, ...]:
    if ticks < max(HORIZONS):
        raise ValueError(f"scenario corpus requires at least {max(HORIZONS)} ticks")
    scenarios: list[Scenario] = []

    state = _base_state()
    scenarios.append(Scenario("stationary_gravity_drop", "free_body", state, _controls(ticks)))

    state = _base_state()
    state.car_vel[0, 0] = (800.0, -400.0, 200.0)
    scenarios.append(Scenario("arbitrary_linear_velocity", "free_body", state, _controls(ticks)))

    state = _base_state()
    state.car_ang_vel[0, 0] = (0.8, -1.2, 2.0)
    scenarios.append(Scenario("arbitrary_angular_velocity", "free_body", state, _controls(ticks)))

    state = _base_state()
    state.car_vel[0, 0] = (2400.0, 300.0, 0.0)
    scenarios.append(Scenario("car_speed_limit_crossing", "free_body", state, _controls(ticks)))

    state = _base_state()
    state.car_ang_vel[0, 0] = (1.2, -1.8, 2.4)
    scenarios.append(
        Scenario(
            "combined_rotation_integration",
            "free_body",
            state,
            _controls(ticks, pitch=0.3, yaw=-0.2, roll=0.4),
        )
    )

    state = _base_state()
    scenarios.append(
        Scenario("boost_from_rest", "boost_throttle", state, _controls(ticks, boost=True))
    )

    state = _base_state()
    state.car_vel[0, 0] = (900.0, -200.0, 100.0)
    scenarios.append(
        Scenario("boost_nonzero_velocity", "boost_throttle", state, _controls(ticks, boost=True))
    )

    state = _base_state()
    state.boost[0, 0] = 2.0
    scenarios.append(
        Scenario("boost_depletion", "boost_throttle", state, _controls(ticks, boost=True))
    )

    scenarios.append(
        Scenario(
            "forward_air_throttle",
            "boost_throttle",
            _base_state(),
            _controls(ticks, throttle=1.0),
        )
    )
    scenarios.append(
        Scenario(
            "reverse_air_throttle",
            "boost_throttle",
            _base_state(),
            _controls(ticks, throttle=-1.0),
        )
    )
    scenarios.append(
        Scenario(
            "boost_throttle_combined",
            "boost_throttle",
            _base_state(),
            _controls(ticks, throttle=1.0, boost=True),
        )
    )

    state = _post_jump_state()
    scenarios.append(
        Scenario(
            "tap_jump_airborne_phase",
            "jump",
            state,
            _controls(ticks),
            note="initialized immediately after the first-jump impulse; ground contact is excluded",
        )
    )

    state = _post_jump_state()
    scenarios.append(
        Scenario(
            "minimum_three_tick_jump",
            "jump",
            state,
            _controls(ticks, jump_until=3),
            note="initialized immediately after the first-jump impulse; ground contact is excluded",
        )
    )

    state = _post_jump_state()
    scenarios.append(
        Scenario(
            "full_point_two_second_jump",
            "jump",
            state,
            _controls(ticks, jump_until=24),
            note="initialized immediately after the first-jump impulse; ground contact is excluded",
        )
    )

    for delay in (0.05, 0.4, 1.0):
        state = _double_jump_ready(delay)
        scenarios.append(
            Scenario(
                f"double_jump_delay_{delay:.2f}",
                "jump",
                state,
                _controls(ticks, jump_ticks={0}),
            )
        )

    state = _double_jump_ready(1.25)
    scenarios.append(
        Scenario(
            "double_jump_after_timeout",
            "jump",
            state,
            _controls(ticks, jump_ticks={0}),
        )
    )

    state = _double_jump_ready(0.1)
    state.car_quat[0, 0] = _axis_angle((0.0, 1.0, 0.0), 0.6)
    scenarios.append(
        Scenario(
            "tilted_roof_double_jump",
            "jump",
            state,
            _controls(ticks, jump_ticks={0}),
            note="contact-free roof-direction impulse validates the tilted first-jump equation too",
        )
    )

    for axis, values in (
        ("pitch", {"pitch": 1.0}),
        ("yaw", {"yaw": 1.0}),
        ("roll", {"roll": 1.0}),
    ):
        scenarios.append(
            Scenario(
                f"isolated_{axis}",
                "air_torque_dodge",
                _base_state(),
                _controls(ticks, **values),
            )
        )

    scenarios.append(
        Scenario(
            "combined_air_torque",
            "air_torque_dodge",
            _base_state(),
            _controls(ticks, pitch=0.7, yaw=-0.5, roll=0.3),
        )
    )

    state = _double_jump_ready(0.1)
    scenarios.append(
        Scenario(
            "forward_dodge",
            "air_torque_dodge",
            state,
            _controls(ticks, pitch=-1.0, jump_ticks={0}),
        )
    )
    state = _double_jump_ready(0.1)
    state.car_vel[0, 0] = (900.0, 50.0, -100.0)
    scenarios.append(
        Scenario(
            "diagonal_dodge_flip_timer",
            "air_torque_dodge",
            state,
            _controls(ticks, pitch=-0.7, yaw=0.6, jump_ticks={0}),
        )
    )

    state = _base_state()
    state.car_ang_vel[0, 0] = (0.0, 0.0, 5.45)
    scenarios.append(
        Scenario(
            "angular_speed_limit",
            "air_torque_dodge",
            state,
            _controls(ticks, roll=-1.0),
        )
    )

    state = _base_state()
    state.ball_vel[0] = (1800.0, -700.0, 400.0)
    state.ball_ang_vel[0] = (1.0, -2.0, 5.8)
    state.ball_quat[0] = _axis_angle((1.0, 1.0, 0.5), 0.4)
    scenarios.append(
        Scenario(
            "free_ball_drag_rotation_limits",
            "free_body",
            state,
            _controls(ticks),
            compare_ball=True,
        )
    )

    return tuple(scenarios)


def source_backed_jump_checks() -> dict[str, float | int | str]:
    """Checks excluded from a live no-contact oracle because the initial ground is required."""

    return {
        "first_jump_impulse_uu_per_s": float(c.JUMP_IMMEDIATE_FORCE),
        "sticky_accel_uu_per_s2": float(c.JUMP_STICKY_ACCEL),
        "sticky_ticks": c.JUMP_STICKY_TICKS,
        "jump_hold_accel_uu_per_s2": float(c.JUMP_ACCEL),
        "jump_hold_max_s": float(c.JUMP_MAX_TIME),
        "pre_min_hold_scale": float(c.JUMP_PRE_MIN_ACCEL_SCALE),
        "status": "source_backed",
    }


def _base_state() -> StateSnapshot:
    state = StateSnapshot.empty(1)
    state.car_pos[0, 0] = (0.0, 0.0, 1000.0)
    state.car_pos[0, 1] = (2800.0, 2800.0, 1500.0)
    state.ball_pos[0] = (-2800.0, -2800.0, 1400.0)
    state.ball_vel[0] = (0.001, 0.0, 0.0)  # prevent RocketSim's exact-zero sleep shortcut
    return state


def _post_jump_state() -> StateSnapshot:
    state = _base_state()
    state.car_vel[0, 0, 2] = c.JUMP_IMMEDIATE_FORCE
    state.has_jumped[0, 0] = 1
    state.is_jumping[0, 0] = 1
    state.jump_time[0, 0] = 0.0
    # Sticky force requires a world contact; live no-contact parity begins after the impulse.
    state.sticky_ticks[0, 0] = 0
    return state


def _double_jump_ready(delay: float) -> StateSnapshot:
    state = _base_state()
    state.has_jumped[0, 0] = 1
    state.is_jumping[0, 0] = 0
    state.jump_time[0, 0] = c.JUMP_MAX_TIME
    state.air_time[0, 0] = delay + float(c.JUMP_MAX_TIME)
    state.air_time_since_jump[0, 0] = delay
    return state


def _controls(
    ticks: int,
    *,
    throttle: float = 0.0,
    steer: float = 0.0,
    pitch: float = 0.0,
    yaw: float = 0.0,
    roll: float = 0.0,
    boost: bool = False,
    jump_until: int = 0,
    jump_ticks: set[int] | None = None,
) -> tuple[ControlBatch, ...]:
    result: list[ControlBatch] = []
    jump_ticks = jump_ticks or set()
    for tick in range(ticks):
        controls = ControlBatch.zeros(1)
        controls.throttle[0, 0] = throttle
        controls.steer[0, 0] = steer
        controls.pitch[0, 0] = pitch
        controls.yaw[0, 0] = yaw
        controls.roll[0, 0] = roll
        controls.boost[0, 0] = boost
        controls.jump[0, 0] = tick < jump_until or tick in jump_ticks
        result.append(controls)
    return tuple(result)


def _axis_angle(axis: tuple[float, float, float], angle: float) -> np.ndarray:
    vector = np.asarray(axis, dtype=np.float32)
    vector /= np.linalg.norm(vector)
    half = np.float32(angle * 0.5)
    result = np.empty(4, dtype=np.float32)
    result[:3] = vector * np.sin(half)
    result[3] = np.cos(half)
    return result
