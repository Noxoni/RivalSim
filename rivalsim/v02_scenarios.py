"""Contact-rich, control-rich v0.2 parity scenario matrix."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from rivalsim.controls import ControlBatch
from rivalsim.state import StateSnapshot

V02_HORIZONS = (1, 4, 8, 30, 60, 120, 300, 600)


@dataclass(frozen=True, slots=True)
class StaticWorldScenario:
    name: str
    family: str
    initial: StateSnapshot
    schedule: tuple[tuple[int, ControlBatch], ...]

    def controls_at(self, tick: int) -> ControlBatch:
        result = self.schedule[0][1]
        for start, controls in self.schedule:
            if start > tick:
                break
            result = controls
        return result


def make_v02_scenarios() -> tuple[StaticWorldScenario, ...]:
    scenarios: list[StaticWorldScenario] = []

    def add(
        name: str,
        family: str,
        state: StateSnapshot,
        controls: ControlBatch,
        *changes: tuple[int, ControlBatch],
    ) -> None:
        scenarios.append(StaticWorldScenario(name, family, state, ((0, controls), *changes)))

    zero = ControlBatch.zeros(1)
    add("floor_settle", "settle_and_rest", _state(pos=(0, 0, 140), vel=(0, 0, -300)), zero)
    add("floor_rest", "settle_and_rest", _state(pos=(0, 0, 17)), zero)
    add("level_landing", "landings", _state(pos=(500, 200, 180), vel=(80, 0, -600)), zero)
    add(
        "tilted_landing",
        "landings",
        _state(pos=(-500, 200, 180), vel=(100, -40, -550), pyr=(0.25, 0.0, 0.18)),
        zero,
    )
    add(
        "partial_two_wheel_landing",
        "landings",
        _state(pos=(0, -500, 90), vel=(200, 20, -250), pyr=(0.05, 0.0, 0.48)),
        zero,
    )
    add("throttle_forward", "longitudinal", _state(pos=(0, 0, 17)), _controls(throttle=1))
    add("throttle_reverse", "longitudinal", _state(pos=(0, 0, 17)), _controls(throttle=-1))
    add("coast_from_speed", "longitudinal", _state(pos=(0, 0, 17), vel=(1000, 0, 0)), zero)
    add(
        "brake_to_reverse",
        "longitudinal",
        _state(pos=(0, 0, 17), vel=(1000, 0, 0)),
        _controls(throttle=-1),
    )
    add(
        "ground_boost", "boost", _state(pos=(0, 0, 17), boost=80), _controls(throttle=1, boost=True)
    )

    for speed, label in ((250, "low"), (1000, "medium"), (1800, "high")):
        for amount, magnitude in ((0.5, "partial"), (1.0, "full")):
            for sign, direction in ((-1.0, "right"), (1.0, "left")):
                add(
                    f"steer_{label}_{magnitude}_{direction}",
                    "steering",
                    _state(pos=(0, 0, 17), vel=(speed, 0, 0)),
                    _controls(throttle=0.5, steer=amount * sign),
                )

    add(
        "powerslide_initiation",
        "powerslide",
        _state(pos=(0, 0, 17), vel=(900, 0, 0)),
        _controls(throttle=0.5, steer=1, handbrake=True),
    )
    add(
        "powerslide_hold",
        "powerslide",
        _state(pos=(0, 0, 17), vel=(1200, 0, 0)),
        _controls(throttle=1, steer=-1, handbrake=True),
    )
    add(
        "powerslide_release",
        "powerslide",
        _state(pos=(0, 0, 17), vel=(1000, 0, 0)),
        _controls(throttle=0.7, steer=1, handbrake=True),
        (60, _controls(throttle=0.7, steer=1, handbrake=False)),
    )

    add(
        "side_wall_transition",
        "arena_surfaces",
        _state(pos=(3920, 0, 260), vel=(700, 100, -50), pyr=(-0.5, 0.0, 0.0)),
        _controls(throttle=1),
    )
    add(
        "ramp_transition",
        "arena_surfaces",
        _state(pos=(3800, 4300, 120), vel=(500, 500, -100), pyr=(-0.3, 0.6, 0.0)),
        _controls(throttle=0.8, steer=0.4),
    )
    add(
        "back_wall_transition",
        "arena_surfaces",
        _state(pos=(700, 5050, 300), vel=(100, 700, -40), pyr=(0.0, 0.0, 0.0)),
        _controls(throttle=1),
    )
    add(
        "corner_transition",
        "arena_surfaces",
        _state(pos=(3800, 4700, 180), vel=(500, 500, -100), pyr=(-0.25, 0.7, 0.2)),
        _controls(throttle=0.8, steer=-0.5),
    )
    add(
        "ceiling_contact",
        "arena_surfaces",
        _state(pos=(0, 0, 2028), vel=(200, 0, 80), pyr=(0.0, 0.0, np.pi)),
        _controls(throttle=0.5),
    )

    add(
        "nose_impact",
        "body_contacts",
        _state(pos=(0, 0, 25), vel=(900, 0, -300), pyr=(0.65, 0, 0)),
        zero,
    )
    add(
        "side_impact",
        "body_contacts",
        _state(pos=(0, 0, 10), vel=(200, 800, -200), pyr=(0, 0, 0.8)),
        zero,
    )
    add(
        "roof_impact",
        "body_contacts",
        _state(pos=(0, 0, 22), vel=(100, 0, -500), pyr=(0, 0, np.pi)),
        zero,
    )
    add(
        "off_center_impact",
        "body_contacts",
        _state(pos=(0, 0, 5), vel=(500, 300, -350), pyr=(0.5, 0.2, 0.4)),
        zero,
    )
    add(
        "wall_scrape",
        "body_contacts",
        _state(pos=(4075, 1500, 650), vel=(100, 900, -50), pyr=(-np.pi / 2, 0.4, 0.0)),
        _controls(throttle=0.4, steer=0.5),
    )
    return tuple(scenarios)


def _state(
    *,
    pos: tuple[float, float, float],
    vel: tuple[float, float, float] = (0.0, 0.0, 0.0),
    pyr: tuple[float, float, float] = (0.0, 0.0, 0.0),
    boost: float = 100.0,
) -> StateSnapshot:
    state = StateSnapshot.empty(1)
    state.car_pos[0, 0] = pos
    state.car_vel[0, 0] = vel
    state.car_quat[0, 0] = _quat_from_euler(*pyr)
    state.car_pos[0, 1] = (2500.0, -2500.0, 17.0)
    state.boost[0, 0] = boost
    state.on_ground[0] = int(pos[2] <= 25.0)
    state.ball_pos[0] = (0.0, 0.0, 1500.0)
    state.validate()
    return state


def _controls(**values: float | bool) -> ControlBatch:
    return ControlBatch.constant(1, **values)


def _quat_from_euler(pitch: float, yaw: float, roll: float) -> np.ndarray:
    # Inputs use RocketSim/RL's pitch-yaw-roll naming; compose XYZ local axes.
    cr = np.cos(roll * 0.5)
    sr = np.sin(roll * 0.5)
    cp = np.cos(pitch * 0.5)
    sp = np.sin(pitch * 0.5)
    cy = np.cos(yaw * 0.5)
    sy = np.sin(yaw * 0.5)
    return np.asarray(
        (
            sr * cp * cy - cr * sp * sy,
            cr * sp * cy + sr * cp * sy,
            cr * cp * sy - sr * sp * cy,
            cr * cp * cy + sr * sp * sy,
        ),
        dtype=np.float32,
    )
