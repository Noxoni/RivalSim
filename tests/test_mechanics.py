from __future__ import annotations

import numpy as np
import pytest

from rivalsim import constants as c
from rivalsim.controls import ControlBatch
from rivalsim.reference.cpu_simple import CpuSimulator
from rivalsim.state import StateSnapshot


def test_gravity_tick_uses_semi_implicit_integration() -> None:
    initial = StateSnapshot.empty(1)
    initial_z = float(initial.car_pos[0, 0, 2])
    sim = CpuSimulator(initial)
    sim.step()
    state = sim.snapshot()
    expected_vel = float(c.GRAVITY_Z * c.DT)
    assert state.car_vel[0, 0, 2] == pytest.approx(expected_vel, abs=1e-6)
    assert state.car_pos[0, 0, 2] == pytest.approx(initial_z + expected_vel * float(c.DT), abs=1e-5)


def test_car_and_ball_velocity_caps() -> None:
    state = StateSnapshot.empty(1)
    state.car_vel[0, 0] = (4000.0, -3000.0, 1000.0)
    state.car_ang_vel[0, 0] = (8.0, -7.0, 6.0)
    state.ball_vel[0] = (8000.0, 1000.0, -500.0)
    state.ball_ang_vel[0] = (3.0, -7.0, 5.0)
    sim = CpuSimulator(state)
    sim.step()
    result = sim.snapshot()
    assert np.linalg.norm(result.car_vel[0, 0]) <= float(c.CAR_MAX_SPEED) + 1e-4
    assert np.linalg.norm(result.car_ang_vel[0, 0]) <= float(c.CAR_MAX_ANG_SPEED) + 1e-6
    assert np.linalg.norm(result.ball_vel[0]) <= float(c.BALL_MAX_SPEED) + 1e-3
    assert np.linalg.norm(result.ball_ang_vel[0]) <= float(c.BALL_MAX_ANG_SPEED) + 1e-6


def test_boost_use_and_depletion() -> None:
    state = StateSnapshot.empty(1)
    state.boost[0, 0] = 1.0
    controls = ControlBatch.zeros(1)
    controls.boost[0, 0] = 1
    sim = CpuSimulator(state, controls)
    sim.step()
    result = sim.snapshot()
    assert result.is_boosting[0, 0] == 1
    assert result.boost[0, 0] == pytest.approx(
        1.0 - float(c.BOOST_USED_PER_SECOND * c.DT), abs=1e-6
    )
    assert result.car_vel[0, 0, 0] > 0.0
    sim.step(20)
    result = sim.snapshot()
    assert result.boost[0, 0] == 0.0
    assert result.is_boosting[0, 0] == 0


def test_jump_edge_sticky_hold_and_timing() -> None:
    state = StateSnapshot.empty(1)
    state.on_ground[0, 0] = 1
    controls = ControlBatch.zeros(1)
    controls.jump[0, 0] = 1
    sim = CpuSimulator(state, controls)
    sim.step()
    result = sim.snapshot()
    assert result.has_jumped[0, 0] == 1
    assert result.is_jumping[0, 0] == 1
    assert result.on_ground[0, 0] == 0
    assert result.sticky_ticks[0, 0] == 2
    assert result.jump_time[0, 0] == pytest.approx(float(c.DT), abs=1e-7)
    expected = float(
        c.JUMP_IMMEDIATE_FORCE
        + (c.JUMP_ACCEL * c.JUMP_PRE_MIN_ACCEL_SCALE - c.JUMP_STICKY_ACCEL + c.GRAVITY_Z) * c.DT
    )
    assert result.car_vel[0, 0, 2] == pytest.approx(expected, abs=2e-5)

    # A held button is not another edge and cannot trigger a double jump.
    sim.step(3)
    result = sim.snapshot()
    assert result.has_double_jumped[0, 0] == 0
    controls.jump[0, 0] = 0
    sim.set_controls(controls)
    sim.step()
    assert sim.snapshot().is_jumping[0, 0] == 0


def test_legal_double_jump_and_timeout() -> None:
    ready = StateSnapshot.empty(1)
    ready.has_jumped[0, 0] = 1
    ready.jump_time[0, 0] = c.JUMP_MAX_TIME
    ready.air_time_since_jump[0, 0] = 0.2
    controls = ControlBatch.zeros(1)
    controls.jump[0, 0] = 1
    legal = CpuSimulator(ready, controls)
    before = float(ready.car_vel[0, 0, 2])
    legal.step()
    result = legal.snapshot()
    assert result.has_double_jumped[0, 0] == 1
    assert result.car_vel[0, 0, 2] > before + 280.0

    expired = ready.copy()
    expired.air_time_since_jump[0, 0] = c.DOUBLEJUMP_MAX_DELAY
    timeout = CpuSimulator(expired, controls)
    timeout.step()
    result = timeout.snapshot()
    assert result.has_double_jumped[0, 0] == 0
    assert result.has_flipped[0, 0] == 0


@pytest.mark.parametrize(
    ("control", "axis", "sign"),
    [("pitch", 1, -1), ("yaw", 2, 1), ("roll", 0, -1)],
)
def test_air_torque_axis_and_sign(control: str, axis: int, sign: int) -> None:
    controls = ControlBatch.zeros(1)
    getattr(controls, control)[0, 0] = 1.0
    sim = CpuSimulator(StateSnapshot.empty(1), controls)
    sim.step()
    angular = sim.snapshot().car_ang_vel[0, 0]
    assert np.sign(angular[axis]) == sign
    other = np.delete(angular, axis)
    assert np.max(np.abs(other)) < 1e-7


def test_orientation_remains_normalized() -> None:
    state = StateSnapshot.random(64, 19)
    controls = ControlBatch.constant(64, pitch=0.7, yaw=-0.4, roll=0.8)
    sim = CpuSimulator(state, controls)
    sim.step(600)
    result = sim.snapshot()
    assert np.allclose(np.linalg.norm(result.car_quat, axis=-1), 1.0, atol=2e-5)
    assert np.allclose(np.linalg.norm(result.ball_quat, axis=-1), 1.0, atol=2e-5)
