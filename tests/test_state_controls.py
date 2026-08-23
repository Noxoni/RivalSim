from __future__ import annotations

import numpy as np

from rivalsim.controls import ControlBatch
from rivalsim.simulator import RivalSim
from rivalsim.state import StateSnapshot


def test_state_shape_and_gpu_allocation() -> None:
    sim = RivalSim(8, randomize=False)
    assert sim.state.car_count == 16
    assert sim.state.car_pos.shape == (16,)
    assert sim.state.ball_pos.shape == (8,)
    snapshot = sim.snapshot()
    assert snapshot.car_pos.shape == (8, 2, 3)
    assert snapshot.car_quat.shape == (8, 2, 4)
    assert snapshot.ball_pos.shape == (8, 3)
    assert snapshot.ball_quat.shape == (8, 4)
    assert sim.logical_state_bytes == snapshot.nbytes


def test_deterministic_reset() -> None:
    expected = StateSnapshot.random(16, seed=0xC0FFEE)
    duplicate = StateSnapshot.random(16, seed=0xC0FFEE)
    for name in expected.__dataclass_fields__:
        assert np.array_equal(getattr(expected, name), getattr(duplicate, name))

    sim = RivalSim(16, randomize=False)
    sim.reset(seed=0xC0FFEE)
    first = sim.snapshot()
    sim.step(4)
    sim.reset(seed=0xC0FFEE)
    second = sim.snapshot()
    for name in expected.__dataclass_fields__:
        assert np.array_equal(getattr(first, name), getattr(second, name))


def test_control_clamping_and_previous_controls() -> None:
    controls = ControlBatch.constant(
        2,
        throttle=7.0,
        steer=-4.0,
        pitch=2.0,
        yaw=-3.0,
        roll=9.0,
        jump=True,
        boost=True,
        handbrake=True,
    )
    assert np.all(controls.throttle == 1.0)
    assert np.all(controls.steer == -1.0)
    assert np.all(controls.pitch == 1.0)
    assert np.all(controls.yaw == -1.0)
    assert np.all(controls.roll == 1.0)

    sim = RivalSim(2, randomize=False)
    sim.set_controls(controls)
    sim.step()
    state = sim.snapshot()
    assert np.all(state.prev_throttle == 1.0)
    assert np.all(state.prev_steer == -1.0)
    assert np.all(state.prev_pitch == 1.0)
    assert np.all(state.prev_yaw == -1.0)
    assert np.all(state.prev_roll == 1.0)
    assert np.all(state.prev_jump == 1)
    assert np.all(state.prev_boost == 1)
    assert np.all(state.prev_handbrake == 1)
