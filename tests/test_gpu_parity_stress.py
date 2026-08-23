from __future__ import annotations

import numpy as np

from rivalsim.controls import ControlBatch
from rivalsim.reference.cpu_simple import CpuSimulator
from rivalsim.simulator import RivalSim
from rivalsim.state import StateSnapshot


def test_cpu_gpu_same_equation_parity() -> None:
    state = StateSnapshot.random(256, 123)
    controls = ControlBatch.constant(
        256,
        throttle=0.6,
        pitch=-0.4,
        yaw=0.3,
        roll=-0.2,
        boost=True,
    )
    cpu = CpuSimulator(state, controls)
    gpu = RivalSim(256, randomize=False)
    gpu.reset(state)
    gpu.set_controls(controls)
    cpu.step(300)
    gpu.step(300)
    left = cpu.snapshot()
    right = gpu.snapshot()
    assert np.max(np.abs(left.car_pos - right.car_pos)) < 0.005
    assert np.max(np.abs(left.car_vel - right.car_vel)) < 0.002
    assert np.max(np.abs(left.car_quat - right.car_quat)) < 2e-5
    assert np.max(np.abs(left.car_ang_vel - right.car_ang_vel)) < 2e-5
    assert np.array_equal(left.has_flipped, right.has_flipped)
    assert np.array_equal(left.has_double_jumped, right.has_double_jumped)


def test_no_nans_over_long_contact_free_random_stress() -> None:
    state = StateSnapshot.random(1024, 8675309)
    rng = np.random.default_rng(44)
    controls = ControlBatch.zeros(1024)
    for array in controls.arrays()[:5]:
        array[...] = rng.uniform(-1.0, 1.0, array.shape)
    controls.jump[...] = rng.random(controls.jump.shape) < 0.2
    controls.boost[...] = rng.random(controls.boost.shape) < 0.5
    gpu = RivalSim(1024, randomize=False)
    gpu.reset(state)
    gpu.set_controls(controls)
    gpu.step(2400)
    result = gpu.snapshot()
    result.validate()
    assert np.isfinite(result.car_pos).all()
    assert np.isfinite(result.car_vel).all()
    assert np.isfinite(result.car_quat).all()
    assert np.isfinite(result.ball_pos).all()
    assert np.isfinite(result.ball_vel).all()
