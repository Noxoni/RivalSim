from __future__ import annotations

import os

import numpy as np
import pytest
import warp as wp

from rivalsim.arena import ArenaGeometry, WarpArenaMeshes
from rivalsim.controls import ControlBatch
from rivalsim.state import StateSnapshot
from rivalsim.static_world import ActionTape, StaticWorldSim, make_contact_rich_state


@pytest.fixture(scope="module")
def arena_assets() -> tuple[str, ArenaGeometry, WarpArenaMeshes]:
    root = os.environ.get("RIVALSIM_COLLISION_DIR")
    if not root or not wp.is_cuda_available():
        pytest.skip("exact local CMFs and CUDA are required")
    geometry = ArenaGeometry.load_soccar(root)
    return root, geometry, WarpArenaMeshes(geometry)


def _floor_state(num_envs: int = 1) -> StateSnapshot:
    state = StateSnapshot.empty(num_envs)
    state.car_pos[..., 2] = 17.0
    state.car_pos[:, 1, 1] = 1000.0
    return state


def _sim(
    assets: tuple[str, ArenaGeometry, WarpArenaMeshes],
    state: StateSnapshot,
    *,
    variant: str = "B3",
) -> StaticWorldSim:
    root, geometry, meshes = assets
    return StaticWorldSim(
        state.num_envs,
        root,
        variant=variant,
        initial=state,
        geometry=geometry,
        meshes=meshes,
    )


def test_four_wheel_state_and_suspension_force(arena_assets) -> None:
    sim = _sim(arena_assets, _floor_state())
    sim.step(2, synchronize=True)
    state = sim.snapshot()
    vehicle = sim.vehicle_snapshot()
    assert vehicle.wheel_contact.shape == (2, 4)
    assert vehicle.wheels_with_contact[0] == 4
    assert state.on_ground[0, 0] == 1
    assert np.all(vehicle.wheel_hit_distance[0] > 0.0)
    assert np.all(vehicle.wheel_hit_face[0] != -1)
    assert np.all(vehicle.suspension_clipped_factor[0] >= 1.0)
    assert np.all(vehicle.suspension_force[0] > 0.0)


def test_throttle_reverse_coast_and_brake(arena_assets) -> None:
    forward = _sim(arena_assets, _floor_state())
    forward.set_controls(ControlBatch.constant(1, throttle=1.0))
    forward.step(60, synchronize=True)
    forward_speed = float(forward.snapshot().car_vel[0, 0, 0])

    reverse = _sim(arena_assets, _floor_state())
    reverse.set_controls(ControlBatch.constant(1, throttle=-1.0))
    reverse.step(60, synchronize=True)
    reverse_speed = float(reverse.snapshot().car_vel[0, 0, 0])
    assert forward_speed > 300.0
    assert reverse_speed < -300.0

    moving = _floor_state()
    moving.car_vel[0, 0, 0] = 900.0
    coast = _sim(arena_assets, moving)
    coast.set_controls(ControlBatch.zeros(1))
    coast.step(30, synchronize=True)
    coast_speed = float(coast.snapshot().car_vel[0, 0, 0])

    brake = _sim(arena_assets, moving)
    brake.set_controls(ControlBatch.constant(1, throttle=-1.0))
    brake.step(30, synchronize=True)
    brake_speed = float(brake.snapshot().car_vel[0, 0, 0])
    assert 0.0 < brake_speed < coast_speed < 900.0


def test_steering_symmetry_and_powerslide_state(arena_assets) -> None:
    outputs = []
    for steer in (-1.0, 1.0):
        sim = _sim(arena_assets, _floor_state())
        sim.set_controls(ControlBatch.constant(1, throttle=0.65, steer=steer))
        sim.step(24, synchronize=True)
        outputs.append(sim.snapshot())
    assert np.sign(outputs[0].car_pos[0, 0, 1]) == -np.sign(outputs[1].car_pos[0, 0, 1])
    assert abs(abs(outputs[0].car_pos[0, 0, 1]) - abs(outputs[1].car_pos[0, 0, 1])) < 8.0

    normal = _sim(arena_assets, _floor_state())
    normal.set_controls(ControlBatch.constant(1, throttle=1.0, steer=1.0))
    normal.step(12, synchronize=True)
    normal_vehicle = normal.vehicle_snapshot()
    sliding = _sim(arena_assets, _floor_state())
    sliding.set_controls(ControlBatch.constant(1, throttle=1.0, steer=1.0, handbrake=True))
    sliding.step(12, synchronize=True)
    slide_vehicle = sliding.vehicle_snapshot()
    assert slide_vehicle.handbrake_value[0] == pytest.approx(0.5, abs=1e-5)
    assert slide_vehicle.lateral_friction[0].mean() < normal_vehicle.lateral_friction[0].mean()


def test_obb_broadphase_narrowphase_and_off_center_angular_response(arena_assets) -> None:
    wall = make_contact_rich_state(8)
    sim = _sim(arena_assets, wall)
    sim.step(1, synchronize=True)
    vehicle = sim.vehicle_snapshot()
    assert vehicle.candidate_count.max() > 0
    assert vehicle.contact_count.max() > 0
    assert np.any(vehicle.contact_penetration > 0.0)

    body = _floor_state()
    body.car_pos[0, 0] = (0.0, 0.0, -10.0)
    body.car_vel[0, 0] = (500.0, 0.0, -100.0)
    sim = _sim(arena_assets, body)
    sim.step(1, synchronize=True)
    state = sim.snapshot()
    vehicle = sim.vehicle_snapshot()
    assert vehicle.contact_count[0] >= 1
    assert state.car_vel[0, 0, 2] > -10.0
    assert np.linalg.norm(state.car_ang_vel[0, 0]) > 0.05


def test_floor_rest_and_contact_rich_2400_tick_stress(arena_assets) -> None:
    rest = _sim(arena_assets, _floor_state())
    rest.step(600, synchronize=True)
    rest_state = rest.snapshot()
    assert 15.0 < rest_state.car_pos[0, 0, 2] < 20.0
    assert np.linalg.norm(rest_state.car_vel[0, 0]) < 2.0

    initial = make_contact_rich_state(32)
    stress = _sim(arena_assets, initial)
    stress.set_action_tape(ActionTape.deterministic())
    stress.step(2400, synchronize=True)
    state = stress.snapshot()
    vehicle = stress.vehicle_snapshot()
    assert np.isfinite(state.car_pos).all()
    assert np.isfinite(state.car_vel).all()
    assert np.isfinite(state.car_ang_vel).all()
    assert np.max(np.linalg.norm(state.car_vel, axis=-1)) <= 2300.001
    assert np.max(np.linalg.norm(state.car_ang_vel, axis=-1)) <= 5.5001
    assert np.max(vehicle.penetration_max) < 100.0


def test_contact_rich_stress_is_repeatably_deterministic(arena_assets) -> None:
    initial = make_contact_rich_state(16, seed=77)
    outputs = []
    for _ in range(2):
        sim = _sim(arena_assets, initial.copy())
        sim.set_action_tape(ActionTape.deterministic())
        sim.step(2400, synchronize=True)
        outputs.append((sim.snapshot(), sim.vehicle_snapshot()))
    left_state, left_vehicle = outputs[0]
    right_state, right_vehicle = outputs[1]
    for name in ("car_pos", "car_vel", "car_quat", "car_ang_vel", "boost"):
        np.testing.assert_array_equal(getattr(left_state, name), getattr(right_state, name))
    for name in (
        "wheel_contact",
        "wheel_hit_distance",
        "contact_count",
        "penetration_max",
    ):
        np.testing.assert_array_equal(getattr(left_vehicle, name), getattr(right_vehicle, name))


def test_soccar_boost_pad_pickup_cooldown_and_reset(arena_assets) -> None:
    big = StateSnapshot.empty(1)
    big.car_pos[0, 0] = (-3584.0, 0.0, 73.0)
    big.boost[0, 0] = 0.0
    sim = _sim(arena_assets, big)
    sim.step(1, synchronize=True)
    assert sim.snapshot().boost[0, 0] == pytest.approx(100.0)
    assert float(sim.boost_pad_cooldown.numpy()[0]) == pytest.approx(10.0)

    sim.step(1, synchronize=True)
    assert float(sim.boost_pad_cooldown.numpy()[0]) == pytest.approx(10.0 - 1.0 / 120.0)

    small = StateSnapshot.empty(1)
    small.car_pos[0, 0] = (0.0, -4240.0, 70.0)
    small.boost[0, 0] = 5.0
    sim.reset(small)
    sim.step(1, synchronize=True)
    assert sim.snapshot().boost[0, 0] == pytest.approx(17.0)
    cooldown = sim.boost_pad_cooldown.numpy().reshape(1, -1)
    assert float(cooldown[0, 6]) == pytest.approx(4.0)
