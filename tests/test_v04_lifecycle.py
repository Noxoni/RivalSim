from __future__ import annotations

import os

import numpy as np
import pytest
import warp as wp

from rivalsim import CompleteWorldSim, StateSnapshot, make_standard_kickoff_state
from rivalsim.arena import ArenaGeometry, WarpArenaMeshes
from rivalsim.kernels.boost_pad import SOCCAR_PAD_POSITIONS
from rivalsim.v03_phase_c_corpus import generate_phase_c_cases, phase_c_cases_to_state
from rivalsim.v04_authority import load_cache


@pytest.fixture(scope="module")
def arena_assets() -> tuple[str, ArenaGeometry, WarpArenaMeshes]:
    root = os.environ.get("RIVALSIM_COLLISION_DIR")
    if not root or not wp.is_cuda_available():
        pytest.skip("exact local CMFs and CUDA are required")
    geometry = ArenaGeometry.load_soccar(root)
    return root, geometry, WarpArenaMeshes(geometry)


def _sim(
    assets: tuple[str, ArenaGeometry, WarpArenaMeshes],
    state: StateSnapshot,
    **kwargs: object,
) -> CompleteWorldSim:
    root, geometry, meshes = assets
    return CompleteWorldSim(
        state.num_envs,
        root,
        initial=state,
        geometry=geometry,
        meshes=meshes,
        car_visitation_order="a_then_b",
        **kwargs,
    )


def test_standard_kickoff_state_covers_all_source_layouts() -> None:
    state = make_standard_kickoff_state(5, np.arange(5, dtype=np.int32))
    np.testing.assert_array_equal(
        state.car_pos[:, 0, :2],
        np.asarray(
            (
                (-2048.0, -2559.9998),
                (2048.0, -2559.9998),
                (-256.0, -3839.9998),
                (256.0, -3839.9998),
                (0.0, -4608.0),
            ),
            dtype=np.float32,
        ),
    )
    np.testing.assert_array_equal(state.car_pos[:, 1, :2], -state.car_pos[:, 0, :2])
    assert np.all(state.on_ground == 1)
    assert np.all(state.boost == np.float32(100.0 / 3.0))
    assert np.all(state.ball_pos[:, 2] == np.float32(93.1500015258789))


def test_goal_score_kickoff_reset_is_resident_and_preserves_order(arena_assets) -> None:
    state = StateSnapshot.empty(5)
    state.car_pos[:] = np.asarray((0.0, 0.0, 1500.0), dtype=np.float32)
    state.ball_pos[:] = np.asarray((0.0, 5300.0, 93.15), dtype=np.float32)
    root, geometry, meshes = arena_assets
    order = np.asarray((0, 1, 0, 1, 0), dtype=np.int32)
    sim = CompleteWorldSim(
        5,
        root,
        initial=state,
        geometry=geometry,
        meshes=meshes,
        kickoff_selector=np.arange(5, dtype=np.int32),
        car_visitation_order=order,
    )
    sim.request_demolition(0)
    sim.reset_transfer_counters()
    sim.step(1, synchronize=True)
    assert sim.host_to_device_bytes == 0
    assert sim.device_to_host_bytes == 0
    lifecycle = sim.lifecycle_snapshot()
    output = sim.snapshot()
    assert np.all(lifecycle.goal_scored == 1)
    assert np.all(lifecycle.scoring_team == 0)
    assert np.all(lifecycle.blue_score == 1)
    np.testing.assert_array_equal(lifecycle.kickoff_layout, np.arange(5))
    assert np.all(lifecycle.pad_active == 1)
    assert np.all(lifecycle.car_is_demoed == 0)
    np.testing.assert_array_equal(sim.car_car.visit_order, order)
    assert np.all(output.ball_pos[:, 2] == np.float32(93.1500015258789))


def test_pad_contention_uses_persistent_car_visitation_order(arena_assets) -> None:
    state = StateSnapshot.empty(2)
    state.car_pos[:] = SOCCAR_PAD_POSITIONS[0]
    state.car_vel[:] = 0.0
    state.boost[:] = 0.0
    state.ball_pos[:] = np.asarray((0.0, 0.0, 1500.0), dtype=np.float32)
    root, geometry, meshes = arena_assets
    sim = CompleteWorldSim(
        2,
        root,
        initial=state,
        geometry=geometry,
        meshes=meshes,
        car_visitation_order=np.asarray((0, 1), dtype=np.int32),
        auto_kickoff=False,
    )
    sim.step(1, synchronize=True)
    lifecycle = sim.lifecycle_snapshot()
    np.testing.assert_array_equal(lifecycle.pad_pickup_car[:, 0], (2, 1))
    np.testing.assert_array_equal(sim.snapshot().boost, ((0.0, 100.0), (100.0, 0.0)))


def test_demo_state_freezes_then_respawns_without_membership_change(arena_assets) -> None:
    state = StateSnapshot.empty(1)
    state.car_pos[0, 0] = (123.0, 456.0, 1000.0)
    state.car_vel[0, 0] = (10.0, 20.0, 30.0)
    sim = _sim(arena_assets, state, respawn_selector=2, auto_kickoff=False)
    order = sim.car_car.visit_order
    sim.request_demolition(0)
    sim.step(1, synchronize=True)
    event = sim.snapshot()
    assert sim.lifecycle_snapshot().demo_respawn_timer[0, 0] == np.float32(3.0)
    sim.step(359, synchronize=True)
    before = sim.snapshot()
    np.testing.assert_array_equal(before.car_pos[0, 0], event.car_pos[0, 0])
    np.testing.assert_array_equal(before.car_vel[0, 0], event.car_vel[0, 0])
    assert sim.lifecycle_snapshot().demo_respawn_timer[0, 0] == np.float32(0.008321664296090603)
    sim.step(1, synchronize=True)
    lifecycle = sim.lifecycle_snapshot()
    output = sim.snapshot()
    assert lifecycle.respawn_event[0, 0] == 1
    assert lifecycle.respawn_location[0, 0] == 2
    assert lifecycle.car_is_demoed[0, 0] == 0
    np.testing.assert_array_equal(output.car_pos[0, 0], (2304.0, -4608.0, 36.0))
    np.testing.assert_array_equal(sim.car_car.visit_order, order)


def test_source_backed_car_car_demo_enters_lifecycle(arena_assets) -> None:
    case = generate_phase_c_cases()[1]
    state = phase_c_cases_to_state((case,))
    sim = _sim(arena_assets, state, auto_kickoff=False)
    sim.step(1, synchronize=True)
    lifecycle = sim.lifecycle_snapshot()
    np.testing.assert_array_equal(sim.car_car.event_is_demo.numpy(), (1, 0, 0, 0))
    np.testing.assert_array_equal(lifecycle.car_is_demoed, ((0, 1),))
    assert lifecycle.demo_respawn_timer[0, 1] == np.float32(3.0)


def test_host_full_reset_returns_to_configured_kickoff(arena_assets) -> None:
    state = StateSnapshot.empty(1)
    state.car_pos[:] = np.asarray((100.0, 200.0, 300.0), dtype=np.float32)
    sim = _sim(arena_assets, state, kickoff_selector=3)
    sim.step(2, synchronize=True)
    sim.reset()
    expected = make_standard_kickoff_state(1, 3)
    actual = sim.snapshot()
    np.testing.assert_array_equal(actual.car_pos, expected.car_pos)
    np.testing.assert_array_equal(actual.car_quat, expected.car_quat)
    np.testing.assert_array_equal(actual.ball_pos, expected.ball_pos)
    lifecycle = sim.lifecycle_snapshot()
    assert lifecycle.world_tick[0] == 0
    assert lifecycle.episode_tick[0] == 0
    assert lifecycle.blue_score[0] == 0
    assert lifecycle.orange_score[0] == 0


def test_missing_v04_authority_has_no_live_fallback(tmp_path) -> None:
    collision_root = os.environ.get("RIVALSIM_COLLISION_DIR")
    if not collision_root:
        pytest.skip("exact local CMFs are required")
    with pytest.raises(FileNotFoundError, match="no live fallback"):
        load_cache(collision_root, tmp_path)
