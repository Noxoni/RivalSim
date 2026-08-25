from __future__ import annotations

import os
from dataclasses import fields

import numpy as np
import pytest
import warp as wp

from rivalsim.arena import ArenaGeometry, WarpArenaMeshes
from rivalsim.car_car_state import CarCarState
from rivalsim.reference.v03_phase_c_oracle import (
    PHASE_C_NATIVE_BRANCHES,
    CarCarBatchOracleFrame,
)
from rivalsim.static_world import CarCarWorldSim
from rivalsim.v03_phase_c_cache import (
    _FIELD_SCHEMA,
    PHASE_C_AUTHORITY_SETTINGS,
    PHASE_C_CAPTURE_TICKS,
    PHASE_C_FIELDS,
    phase_c_frame_arrays,
)
from rivalsim.v03_phase_c_corpus import generate_phase_c_cases, phase_c_cases_to_state


def _frame(count: int, offset: float) -> CarCarBatchOracleFrame:
    values = {}
    for field in fields(CarCarBatchOracleFrame):
        dtype, tail = _FIELD_SCHEMA[field.name]
        shape = (count, *tail)
        if np.issubdtype(dtype, np.bool_):
            value = np.full(shape, int(offset) % 2, dtype=dtype)
        else:
            value = np.full(shape, offset, dtype=dtype)
        values[field.name] = value
    return CarCarBatchOracleFrame(**values)


def test_car_visit_order_is_persistent_membership_lifecycle_state() -> None:
    selected = np.asarray((0, 1, 1, 0), dtype=np.int32)
    state = CarCarState(
        4,
        "cpu",
        lifecycle_seed=91,
        pre_tick_first_car=selected,
    )
    snapshot = state.snapshot()
    np.testing.assert_array_equal(snapshot.pre_tick_first_car, selected)
    np.testing.assert_array_equal(snapshot.membership_epoch, np.ones(4, dtype=np.uint64))

    preserved = CarCarState(4, "cpu", **state.lifecycle_copy_kwargs())
    np.testing.assert_array_equal(preserved.visit_order, selected)
    np.testing.assert_array_equal(preserved.membership_epoch, np.ones(4, dtype=np.uint64))

    preserved.membership_changed("a_then_b")
    np.testing.assert_array_equal(preserved.visit_order, np.zeros(4, dtype=np.int32))
    np.testing.assert_array_equal(
        preserved.membership_epoch, np.full(4, 2, dtype=np.uint64)
    )


def test_generic_lifecycle_selection_has_no_physical_or_case_input() -> None:
    first = CarCarState(32, "cpu", lifecycle_seed=20260825)
    second = CarCarState(32, "cpu", lifecycle_seed=20260825)
    np.testing.assert_array_equal(first.visit_order, second.visit_order)
    assert set(first.visit_order.tolist()) == {0, 1}

    before = first.visit_order
    first.membership_changed()
    assert np.any(first.visit_order != before)
    np.testing.assert_array_equal(first.membership_epoch, np.full(32, 2, dtype=np.uint64))


def test_phase_c_authority_preserves_complete_branch_axis() -> None:
    initial = [_frame(3, float(branch)) for branch in range(2)]
    branch_frames = [
        [
            _frame(3, float(branch * 100 + tick))
            for tick in PHASE_C_CAPTURE_TICKS
        ]
        for branch in range(2)
    ]
    arrays = phase_c_frame_arrays(initial, branch_frames)
    assert set(arrays) == {
        *(f"initial_{field}" for field in PHASE_C_FIELDS),
        *PHASE_C_FIELDS,
    }
    assert arrays["initial_car_pos"].shape == (3, 2, 2, 3)
    assert arrays["car_pos"].shape == (3, 2, 12, 2, 3)
    assert arrays["bump_event_count"].shape == (3, 2, 12)
    np.testing.assert_array_equal(arrays["car_pos"][:, 0, 3], 4.0)
    np.testing.assert_array_equal(arrays["car_pos"][:, 1, 3], 104.0)


def test_phase_c_relation_forbids_pointer_and_metric_branch_emulation() -> None:
    relation = PHASE_C_AUTHORITY_SETTINGS["native_multi_outcome_relation"]
    diagnostic = PHASE_C_AUTHORITY_SETTINGS["order_diagnostic"]
    assert relation["branches"] == list(PHASE_C_NATIVE_BRANCHES)
    assert "one complete" in relation["acceptance"]
    assert "one complete labeled branch" in relation["coherence"]
    assert relation["runtime_selection"] == (
        "none; authority comparison never selects a branch inside RivalSim"
    )
    assert diagnostic["native_pointers_exposed"] is False
    assert diagnostic["allocator_addresses_exposed"] is False
    assert diagnostic["behavior_mutation"] is False


def test_gpu_world_preserves_order_across_reset_and_changes_on_membership() -> None:
    collision_root = os.environ.get("RIVALSIM_COLLISION_DIR")
    if not collision_root or not wp.is_cuda_available():
        pytest.skip("exact local CMFs and CUDA are required")
    geometry = ArenaGeometry.load_soccar(collision_root)
    meshes = WarpArenaMeshes(geometry, "cuda:0")
    cases = generate_phase_c_cases()[:2]
    initial = phase_c_cases_to_state(cases)
    sim = CarCarWorldSim(
        2,
        collision_root,
        variant="B3",
        device="cuda:0",
        initial=initial,
        geometry=geometry,
        meshes=meshes,
        car_visitation_order=np.asarray((0, 1), dtype=np.int32),
    )
    sim.step(1, synchronize=True)
    np.testing.assert_array_equal(sim.car_car.snapshot().pre_tick_first_car, (0, 1))

    sim.reset(initial)
    reset = sim.car_car.snapshot()
    np.testing.assert_array_equal(reset.pre_tick_first_car, (0, 1))
    np.testing.assert_array_equal(reset.membership_epoch, (1, 1))

    sim.car_membership_changed(np.asarray((1, 0), dtype=np.int32))
    changed = sim.car_car.snapshot()
    np.testing.assert_array_equal(changed.pre_tick_first_car, (1, 0))
    np.testing.assert_array_equal(changed.membership_epoch, (2, 2))
