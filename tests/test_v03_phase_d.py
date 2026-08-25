from __future__ import annotations

import os
from collections import Counter
from dataclasses import fields

import numpy as np
import pytest
import warp as wp

from rivalsim.arena import ArenaGeometry, WarpArenaMeshes
from rivalsim.reference.v03_phase_d_oracle import (
    PHASE_D_NATIVE_BRANCHES,
    IntegratedBatchOracleFrame,
)
from rivalsim.static_world import IntegratedWorldSim
from rivalsim.v03_phase_d_cache import (
    _FIELD_SCHEMA,
    PHASE_D_AUTHORITY_SETTINGS,
    PHASE_D_CAPTURE_TICKS,
    PHASE_D_FIELDS,
    phase_d_frame_arrays,
)
from rivalsim.v03_phase_d_corpus import (
    PHASE_D_CASE_COUNT,
    PHASE_D_FAMILIES,
    generate_phase_d_cases,
    phase_d_cases_to_state,
    phase_d_controls_at,
    phase_d_corpus_sha256,
    phase_d_representative_indices,
)

EXPECTED_PHASE_D_CORPUS_SHA256 = (
    "82F5156CC001F2B85C4746E753D0A61FDCE2C3FA5116A396CF3647DCB5F01187"
)


def _frame(count: int, offset: float) -> IntegratedBatchOracleFrame:
    values = {}
    for field in fields(IntegratedBatchOracleFrame):
        dtype, tail = _FIELD_SCHEMA[field.name]
        shape = (count, *tail)
        if np.issubdtype(dtype, np.bool_):
            value = np.full(shape, int(offset) % 2, dtype=dtype)
        else:
            value = np.full(shape, offset, dtype=dtype)
        values[field.name] = value
    return IntegratedBatchOracleFrame(**values)


def test_phase_d_corpus_is_frozen_balanced_and_deterministic() -> None:
    cases = generate_phase_d_cases()
    assert len(cases) == PHASE_D_CASE_COUNT
    assert len({case.case_id for case in cases}) == PHASE_D_CASE_COUNT
    assert Counter(case.family for case in cases) == {
        family: PHASE_D_CASE_COUNT // len(PHASE_D_FAMILIES)
        for family in PHASE_D_FAMILIES
    }
    assert phase_d_corpus_sha256(cases) == EXPECTED_PHASE_D_CORPUS_SHA256
    np.testing.assert_array_equal(
        phase_d_cases_to_state(cases[:3]).car_pos,
        phase_d_cases_to_state(generate_phase_d_cases()[:3]).car_pos,
    )
    for tick in (0, 4, 11):
        controls = phase_d_controls_at(cases[:5], tick)
        assert controls.num_envs == 5


def test_phase_d_representative_covers_every_family_and_mode() -> None:
    cases = generate_phase_d_cases()
    indices = phase_d_representative_indices(cases)
    selected = [cases[index] for index in indices]
    assert {case.family for case in selected} == set(PHASE_D_FAMILIES)
    assert {case.mode for case in selected} == {case.mode for case in cases}


def test_phase_d_authority_preserves_complete_branch_axis() -> None:
    initial = [_frame(3, float(branch)) for branch in range(2)]
    branch_frames = [
        [
            _frame(3, float(branch * 100 + tick))
            for tick in PHASE_D_CAPTURE_TICKS
        ]
        for branch in range(2)
    ]
    arrays = phase_d_frame_arrays(initial, branch_frames)
    assert set(arrays) == {
        *(f"initial_{field}" for field in PHASE_D_FIELDS),
        *PHASE_D_FIELDS,
    }
    assert arrays["initial_car_pos"].shape == (3, 2, 2, 3)
    assert arrays["car_pos"].shape == (3, 2, 12, 2, 3)
    assert arrays["bump_event_count"].shape == (3, 2, 12)
    np.testing.assert_array_equal(arrays["car_pos"][:, 0, 3], 4.0)
    np.testing.assert_array_equal(arrays["car_pos"][:, 1, 3], 104.0)


def test_phase_d_relation_forbids_address_and_metric_branch_emulation() -> None:
    relation = PHASE_D_AUTHORITY_SETTINGS["native_multi_outcome_relation"]
    diagnostic = PHASE_D_AUTHORITY_SETTINGS["order_diagnostic"]
    assert relation["branches"] == list(PHASE_D_NATIVE_BRANCHES)
    assert "one complete" in relation["coherence"]
    assert "one complete" in relation["acceptance"]
    assert relation["metric_mixing"] is False
    assert relation["best_match_runtime_selection"] is False
    assert diagnostic["native_pointers_exposed"] is False
    assert diagnostic["allocator_addresses_exposed"] is False
    assert diagnostic["behavior_mutation"] is False


def test_integrated_world_preserves_lifecycle_until_membership_changes() -> None:
    collision_root = os.environ.get("RIVALSIM_COLLISION_DIR")
    if not collision_root or not wp.is_cuda_available():
        pytest.skip("exact local CMFs and CUDA are required")
    geometry = ArenaGeometry.load_soccar(collision_root)
    meshes = WarpArenaMeshes(geometry, "cuda:0")
    initial = phase_d_cases_to_state(generate_phase_d_cases()[:2])
    sim = IntegratedWorldSim(
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
    np.testing.assert_array_equal(sim.car_car.visit_order, (0, 1))

    sim.reset(initial)
    np.testing.assert_array_equal(sim.car_car.visit_order, (0, 1))
    np.testing.assert_array_equal(sim.car_car.membership_epoch, (1, 1))

    sim.car_membership_changed(np.asarray((1, 0), dtype=np.int32))
    np.testing.assert_array_equal(sim.car_car.visit_order, (1, 0))
    np.testing.assert_array_equal(sim.car_car.membership_epoch, (2, 2))
    np.testing.assert_array_equal(
        sim._dynamic_proxy_cell.numpy().reshape(2, 3),
        np.tile(np.asarray((2862, 3066, 3066), dtype=np.int32), (2, 1)),
    )
