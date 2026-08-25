from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from rivalsim.reference.rocketsim_oracle import BallWorldBatchOracleFrame
from rivalsim.v03_corpus import (
    BallWorldCase,
    _analytic_plane_cases,
    phase_a_selection_sha256,
)
from rivalsim.v03_oracle_cache import (
    PHASE_A_FRAME_FIELDS,
    V03_CACHE_CAPTURE_TICKS,
    build_phase_a_identity,
    finalize_phase_a_cache,
    freeze_phase_a_corpus,
    load_phase_a_frames,
    phase_a_frame_arrays,
    phase_cache_dir,
    write_phase_a_chunk,
)


def _case(case_id: str, x: float) -> BallWorldCase:
    return BallWorldCase(
        case_id=case_id,
        case_kind="triangle_face",
        family="ball_triangle",
        mode="normal_impact",
        mesh_index=0,
        mesh_file="mesh_0.cmf",
        target_face=0,
        target_neighbor_face=None,
        target_edge=None,
        edge_class=None,
        analytic_plane=None,
        region_labels=("test",),
        target_point=np.asarray((x, 0.0, 0.0), dtype=np.float32),
        target_normal=np.asarray((0.0, 0.0, 1.0), dtype=np.float32),
        position=np.asarray((x, 0.0, 91.0), dtype=np.float32),
        velocity=np.asarray((0.0, 0.0, -100.0), dtype=np.float32),
        quaternion=np.asarray((0.0, 0.0, 0.0, 1.0), dtype=np.float32),
        angular_velocity=np.zeros(3, dtype=np.float32),
    )


def _geometry(tmp_path: Path) -> SimpleNamespace:
    cmf = tmp_path / "mesh_0.cmf"
    cmf.write_bytes(b"v03-test-cmf")
    mesh = SimpleNamespace(path=cmf, sha256="AB" * 32)
    return SimpleNamespace(content_sha256="CD" * 32, meshes=(mesh,))


def _frame(count: int, offset: float) -> BallWorldBatchOracleFrame:
    matrix = np.repeat(np.eye(3, dtype=np.float32)[None], count, axis=0)
    return BallWorldBatchOracleFrame(
        ball_pos=np.full((count, 3), offset, dtype=np.float32),
        ball_vel=np.full((count, 3), offset + 1.0, dtype=np.float32),
        ball_matrix=matrix,
        ball_ang_vel=np.full((count, 3), offset + 2.0, dtype=np.float32),
    )


def test_soccar_analytic_ceiling_uses_pinned_2048_uu() -> None:
    ceiling = [case for case in _analytic_plane_cases() if case.analytic_plane == "ceiling"]
    assert len(ceiling) == 5
    assert all(float(case.target_point[2]) == 2048.0 for case in ceiling)


def test_phase_a_selection_hash_binds_order_and_rejects_duplicates() -> None:
    cases = (_case("A", 1.0), _case("B", 2.0))
    assert phase_a_selection_sha256(cases, (0, 1)) != phase_a_selection_sha256(
        cases, (1, 0)
    )
    with pytest.raises(ValueError, match="duplicate"):
        phase_a_selection_sha256(cases, (0, 0))


def test_phase_a_cache_preserves_post_set_state_readback(tmp_path: Path) -> None:
    cases = (_case("A", 1.0), _case("B", 2.0))
    identity = build_phase_a_identity(_geometry(tmp_path), cases)
    cache_root = tmp_path / "cache"
    cache_dir = phase_cache_dir(cache_root, identity)
    frozen = freeze_phase_a_corpus(cache_dir, identity, cases)
    initial = _frame(2, 0.25)
    frames = [
        _frame(2, float(tick))
        for tick in V03_CACHE_CAPTURE_TICKS
    ]
    arrays = phase_a_frame_arrays(initial, frames)
    assert set(arrays) == set(PHASE_A_FRAME_FIELDS)
    write_phase_a_chunk(cache_dir, identity, 0, ("A", "B"), arrays)
    finalize_phase_a_cache(cache_dir, identity, cases, frozen)

    loaded = load_phase_a_frames(cache_root, identity, cases, (1, 0))
    np.testing.assert_array_equal(
        loaded["initial_ball_pos"], initial.ball_pos[[1, 0]]
    )
    np.testing.assert_array_equal(
        loaded["ball_vel"][:, 3], frames[3].ball_vel[[1, 0]]
    )
    assert identity["identity_inputs"]["authority_tooling"]
    assert identity["identity_inputs"]["authority_settings"][
        "initial_state_custody"
    ]
