from __future__ import annotations

from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from rivalsim.dfh_breadth import BreadthCase, corpus_sha256
from rivalsim.math import bullet_matrix_to_quat, matrix_to_quat, quat_to_matrix
from rivalsim.state import StateSnapshot
from rivalsim.v022_oracle_cache import (
    AUTHORITY_SETTINGS,
    CACHE_CAPTURE_TICKS,
    FRAME_FIELDS,
    RocketSimAuthorityCache,
    authority_cache_dir,
    authority_chunk_path,
    build_authority_identity,
    canonical_json_bytes,
    finalize_authority_cache,
    sha256_bytes,
    validate_authority_chunk,
    validate_or_write_identity,
    verify_complete_authority_cache,
    write_authority_chunk,
    write_frozen_corpus,
)


def _case(case_id: str, position_x: float) -> BreadthCase:
    vector = np.asarray((0.0, 0.0, 1.0), dtype=np.float32)
    return BreadthCase(
        case_id=case_id,
        case_kind="triangle_face",
        family="triangle_chassis",
        mode="normal_impact",
        contact_path="chassis",
        mesh_index=0,
        mesh_file="mesh_0.cmf",
        target_face=0,
        target_neighbor_face=None,
        target_edge=None,
        edge_class=None,
        analytic_plane=None,
        expected_plane_face=None,
        target_point=np.zeros(3, dtype=np.float32),
        target_normal=vector,
        edge_start=None,
        edge_end=None,
        region_labels=("test",),
        position=np.asarray((position_x, 0.0, 20.0), dtype=np.float32),
        quaternion=np.asarray((0.0, 0.0, 0.0, 1.0), dtype=np.float32),
        velocity=np.zeros(3, dtype=np.float32),
        angular_velocity=np.zeros(3, dtype=np.float32),
        controls=(0.0, 0.0, 0.0, 0.0, 0.0, 0, 0, 0),
    )


def _identity(tmp_path: Path, cases: tuple[BreadthCase, ...]) -> dict[str, object]:
    cmf = tmp_path / "mesh_0.cmf"
    cmf.write_bytes(b"frozen-cmf")
    mesh = SimpleNamespace(path=cmf, sha256="AB" * 32)
    geometry = SimpleNamespace(content_sha256="CD" * 32, meshes=(mesh,))
    return build_authority_identity(geometry, cases)


def _frames(count: int) -> dict[str, np.ndarray]:
    ticks = len(CACHE_CAPTURE_TICKS)
    result = {
        "car_pos": np.zeros((ticks, count, 3), dtype=np.float32),
        "car_vel": np.zeros((ticks, count, 3), dtype=np.float32),
        "car_matrix": np.zeros((ticks, count, 3, 3), dtype=np.float32),
        "car_ang_vel": np.zeros((ticks, count, 3), dtype=np.float32),
        "boost": np.full((ticks, count), 100.0, dtype=np.float32),
        "handbrake_value": np.zeros((ticks, count), dtype=np.float32),
        "on_ground": np.zeros((ticks, count), dtype=np.bool_),
        "wheel_contacts": np.zeros((ticks, count, 4), dtype=np.bool_),
        "has_world_contact": np.zeros((ticks, count), dtype=np.bool_),
        "world_contact_normal": np.zeros((ticks, count, 3), dtype=np.float32),
    }
    result["car_pos"][:, 0, 0] = np.arange(ticks, dtype=np.float32)
    if count > 1:
        result["car_pos"][:, 1, 0] = np.arange(ticks, dtype=np.float32) + 100.0
    return result


def test_bullet_matrix_readback_matches_traced_simd_rounding() -> None:
    source = np.asarray(
        (-0.357999176, 0.629101992, -0.015083136, 0.689811409),
        dtype=np.float32,
    )
    expected = np.asarray(
        (-0.357999235, 0.629102051, -0.0150831491, 0.689811409),
        dtype=np.float32,
    )
    np.testing.assert_array_equal(
        bullet_matrix_to_quat(quat_to_matrix(source)),
        expected,
    )


def test_authority_identity_binds_requested_inputs(tmp_path: Path) -> None:
    cases = (_case("A", 1.0),)
    identity = _identity(tmp_path, cases)
    inputs = identity["identity_inputs"]

    assert identity["authority_identity_sha256"] == sha256_bytes(canonical_json_bytes(inputs))
    assert inputs["rocketsim"]["primary_commit"]
    assert inputs["collision_assets"]["combined_content_sha256"] == "CD" * 32
    assert inputs["collision_assets"]["files"][0]["sha256"] == "AB" * 32
    assert inputs["corpus"]["generator_source_sha256"]
    assert inputs["corpus"]["generator_config_sha256"]
    assert inputs["corpus"]["seed"] == 20260823
    assert inputs["corpus"]["corpus_sha256"] == corpus_sha256(cases)
    assert inputs["authority_settings"] == AUTHORITY_SETTINGS
    assert inputs["authority_settings_sha256"] == sha256_bytes(
        canonical_json_bytes(AUTHORITY_SETTINGS)
    )

    changed_cases = (replace(cases[0], position=np.asarray((2.0, 0.0, 20.0), dtype=np.float32)),)
    assert (
        _identity(tmp_path, changed_cases)["authority_identity_sha256"]
        != identity["authority_identity_sha256"]
    )


def test_complete_cache_round_trip_and_reordered_selection(tmp_path: Path) -> None:
    source_quaternion = np.asarray(
        (-0.147855073, 0.491985708, 0.543017209, 0.664244890),
        dtype=np.float32,
    )
    source_velocity = np.asarray((3.25, -7.5, 11.0), dtype=np.float32)
    source_angular_velocity = np.asarray((0.25, -0.5, 0.75), dtype=np.float32)
    cases = (
        _case("A", 1.0),
        replace(
            _case("B", 2.0),
            quaternion=source_quaternion,
            velocity=source_velocity,
            angular_velocity=source_angular_velocity,
        ),
    )
    identity = _identity(tmp_path, cases)
    cache_root = tmp_path / "cache"
    cache_dir = authority_cache_dir(cache_root, identity)
    validate_or_write_identity(cache_dir, identity)
    corpus_artifact = write_frozen_corpus(cache_dir, identity, cases)
    initial = StateSnapshot.empty(2)
    initial.car_pos[:, 0, 0] = (1.0, 2.0)
    initial.car_pos[1, 0, 0] = np.nextafter(np.float32(2.0), np.float32(0.0))
    initial.car_vel[1, 0] = np.nextafter(source_velocity, np.float32(0.0))
    initial.car_quat[1, 0] = matrix_to_quat(quat_to_matrix(source_quaternion))
    initial.car_ang_vel[1, 0] = np.nextafter(
        source_angular_velocity, np.float32(0.0)
    )
    frames = _frames(2)
    chunk = authority_chunk_path(cache_dir, 0, 2)
    write_authority_chunk(
        chunk,
        identity,
        0,
        tuple(case.case_id for case in cases),
        initial,
        frames,
    )
    validate_authority_chunk(chunk, identity, 0, ("A", "B"))
    finalize_authority_cache(cache_dir, identity, cases, corpus_artifact)
    verify_complete_authority_cache(cache_root, identity, cases)

    cached = RocketSimAuthorityCache(cache_root, identity, cases).load((1, 0))
    np.testing.assert_array_equal(
        cached.authoritative_snapshot.car_pos[:, 0, 0],
        np.asarray((2.0, 1.0), dtype=np.float32),
    )
    np.testing.assert_array_equal(
        cached.authoritative_snapshot.car_quat[0, 0],
        source_quaternion,
    )
    np.testing.assert_array_equal(
        cached.authoritative_snapshot.car_vel[0, 0],
        source_velocity,
    )
    np.testing.assert_array_equal(
        cached.authoritative_snapshot.car_ang_vel[0, 0],
        source_angular_velocity,
    )
    assert not np.array_equal(initial.car_quat[1, 0], source_quaternion)
    assert set(cached.frames) == set(FRAME_FIELDS)
    np.testing.assert_array_equal(
        cached.frame(4).car_pos[:, 0],
        np.asarray((103.0, 3.0), dtype=np.float32),
    )
