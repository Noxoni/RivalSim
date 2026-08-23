from __future__ import annotations

import struct
from pathlib import Path

import numpy as np
import pytest

from rivalsim.arena import (
    ArenaGeometry,
    raycast_soccar_cpu,
    raycast_triangles_cpu,
    read_cmf,
)


def _write_test_cmf(path: Path) -> None:
    triangles = np.array(((0, 1, 2),), dtype="<i4")
    vertices = np.array(((0, 0, 0), (1, 0, 0), (0, 1, 0)), dtype="<f4")
    path.write_bytes(struct.pack("<ii", 1, 3) + triangles.tobytes() + vertices.tobytes())


def test_cmf_parser_counts_bounds_and_determinism(tmp_path: Path) -> None:
    path = tmp_path / "mesh_0.cmf"
    _write_test_cmf(path)
    first = read_cmf(path)
    second = read_cmf(path)
    assert first.triangle_count == 1
    assert first.vertex_count == 3
    assert first.sha256 == second.sha256
    assert first.rocketsim_hash == second.rocketsim_hash
    assert first.vertices_uu.max() == 50.0


def test_cmf_parser_rejects_trailing_bytes_and_bad_indices(tmp_path: Path) -> None:
    path = tmp_path / "bad.cmf"
    _write_test_cmf(path)
    path.write_bytes(path.read_bytes() + b"x")
    with pytest.raises(ValueError, match="invalid CMF size"):
        read_cmf(path)

    triangles = np.array(((0, 1, 3),), dtype="<i4")
    vertices = np.zeros((3, 3), dtype="<f4")
    path.write_bytes(struct.pack("<ii", 1, 3) + triangles.tobytes() + vertices.tobytes())
    with pytest.raises(ValueError, match="out-of-range"):
        read_cmf(path)


def test_cpu_triangle_raycast_hit_miss_distance_and_winding_normal() -> None:
    vertices = np.array(((0, 0, 0), (10, 0, 0), (0, 10, 0)), dtype=np.float32)
    triangles = np.array(((0, 1, 2),), dtype=np.int32)
    origins = np.array(((1, 1, 5), (20, 20, 5)), dtype=np.float32)
    directions = np.array(((0, 0, -1), (0, 0, -1)), dtype=np.float32)
    hit, distance, normal, face = raycast_triangles_cpu(
        vertices, triangles, origins, directions, 20.0
    )
    np.testing.assert_array_equal(hit, (1, 0))
    np.testing.assert_allclose(distance, (5.0, 20.0))
    np.testing.assert_allclose(normal[0], (0.0, 0.0, 1.0))
    np.testing.assert_array_equal(face, (0, -1))


def test_exact_local_soccar_manifest_when_configured(monkeypatch: pytest.MonkeyPatch) -> None:
    import os

    configured = os.environ.get("RIVALSIM_COLLISION_DIR")
    if not configured:
        pytest.skip("RIVALSIM_COLLISION_DIR is not configured")
    geometry = ArenaGeometry.load_soccar(configured)
    assert geometry.vertex_count == 4468
    assert geometry.triangle_count == 8020
    assert len(geometry.meshes) == 16
    assert geometry.content_sha256 == (
        "2239556BDC74D205CAA6E46A0F6E91FA2C6E4257E84D4F608BA775958B0A5538"
    )
    assert geometry.meshes[0].rocketsim_hash == 0xA160BAF9
    assert geometry.meshes[-1].rocketsim_hash == 0x255BA8C1


def test_soccar_reference_planes_are_part_of_world_query(tmp_path: Path) -> None:
    soccar = tmp_path / "soccar"
    soccar.mkdir()
    _write_test_cmf(soccar / "mesh_0.cmf")
    geometry = ArenaGeometry.load_soccar(tmp_path)
    origins = np.array(((100, 100, 1000), (100, 100, 1000)), dtype=np.float32)
    directions = np.array(((0, 0, -1), (0, 0, 1)), dtype=np.float32)
    hit, distance, normal, face = raycast_soccar_cpu(geometry, origins, directions, 5000.0)
    np.testing.assert_array_equal(hit, (1, 1))
    np.testing.assert_allclose(distance, (1000.0, 1048.0))
    np.testing.assert_allclose(normal, ((0, 0, 1), (0, 0, -1)))
    np.testing.assert_array_equal(face, (-2, -3))
