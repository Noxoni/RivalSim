from __future__ import annotations

import os

import numpy as np
import pytest
import warp as wp

from rivalsim.arena import ArenaGeometry, WarpArenaMeshes, raycast_soccar_cpu
from rivalsim.kernels.world_queries import query_soccar_rays
from rivalsim.ray_corpus import make_soccar_ray_corpus


@pytest.mark.parametrize("constructor", ["default", "cubql"])
def test_gpu_soccar_rays_match_cpu_reference(constructor: str) -> None:
    collision_root = os.environ.get("RIVALSIM_COLLISION_DIR")
    if not collision_root or not wp.is_cuda_available():
        pytest.skip("exact local CMFs and CUDA are required")
    geometry = ArenaGeometry.load_soccar(collision_root)
    meshes = WarpArenaMeshes(geometry)
    corpus = make_soccar_ray_corpus(rays_per_family=16)
    expected = raycast_soccar_cpu(
        geometry,
        corpus.origins,
        corpus.directions,
        corpus.max_distances,
    )
    actual_arrays = query_soccar_rays(
        getattr(meshes, constructor),
        corpus.origins,
        corpus.directions,
        corpus.max_distances,
        device=meshes.device,
    )
    wp.synchronize_device(meshes.device)
    actual = tuple(array.numpy() for array in actual_arrays)
    np.testing.assert_array_equal(actual[0], expected[0])
    np.testing.assert_allclose(actual[1], expected[1], atol=2e-2, rtol=2e-5)
    # Warp consistently returns the geometric winding normal. Only compare
    # hits away from exact shared-edge ambiguity.
    unambiguous = (expected[0] == 1) & (np.abs(actual[1] - expected[1]) < 2e-2)
    dots = np.einsum("ij,ij->i", actual[2][unambiguous], expected[2][unambiguous])
    assert np.all(dots > 0.999)


def test_ray_corpus_covers_every_required_family() -> None:
    corpus = make_soccar_ray_corpus(rays_per_family=2)
    assert set(corpus.categories) == {
        "floor",
        "ceiling",
        "side_walls",
        "ramps_and_curves",
        "goals_and_back_wall",
        "corners",
        "boundary_edges",
        "misses",
    }
    assert corpus.count == 16
