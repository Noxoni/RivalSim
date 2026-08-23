"""Deterministic static-world ray corpus spanning Soccar surface families."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True, slots=True)
class RayCorpus:
    origins: np.ndarray
    directions: np.ndarray
    max_distances: np.ndarray
    categories: tuple[str, ...]

    @property
    def count(self) -> int:
        return len(self.origins)


def make_soccar_ray_corpus(seed: int = 20260823, rays_per_family: int = 96) -> RayCorpus:
    """Build rays for planes, curved CMFs, goals, boundary edges, and misses."""

    if rays_per_family <= 0:
        raise ValueError("rays_per_family must be positive")
    rng = np.random.default_rng(seed)
    origins: list[np.ndarray] = []
    directions: list[np.ndarray] = []
    categories: list[str] = []

    def add(category: str, source: np.ndarray, target: np.ndarray) -> None:
        count = len(source)
        origins.append(source.astype(np.float32))
        directions.append((target - source).astype(np.float32))
        categories.extend((category,) * count)

    n = rays_per_family
    source = np.column_stack(
        (rng.uniform(-3500, 3500, n), rng.uniform(-4500, 4500, n), rng.uniform(300, 1800, n))
    )
    target = source.copy()
    target[:, 2] = 0
    add("floor", source, target)

    target = source.copy()
    target[:, 2] = 2048
    add("ceiling", source, target)

    signs = np.where(np.arange(n) % 2 == 0, -1.0, 1.0)
    target = source.copy()
    target[:, 0] = signs * 4096
    add("side_walls", source, target)

    source = np.column_stack(
        (rng.uniform(-3000, 3000, n), rng.uniform(-3500, 3500, n), rng.uniform(100, 900, n))
    )
    signs = np.where(np.arange(n) % 2 == 0, -1.0, 1.0)
    target = np.column_stack(
        (signs * rng.uniform(3850, 4080, n), rng.uniform(-4800, 4800, n), rng.uniform(40, 750, n))
    )
    add("ramps_and_curves", source, target)

    source = np.column_stack(
        (rng.uniform(-1000, 1000, n), rng.uniform(-3500, 3500, n), rng.uniform(100, 1200, n))
    )
    signs = np.where(np.arange(n) % 2 == 0, -1.0, 1.0)
    target = np.column_stack(
        (rng.uniform(-750, 750, n), signs * rng.uniform(5000, 5900, n), rng.uniform(50, 900, n))
    )
    add("goals_and_back_wall", source, target)

    source = np.column_stack(
        (rng.uniform(-3000, 3000, n), rng.uniform(-3500, 3500, n), rng.uniform(100, 1300, n))
    )
    signs_x = np.where(np.arange(n) % 2 == 0, -1.0, 1.0)
    signs_y = np.where((np.arange(n) // 2) % 2 == 0, -1.0, 1.0)
    target = np.column_stack(
        (
            signs_x * rng.uniform(3800, 4100, n),
            signs_y * rng.uniform(4700, 5200, n),
            rng.uniform(50, 1200, n),
        )
    )
    add("corners", source, target)

    # Vertex-addressed rays put floating-point edge behavior under direct test.
    edge_targets = np.array(
        (
            (4096, 5120, 0),
            (-4096, 5120, 0),
            (4096, -5120, 0),
            (-4096, -5120, 0),
            (0, 5120, 640),
            (0, -5120, 640),
            (4096, 0, 2048),
            (-4096, 0, 2048),
        ),
        dtype=np.float64,
    )
    target = edge_targets[np.arange(n) % len(edge_targets)]
    source = target * 0.55 + np.array((0.0, 0.0, 500.0))
    source += rng.normal(0.0, 0.01, source.shape)
    add("boundary_edges", source, target)

    source = np.column_stack(
        (rng.uniform(-500, 500, n), rng.choice((-7000.0, 7000.0), n), rng.uniform(300, 1700, n))
    )
    target = source.copy()
    target[:, 1] += np.sign(source[:, 1]) * 2000.0
    add("misses", source, target)

    origin_array = np.ascontiguousarray(np.concatenate(origins), dtype=np.float32)
    direction_array = np.ascontiguousarray(np.concatenate(directions), dtype=np.float32)
    direction_array /= np.linalg.norm(direction_array, axis=1, keepdims=True)
    return RayCorpus(
        origins=origin_array,
        directions=direction_array,
        max_distances=np.full(len(origin_array), 12000.0, dtype=np.float32),
        categories=tuple(categories),
    )
