from __future__ import annotations

import torch

from rivalsim.rival2_aerial_intercept_distillation import (
    DISTILLATION_VERSION,
    DistillationDataset,
    physical_gate,
)


def _dataset() -> DistillationDataset:
    observations = []
    actions = []
    ticks = []
    trajectories = []
    offset = 0
    for pack in range(3):
        for side in range(2):
            for world in range(2):
                length = 5 + world
                observations.append(torch.full((length, 182), float(pack * 10 + side * 2 + world)))
                actions.append(torch.full((length, 8), float(world)))
                ticks.append(torch.arange(29, 29 + length))
                trajectories.append(
                    {
                        "pack": pack,
                        "side": side,
                        "world": world,
                        "seed": 100 + pack * 10 + side,
                        "offset": offset,
                        "length": length,
                        "first_touch_tick": 29 + length - 1,
                        "outcome": "qualified_high_touch",
                        "weight": float(world + 1),
                    }
                )
                offset += length
    return DistillationDataset(
        {
            "format": f"{DISTILLATION_VERSION}_TRAJECTORIES",
            "observation": torch.cat(observations),
            "action": torch.cat(actions),
            "tick": torch.cat(ticks),
            "trajectories": trajectories,
        }
    )


def test_balanced_sampler_covers_every_pack_side_and_respects_world_cap() -> None:
    dataset = _dataset()
    selected = dataset.sample(
        120,
        generator=torch.Generator().manual_seed(77),
        maximum_samples_per_world=10,
    )
    assert selected.shape == (120,)
    selected_set = selected.tolist()
    category_counts = {(pack, side): 0 for pack in range(3) for side in range(2)}
    for trajectory in dataset.trajectories:
        count = sum(
            trajectory.offset <= row < trajectory.offset + trajectory.length for row in selected_set
        )
        assert count <= 10
        category_counts[trajectory.category] += count
    assert set(category_counts.values()) == {20}


def _authority() -> dict:
    return {
        "optimization": {"human_validation_rmse_max": 0.3},
        "selection": {
            "center_pop": {
                "high_touch_fraction_min": 0.15,
                "qualified_goal_fraction_min": 0.05,
            },
            "lateral_pop": {
                "high_touch_fraction_min": 0.5,
                "goalward_first_touch_fraction_min": 0.45,
            },
            "airborne_possession": {
                "high_touch_fraction_min": 0.35,
                "goalward_first_touch_fraction_min": 0.3,
            },
        },
    }


def _physical() -> list[dict]:
    rows = []
    for pack in ("center_pop", "lateral_pop", "airborne_possession"):
        for side in (0, 1):
            rows.append(
                {
                    "pack": pack,
                    "side": side,
                    "finite_observation": True,
                    "fractions": {
                        "high_touch": 0.6,
                        "goalward_first_touch": 0.5,
                        "goal": 0.1,
                    },
                }
            )
    return rows


def test_physical_gate_requires_every_side_and_human_guard() -> None:
    rows = _physical()
    assert physical_gate(rows, [0.2, 0.2], _authority())
    rows[3]["fractions"]["goalward_first_touch"] = 0.44
    assert not physical_gate(rows, [0.2, 0.2], _authority())
    rows = _physical()
    assert not physical_gate(rows, [0.2, 0.31], _authority())
