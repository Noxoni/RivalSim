from __future__ import annotations

import torch

from rivalsim.rival2_aerial_intercept_dagger import (
    DAGGER_VERSION,
    CorrectionDataset,
    evenly_spaced_indices,
)


def _round(round_index: int) -> dict:
    observation = []
    action = []
    tick = []
    trajectories = []
    offset = 0
    for pack in range(3):
        for side in range(2):
            for world in range(2):
                local_tick = torch.tensor((29, 31, 35, 40), dtype=torch.int64)
                observation.append(torch.full((4, 182), float(pack + side + round_index)))
                action.append(torch.full((4, 8), float(world)))
                tick.append(local_tick)
                trajectories.append(
                    {
                        "pack": pack,
                        "side": side,
                        "world": world,
                        "round": round_index,
                        "seed": 1000 + round_index,
                        "offset": offset,
                        "length": 4,
                        "success": world == 0,
                        "weight": 1.0 if world == 0 else 2.0,
                    }
                )
                offset += 4
    return {
        "format": f"{DAGGER_VERSION}_CORRECTIONS",
        "observation": torch.cat(observation),
        "action": torch.cat(action),
        "tick": torch.cat(tick),
        "trajectories": trajectories,
    }


def test_correction_dataset_merges_rounds_and_balances_categories() -> None:
    dataset = CorrectionDataset([_round(1), _round(2)])
    selected = dataset.sample(120, generator=torch.Generator().manual_seed(9))
    assert selected.shape == (120,)
    category_count = {(pack, side): 0 for pack in range(3) for side in range(2)}
    for row in selected.tolist():
        for trajectory in dataset.trajectories:
            if trajectory.offset <= row < trajectory.offset + trajectory.length:
                category_count[trajectory.category] += 1
                break
    assert set(category_count.values()) == {20}


def test_evenly_spaced_indices_preserve_order_and_endpoints() -> None:
    source = torch.arange(29, 229)
    selected = evenly_spaced_indices(source, 96)
    assert selected.shape == (96,)
    assert int(selected[0]) == 29
    assert int(selected[-1]) == 228
    assert bool((selected[1:] > selected[:-1]).all())
    assert torch.equal(evenly_spaced_indices(source[:5], 96), source[:5])
