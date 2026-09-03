from __future__ import annotations

import numpy as np

from rivalsim.rival2_ground_to_air_entry_v11 import SETUP_NAMES
from rivalsim.rival2_ground_to_air_integrated_selfplay_v17 import (
    CATEGORY_ORDINARY,
    CATEGORY_V11_CONTROLLED,
    CATEGORY_V16_HIGH_SPEED,
    build_integrated_selfplay_initial_state,
    integrated_state_summary,
)


def _build(seed: int = 41):
    return build_integrated_selfplay_initial_state(
        400,
        seed=seed,
        v11_fraction=0.25,
        high_speed_fraction=0.25,
        setup_weights=(0.35, 0.15, 0.20, 0.30),
        difficulty=0.0,
    )


def test_integrated_mix_is_bounded_balanced_and_contains_all_setups() -> None:
    batch = _build()
    summary = integrated_state_summary(batch)
    assert summary["by_category"] == {
        "ordinary_v23_selfplay": 200,
        "v11_controlled": 100,
        "v16_high_speed": 100,
    }
    assert summary["by_category_and_attacker_side"] == {
        "v11_controlled": {"0": 50, "1": 50},
        "v16_high_speed": {"0": 50, "1": 50},
    }
    assert set(summary["v11_by_setup"]) == set(SETUP_NAMES)
    assert all(value > 0 for value in summary["v11_by_setup"].values())
    assert set(np.unique(batch.category)) == {
        CATEGORY_ORDINARY,
        CATEGORY_V11_CONTROLLED,
        CATEGORY_V16_HIGH_SPEED,
    }


def test_integrated_mix_is_deterministic_and_changes_with_seed() -> None:
    first = _build()
    second = _build()
    different = _build(42)
    for name in first.state.__dataclass_fields__:
        assert np.array_equal(getattr(first.state, name), getattr(second.state, name))
    assert np.array_equal(first.category, second.category)
    assert np.array_equal(first.attacker_side, second.attacker_side)
    assert np.array_equal(first.setup, second.setup)
    assert not np.array_equal(first.state.ball_pos, different.state.ball_pos)


def test_integrated_mix_rejects_no_ordinary_worlds() -> None:
    try:
        build_integrated_selfplay_initial_state(
            100,
            seed=1,
            v11_fraction=0.5,
            high_speed_fraction=0.5,
            setup_weights=(0.35, 0.15, 0.20, 0.30),
            difficulty=0.0,
        )
    except ValueError as error:
        assert "ordinary" in str(error)
    else:
        raise AssertionError("curriculum without ordinary self-play was accepted")
