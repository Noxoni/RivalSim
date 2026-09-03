"""Integrated low-speed, high-speed, and ordinary aerial self-play states.

The V17 curriculum extends the V14 mixed state builder without changing the
production reward. It keeps ordinary V23 kickoff self-play in half of the
worlds, retains all four V11 aerial feed families, and adds action-free V16
high-speed rising-ball states derived from measured natural handoffs.
"""

from __future__ import annotations

from dataclasses import dataclass, fields
from typing import Any

import numpy as np

from rivalsim.rival2_ground_to_air_entry_v11 import (
    DEFENDER_LIVE,
    SETUP_NAMES,
    build_ground_to_air_entry_scenarios,
)
from rivalsim.rival2_ground_to_air_high_speed_v16 import (
    build_high_speed_ground_to_air_scenarios,
)
from rivalsim.rival2_ground_to_air_mixed_selfplay_v14 import (
    AerialOptionMixedSelfPlayTrainerV14,
)
from rivalsim.state import StateSnapshot
from rivalsim.static_world import make_standard_kickoff_state

GROUND_TO_AIR_INTEGRATED_SELFPLAY_V17_VERSION = (
    "RIVAL2_GROUND_TO_AIR_INTEGRATED_SELFPLAY_V17"
)

CATEGORY_ORDINARY = 0
CATEGORY_V11_CONTROLLED = 1
CATEGORY_V16_HIGH_SPEED = 2
CATEGORY_NAMES = {
    CATEGORY_ORDINARY: "ordinary_v23_selfplay",
    CATEGORY_V11_CONTROLLED: "v11_controlled",
    CATEGORY_V16_HIGH_SPEED: "v16_high_speed",
}


@dataclass(frozen=True, slots=True)
class IntegratedSelfPlayInitialState:
    state: StateSnapshot
    category: np.ndarray
    attacker_side: np.ndarray
    setup: np.ndarray
    kickoff_selector: np.ndarray


class AerialOptionIntegratedSelfPlayTrainerV17(
    AerialOptionMixedSelfPlayTrainerV14
):
    """V14 physical continuation trainer with V17 checkpoint identity."""

    def checkpoint_payload(self, provenance: dict[str, Any]) -> dict[str, Any]:
        payload = super().checkpoint_payload(provenance)
        payload["format"] = (
            f"{GROUND_TO_AIR_INTEGRATED_SELFPLAY_V17_VERSION}_CHECKPOINT"
        )
        return payload


def _overlay_rows(
    destination: StateSnapshot,
    rows: np.ndarray,
    source: StateSnapshot,
) -> None:
    for item in fields(StateSnapshot):
        getattr(destination, item.name)[rows] = getattr(source, item.name)


def _bounded_even_count(worlds: int, fraction: float) -> int:
    count = round(worlds * fraction)
    count = max(2, min(worlds - 2, count))
    return count - count % 2


def build_integrated_selfplay_initial_state(
    worlds: int,
    *,
    seed: int,
    v11_fraction: float,
    high_speed_fraction: float,
    setup_weights: tuple[float, ...],
    difficulty: float,
) -> IntegratedSelfPlayInitialState:
    """Build deterministic side-balanced integrated aerial/self-play states."""

    if worlds < 8:
        raise ValueError("integrated self-play requires at least eight worlds")
    if not 0.0 < v11_fraction < 1.0:
        raise ValueError("V11 fraction must be inside (0,1)")
    if not 0.0 < high_speed_fraction < 1.0:
        raise ValueError("high-speed fraction must be inside (0,1)")
    if v11_fraction + high_speed_fraction >= 1.0:
        raise ValueError("ordinary self-play worlds must remain present")
    if len(setup_weights) != len(SETUP_NAMES):
        raise ValueError("one weight is required for every V11 setup")

    v11_count = _bounded_even_count(worlds, v11_fraction)
    high_speed_count = _bounded_even_count(worlds, high_speed_fraction)
    if v11_count + high_speed_count > worlds - 2:
        raise ValueError("requested curricula leave fewer than two ordinary worlds")

    rng = np.random.default_rng(seed)
    order = rng.permutation(worlds).astype(np.int64)
    v11_rows = np.sort(order[:v11_count])
    high_speed_rows = np.sort(order[v11_count : v11_count + high_speed_count])
    v11_half = v11_count // 2
    high_speed_half = high_speed_count // 2

    kickoff_selector = (
        np.arange(worlds, dtype=np.int32) + np.int32(seed % 5)
    ) % 5
    state = make_standard_kickoff_state(worlds, kickoff_selector)

    v11_side_zero = build_ground_to_air_entry_scenarios(
        v11_half,
        seed=seed ^ 0x13579BDF,
        attacker_side=0,
        setup_weights=setup_weights,
        difficulty=difficulty,
        defender_mode=DEFENDER_LIVE,
    )
    v11_side_one = build_ground_to_air_entry_scenarios(
        v11_half,
        seed=seed ^ 0x2468ACE0,
        attacker_side=1,
        setup_weights=setup_weights,
        difficulty=difficulty,
        defender_mode=DEFENDER_LIVE,
    )
    _overlay_rows(state, v11_rows[:v11_half], v11_side_zero.state)
    _overlay_rows(state, v11_rows[v11_half:], v11_side_one.state)

    high_speed_side_zero = build_high_speed_ground_to_air_scenarios(
        high_speed_half,
        seed=seed ^ 0x51F15EED,
        attacker_side=0,
    )
    high_speed_side_one = build_high_speed_ground_to_air_scenarios(
        high_speed_half,
        seed=seed ^ 0xA17EA11E,
        attacker_side=1,
    )
    _overlay_rows(
        state,
        high_speed_rows[:high_speed_half],
        high_speed_side_zero.state,
    )
    _overlay_rows(
        state,
        high_speed_rows[high_speed_half:],
        high_speed_side_one.state,
    )
    state.validate()

    category = np.full(worlds, CATEGORY_ORDINARY, dtype=np.int8)
    category[v11_rows] = CATEGORY_V11_CONTROLLED
    category[high_speed_rows] = CATEGORY_V16_HIGH_SPEED
    attacker = np.full(worlds, -1, dtype=np.int8)
    attacker[v11_rows[:v11_half]] = 0
    attacker[v11_rows[v11_half:]] = 1
    attacker[high_speed_rows[:high_speed_half]] = 0
    attacker[high_speed_rows[high_speed_half:]] = 1
    setup = np.full(worlds, -1, dtype=np.int8)
    setup[v11_rows[:v11_half]] = v11_side_zero.setup.astype(np.int8)
    setup[v11_rows[v11_half:]] = v11_side_one.setup.astype(np.int8)
    return IntegratedSelfPlayInitialState(
        state=state,
        category=category,
        attacker_side=attacker,
        setup=setup,
        kickoff_selector=kickoff_selector,
    )


def integrated_state_summary(batch: IntegratedSelfPlayInitialState) -> dict[str, Any]:
    category = batch.category
    attacker = batch.attacker_side
    setup = batch.setup
    result: dict[str, Any] = {
        "worlds": int(category.size),
        "by_category": {
            name: int((category == value).sum())
            for value, name in CATEGORY_NAMES.items()
        },
        "by_category_and_attacker_side": {},
        "v11_by_setup": {},
    }
    for value, name in CATEGORY_NAMES.items():
        if value == CATEGORY_ORDINARY:
            continue
        result["by_category_and_attacker_side"][name] = {
            str(side): int(((category == value) & (attacker == side)).sum())
            for side in (0, 1)
        }
    for index, name in enumerate(SETUP_NAMES):
        result["v11_by_setup"][name] = int(
            ((category == CATEGORY_V11_CONTROLLED) & (setup == index)).sum()
        )
    return result


__all__ = [
    "CATEGORY_NAMES",
    "CATEGORY_ORDINARY",
    "CATEGORY_V11_CONTROLLED",
    "CATEGORY_V16_HIGH_SPEED",
    "GROUND_TO_AIR_INTEGRATED_SELFPLAY_V17_VERSION",
    "AerialOptionIntegratedSelfPlayTrainerV17",
    "IntegratedSelfPlayInitialState",
    "build_integrated_selfplay_initial_state",
    "integrated_state_summary",
]
