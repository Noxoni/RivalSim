from __future__ import annotations

import torch

from rivalsim.rival2_contracts import (
    OBS_DIM,
    REWARD_SSL_FOUNDATION_V1_CONTRACT,
    RIVAL2_REWARD_SSL_FOUNDATION_V1_VERSION,
    contract_hashes_for_reward,
)
from rivalsim.ssl_foundation_v1 import (
    SCENARIO_NAMES,
    SSL_FOUNDATION_GAMMA,
    SSL_FOUNDATION_WEIGHTS,
    build_ssl_foundation_scenarios,
    ssl_foundation_potentials,
    ssl_foundation_shaping,
)


def test_reward_contract_has_only_goals_and_four_potential_differences() -> None:
    contract = REWARD_SSL_FOUNDATION_V1_CONTRACT
    assert contract["terminal"] == {
        "goal_for": 10.0,
        "goal_against": -10.0,
        "timeout": 0.0,
    }
    assert set(contract["potentials"]) == {
        "field",
        "free_ball_access",
        "control_advantage",
        "defensive_coverage",
    }
    assert contract["named_mechanics_reward"] == 0.0
    assert contract["named_mechanics_hot_path"] is False
    assert len(contract["direct_reward_exactly_zero"]) == 14
    hashes = contract_hashes_for_reward(RIVAL2_REWARD_SSL_FOUNDATION_V1_VERSION)
    assert RIVAL2_REWARD_SSL_FOUNDATION_V1_VERSION in hashes


def test_potentials_are_bounded_and_terminal_successor_is_absorbing() -> None:
    generator = torch.Generator().manual_seed(7)
    before = torch.randn(5, 2, OBS_DIM, generator=generator)
    after = torch.randn(5, 2, OBS_DIM, generator=generator)
    terminated = torch.tensor([False, True, False, True, False])
    potentials = ssl_foundation_potentials(before)
    for name in ("field", "access", "control", "defense"):
        value = getattr(potentials, name)
        assert torch.isfinite(value).all()
        assert (value.abs() <= 1.0 + 1.0e-6).all()
    shaping = ssl_foundation_shaping(before, after, terminated)
    for name, weight in SSL_FOUNDATION_WEIGHTS.items():
        expected = -weight * getattr(potentials, name)[terminated]
        torch.testing.assert_close(shaping[name][terminated], expected)


def test_nonterminal_potential_shaping_telescopes() -> None:
    generator = torch.Generator().manual_seed(11)
    states = torch.randn(9, 1, 2, OBS_DIM, generator=generator)
    discounted = torch.zeros(1, 2)
    for tick in range(8):
        shaping = ssl_foundation_shaping(
            states[tick], states[tick + 1], torch.zeros(1, dtype=torch.bool)
        )["total"]
        discounted += (SSL_FOUNDATION_GAMMA**tick) * shaping
    first = ssl_foundation_potentials(states[0]).weighted_total
    last = ssl_foundation_potentials(states[-1]).weighted_total
    expected = -first + (SSL_FOUNDATION_GAMMA**8) * last
    torch.testing.assert_close(discounted, expected, atol=2.0e-6, rtol=2.0e-6)


def test_reset_curriculum_is_deterministic_balanced_and_has_no_task_feature() -> None:
    first = build_ssl_foundation_scenarios(160, seed=123)
    second = build_ssl_foundation_scenarios(160, seed=123)
    assert first.summary() == second.summary()
    assert first.summary()["counts"] == {
        name: count
        for name, count in zip(SCENARIO_NAMES, (40, 24, 24, 16, 16, 16, 16, 8), strict=True)
    }
    assert first.summary()["focal_side_counts"] == {"0": 80, "1": 80}
    assert first.summary()["task_or_scenario_id_in_observation"] is False
    for name in first.state.__dataclass_fields__:
        torch.testing.assert_close(
            torch.from_numpy(getattr(first.state, name)),
            torch.from_numpy(getattr(second.state, name)),
        )
