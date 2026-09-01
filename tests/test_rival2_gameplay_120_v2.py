from __future__ import annotations

import os
from pathlib import Path

import pytest
import torch
import warp as wp

from rivalsim.arena import ArenaGeometry, WarpArenaMeshes
from rivalsim.gameplay_120_v2 import (
    OUTCOME_EXEMPT_CONTESTED_50,
    OUTCOME_EXEMPT_POWER_CONTACT,
    OUTCOME_EXEMPT_RETAINED_CONTROL,
    OUTCOME_UNNECESSARY_FLIP_THROUGH_CONTACT,
    physical_control_score,
    physical_flip_outcome,
)
from rivalsim.rival2_contracts import (
    GAMEPLAY_120_V2_BOOST_USE_REWARD,
    GAMEPLAY_120_V2_CONTROL_REWARD,
    GAMEPLAY_120_V2_SPEED_COEFFICIENT,
    GAMEPLAY_120_V2_SUPERSONIC_REWARD,
    GAMEPLAY_120_V2_UNNECESSARY_FLIP_PENALTY,
    REWARD_GAMEPLAY_120_V1_CONTRACT_HASH,
    REWARD_GAMEPLAY_120_V2_CONTRACT,
    REWARD_GAMEPLAY_120_V2_CONTRACT_HASH,
    RIVAL2_REWARD_GAMEPLAY_120_V2_VERSION,
    contract_hashes_for_reward,
)
from rivalsim.rival2_env import Rival2Env
from rivalsim.state import StateSnapshot


@pytest.fixture(scope="module")
def arena_assets() -> tuple[str, ArenaGeometry, WarpArenaMeshes]:
    configured = os.environ.get("RIVALSIM_COLLISION_DIR")
    candidates = [Path(configured)] if configured else []
    candidates.append(Path("G:/dev/RLBot-Rival/bot/collision_meshes"))
    root = next(
        (p for p in candidates if (p / "soccar" / "mesh_0.cmf").is_file()),
        None,
    )
    if root is None or not wp.is_cuda_available() or not torch.cuda.is_available():
        pytest.skip("exact local CMFs and CUDA are required")
    geometry = ArenaGeometry.load_soccar(root)
    return str(root), geometry, WarpArenaMeshes(geometry)


def _env(
    assets: tuple[str, ArenaGeometry, WarpArenaMeshes],
    state: StateSnapshot,
) -> Rival2Env:
    root, geometry, meshes = assets
    return Rival2Env(
        state.car_pos.shape[0],
        root,
        reward_version=RIVAL2_REWARD_GAMEPLAY_120_V2_VERSION,
        initial=state,
        geometry=geometry,
        meshes=meshes,
        car_visitation_order="a_then_b",
        device="cuda:0",
    )


def test_v2_contract_is_separate_and_disables_untrusted_dense_terms() -> None:
    assert REWARD_GAMEPLAY_120_V1_CONTRACT_HASH == (
        "0D4C9A78803BBAF851AB4FDD7D9AC4196AB08E42B51DC0A173A1EAEC066AFAED"
    )
    assert REWARD_GAMEPLAY_120_V2_CONTRACT_HASH == (
        "E63920316F04ED66F02065D0DEDBEF500CDAF8F485BD2602E21AFECBA72EFF6C"
    )
    assert GAMEPLAY_120_V2_SPEED_COEFFICIENT == 0.0
    assert GAMEPLAY_120_V2_BOOST_USE_REWARD == 0.0
    assert GAMEPLAY_120_V2_SUPERSONIC_REWARD == 0.000075
    assert REWARD_GAMEPLAY_120_V2_CONTRACT["named_mechanics_hot_path"] is False
    assert REWARD_GAMEPLAY_120_V2_CONTRACT["named_mechanics_reward"] == 0.0
    assert contract_hashes_for_reward(RIVAL2_REWARD_GAMEPLAY_120_V2_VERSION)[
        RIVAL2_REWARD_GAMEPLAY_120_V2_VERSION
    ] == REWARD_GAMEPLAY_120_V2_CONTRACT_HASH


@pytest.mark.parametrize(
    ("distance", "relative_speed", "expected"),
    [
        (500.0, 0.0, 0.0),
        (600.0, 0.0, 0.0),
        (150.0, 0.0, 1.0),
        (150.0, 1200.0, 0.0),
        (150.0, 600.0, 0.5),
        (325.0, 600.0, 0.25),
    ],
)
def test_control_formula(distance: float, relative_speed: float, expected: float) -> None:
    assert physical_control_score(
        distance=distance, relative_speed=relative_speed
    ) == pytest.approx(expected, rel=0.0, abs=1.0e-12)


def test_physical_exemption_precedence_and_penalty() -> None:
    assert GAMEPLAY_120_V2_UNNECESSARY_FLIP_PENALTY == -0.005
    assert physical_flip_outcome(
        contested_50=True, power_contact=True, retained_control=True
    ) == OUTCOME_EXEMPT_CONTESTED_50
    assert physical_flip_outcome(
        contested_50=False, power_contact=True, retained_control=True
    ) == OUTCOME_EXEMPT_POWER_CONTACT
    assert physical_flip_outcome(
        contested_50=False, power_contact=False, retained_control=True
    ) == OUTCOME_EXEMPT_RETAINED_CONTROL
    assert physical_flip_outcome(
        contested_50=False, power_contact=False, retained_control=False
    ) == OUTCOME_UNNECESSARY_FLIP_THROUGH_CONTACT


def test_supersonic_and_control_components_are_authoritative_and_zero_sum(
    arena_assets,
) -> None:
    state = StateSnapshot.empty(1)
    state.car_pos[0, 0] = (0.0, 0.0, 1000.0)
    state.car_pos[0, 1] = (3000.0, 0.0, 1000.0)
    state.ball_pos[0] = (0.0, 0.0, 1300.0)
    state.car_vel[0, 0] = (0.0, 2300.0, 0.0)
    state.ball_vel[0] = (0.0, 2300.0, 0.0)
    state.is_supersonic[0, 0] = 1
    state.supersonic_time[0, 0] = 1.0
    env = _env(arena_assets, state)
    transition = env.step(torch.zeros((1, 2, 8), device=env.device))
    torch.cuda.synchronize()

    assert float(env.bridge.views["rival2.speed_component"].item()) == 0.0
    assert float(env.bridge.views["rival2.boost_use_component"].item()) == 0.0
    assert float(env.bridge.views["rival2.supersonic_component"].item()) == (
        pytest.approx(GAMEPLAY_120_V2_SUPERSONIC_REWARD, abs=1.0e-9)
    )
    scores = env.bridge.views["gameplay_120.control_score_current"].reshape(1, 2)
    expected = GAMEPLAY_120_V2_CONTROL_REWARD * float(
        scores[0, 0] - scores[0, 1]
    )
    assert float(env.bridge.views["gameplay_120.control_component"].item()) == (
        pytest.approx(expected, abs=1.0e-9)
    )
    torch.testing.assert_close(
        transition.reward[:, 0], -transition.reward[:, 1], rtol=0.0, atol=0.0
    )


def test_reset_transition_cannot_farm_control_reward(arena_assets) -> None:
    state = StateSnapshot.empty(1)
    state.car_pos[0, 0] = (0.0, 5300.0, 100.0)
    state.car_pos[0, 1] = (3000.0, 0.0, 1000.0)
    state.ball_pos[0] = (0.0, 5300.0, 93.15)
    env = _env(arena_assets, state)
    transition = env.step(torch.zeros((1, 2, 8), device=env.device))
    torch.cuda.synchronize()
    assert bool(transition.reset_mask.item())
    assert float(env.bridge.views["gameplay_120.control_component"].item()) == 0.0
    assert int(
        env.bridge.views["gameplay_120.control_score_tick_total"].sum().item()
    ) == 0


@pytest.mark.parametrize(("retained", "expected_bad"), [(True, 0), (False, 1)])
def test_pending_contact_uses_authoritative_post_contact_control(
    arena_assets, retained: bool, expected_bad: int
) -> None:
    state = StateSnapshot.empty(1)
    state.car_pos[0, 0] = (0.0, 0.0, 1000.0)
    state.car_pos[0, 1] = (3000.0, 0.0, 1000.0)
    state.ball_pos[0] = (0.0, 0.0, 1300.0) if retained else (0.0, 0.0, 2000.0)
    env = _env(arena_assets, state)
    guard = env.world.gameplay_120
    wp.to_torch(guard.pending_active)[0] = 1
    wp.to_torch(guard.pending_self_contact_tick)[0] = 0
    action = torch.zeros((1, 2, 8), device=env.device)
    for _ in range(12):
        env.step(action)
    torch.cuda.synchronize()

    assert int(wp.to_torch(guard.pending_active)[0].item()) == 0
    assert int(wp.to_torch(guard.bad_flip_total)[0].item()) == expected_bad
    assert int(wp.to_torch(guard.retained_control_exempt_total)[0].item()) == (
        1 - expected_bad
    )
    expected_component = (
        GAMEPLAY_120_V2_UNNECESSARY_FLIP_PENALTY if expected_bad else 0.0
    )
    assert float(wp.to_torch(guard.bad_flip_component)[0].item()) == pytest.approx(
        expected_component, abs=1.0e-9
    )


def test_v2_allocates_no_named_mechanics_state(arena_assets) -> None:
    env = _env(arena_assets, StateSnapshot.empty(1))
    inventory = env.world.gameplay_120.memory_inventory()
    assert env.world.gameplay_v3 is None
    assert inventory["named_mechanics_arrays"] == 0
    assert inventory["controlled_flick_arrays"] == 0
