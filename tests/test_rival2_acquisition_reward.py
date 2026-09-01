from __future__ import annotations

import os

import pytest
import torch
import warp as wp

from rivalsim.arena import ArenaGeometry, WarpArenaMeshes
from rivalsim.kernels.rival2 import (
    NO_TOUCH_TIMEOUT_TICKS,
    PHYSICS_TICKS_PER_DECISION,
)
from rivalsim.rival2_contracts import (
    REWARD_ACQUISITION_120_V1_CONTRACT,
    REWARD_ACQUISITION_120_V1_CONTRACT_HASH,
    REWARD_ACQUISITION_V1_CONTRACT,
    REWARD_ACQUISITION_V1_CONTRACT_HASH,
    RIVAL2_ACTION_V2_120HZ_VERSION,
    RIVAL2_EPISODE_VERSION,
    RIVAL2_OBS_V2_120HZ_VERSION,
    RIVAL2_REWARD_ACQUISITION_120_V1_VERSION,
    RIVAL2_REWARD_ACQUISITION_V1_VERSION,
    RIVAL2_REWARD_V2_VERSION,
    contract_hashes_for_reward,
)
from rivalsim.rival2_env import Rival2Env
from rivalsim.v03_corpus import generate_phase_b_cases, phase_b_cases_to_state


@pytest.fixture(scope="module")
def arena_assets() -> tuple[str, ArenaGeometry, WarpArenaMeshes]:
    root = os.environ.get("RIVALSIM_COLLISION_DIR")
    if not root or not wp.is_cuda_available() or not torch.cuda.is_available():
        pytest.skip("exact local CMFs, Warp CUDA, and PyTorch CUDA are required")
    geometry = ArenaGeometry.load_soccar(root)
    return root, geometry, WarpArenaMeshes(geometry)


def _env(
    assets: tuple[str, ArenaGeometry, WarpArenaMeshes],
    *,
    reward_version: str,
    **kwargs: object,
) -> Rival2Env:
    root, geometry, meshes = assets
    return Rival2Env(
        1,
        root,
        reward_version=reward_version,
        geometry=geometry,
        meshes=meshes,
        car_visitation_order="a_then_b",
        **kwargs,
    )


def test_acquisition_contract_is_explicit_and_content_addressed() -> None:
    assert REWARD_ACQUISITION_V1_CONTRACT["goal"] == {
        "score": 10.0,
        "concede": -10.0,
        "zero_sum": True,
    }
    assert REWARD_ACQUISITION_V1_CONTRACT["unique_touch_per_player"]["reward"] == 0.20
    assert (
        REWARD_ACQUISITION_V1_CONTRACT["first_legitimate_touch_per_player_per_episode"][
            "reward"
        ]
        == 1.0
    )
    assert (
        REWARD_ACQUISITION_V1_CONTRACT["no_touch_failure"][
            "reward_per_player_without_any_episode_touch"
        ]
        == -0.5
    )
    hashes = contract_hashes_for_reward(
        RIVAL2_REWARD_ACQUISITION_V1_VERSION, RIVAL2_EPISODE_VERSION
    )
    assert (
        hashes[RIVAL2_REWARD_ACQUISITION_V1_VERSION]
        == REWARD_ACQUISITION_V1_CONTRACT_HASH
    )


def test_acquisition_120_binding_preserves_coefficients_and_uses_120hz_contracts() -> None:
    assert REWARD_ACQUISITION_120_V1_CONTRACT["cadence_hz"] == 120
    assert REWARD_ACQUISITION_120_V1_CONTRACT["goal"] == (
        REWARD_ACQUISITION_V1_CONTRACT["goal"]
    )
    assert REWARD_ACQUISITION_120_V1_CONTRACT["progress"] == (
        REWARD_ACQUISITION_V1_CONTRACT["progress"]
    )
    assert REWARD_ACQUISITION_120_V1_CONTRACT["unique_touch_per_player"] == (
        REWARD_ACQUISITION_V1_CONTRACT["unique_touch_per_player"]
    )
    hashes = contract_hashes_for_reward(
        RIVAL2_REWARD_ACQUISITION_120_V1_VERSION,
        RIVAL2_EPISODE_VERSION,
    )
    assert hashes[RIVAL2_REWARD_ACQUISITION_120_V1_VERSION] == (
        REWARD_ACQUISITION_120_V1_CONTRACT_HASH
    )
    assert RIVAL2_OBS_V2_120HZ_VERSION in hashes
    assert RIVAL2_ACTION_V2_120HZ_VERSION in hashes


def test_first_touch_stacks_once_and_continuous_contact_is_latched(
    arena_assets: tuple[str, ArenaGeometry, WarpArenaMeshes],
) -> None:
    initial = phase_b_cases_to_state((generate_phase_b_cases()[0],))
    acquisition = _env(
        arena_assets,
        reward_version=RIVAL2_REWARD_ACQUISITION_V1_VERSION,
        initial=initial,
    )
    base = _env(
        arena_assets,
        reward_version=RIVAL2_REWARD_V2_VERSION,
        initial=initial,
    )
    zero = torch.zeros((1, 2, 8), device=acquisition.device)
    first_acquisition = acquisition.step(zero)
    first_base = base.step(zero)
    second_acquisition = acquisition.step(zero)
    second_base = base.step(zero)
    torch.cuda.synchronize()

    touched_side = int(
        acquisition.bridge.views["rival2.episode_player_touched"]
        .reshape(1, 2)[0]
        .argmax()
        .item()
    )
    assert acquisition.bridge.views["rival2.episode_player_touched"].sum().item() == 1
    # Acquisition adds +1 first contact and replaces V1's +0.05 zero-sum
    # unique-touch term with +0.20 for only the player who touched.
    expected_delta = torch.full((2,), 0.05, device=acquisition.device)
    expected_delta[touched_side] = 1.15
    torch.testing.assert_close(
        (first_acquisition.reward - first_base.reward)[0],
        expected_delta,
        rtol=0.0,
        atol=2.0e-6,
    )
    # Sustained overlap is not a second event and cannot receive either bonus.
    torch.testing.assert_close(
        second_acquisition.reward - second_base.reward,
        torch.zeros_like(second_acquisition.reward),
        rtol=0.0,
        atol=2.0e-6,
    )


def test_no_touch_failure_is_per_player_and_uses_original_truncation(
    arena_assets: tuple[str, ArenaGeometry, WarpArenaMeshes],
) -> None:
    acquisition = _env(
        arena_assets, reward_version=RIVAL2_REWARD_ACQUISITION_V1_VERSION
    )
    base = _env(arena_assets, reward_version=RIVAL2_REWARD_V2_VERSION)
    for env in (acquisition, base):
        env.bridge.views["rival2.no_touch_ticks"].fill_(
            NO_TOUCH_TIMEOUT_TICKS - PHYSICS_TICKS_PER_DECISION
        )
    acquisition.bridge.views["rival2.episode_player_touched"].reshape(1, 2)[0, 0] = 1
    zero = torch.zeros((1, 2, 8), device=acquisition.device)
    acquisition_result = acquisition.step(zero)
    base_result = base.step(zero)
    torch.cuda.synchronize()

    assert acquisition_result.truncated.item() and acquisition_result.reset_mask.item()
    torch.testing.assert_close(
        (acquisition_result.reward - base_result.reward)[0],
        torch.tensor((0.0, -0.5), device=acquisition.device),
        rtol=0.0,
        atol=2.0e-6,
    )
    assert acquisition.bridge.views["rival2.episode_player_touched"].sum().item() == 0
