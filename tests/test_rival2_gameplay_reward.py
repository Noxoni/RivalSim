from __future__ import annotations

import math
import os

import pytest
import torch
import warp as wp

from rivalsim.arena import ArenaGeometry, WarpArenaMeshes
from rivalsim.kernels.boost_pad import SOCCAR_PAD_POSITIONS
from rivalsim.rival2_contracts import (
    GAMEPLAY_BIG_PAD_PICKUP_REWARD,
    GAMEPLAY_BOOST_USE_REWARD,
    GAMEPLAY_SAVE_REWARD,
    GAMEPLAY_SMALL_PAD_PICKUP_REWARD,
    REWARD_CONTRACT_HASH,
    REWARD_GAMEPLAY_V1_CONTRACT,
    REWARD_GAMEPLAY_V1_CONTRACT_HASH,
    RIVAL2_EPISODE_VERSION,
    RIVAL2_REWARD_GAMEPLAY_V1_VERSION,
    contract_hashes_for_reward,
)
from rivalsim.rival2_env import Rival2Env
from rivalsim.state import StateSnapshot

HISTORICAL_V1_HASH = "E3C97C7B3EA97D15F6AFB3AF21C40BAFBD206F0ED1124BAD6EA2C5A2ED14786F"
GAMEPLAY_COMPONENTS = (
    "v1_goal_component",
    "v1_progress_component",
    "v1_touch_component",
    "v1_demo_component",
    "speed_component",
    "supersonic_component",
    "boost_use_component",
    "boost_pickup_component",
    "save_component",
)


@pytest.fixture(scope="module")
def arena_assets() -> tuple[str, ArenaGeometry, WarpArenaMeshes]:
    root = os.environ.get("RIVALSIM_COLLISION_DIR")
    if not root or not wp.is_cuda_available() or not torch.cuda.is_available():
        pytest.skip("exact local CMFs, Warp CUDA, and PyTorch CUDA are required")
    geometry = ArenaGeometry.load_soccar(root)
    return root, geometry, WarpArenaMeshes(geometry)


def _env(
    assets: tuple[str, ArenaGeometry, WarpArenaMeshes],
    initial: StateSnapshot,
) -> Rival2Env:
    root, geometry, meshes = assets
    return Rival2Env(
        initial.car_pos.shape[0],
        root,
        reward_version=RIVAL2_REWARD_GAMEPLAY_V1_VERSION,
        initial=initial,
        geometry=geometry,
        meshes=meshes,
        car_visitation_order="a_then_b",
    )


def test_gameplay_contract_is_new_zero_sum_identity_and_v1_is_immutable() -> None:
    assert REWARD_CONTRACT_HASH == HISTORICAL_V1_HASH
    assert REWARD_GAMEPLAY_V1_CONTRACT["base_reward"]["version"] == "RIVAL2_REWARD_V1"
    assert REWARD_GAMEPLAY_V1_CONTRACT["zero_sum"] is True
    assert REWARD_GAMEPLAY_V1_CONTRACT["approach_reward"] is None
    assert REWARD_GAMEPLAY_V1_CONTRACT["direct_mechanic_rewards_or_costs"] == []
    assert REWARD_GAMEPLAY_V1_CONTRACT["save"]["event_reward"] == GAMEPLAY_SAVE_REWARD
    hashes = contract_hashes_for_reward(RIVAL2_REWARD_GAMEPLAY_V1_VERSION, RIVAL2_EPISODE_VERSION)
    assert hashes[RIVAL2_REWARD_GAMEPLAY_V1_VERSION] == (REWARD_GAMEPLAY_V1_CONTRACT_HASH)


def test_gameplay_save_uses_pre_touch_threat_and_post_touch_clearance(
    arena_assets: tuple[str, ArenaGeometry, WarpArenaMeshes],
) -> None:
    state = StateSnapshot.empty(1)
    yaw = math.pi / 2.0
    state.car_quat[0, 0] = (0.0, 0.0, math.sin(yaw / 2.0), math.cos(yaw / 2.0))
    state.car_pos[0, 0] = (0.0, -4500.0, 72.395)
    state.car_vel[0, 0] = (0.0, 1600.0, 0.0)
    state.car_pos[0, 1] = (3000.0, 0.0, 1000.0)
    state.ball_pos[0] = (28.177305, -4334.3708, 93.15)
    state.ball_vel[0] = (0.0, -1000.0, 0.0)
    env = _env(arena_assets, state)
    transition = env.step(torch.zeros((1, 2, 8), device=env.device))
    torch.cuda.synchronize()

    assert env.bridge.views["rival2.touch_count"].reshape(1, 2).cpu().tolist() == [[1, 0]]
    assert env.bridge.views["rival2.save_count"].reshape(1, 2).cpu().tolist() == [[1, 0]]
    assert float(env.bridge.views["rival2.v1_touch_component"].item()) == pytest.approx(0.05)
    assert float(env.bridge.views["rival2.save_component"].item()) == pytest.approx(0.75)
    blue_components = sum(env.bridge.views[f"rival2.{name}"] for name in GAMEPLAY_COMPONENTS)
    torch.testing.assert_close(transition.reward[:, 0], blue_components, rtol=0.0, atol=2.0e-6)
    torch.testing.assert_close(transition.reward[:, 1], -blue_components, rtol=0.0, atol=2.0e-6)


def test_boost_reward_requires_physical_use_and_positive_pad_resource_gain(
    arena_assets: tuple[str, ArenaGeometry, WarpArenaMeshes],
) -> None:
    state = StateSnapshot.empty(3)
    state.ball_pos[:] = (0.0, 0.0, 1500.0)
    # World 0: Blue actually uses boost. Orange holds boost while empty.
    state.boost[0] = (50.0, 0.0)
    # World 1: Blue gains a full pad from empty.
    state.car_pos[1, 0] = SOCCAR_PAD_POSITIONS[0]
    state.boost[1, 0] = 0.0
    # World 2: Orange crosses a small pad while full; the source pad event may
    # consume the pad, but zero resource gain must earn zero gameplay reward.
    state.car_pos[2, 1] = SOCCAR_PAD_POSITIONS[6]
    state.boost[2, 1] = 100.0
    env = _env(arena_assets, state)
    action = torch.zeros((3, 2, 8), device=env.device)
    action[0, :, 6] = 1.0
    transition = env.step(action)
    torch.cuda.synchronize()

    boost_use = env.bridge.views["rival2.boost_use_event"].reshape(3, 2)
    assert boost_use.cpu().tolist()[0] == [1, 0]
    assert float(env.bridge.views["rival2.boost_use_component"][0].item()) == pytest.approx(
        GAMEPLAY_BOOST_USE_REWARD
    )
    big = env.bridge.views["rival2.big_pad_pickup_count"].reshape(3, 2)
    small = env.bridge.views["rival2.small_pad_pickup_count"].reshape(3, 2)
    assert big.cpu().tolist()[1] == [1, 0]
    assert float(env.bridge.views["rival2.boost_pickup_component"][1].item()) == pytest.approx(
        GAMEPLAY_BIG_PAD_PICKUP_REWARD
    )
    assert small.cpu().tolist()[2] == [0, 0]
    assert float(env.bridge.views["rival2.boost_pickup_component"][2].item()) == 0.0
    torch.testing.assert_close(
        transition.reward[:, 0], -transition.reward[:, 1], rtol=0.0, atol=0.0
    )


def test_small_pad_positive_gain_uses_small_event_scale(
    arena_assets: tuple[str, ArenaGeometry, WarpArenaMeshes],
) -> None:
    state = StateSnapshot.empty(1)
    state.ball_pos[0] = (0.0, 0.0, 1500.0)
    state.car_pos[0, 1] = SOCCAR_PAD_POSITIONS[6]
    state.boost[0, 1] = 5.0
    env = _env(arena_assets, state)
    env.step(torch.zeros((1, 2, 8), device=env.device))
    torch.cuda.synchronize()
    assert env.bridge.views["rival2.small_pad_pickup_count"].reshape(1, 2).cpu().tolist() == [
        [0, 1]
    ]
    assert float(env.bridge.views["rival2.boost_pickup_component"].item()) == pytest.approx(
        -GAMEPLAY_SMALL_PAD_PICKUP_REWARD
    )
