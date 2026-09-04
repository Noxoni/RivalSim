from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pytest
import torch
import warp as wp

from rivalsim.arena import ArenaGeometry, WarpArenaMeshes
from rivalsim.kernels.rival2 import NO_TOUCH_TIMEOUT_TICKS
from rivalsim.rival2_contracts import (
    OBS_DIM,
    REWARD_SSL_FOUNDATION_V1_CONTRACT,
    RIVAL2_REWARD_SSL_FOUNDATION_V1_VERSION,
    contract_hashes_for_reward,
)
from rivalsim.rival2_env import Rival2Env
from rivalsim.ssl_foundation_v1 import (
    SCENARIO_NAMES,
    SSL_FOUNDATION_GAMMA,
    SSL_FOUNDATION_WEIGHTS,
    build_ssl_foundation_scenarios,
    ssl_foundation_potentials,
    ssl_foundation_shaping,
)


def test_reward_contract_has_only_goals_and_six_potential_differences() -> None:
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
        "car_ball_target_approach_alignment",
        "boost_reserve",
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
    for name in ("field", "access", "control", "defense", "alignment", "boost"):
        value = getattr(potentials, name)
        assert torch.isfinite(value).all()
        assert (value.abs() <= 1.0 + 1.0e-6).all()
    shaping = ssl_foundation_shaping(before, after, terminated)
    for name, weight in SSL_FOUNDATION_WEIGHTS.items():
        expected = -weight * getattr(potentials, name)[terminated]
        torch.testing.assert_close(shaping[name][terminated], expected)
    component_sum = sum(shaping[name] for name in SSL_FOUNDATION_WEIGHTS)
    torch.testing.assert_close(shaping["total"], component_sum)


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
        for name, count in zip(SCENARIO_NAMES, (24, 16, 24, 24, 16, 16, 16, 16, 8), strict=True)
    }
    assert first.summary()["focal_side_counts"] == {"0": 80, "1": 80}
    assert first.summary()["standard_kickoff_starts"] == 16
    assert sorted(first.summary()["standard_kickoff_layout_counts"].values()) == [3, 3, 3, 3, 4]
    assert set(first.kickoff_layout[first.kickoff_indicator != 0]) == set(range(5))
    assert sorted(first.summary()["wall_aerial_variant_counts"].values()) == [5, 5, 6]
    assert first.summary()["ground_heading_momentum_semantics"] == (
        "coherent_with_off_angle_coverage"
    )
    assert first.summary()["task_or_scenario_id_in_observation"] is False
    for name in first.state.__dataclass_fields__:
        torch.testing.assert_close(
            torch.from_numpy(getattr(first.state, name)),
            torch.from_numpy(getattr(second.state, name)),
        )


def _quaternion_forward(quaternion: np.ndarray) -> np.ndarray:
    x, y, z, w = np.moveaxis(quaternion, -1, 0)
    return np.stack(
        (
            1.0 - 2.0 * (y * y + z * z),
            2.0 * (x * y + w * z),
            2.0 * (x * z - w * y),
        ),
        axis=-1,
    )


def test_ground_routes_have_coherent_heading_and_cover_off_angle_approaches() -> None:
    batch = build_ssl_foundation_scenarios(900, seed=77123)
    state = batch.state
    forward = _quaternion_forward(state.car_quat)
    speed = np.linalg.norm(state.car_vel[..., :2], axis=-1)
    moving_ground = (state.on_ground != 0) & (speed > 50.0)
    velocity_direction = state.car_vel[..., :2] / np.maximum(speed[..., None], 1.0)
    heading_velocity_alignment = np.sum(forward[..., :2] * velocity_direction, axis=-1)
    assert np.min(heading_velocity_alignment[moving_ground]) > 0.95

    natural = batch.family == SCENARIO_NAMES.index("natural_ongoing")
    car_to_ball = state.ball_pos[natural, None, :2] - state.car_pos[natural, :, :2]
    car_to_ball /= np.maximum(np.linalg.norm(car_to_ball, axis=-1, keepdims=True), 1.0)
    approach_alignment = np.sum(forward[natural, :, :2] * car_to_ball, axis=-1)
    # Natural play contains ordinary good and off-angle approaches; cars are
    # no longer globally handed an exact face-the-ball orientation.
    assert float(np.quantile(approach_alignment, 0.10)) < 0.55
    assert float(np.quantile(approach_alignment, 0.90)) > 0.90
    assert not np.allclose(approach_alignment, 1.0, rtol=0.0, atol=1.0e-5)


def test_wall_aerial_family_contains_ground_wall_and_airborne_car_states() -> None:
    batch = build_ssl_foundation_scenarios(900, seed=77124)
    state = batch.state
    rows = np.flatnonzero(batch.family == SCENARIO_NAMES.index("wall_aerial"))
    assert set(batch.wall_aerial_variant[rows]) == {0, 1, 2}

    focal = batch.focal_side[rows].astype(np.int64)
    car_pos = state.car_pos[rows, focal]
    car_vel = state.car_vel[rows, focal]
    on_ground = state.on_ground[rows, focal]
    has_jumped = state.has_jumped[rows, focal]
    variants = batch.wall_aerial_variant[rows]

    grounded = variants == 0
    assert np.all(on_ground[grounded] == 1)
    assert np.allclose(car_pos[grounded, 2], 17.0)
    assert np.all(state.ball_pos[rows[grounded], 2] >= 450.0)

    wall = variants == 1
    assert np.all(on_ground[wall] == 0)
    assert np.all(np.abs(car_pos[wall, 0]) >= 4000.0)
    assert np.all((car_pos[wall, 2] >= 250.0) & (car_pos[wall, 2] <= 1050.0))
    assert np.all(np.linalg.norm(car_vel[wall], axis=-1) >= 650.0)

    airborne = variants == 2
    assert np.all(on_ground[airborne] == 0)
    assert np.all(has_jumped[airborne] == 1)
    assert np.all(car_pos[airborne, 2] > 45.0)
    assert np.all(np.linalg.norm(car_vel[airborne], axis=-1) >= 550.0)


@pytest.fixture(scope="module")
def ssl_arena_assets() -> tuple[str, ArenaGeometry, WarpArenaMeshes]:
    configured = os.environ.get("RIVALSIM_COLLISION_DIR")
    candidates = [Path(configured)] if configured else []
    candidates.append(Path("G:/dev/RLBot-Rival/bot/collision_meshes"))
    root = next(
        (candidate for candidate in candidates if (candidate / "soccar" / "mesh_0.cmf").is_file()),
        None,
    )
    if root is None or not wp.is_cuda_available() or not torch.cuda.is_available():
        pytest.skip("exact local CMFs and CUDA are required")
    geometry = ArenaGeometry.load_soccar(root)
    return str(root), geometry, WarpArenaMeshes(geometry)


def _force_blue_goal(env: Rival2Env, row: int) -> None:
    scoring_position = torch.tensor((0.0, 5300.0, 93.15), device=env.device)
    env.bridge.views["ball_pos"].reshape(env.num_envs, 3)[row].copy_(scoring_position)
    wp.to_torch(env.world.ball_world.position_bt).reshape(env.num_envs, 3)[row].copy_(
        scoring_position * 0.02
    )


def _force_orange_goal(env: Rival2Env, row: int) -> None:
    scoring_position = torch.tensor((0.0, -5300.0, 93.15), device=env.device)
    env.bridge.views["ball_pos"].reshape(env.num_envs, 3)[row].copy_(scoring_position)
    wp.to_torch(env.world.ball_world.position_bt).reshape(env.num_envs, 3)[row].copy_(
        scoring_position * 0.02
    )


def test_ssl_120hz_goal_timeout_and_curriculum_reset_are_single_tick(
    ssl_arena_assets: tuple[str, ArenaGeometry, WarpArenaMeshes],
) -> None:
    root, geometry, meshes = ssl_arena_assets
    # Ten copies of every family ensure the live physics smoke contains all
    # three wall/aerial variants as well as every kickoff layout.
    worlds = len(SCENARIO_NAMES) * 10
    batch = build_ssl_foundation_scenarios(worlds, seed=99120)
    env = Rival2Env(
        worlds,
        root,
        geometry=geometry,
        meshes=meshes,
        device="cuda:0",
        seed=99120,
        car_visitation_order="a_then_b",
        reward_version=RIVAL2_REWARD_SSL_FOUNDATION_V1_VERSION,
        ssl_foundation_scenarios=batch,
    )
    assert env.physics_ticks_per_decision == 1
    assert env.world.physics_ticks_per_decision == 1
    reset_bank = env.world.ssl_foundation_reset
    assert reset_bank is not None

    _force_blue_goal(env, 0)
    _force_orange_goal(env, 2)
    env.bridge.views["rival2.no_touch_ticks"][1] = NO_TOUCH_TIMEOUT_TICKS - 1
    first = env.step(torch.zeros((worlds, 2, 8), device=env.device))

    assert bool(first.terminated[0])
    assert not bool(first.truncated[0])
    assert bool(first.reset_mask[0])
    assert not bool(first.terminated[1])
    assert bool(first.truncated[1])
    assert bool(first.reset_mask[1])
    assert bool(first.terminated[2]) and bool(first.reset_mask[2])
    blue_terminal_only = first.reward[0] - env.last_ssl_foundation_components["total"][0]
    orange_terminal_only = first.reward[2] - env.last_ssl_foundation_components["total"][2]
    torch.testing.assert_close(
        blue_terminal_only,
        torch.tensor((10.0, -10.0), device=env.device),
        rtol=0,
        atol=1.0e-5,
    )
    torch.testing.assert_close(
        orange_terminal_only,
        torch.tensor((-10.0, 10.0), device=env.device),
        rtol=0,
        atol=1.0e-5,
    )
    assert int(wp.to_torch(reset_bank.reset_generation)[0].item()) == 1
    first_source = int(wp.to_torch(reset_bank.current_source_index)[0].item())
    assert first_source == reset_bank.reset_stride % worlds
    np.testing.assert_allclose(
        env.bridge.views["ball_pos"].reshape(worlds, 3)[0].cpu().numpy(),
        batch.state.ball_pos[first_source],
        rtol=0,
        atol=1.0e-4,
    )
    assert int(wp.to_torch(env.world.lifecycle.reset_required)[0].item()) == 0

    _force_blue_goal(env, 0)
    second = env.step(torch.zeros((worlds, 2, 8), device=env.device))
    assert bool(second.terminated[0]) and bool(second.reset_mask[0])
    assert int(wp.to_torch(reset_bank.reset_generation)[0].item()) == 2
    second_source = int(wp.to_torch(reset_bank.current_source_index)[0].item())
    assert second_source == (2 * reset_bank.reset_stride) % worlds
    assert second_source != first_source
