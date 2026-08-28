from __future__ import annotations

import inspect
import math
import os
from pathlib import Path

import numpy as np
import pytest
import torch
import warp as wp

from rivalsim.arena import ArenaGeometry, WarpArenaMeshes
from rivalsim.gameplay_v3 import (
    CONTEST_CONTACT_WINDOW_TICKS,
    OUTCOME_EXEMPT_CONTESTED_50,
    OUTCOME_EXEMPT_CONTROLLED_FLICK,
    OUTCOME_EXEMPT_POWER_CONTACT,
    OUTCOME_EXEMPT_RECOGNIZED_MECHANIC,
    OUTCOME_UNNECESSARY_FLIP_THROUGH_CONTACT,
    Rival2GameplayV3State,
    contest_convergence_exempt,
    controlled_flick_exempt,
    flip_contact_candidate,
    gameplay_v3_compose_reward,
    power_contact_exempt,
    primary_flip_outcome,
)
from rivalsim.kernels.boost_pad import SOCCAR_PAD_POSITIONS
from rivalsim.kernels.rival2 import (
    REWARD_MODE_GAMEPLAY_V3,
)
from rivalsim.rival2_contracts import (
    ACTION_CONTRACT_HASH,
    EPISODE_CONTRACT_HASH,
    GAMEPLAY_V3_MAX_PAID_MECHANICS_EVENTS,
    GAMEPLAY_V3_MECHANICS_EPISODE_BUDGET,
    GAMEPLAY_V3_MECHANICS_EVENT_REWARD,
    GAMEPLAY_V3_UNNECESSARY_FLIP_PENALTY,
    OBSERVATION_SCHEMA_HASH,
    REWARD_GAMEPLAY_V1_CONTRACT_HASH,
    REWARD_GAMEPLAY_V2_CONTRACT_HASH,
    REWARD_GAMEPLAY_V3_CONTRACT,
    REWARD_GAMEPLAY_V3_CONTRACT_HASH,
    RIVAL2_REWARD_GAMEPLAY_V1_VERSION,
    RIVAL2_REWARD_GAMEPLAY_V2_VERSION,
    RIVAL2_REWARD_GAMEPLAY_V3_VERSION,
    contract_hashes_for_reward,
)
from rivalsim.rival2_env import Rival2Env, Rival2WorldSim
from rivalsim.state import StateSnapshot

EXPECTED_V1_HASH = "48AAC000B97D2652507F677184A3FE4F0A3A86CED136B680C933EFF33CD9F072"
EXPECTED_V2_HASH = "4073E29C1013458D5784435061FE47C639525BE37E8CD519783889C69BA87D41"


def _collision_root() -> Path | None:
    configured = os.environ.get("RIVALSIM_COLLISION_DIR")
    candidates = (
        Path(configured) if configured else None,
        Path("G:/dev/RLBot-Rival/bot/collision_meshes"),
    )
    for candidate in candidates:
        if candidate is not None and (candidate / "soccar" / "mesh_0.cmf").is_file():
            return candidate
    return None


def test_v3_contract_identity_and_historical_hashes_are_frozen() -> None:
    assert REWARD_GAMEPLAY_V1_CONTRACT_HASH == EXPECTED_V1_HASH
    assert REWARD_GAMEPLAY_V2_CONTRACT_HASH == EXPECTED_V2_HASH
    assert len(REWARD_GAMEPLAY_V3_CONTRACT_HASH) == 64
    assert REWARD_GAMEPLAY_V3_CONTRACT["unconditional_unique_touch"] == 0.0
    assert REWARD_GAMEPLAY_V3_CONTRACT["gameplay_v2_standalone_double_dash_reward"] == 0.0
    assert REWARD_GAMEPLAY_V3_CONTRACT["mechanics"]["event_reward"] == 0.005
    assert REWARD_GAMEPLAY_V3_CONTRACT["mechanics"]["episode_budget"] == 0.05
    assert REWARD_GAMEPLAY_V3_CONTRACT["mechanics"]["max_paid_events_per_player_episode"] == 10
    assert contract_hashes_for_reward(RIVAL2_REWARD_GAMEPLAY_V3_VERSION) == {
        "RIVAL2_OBS_V1": OBSERVATION_SCHEMA_HASH,
        "RIVAL2_ACTION_V1": ACTION_CONTRACT_HASH,
        RIVAL2_REWARD_GAMEPLAY_V3_VERSION: REWARD_GAMEPLAY_V3_CONTRACT_HASH,
        "RIVAL2_EPISODE_V1": EPISODE_CONTRACT_HASH,
    }


def test_v3_classifier_calibration_boundaries_and_primary_precedence() -> None:
    assert CONTEST_CONTACT_WINDOW_TICKS == 2
    assert contest_convergence_exempt(
        opponent_distance=500.0,
        self_closing_speed=150.0,
        opponent_closing_speed=150.0,
        time_to_ball_delta=0.12,
    )
    assert not contest_convergence_exempt(
        opponent_distance=500.01,
        self_closing_speed=400.0,
        opponent_closing_speed=400.0,
        time_to_ball_delta=0.01,
    )
    assert not contest_convergence_exempt(
        opponent_distance=200.0,
        self_closing_speed=300.0,
        opponent_closing_speed=-1.0,
        time_to_ball_delta=0.01,
    )
    assert power_contact_exempt(
        total_closing_speed=300.0,
        rotational_closing_speed=100.0,
        rotational_share=0.18,
        ball_delta_v=175.0,
    )
    assert not power_contact_exempt(
        total_closing_speed=800.0,
        rotational_closing_speed=99.99,
        rotational_share=0.5,
        ball_delta_v=500.0,
    )
    assert controlled_flick_exempt(
        control_ticks=4,
        control_max_distance=220.0,
        control_max_relative_speed=260.0,
        release_distance=245.0,
        ball_delta_v=120.0,
    )
    assert not controlled_flick_exempt(
        control_ticks=3,
        control_max_distance=100.0,
        control_max_relative_speed=100.0,
        release_distance=300.0,
        ball_delta_v=200.0,
    )
    assert primary_flip_outcome(
        recognized_mechanic=True,
        controlled_flick=True,
        contested_50=True,
        power_contact=True,
    ) == OUTCOME_EXEMPT_RECOGNIZED_MECHANIC
    assert primary_flip_outcome(
        recognized_mechanic=False,
        controlled_flick=True,
        contested_50=True,
        power_contact=True,
    ) == OUTCOME_EXEMPT_CONTROLLED_FLICK
    assert primary_flip_outcome(
        recognized_mechanic=False,
        controlled_flick=False,
        contested_50=True,
        power_contact=True,
    ) == OUTCOME_EXEMPT_CONTESTED_50
    assert primary_flip_outcome(
        recognized_mechanic=False,
        controlled_flick=False,
        contested_50=False,
        power_contact=True,
    ) == OUTCOME_EXEMPT_POWER_CONTACT
    assert primary_flip_outcome(
        recognized_mechanic=False,
        controlled_flick=False,
        contested_50=False,
        power_contact=False,
    ) == OUTCOME_UNNECESSARY_FLIP_THROUGH_CONTACT


@pytest.mark.parametrize(
    ("touch", "flipping", "flipped", "torque", "expected"),
    (
        (True, True, True, 0.26, True),
        (True, False, True, 1.0, False),
        (True, True, False, 1.0, False),
        (True, True, True, 0.25, False),
        (False, True, True, 1.0, False),
    ),
)
def test_bad_flip_candidate_is_same_contact_active_directional_dodge_only(
    touch: bool,
    flipping: bool,
    flipped: bool,
    torque: float,
    expected: bool,
) -> None:
    assert flip_contact_candidate(
        touch_onset=touch,
        is_flipping=flipping,
        has_flipped=flipped,
        directional_torque_norm=torque,
    ) is expected


def test_v3_reward_compose_budget_is_integer_exact_independent_and_zero_sum() -> None:
    device = "cpu"
    interval_tick = wp.array(np.asarray((4,), dtype=np.int32), device=device)
    reward = wp.array(np.asarray((1.25, -1.25), dtype=np.float32), device=device)
    requested = wp.array(np.asarray((11, 2), dtype=np.int32), device=device)
    detected_values = np.zeros(20, dtype=np.int32)
    detected_values[0] = 11
    detected_values[10] = 2
    detected = wp.array(detected_values, device=device)
    paid = wp.zeros(2, dtype=wp.int32, device=device)
    suppressed = wp.zeros(2, dtype=wp.int32, device=device)
    bad = wp.array(np.asarray((1, 0), dtype=np.int32), device=device)
    paid_episode = wp.array(np.asarray((0, 9), dtype=np.int32), device=device)
    exhausted_latched = wp.zeros(2, dtype=wp.int32, device=device)
    exhausted_total = wp.zeros(2, dtype=wp.int32, device=device)
    total_paid = wp.zeros(20, dtype=wp.int32, device=device)
    total_suppressed = wp.zeros(2, dtype=wp.int32, device=device)
    mechanics = wp.zeros(1, dtype=wp.float32, device=device)
    bad_component = wp.zeros(1, dtype=wp.float32, device=device)
    total = wp.zeros(1, dtype=wp.float32, device=device)

    wp.launch(
        gameplay_v3_compose_reward,
        dim=1,
        inputs=[
            interval_tick,
            reward,
            requested,
            detected,
            paid,
            suppressed,
            bad,
            paid_episode,
            exhausted_latched,
            exhausted_total,
            total_paid,
            total_suppressed,
            mechanics,
            bad_component,
            total,
        ],
        device=device,
    )
    value = np.asarray(reward.numpy())
    assert GAMEPLAY_V3_MECHANICS_EVENT_REWARD == 0.005
    assert GAMEPLAY_V3_UNNECESSARY_FLIP_PENALTY == -0.01
    assert GAMEPLAY_V3_MAX_PAID_MECHANICS_EVENTS == 10
    assert GAMEPLAY_V3_MECHANICS_EPISODE_BUDGET == 0.05
    np.testing.assert_array_equal(paid.numpy(), np.asarray((10, 1), dtype=np.int32))
    np.testing.assert_array_equal(suppressed.numpy(), np.asarray((1, 1), dtype=np.int32))
    np.testing.assert_allclose(mechanics.numpy(), np.asarray((0.045,), dtype=np.float32))
    np.testing.assert_allclose(bad_component.numpy(), np.asarray((-0.01,), dtype=np.float32))
    np.testing.assert_allclose(value, np.asarray((1.285, -1.285), dtype=np.float32))
    np.testing.assert_allclose(value[1], -value[0])
    np.testing.assert_array_equal(exhausted_total.numpy(), np.asarray((1, 1), dtype=np.int32))


def test_production_module_has_no_host_export_in_tick_or_compose_path() -> None:
    source = inspect.getsource(Rival2GameplayV3State.launch_tick)
    source += inspect.getsource(Rival2GameplayV3State.compose_reward)
    assert ".cpu(" not in source
    assert ".numpy(" not in source


@pytest.mark.skipif(not wp.is_cuda_available(), reason="CUDA required")
def test_reward_identity_construction_step_matrix_and_v3_state_lifetime() -> None:
    collision_root = _collision_root()
    if collision_root is None:
        pytest.skip("RocketSim collision meshes unavailable")
    for reward_version in (
        RIVAL2_REWARD_GAMEPLAY_V1_VERSION,
        RIVAL2_REWARD_GAMEPLAY_V2_VERSION,
        RIVAL2_REWARD_GAMEPLAY_V3_VERSION,
    ):
        env = Rival2Env(2, str(collision_root), reward_version=reward_version)
        assert env.observation.shape == (2, 2, 182)
        expected_v3 = reward_version == RIVAL2_REWARD_GAMEPLAY_V3_VERSION
        assert (env.world.gameplay_v3 is not None) is expected_v3
        assert ("gameplay_v3.total_component" in env.bridge.views) is expected_v3
        if expected_v3:
            inventory = env.world.gameplay_v3.memory_inventory()
            assert inventory["calibration_evidence_buffers_allocated"] is False
            assert not any(name.startswith("evidence_") for name in inventory["arrays"])
        step = env.step(torch.zeros((2, 2, 8), dtype=torch.float32, device=env.device))
        torch.cuda.synchronize(env.device)
        assert step.observation.shape == (2, 2, 182)
        torch.testing.assert_close(step.reward[:, 1], -step.reward[:, 0], rtol=0.0, atol=1e-6)
        if expected_v3:
            components = (
                env.bridge.views["rival2.v1_goal_component"]
                + env.bridge.views["rival2.v1_progress_component"]
                + env.bridge.views["rival2.v1_touch_component"]
                + env.bridge.views["rival2.v1_demo_component"]
                + env.bridge.views["rival2.speed_component"]
                + env.bridge.views["rival2.supersonic_component"]
                + env.bridge.views["rival2.boost_use_component"]
                + env.bridge.views["rival2.boost_pickup_component"]
                + env.bridge.views["rival2.save_component"]
                + env.bridge.views["gameplay_v3.mechanics_component"]
                + env.bridge.views["gameplay_v3.bad_flip_component"]
            )
            torch.testing.assert_close(step.reward[:, 0], components, rtol=0.0, atol=1e-6)
            assert bool((env.bridge.views["rival2.v1_touch_component"] == 0).all())


@pytest.mark.skipif(not wp.is_cuda_available(), reason="CUDA required")
def test_unknown_mode_fails_closed_and_v3_requires_fresh_state() -> None:
    collision_root = _collision_root()
    if collision_root is None:
        pytest.skip("RocketSim collision meshes unavailable")
    with pytest.raises(ValueError, match=r"unsupported Rival 2\.0 reward mode"):
        Rival2WorldSim(1, str(collision_root), reward_mode=99, device="cuda:0")
    env = Rival2Env(
        1,
        str(collision_root),
        reward_version=RIVAL2_REWARD_GAMEPLAY_V2_VERSION,
    )
    with pytest.raises(ValueError, match="freshly constructed"):
        env.set_reward_version(RIVAL2_REWARD_GAMEPLAY_V3_VERSION)
    assert env.world.reward_mode != REWARD_MODE_GAMEPLAY_V3


@pytest.mark.skipif(not wp.is_cuda_available(), reason="CUDA required")
def test_v3_retains_v1_boost_pad_and_save_event_accounting_but_zeroes_touch() -> None:
    collision_root = _collision_root()
    if collision_root is None:
        pytest.skip("RocketSim collision meshes unavailable")
    geometry = ArenaGeometry.load_soccar(collision_root)
    meshes = WarpArenaMeshes(geometry, "cuda:0")

    def initial_state() -> StateSnapshot:
        state = StateSnapshot.empty(4)
        state.ball_pos[:] = (0.0, 0.0, 1500.0)
        state.boost[1] = (50.0, 0.0)
        state.car_pos[2, 0] = SOCCAR_PAD_POSITIONS[0]
        state.boost[2, 0] = 0.0
        state.car_pos[3, 1] = SOCCAR_PAD_POSITIONS[6]
        state.boost[3, 1] = 5.0

        yaw = math.pi / 2.0
        state.car_quat[0, 0] = (0.0, 0.0, math.sin(yaw / 2.0), math.cos(yaw / 2.0))
        state.car_pos[0, 0] = (0.0, -4500.0, 72.395)
        state.car_vel[0, 0] = (0.0, 1600.0, 0.0)
        state.car_pos[0, 1] = (3000.0, 0.0, 1000.0)
        state.ball_pos[0] = (28.177305, -4334.3708, 93.15)
        state.ball_vel[0] = (0.0, -1000.0, 0.0)
        return state

    outputs: dict[str, dict[str, torch.Tensor]] = {}
    for version in (
        RIVAL2_REWARD_GAMEPLAY_V1_VERSION,
        RIVAL2_REWARD_GAMEPLAY_V3_VERSION,
    ):
        env = Rival2Env(
            4,
            str(collision_root),
            initial=initial_state(),
            geometry=geometry,
            meshes=meshes,
            car_visitation_order="a_then_b",
            reward_version=version,
        )
        action = torch.zeros((4, 2, 8), device=env.device)
        action[1, :, 6] = 1.0
        env.step(action)
        torch.cuda.synchronize(env.device)
        outputs[version] = {
            name: env.bridge.views[f"rival2.{name}"].clone()
            for name in (
                "boost_use_event",
                "small_pad_pickup_count",
                "big_pad_pickup_count",
                "save_count",
                "boost_use_component",
                "boost_pickup_component",
                "save_component",
                "v1_touch_component",
            )
        }

    v1 = outputs[RIVAL2_REWARD_GAMEPLAY_V1_VERSION]
    v3 = outputs[RIVAL2_REWARD_GAMEPLAY_V3_VERSION]
    for name in (
        "boost_use_event",
        "small_pad_pickup_count",
        "big_pad_pickup_count",
        "save_count",
        "boost_use_component",
        "boost_pickup_component",
        "save_component",
    ):
        torch.testing.assert_close(v3[name], v1[name], rtol=0.0, atol=0.0)
    assert float(v1["v1_touch_component"][0].item()) == pytest.approx(0.05)
    assert float(v3["v1_touch_component"][0].item()) == 0.0


@pytest.mark.skipif(not wp.is_cuda_available(), reason="CUDA required")
def test_v3_interval_and_episode_state_have_distinct_lifetimes() -> None:
    collision_root = _collision_root()
    if collision_root is None:
        pytest.skip("RocketSim collision meshes unavailable")
    env = Rival2Env(
        2,
        str(collision_root),
        reward_version=RIVAL2_REWARD_GAMEPLAY_V3_VERSION,
    )
    v3 = env.world.gameplay_v3
    assert v3 is not None
    requested = env.bridge.views["gameplay_v3.interval_requested"]
    paid_episode = env.bridge.views["gameplay_v3.mechanics_paid_episode"]
    pending = wp.to_torch(v3.pending_active)
    history = wp.to_torch(v3.history_count)
    requested.fill_(3)
    paid_episode.fill_(7)
    pending.fill_(1)
    history.fill_(5)

    env.world.begin_decision()
    torch.cuda.synchronize(env.device)
    assert bool((requested == 0).all())
    assert bool((paid_episode == 7).all())
    assert bool((pending == 1).all())
    assert bool((history == 5).all())

    reset_mask = env.bridge.views["rival2.reset_mask"]
    reset_mask.copy_(torch.tensor((1, 0), dtype=torch.int32, device=env.device))
    v3.reset(env.world.rival2.reset_mask)
    torch.cuda.synchronize(env.device)
    assert paid_episode.tolist() == [0, 0, 7, 7]
    assert pending.tolist() == [0, 0, 1, 1]
    assert history.tolist() == [1, 1, 5, 5]
