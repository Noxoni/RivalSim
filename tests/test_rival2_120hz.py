from __future__ import annotations

import inspect
import os
from pathlib import Path

import pytest
import torch
import warp as wp

from rivalsim.arena import ArenaGeometry, WarpArenaMeshes
from rivalsim.gameplay_120 import (
    OUTCOME_EXEMPT_CONTESTED_50,
    OUTCOME_EXEMPT_POWER_CONTACT,
    OUTCOME_UNNECESSARY_FLIP_THROUGH_CONTACT,
    physical_flip_outcome,
)
from rivalsim.human_demo.analysis import human_action_alignment_report
from rivalsim.rival2_contracts import (
    ACTION_CONTRACT_HASH,
    ACTION_CONTRACT_V2_120HZ,
    ACTION_CONTRACT_V2_120HZ_HASH,
    GAMEPLAY_120_BOOST_USE_REWARD,
    GAMEPLAY_120_SPEED_COEFFICIENT,
    GAMEPLAY_120_SUPERSONIC_REWARD,
    GAMEPLAY_BOOST_USE_REWARD,
    GAMEPLAY_SPEED_COEFFICIENT,
    GAMEPLAY_SUPERSONIC_REWARD,
    OBS_FIELD_NAMES,
    OBSERVATION_SCHEMA_HASH,
    OBSERVATION_SCHEMA_V2_120HZ,
    OBSERVATION_SCHEMA_V2_120HZ_HASH,
    REWARD_GAMEPLAY_120_V1_CONTRACT,
    REWARD_GAMEPLAY_120_V1_CONTRACT_HASH,
    REWARD_GAMEPLAY_V3_CONTRACT_HASH,
    RIVAL2_REWARD_GAMEPLAY_120_V1_VERSION,
    RIVAL2_REWARD_GAMEPLAY_V1_VERSION,
    contract_hashes_for_reward,
)
from rivalsim.rival2_env import Rival2Env
from rivalsim.rival2_full_match_env import Rival2FullMatchEnv
from rivalsim.rival2_opponent_curriculum import (
    OPPONENT_HISTORICAL,
    OPPONENT_NEXTO,
    OPPONENT_WISP,
    Rival2OpponentCurriculumTrainer,
)
from rivalsim.rival2_policy import sample_hybrid_action
from rivalsim.rival2_ppo import (
    RIVAL2_PPO_120HZ_CONTRACT_HASH,
    RIVAL2_PPO_120HZ_GAE_LAMBDA,
    RIVAL2_PPO_120HZ_GAMMA,
    Rival2PPOConfig,
    rival2_ppo_120hz_config,
)
from rivalsim.viewer.spectator import RivalVisSession

EXPECTED_HISTORICAL_OBS_HASH = (
    "10669E7D240D553BEA601F8AD7AEF9F9029310E55EA6DC4072E866F34BB218AF"
)
EXPECTED_HISTORICAL_ACTION_HASH = (
    "145AF5B49E1A0C85209022A6DE862F05EF996EB89B90B124072A59BC9936727B"
)
EXPECTED_HISTORICAL_GAMEPLAY_V3_HASH = (
    "174D94E19B3F053E250147F98835C18CF65260A82E23B6E58F234F6E81E0D4E7"
)
EXPECTED_120HZ_OBS_HASH = (
    "BF9E141E5A1E5D2F15581C8BBB10F31F11FC5AA6736B327E61C03DD8D2388237"
)
EXPECTED_120HZ_ACTION_HASH = (
    "5E3747CCF9F59BA18D81D07014D60637F7D886907A0F44B0CA681C74F20EF91A"
)
EXPECTED_120HZ_REWARD_HASH = (
    "0D4C9A78803BBAF851AB4FDD7D9AC4196AB08E42B51DC0A173A1EAEC066AFAED"
)
EXPECTED_120HZ_PPO_IDENTITY_HASH = (
    "F5DF4C30EE80BC39E52CFCB4E2813E03D8D26C795A9FC18BC62DD2140D9C9F8A"
)
EXPECTED_120HZ_PPO_CONFIG_HASH = (
    "02A4FBC09D79AD3BA677C3D2F942FE4719DA621EE87E4417D28318BFECA87F93"
)


@pytest.fixture(scope="module")
def arena_assets() -> tuple[str, ArenaGeometry, WarpArenaMeshes]:
    configured = os.environ.get("RIVALSIM_COLLISION_DIR")
    candidates = [Path(configured)] if configured else []
    candidates.append(Path("G:/dev/RLBot-Rival/bot/collision_meshes"))
    root = next(
        (
            candidate
            for candidate in candidates
            if (candidate / "soccar" / "mesh_0.cmf").is_file()
        ),
        None,
    )
    if root is None or not wp.is_cuda_available() or not torch.cuda.is_available():
        pytest.skip("exact local CMFs and CUDA are required")
    geometry = ArenaGeometry.load_soccar(root)
    return str(root), geometry, WarpArenaMeshes(geometry)


def _env(
    assets: tuple[str, ArenaGeometry, WarpArenaMeshes],
    reward_version: str,
    *,
    count: int = 1,
    seed: int = 120,
) -> Rival2Env:
    root, geometry, meshes = assets
    return Rival2Env(
        count,
        root,
        geometry=geometry,
        meshes=meshes,
        device="cuda:0",
        seed=seed,
        car_visitation_order="a_then_b",
        reward_version=reward_version,
    )


def test_120hz_contracts_are_new_and_historical_contracts_are_immutable() -> None:
    assert OBSERVATION_SCHEMA_HASH == EXPECTED_HISTORICAL_OBS_HASH
    assert ACTION_CONTRACT_HASH == EXPECTED_HISTORICAL_ACTION_HASH
    assert REWARD_GAMEPLAY_V3_CONTRACT_HASH == EXPECTED_HISTORICAL_GAMEPLAY_V3_HASH
    assert OBSERVATION_SCHEMA_V2_120HZ_HASH == EXPECTED_120HZ_OBS_HASH
    assert ACTION_CONTRACT_V2_120HZ_HASH == EXPECTED_120HZ_ACTION_HASH
    assert REWARD_GAMEPLAY_120_V1_CONTRACT_HASH == EXPECTED_120HZ_REWARD_HASH
    assert OBSERVATION_SCHEMA_V2_120HZ_HASH != OBSERVATION_SCHEMA_HASH
    assert ACTION_CONTRACT_V2_120HZ_HASH != ACTION_CONTRACT_HASH
    assert ACTION_CONTRACT_V2_120HZ["physics_hz"] == 120
    assert ACTION_CONTRACT_V2_120HZ["policy_hz"] == 120
    assert ACTION_CONTRACT_V2_120HZ["hold_ticks"] == 1
    assert OBSERVATION_SCHEMA_V2_120HZ["shape"] == ["world", "agent", 182]
    assert "immediately preceding 120 Hz tick" in OBSERVATION_SCHEMA_V2_120HZ[
        "temporal_semantics"
    ]["previous_action.*"]
    hashes = contract_hashes_for_reward(RIVAL2_REWARD_GAMEPLAY_120_V1_VERSION)
    assert hashes["RIVAL2_OBS_V2_120HZ"] == OBSERVATION_SCHEMA_V2_120HZ_HASH
    assert hashes["RIVAL2_ACTION_V2_120HZ"] == ACTION_CONTRACT_V2_120HZ_HASH
    assert (
        hashes[RIVAL2_REWARD_GAMEPLAY_120_V1_VERSION]
        == REWARD_GAMEPLAY_120_V1_CONTRACT_HASH
    )


def test_reward_coefficients_and_physical_only_bad_flip_contract() -> None:
    assert GAMEPLAY_120_SPEED_COEFFICIENT == GAMEPLAY_SPEED_COEFFICIENT / 4
    assert GAMEPLAY_120_SUPERSONIC_REWARD == GAMEPLAY_SUPERSONIC_REWARD / 4
    assert GAMEPLAY_120_BOOST_USE_REWARD == GAMEPLAY_BOOST_USE_REWARD / 4
    assert REWARD_GAMEPLAY_120_V1_CONTRACT["unconditional_unique_touch"] == 0.0
    assert REWARD_GAMEPLAY_120_V1_CONTRACT["named_mechanics_reward"] == 0.0
    assert REWARD_GAMEPLAY_120_V1_CONTRACT["named_mechanics_hot_path"] is False
    guard = REWARD_GAMEPLAY_120_V1_CONTRACT["bad_flip_guard"]
    assert guard["active_exemptions_in_precedence_order"] == [
        "EXEMPT_CONTESTED_50",
        "EXEMPT_POWER_CONTACT",
    ]
    assert guard["recognized_mechanic_exemption"] is False
    assert guard["controlled_flick_exemption"] is False
    assert physical_flip_outcome(contested_50=True, power_contact=True) == (
        OUTCOME_EXEMPT_CONTESTED_50
    )
    assert physical_flip_outcome(contested_50=False, power_contact=True) == (
        OUTCOME_EXEMPT_POWER_CONTACT
    )
    assert physical_flip_outcome(contested_50=False, power_contact=False) == (
        OUTCOME_UNNECESSARY_FLIP_THROUGH_CONTACT
    )


def test_active_120hz_consumers_do_not_reference_the_legacy_rival_hold_constant() -> None:
    env_step = inspect.getsource(Rival2Env._step_impl)
    viewer_tick = inspect.getsource(RivalVisSession.advance_physics_tick)
    full_match_init = inspect.getsource(Rival2FullMatchEnv.__init__)
    assert "self.physics_ticks_per_decision" in env_step
    assert "PHYSICS_TICKS_PER_DECISION" not in env_step
    assert "self.env.physics_ticks_per_decision" in viewer_tick
    assert "PHYSICS_TICKS_PER_DECISION" not in viewer_tick
    assert "action_version == RIVAL2_ACTION_V2_120HZ_VERSION" in full_match_init


def test_120hz_ppo_physical_time_equivalence() -> None:
    config = rival2_ppo_120hz_config()
    assert config.rollout_horizon == 128
    assert config.gamma == RIVAL2_PPO_120HZ_GAMMA
    assert config.gae_lambda == RIVAL2_PPO_120HZ_GAE_LAMBDA
    assert config.entropy_coefficient == 0.0
    assert config.gamma**4 == pytest.approx(0.995, rel=0.0, abs=2.0e-15)
    assert config.gae_lambda**4 == pytest.approx(0.95, rel=0.0, abs=2.0e-15)
    assert (config.gamma * config.gae_lambda) ** 4 == pytest.approx(
        0.995 * 0.95, rel=0.0, abs=3.0e-15
    )
    assert RIVAL2_PPO_120HZ_CONTRACT_HASH == EXPECTED_120HZ_PPO_IDENTITY_HASH
    assert config.content_hash == EXPECTED_120HZ_PPO_CONFIG_HASH


def test_human_action_alignment_is_direct_and_keeps_observation_quality_separate() -> None:
    report = human_action_alignment_report()
    assert report["target"]["action_contract_sha256"] == ACTION_CONTRACT_V2_120HZ_HASH
    assert report["temporal_reduction"] is None
    assert report["averaging"] is False
    assert report["subsampling"] is False
    assert report["four_frame_combination"] is False


@pytest.mark.usefixtures("arena_assets")
def test_action_and_previous_action_advance_on_every_physics_tick(arena_assets) -> None:
    env = _env(arena_assets, RIVAL2_REWARD_GAMEPLAY_120_V1_VERSION)
    assert env.policy_hz == 120
    assert env.physics_ticks_per_decision == 1
    assert env.world.gameplay_v3 is None
    assert env.world.gameplay_120 is not None
    assert env.world.gameplay_120.memory_inventory()["named_mechanics_arrays"] == 0
    action_a = torch.zeros((1, 2, 8), device=env.device)
    action_a[0, 0, 0] = 1.0
    action_b = torch.zeros_like(action_a)
    action_b[0, 0, 1] = -0.75
    first = env.step(action_a)
    assert env.world.tick_count == 1
    previous_slice = slice(
        OBS_FIELD_NAMES.index("previous_action.throttle"),
        OBS_FIELD_NAMES.index("previous_action.handbrake") + 1,
    )
    torch.testing.assert_close(
        first.observation[0, 0, previous_slice], action_a[0, 0], rtol=0, atol=0
    )
    second = env.step(action_b)
    assert env.world.tick_count == 2
    torch.testing.assert_close(
        second.observation[0, 0, previous_slice], action_b[0, 0], rtol=0, atol=0
    )


@pytest.mark.usefixtures("arena_assets")
def test_four_tick_held_action_physics_equivalence(arena_assets) -> None:
    old = _env(arena_assets, RIVAL2_REWARD_GAMEPLAY_V1_VERSION, seed=48120)
    new = _env(arena_assets, RIVAL2_REWARD_GAMEPLAY_120_V1_VERSION, seed=48120)
    action = torch.tensor(
        [[[0.75, -0.25, 0.2, -0.1, 0.15, 0.0, 1.0, 0.0],
          [0.5, 0.4, -0.2, 0.1, -0.15, 0.0, 0.0, 0.0]]],
        dtype=torch.float32,
        device=old.device,
    )
    old.step(action)
    for _ in range(4):
        new.step(action)
    torch.cuda.synchronize()
    for name in (
        "car_pos",
        "car_vel",
        "car_quat",
        "car_ang_vel",
        "boost",
        "on_ground",
        "has_jumped",
        "has_double_jumped",
        "has_flipped",
        "is_flipping",
        "wheel_contact",
        "ball_pos",
        "ball_vel",
        "ball_quat",
        "ball_ang_vel",
    ):
        old_value = (
            old.bridge.views[name]
            if name in old.bridge.views
            else wp.to_torch(getattr(old.world.state, name))
        )
        new_value = (
            new.bridge.views[name]
            if name in new.bridge.views
            else wp.to_torch(getattr(new.world.state, name))
        )
        torch.testing.assert_close(
            old_value, new_value, rtol=0.0, atol=2.0e-6
        )


@pytest.mark.usefixtures("arena_assets")
def test_four_tick_progress_and_boost_occupancy_reward_integration(arena_assets) -> None:
    old = _env(arena_assets, RIVAL2_REWARD_GAMEPLAY_V1_VERSION, seed=120481)
    new = _env(arena_assets, RIVAL2_REWARD_GAMEPLAY_120_V1_VERSION, seed=120481)
    action = torch.zeros((1, 2, 8), device=old.device)
    action[0, 0, 6] = 1.0
    old_step = old.step(action)
    new_steps = [new.step(action) for _ in range(4)]
    old_progress = old.bridge.views["rival2.v1_progress_component"].clone()
    old_boost = old.bridge.views["rival2.boost_use_component"].clone()
    # Component views are interval-local, so replay and collect each tick.
    replay = _env(arena_assets, RIVAL2_REWARD_GAMEPLAY_120_V1_VERSION, seed=120481)
    progress_components = []
    boost_components = []
    for _ in range(4):
        replay.step(action)
        progress_components.append(
            replay.bridge.views["rival2.v1_progress_component"].clone()
        )
        boost_components.append(replay.bridge.views["rival2.boost_use_component"].clone())
    new_progress = torch.stack(progress_components).sum(dim=0)
    new_boost = torch.stack(boost_components).sum(dim=0)
    torch.testing.assert_close(old_progress, new_progress, rtol=0, atol=2.0e-7)
    torch.testing.assert_close(old_boost, new_boost, rtol=0, atol=2.0e-7)
    assert len(new_steps) == 4 and torch.isfinite(old_step.reward).all()


@pytest.mark.usefixtures("arena_assets")
def test_goal_terminates_and_resets_on_the_single_120hz_tick(arena_assets) -> None:
    env = _env(arena_assets, RIVAL2_REWARD_GAMEPLAY_120_V1_VERSION, seed=99120)
    scoring_position = torch.tensor(((0.0, 5300.0, 93.15),), device=env.device)
    env.bridge.views["ball_pos"].reshape(1, 3).copy_(scoring_position)
    wp.to_torch(env.world.ball_world.position_bt).reshape(1, 3).copy_(
        scoring_position * 0.02
    )
    transition = env.step(torch.zeros((1, 2, 8), device=env.device))
    assert bool(transition.terminated.item())
    assert bool(transition.reset_mask.item())
    assert env.world.tick_count == 1
    assert env.decision_count == 1
    assert float(transition.reward[0, 0].item()) == pytest.approx(10.0, abs=0.01)
    assert int(env.bridge.views["rival2.kickoff_indicator"].item()) == 1


@pytest.mark.usefixtures("arena_assets")
def test_legacy_historical_policy_is_evaluated_once_and_held_four_ticks(
    arena_assets,
) -> None:
    env = _env(
        arena_assets,
        RIVAL2_REWARD_GAMEPLAY_120_V1_VERSION,
        count=4,
        seed=77120,
    )
    trainer = Rival2OpponentCurriculumTrainer(
        env,
        ppo_config=Rival2PPOConfig(rollout_horizon=4, minibatch_size=8, epochs=1),
        seed=77120,
    )
    trainer.opponent_pool.add(
        trainer.model,
        77,
        policy_hz=30,
        action_version="RIVAL2_ACTION_V1",
    )
    trainer.opponent_family.fill_(OPPONENT_HISTORICAL)
    trainer.rival_side.zero_()
    trainer.opponent_assignment.fill_(77)
    emitted = []
    for _ in range(4):
        actor, _value, _version, _mask = trainer._policy_outputs(env.observation)
        sampled = sample_hybrid_action(
            actor, generator=trainer.policy_generator, config=trainer.policy_config
        ).action
        emitted.append(trainer._apply_historical_policy_cadence(sampled)[:, 1].clone())
    assert trainer.historical_policy_evaluation_calls == 1
    for action in emitted[1:]:
        torch.testing.assert_close(action, emitted[0], rtol=0, atol=0)
    state = trainer.opponent_pool.checkpoint_state()[0]
    assert state["policy_hz"] == 30
    assert state["action_version"] == "RIVAL2_ACTION_V1"


@pytest.mark.usefixtures("arena_assets")
def test_nexto_wisp_physical_tick_cadence_matches_old_outer_step(arena_assets) -> None:
    old = Rival2OpponentCurriculumTrainer(
        _env(arena_assets, RIVAL2_REWARD_GAMEPLAY_V1_VERSION, count=2, seed=92120),
        ppo_config=Rival2PPOConfig(rollout_horizon=1),
        seed=92120,
    )
    new = Rival2OpponentCurriculumTrainer(
        _env(
            arena_assets,
            RIVAL2_REWARD_GAMEPLAY_120_V1_VERSION,
            count=2,
            seed=92120,
        ),
        ppo_config=Rival2PPOConfig(rollout_horizon=4),
        seed=92120,
    )
    family = torch.tensor(
        [OPPONENT_NEXTO, OPPONENT_WISP], dtype=torch.int64, device=old.device
    )
    active = torch.ones(2, dtype=torch.bool, device=old.device)
    for trainer in (old, new):
        trainer.opponent_family.copy_(family)
        trainer.rival_side.zero_()
        trainer.opponent_assignment.fill_(-1)
        trainer.nexto.set_player_index(torch.ones(2, dtype=torch.long, device=trainer.device))
        trainer.wisp.set_player_index(torch.ones(2, dtype=torch.long, device=trainer.device))
        trainer.nexto.activate(active)
        trainer.wisp.activate(active)

    old_nexto: list[torch.Tensor] = []
    old_wisp: list[torch.Tensor] = []
    new_nexto: list[torch.Tensor] = []
    new_wisp: list[torch.Tensor] = []

    def instrument(trainer, nexto_rows, wisp_rows) -> None:
        nexto_tick = trainer.nexto.tick_action
        wisp_tick = trainer.wisp.tick_action

        def capture_nexto(*args, **kwargs):
            result = nexto_tick(*args, **kwargs)
            nexto_rows.append(result[0].clone())
            return result

        def capture_wisp(*args, **kwargs):
            result = wisp_tick(*args, **kwargs)
            wisp_rows.append(result[0].clone())
            return result

        trainer.nexto.tick_action = capture_nexto
        trainer.wisp.tick_action = capture_wisp

    instrument(old, old_nexto, old_wisp)
    instrument(new, new_nexto, new_wisp)
    fixed = torch.zeros((2, 2, 8), device=old.device)
    fixed[:, 0, 0] = 0.5
    old._step_with_frozen_opponents(fixed)
    for _ in range(4):
        new._step_with_frozen_opponents(fixed)
    assert len(old_nexto) == len(old_wisp) == len(new_nexto) == len(new_wisp) == 4
    for old_action, new_action in zip(old_nexto, new_nexto, strict=True):
        torch.testing.assert_close(old_action, new_action, rtol=0, atol=0)
    for old_action, new_action in zip(old_wisp, new_wisp, strict=True):
        torch.testing.assert_close(old_action, new_action, rtol=0, atol=0)
    for name in ("car_pos", "car_vel", "car_quat", "ball_pos", "ball_vel"):
        torch.testing.assert_close(
            old.env.bridge.views[name], new.env.bridge.views[name], rtol=0, atol=2.0e-6
        )
    torch.testing.assert_close(old.nexto.previous_action, new.nexto.previous_action)
    torch.testing.assert_close(old.nexto.neural_counter, new.nexto.neural_counter)
    torch.testing.assert_close(old.wisp.previous_action, new.wisp.previous_action)
    torch.testing.assert_close(old.wisp.ticks, new.wisp.ticks)
    assert old.nexto._cadence_tick == new.nexto._cadence_tick
