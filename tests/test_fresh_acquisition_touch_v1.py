from dataclasses import replace
from pathlib import Path

import pytest
import torch
import warp as wp

from rivalsim.fresh_acquisition_touch_v1 import (
    FirstTouchEnv, FirstTouchCollector, first_touch_payment, authority,
    reward_authority, FIRST_TOUCH_REWARD, new_model, training_bank, SEED,
)
from rivalsim.fresh_acquisition_only_v1 import authority as prior_authority
from rivalsim.sustained_gameplay_v1 import PHYSICS_GAMMA, ppo_config, IDLE_TICKS
from rivalsim.sustained_acquisition_v1 import FAMILY
from rivalsim.ssl_entity_training import fresh_entity_optimizer
from rivalsim.ssl_entity_mixed_training import mixed_joint_ppo_update
from benchmarks.run_rival2_fresh_ground_30hz_v1 import tensor_hash
from tests.test_fresh_ground_30hz import force_goal


@pytest.fixture(scope="module", autouse=True)
def threads():
    torch.set_num_threads(4)


@pytest.mark.parametrize("tick", [0, 1, 2, 3])
def test_first_own_touch_native_substep_discount(tick):
    first = torch.tensor([[tick, -1], [-1, tick], [tick, tick]])
    awarded, reward = first_touch_payment(first, torch.full((3,), -1),
        torch.zeros(3, 2, dtype=torch.bool), torch.ones(3, dtype=torch.bool))
    assert torch.equal(awarded, first >= 0)
    torch.testing.assert_close(reward, (first >= 0).float()*PHYSICS_GAMMA**tick)
    # Sustained/repeated contact in the same episode has no second award.
    assert not first_touch_payment(first, torch.full((3,), -1), awarded,
                                   torch.ones(3, dtype=torch.bool))[1].any()


def test_opponent_first_does_not_remove_own_eligibility_and_no_other_family():
    first = torch.tensor([[0, 1], [0, 1]])
    paid = torch.tensor([[True, False], [False, False]])
    awarded, reward = first_touch_payment(first, torch.tensor([-1, -1]), paid,
                                         torch.tensor([True, False]))
    assert awarded.tolist() == [[False, True], [False, False]]
    assert reward[0, 1] == pytest.approx(PHYSICS_GAMMA)


def test_touch_goal_same_tick_allowed_post_goal_rejected():
    first = torch.tensor([[1, 2], [0, 3], [-1, -1]])
    awarded, reward = first_touch_payment(first, torch.tensor([1, 3, -1]),
        torch.zeros(3, 2, dtype=torch.bool), torch.ones(3, dtype=torch.bool))
    assert awarded.tolist() == [[True, False], [True, True], [False, False]]
    assert reward.sum() > 2.99


def test_only_requested_reward_changes_and_fresh_determinism():
    new, old = authority(), prior_authority()
    assert new["reward"]["base"] == old["reward"]
    assert FIRST_TOUCH_REWARD == 1 and new["reward"]["base"]["goal"] == 10
    for key in ("seed", "ppo", "opponent_worlds", "scenarios", "evaluation_offsets", "max_updates"):
        assert new[key] == old[key]
    a, b = new_model(), new_model()
    assert tensor_hash(a.state_dict()) == tensor_hash(b.state_dict())
    assert not fresh_entity_optimizer(a).state and new["parent"] is None


@pytest.fixture()
def env():
    root = Path("G:/dev/RLBot-Rival/bot/collision_meshes")
    if not torch.cuda.is_available() or not root.exists():
        pytest.skip("Native CUDA and arena required")
    return FirstTouchEnv(32, root, device="cuda:0", seed=SEED,
                         ssl_foundation_scenarios=training_bank(32))


def place_ball(env, row, position):
    env.bridge.views["ball_pos"][row] = position
    wp.to_torch(env.world.ball_world.position_bt).reshape(-1, 3)[row] = position*.02
    env.bridge.views["ball_vel"][row].zero_()
    wp.to_torch(env.world.ball_world.velocity_bt).reshape(-1, 3)[row].zero_()


def native_contact(env, row, side):
    # Disposable native-physics collision fixture, never a training prefix.
    car = env.bridge.views["car_pos"].reshape(32, 2, 3)[row, side]
    place_ball(env, row, car + torch.tensor([0., 0., 75.], device=env.device))


def test_native_contact_once_no_touch_reset_rearmed_only_after_episode(env):
    zeros = torch.zeros(32, 2, 8, device=env.device)
    generation = wp.to_torch(env.world.ssl_foundation_reset.reset_generation)
    before = generation.clone()
    native_contact(env, 0, 0)
    tr = env.step(zeros)
    assert env.last_native["first_touch_tick"][0, 0] >= 0
    assert env.last_components["first_touch"][0, 0] > .999
    assert env.first_touch_paid[0, 0] and not tr.reset_mask[0]
    assert generation[0] == before[0]
    expected = sum(v for k,v in env.last_components.items() if k != "total")
    torch.testing.assert_close(tr.reward, expected, rtol=1e-6, atol=1e-6)
    # Separate and collide again: no repeated reward, even in a new decision.
    place_ball(env, 0, torch.tensor([0., 1000., 93.15], device=env.device))
    env.step(zeros)
    native_contact(env, 0, 0)
    env.step(zeros)
    assert env.last_native["first_touch_tick"][0, 0] >= 0
    assert env.last_components["first_touch"][0, 0] == 0
    # The other player can still earn its own first contact in this episode.
    place_ball(env, 0, torch.tensor([0., 1000., 93.15], device=env.device)); env.step(zeros)
    native_contact(env, 0, 1); env.step(zeros)
    assert env.last_components["first_touch"][0, 1] > .999
    # Scoring remains available, with no bonus itself on a non-contact goal.
    force_goal(env, 0); tr = env.step(zeros)
    assert tr.terminated[0] and not env.first_touch_paid[0].any()
    assert env.last_components["terminal_goal"][0].abs().min() > 9.99
    native_contact(env, 0, 0); env.step(zeros)
    assert env.last_components["first_touch"][0, 0] > .999


def test_idle_clears_latches_but_regular_decision_does_not(env):
    zero = torch.zeros(32, 2, 8, device=env.device)
    env.first_touch_paid[0] = True
    env.step(zero)
    assert env.first_touch_paid[0].all()
    env.idle_ticks[0] = IDLE_TICKS
    tr = env.step(zero)
    assert tr.truncated[0] and not tr.terminated[0]
    assert not env.first_touch_paid[0].any()
    assert env.last_components["inactivity"][0].max() < -.999


def test_collector_records_bonus_with_current_only_masks_and_ppo(env):
    model = new_model().cuda()
    collector = FirstTouchCollector(env, model, seed=SEED)
    collector.config = replace(ppo_config(), rollout_horizon=8, epochs=1, minibatch_size=256)
    native_contact(env, 0, int(env.focal[0])); native_contact(env, 1, 0)
    initial = tensor_hash(model.state_dict())
    rollout = collector.collect()
    metrics = collector.last_metrics
    assert metrics["first_touch_bonus"]["learner_awards"] >= 2
    assert metrics["potential_reward_components"]["first_touch"] > 1.99
    assert metrics["nexto_training_sample_count"]*3 == metrics["trainable_agent_samples"]
    result = mixed_joint_ppo_update(model, fresh_entity_optimizer(model), rollout, collector.config,
                                   torch.Generator(device=env.device).manual_seed(SEED+2))
    assert result["optimizer_steps"] > 0 and result["kl_rejections"] == 0
    assert tensor_hash(model.state_dict()) != initial
