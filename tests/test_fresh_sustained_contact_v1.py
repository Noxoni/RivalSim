from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest
import torch
import warp as wp

from rivalsim.fresh_sustained_contact_v1 import (
    ContactEnv, ContactCollector, contact_bonus_tick, authority, training_bank,
    new_model, SEED, PHYSICS_GAMMA, MIN_BALL_CENTER_BT, MAX_AIR_TOUCHES,
)
from rivalsim.sustained_gameplay_v1 import reward_authority as base_reward, ppo_config, IDLE_TICKS
from rivalsim.sustained_acquisition_v1 import AcquisitionEnv
from rivalsim.static_world import make_standard_kickoff_state
from rivalsim.ssl_entity_training import fresh_entity_optimizer
from rivalsim.ssl_entity_mixed_training import mixed_joint_ppo_update
from rivalsim.fresh_ground_30hz import scenario_hash
from tests.test_fresh_ground_30hz import force_goal
from tests.test_fresh_acquisition_touch_v1 import place_ball


@pytest.fixture(scope="module", autouse=True)
def threads():
    torch.set_num_threads(4)


def test_full_mix_standing_kickoffs_and_no_trained_parent():
    b = training_bank(128)
    assert set(b.family) == set(range(6))
    assert (b.family == 5).sum() == 64
    assert scenario_hash(b) == scenario_hash(training_bank(128))
    rows = np.flatnonzero(b.kickoff_indicator)
    native = make_standard_kickoff_state(len(rows), b.kickoff_layout[rows])
    for name in b.state.__dataclass_fields__:
        np.testing.assert_array_equal(getattr(b.state,name)[rows],getattr(native,name))
    assert not b.state.car_vel[rows].any() and not b.state.ball_vel[rows].any()
    assert authority()["reward"]["base"] == base_reward()
    assert authority()["parent"] is None
    assert not fresh_entity_optimizer(new_model()).state


@pytest.fixture()
def env():
    root=Path("G:/dev/RLBot-Rival/bot/collision_meshes")
    if not torch.cuda.is_available() or not root.exists(): pytest.skip("Native GPU physics required")
    return ContactEnv(32,root,device="cuda:0",seed=SEED,ssl_foundation_scenarios=training_bank(32))


@pytest.mark.parametrize("substep", [0,1,2,3])
def test_native_kernel_height_timing_budget_and_independent_player(substep):
    device="cuda:0"; n=3
    count=wp.ones(n*2,dtype=wp.int32,device=device)
    pos=wp.array(np.array([[0,0,MIN_BALL_CENTER_BT-.0001],[0,0,MIN_BALL_CENTER_BT+.01],[0,0,5]],np.float32),dtype=wp.vec3,device=device)
    goal=wp.zeros(n,dtype=wp.int32,device=device)
    interval=wp.full(n,substep+1,dtype=wp.int32,device=device)
    prior_goal=wp.zeros(n,dtype=wp.int32,device=device)
    names=("prior_count","first_paid","air_paid","first_reward","air_reward","first_events","air_events","air_eligible","repeats")
    arrays={k:wp.zeros(n*2,dtype=wp.float32 if k.endswith("reward") else wp.int32,device=device) for k in names}
    def launch():
        wp.launch(contact_bonus_tick,dim=n,inputs=[count,pos,pos,goal,interval,prior_goal,*[arrays[k] for k in names],PHYSICS_GAMMA,MIN_BALL_CENTER_BT],device=device)
        wp.synchronize_device(device)
    launch()
    np.testing.assert_allclose(arrays["first_reward"].numpy(),PHYSICS_GAMMA**substep,rtol=1e-6)
    np.testing.assert_allclose(arrays["air_reward"].numpy()[:2],0)
    np.testing.assert_allclose(arrays["air_reward"].numpy()[2:],.25*PHYSICS_GAMMA**substep,rtol=1e-6)
    saved=arrays["air_reward"].numpy().copy();launch()
    np.testing.assert_array_equal(saved,arrays["air_reward"].numpy()) # continuous contact
    for i in range(15):
        wp.to_torch(count).add_(1);launch()
    assert arrays["first_events"].numpy().max()==1
    assert arrays["air_paid"].numpy().max()==MAX_AIR_TOUCHES
    assert arrays["air_eligible"].numpy().max()>MAX_AIR_TOUCHES
    assert arrays["air_reward"].numpy().max()==pytest.approx(2.5*PHYSICS_GAMMA**substep,rel=1e-6)
    # Current goal tick may stack; subsequent substeps must not award.
    for a in arrays.values():a.zero_()
    prior_goal.zero_();wp.to_torch(goal).fill_(1);launch()
    assert arrays["first_events"].numpy().sum()==6
    wp.to_torch(count).add_(1);saved=arrays["air_reward"].numpy().copy();launch()
    np.testing.assert_array_equal(saved,arrays["air_reward"].numpy())


def collide(env,row,side,height):
    car=env.bridge.views["car_pos"].reshape(32,2,3)[row,side]
    p=car.clone();p[2]=height
    place_ball(env,row,p)


def test_actual_ground_air_contact_preserves_goal_and_episode(env):
    zero=torch.zeros(32,2,8,device=env.device)
    generation=wp.to_torch(env.world.ssl_foundation_reset.reset_generation).clone()
    collide(env,0,0,91.25);tr=env.step(zero)
    assert env.last_native["first_touch_tick"][0,0]>=0
    assert env.last_components["first_touch"][0,0]>.999
    assert env.last_components["air_touch"][0,0]==0 # no retroactive pop bonus
    assert not tr.reset_mask[0]
    place_ball(env,0,torch.tensor([0.,1000.,93.15],device=env.device));env.step(zero)
    collide(env,0,0,117.);tr=env.step(zero)
    assert env.last_native["first_touch_tick"][0,0]>=0
    assert env.last_components["first_touch"][0,0]==0
    assert env.last_components["air_touch"][0,0]>.249
    assert not tr.reset_mask[0]
    expected=sum(v for k,v in env.last_components.items() if k!="total")
    torch.testing.assert_close(tr.reward,expected,rtol=1e-6,atol=1e-6)
    force_goal(env,0);tr=env.step(zero)
    assert tr.terminated[0]
    assert env.last_components["terminal_goal"][0].abs().min()>9.99
    assert not env.bonus_views["first_paid"][0].any()
    assert not env.bonus_views["air_paid"][0].any()
    assert wp.to_torch(env.world.ssl_foundation_reset.reset_generation)[0]==generation[0]+1
    collide(env,0,0,117.);env.step(zero)
    assert env.last_components["first_touch"][0,0]>.999
    assert env.last_components["air_touch"][0,0]>.249


def test_idle_reset_clears_budgets_not_regular_decision(env):
    zero=torch.zeros(32,2,8,device=env.device)
    env.bonus_views["first_paid"][0]=1;env.bonus_views["air_paid"][0]=10
    env.step(zero)
    assert env.bonus_views["first_paid"][0].min()==1
    assert env.bonus_views["air_paid"][0].min()==10
    env.idle_ticks[0]=IDLE_TICKS;tr=env.step(zero)
    assert tr.truncated[0] and not tr.terminated[0]
    assert not env.bonus_views["air_paid"][0].any()
    assert env.last_components["inactivity"][0].max()<-.999


def test_exact_original_reward_and_physics_parity(env):
    base=AcquisitionEnv(32,"G:/dev/RLBot-Rival/bot/collision_meshes",device="cuda:0",seed=SEED,ssl_foundation_scenarios=training_bank(32))
    zero=torch.zeros(32,2,8,device=env.device)
    for _ in range(4):
        a=base.step(zero);b=env.step(zero)
        torch.testing.assert_close(a.observation,b.observation,rtol=0,atol=0)
        for k,v in base.last_components.items():
            if k!="total":torch.testing.assert_close(v,env.last_components[k],rtol=0,atol=0)
        torch.testing.assert_close(b.reward,a.reward+env.last_components["first_touch"]+env.last_components["air_touch"],rtol=0,atol=0)
        assert torch.equal(a.reset_mask,b.reset_mask)


def test_collector_both_rewards_reach_learner_and_disposable_ppo(env):
    model=new_model().cuda();collector=ContactCollector(env,model,seed=SEED)
    collector.config=replace(ppo_config(),rollout_horizon=8,epochs=1,minibatch_size=256)
    collide(env,0,int(env.focal[0]),117.);collide(env,1,0,117.)
    rollout=collector.collect();m=collector.last_metrics
    assert m["contact_bonuses"]["learner_first_awards"]>=2
    assert m["contact_bonuses"]["learner_air_awards"]>=2
    assert m["nexto_training_sample_count"]*3==m["trainable_agent_samples"]
    result=mixed_joint_ppo_update(model,fresh_entity_optimizer(model),rollout,collector.config,torch.Generator(device=env.device).manual_seed(SEED+2))
    assert result["optimizer_steps"]>0 and result["kl_rejections"]==0
