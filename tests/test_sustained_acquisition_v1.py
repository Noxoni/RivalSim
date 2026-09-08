from dataclasses import replace
import numpy as np
import pytest
import torch
import warp as wp

from rivalsim.sustained_acquisition_v1 import (
    AcquisitionEnv, AcquisitionCollector, Retirement, acquisition_starts,
    training_starts, base_scenarios, FAMILY, NAMES,
)
from rivalsim.fresh_ground_30hz import scenario_hash
from rivalsim.sustained_gameplay_v1 import fresh_model, ppo_config, IDLE_TICKS
from rivalsim.ssl_entity_training import fresh_entity_optimizer
from rivalsim.ssl_entity_mixed_training import mixed_joint_ppo_update
from tests.test_fresh_ground_30hz import force_goal


def test_deterministic_bank_valid_geometry_and_retirement_restores_original():
    a=training_starts(256); b=training_starts(256)
    assert scenario_hash(a)==scenario_hash(b)
    original=base_scenarios(256)
    assert scenario_hash(training_starts(256,retired=True))==scenario_hash(original)
    assert (a.family==FAMILY).sum()==128
    assert set(a.family)==set(range(len(NAMES)))
    retained=a.family!=FAMILY
    for key in a.state.__dataclass_fields__:
        np.testing.assert_array_equal(getattr(a.state,key)[retained],getattr(original.state,key)[retained])
    easy=acquisition_starts(256)
    rows=np.arange(256); s=easy.state
    distances=np.linalg.norm(s.car_pos[rows,easy.focal_side,:2]-s.ball_pos[:,:2],axis=1)
    assert distances[::2].min()>=350 and distances[::2].max()<=850
    assert distances[1::2].min()>=850 and distances[1::2].max()<=1800
    assert np.linalg.norm(s.car_pos[rows,1-easy.focal_side,:2]-s.ball_pos[:,:2],axis=1).min()>=3000
    assert (np.abs(s.car_pos[...,0])<3900).all() and (np.abs(s.car_pos[...,1])<4800).all()
    assert (s.on_ground==1).all() and (s.ball_pos[:,2]==np.float32(93.15)).all()
    assert not s.prev_throttle.any() and not s.prev_jump.any()
    assert set(easy.focal_side[::2])=={0,1} and set(easy.focal_side[1::2])=={0,1}
    s.validate()


def probe(easy=.96,varied=.91):
    return dict(easy=dict(success_fraction=easy),varied=dict(success_fraction=varied),model_unchanged=True,optimizer_steps=0)


def test_retirement_requires_two_successes_and_same_checkpoint_real_gameplay():
    r=Retirement()
    assert not r.observe(27,probe())
    with pytest.raises(ValueError):r.observe(27,probe())
    assert not r.observe(37,probe(.9)) and r.streak==0
    assert not r.observe(47,probe())
    assert r.observe(57,probe())
    assert not r.confirm_match(47,dict(touches_per_minute=20,matches_without_rival_touch=0))
    assert not r.confirm_match(57,dict(touches_per_minute=.38,matches_without_rival_touch=0))
    assert not r.confirm_match(57,dict(touches_per_minute=20,matches_without_rival_touch=1))
    assert r.confirm_match(57,dict(touches_per_minute=5,matches_without_rival_touch=0))
    assert r.retired and r.retired_at==57
    assert not Retirement(**vars(r)).observe(67,probe(0,0))
    assert r.retired


@pytest.fixture(scope="module")
def env():
    torch.set_num_threads(4)
    return AcquisitionEnv(32,"G:/dev/RLBot-Rival/bot/collision_meshes",device="cuda:0",seed=77,
                          ssl_foundation_scenarios=acquisition_starts(32))


def test_native_physical_contact_does_not_reset_or_award_bonus(env):
    # Test-only forward input, never present in training initialization/controller.
    actions=torch.zeros((32,2,8),device=env.device)
    side=env.focal.long().clone(); rows=torch.arange(32,device=env.device)
    actions[rows,side,0]=1
    contacts=0
    for _ in range(90):
        tr=env.step(actions)
        touched=env.last_native["touch_count"][rows,side]>0
        contacts+=int(touched.sum())
        assert not (tr.reset_mask & touched).any()
        assert not tr.reset_mask.any()
        assert "first_touch" not in env.last_components
    assert contacts>0


def test_retirement_only_changes_future_reset_sources(env):
    collector=AcquisitionCollector(env,fresh_model().cuda())
    collector.hidden.fill_(.25)
    before=env.world.state.snapshot()
    family=env.family.clone();hidden=collector.hidden.clone();obs=env.observation.clone()
    generation=wp.to_torch(env.world.ssl_foundation_reset.reset_generation).clone()
    env.retire_starts(training_starts(32,retired=True))
    after=env.world.state.snapshot()
    for key in before.__dataclass_fields__:
        np.testing.assert_array_equal(getattr(before,key),getattr(after,key))
    assert torch.equal(hidden,collector.hidden) and torch.equal(obs,env.observation)
    assert torch.equal(family,env.family)
    assert torch.equal(generation,wp.to_torch(env.world.ssl_foundation_reset.reset_generation))
    action=torch.zeros((32,2,8),device=env.device)
    force_goal(env,0)
    env.idle_ticks[1]=IDLE_TICKS
    tr=env.step(action)
    assert tr.terminated[0] and tr.truncated[1]
    assert (env.family[:2]<FAMILY).all() and (env.family[2:]==FAMILY).all()


def test_six_family_collector_and_disposable_ppo(env):
    model=fresh_model().cuda();collector=AcquisitionCollector(env,model)
    collector.config=replace(ppo_config(),rollout_horizon=8,epochs=1,minibatch_size=256)
    rollout=collector.collect()
    assert len(collector.last_metrics["by_start_family_and_opponent"])==12
    assert collector.last_metrics["nexto_training_sample_count"]*3==collector.last_metrics["trainable_agent_samples"]
    result=mixed_joint_ppo_update(model,fresh_entity_optimizer(model),rollout,collector.config,
                                torch.Generator(device=env.device).manual_seed(83))
    assert result["optimizer_steps"]>0 and result["kl_rejections"]==0
