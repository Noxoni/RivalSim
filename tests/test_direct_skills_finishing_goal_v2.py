from pathlib import Path

import pytest
import torch

from rivalsim.direct_skills_finishing_goal_v2 import FinishingGoalEvents,FinishingGoalEnv,reward_authority
from rivalsim.direct_skills_v1 import SkillEvents,DirectSkillsEnv
from rivalsim.direct_skills_shooting_curriculum_v1 import scenarios
from rivalsim.rival2_contracts import OBS_FIELD_NAMES
from rivalsim.fresh_ground_30hz import content_hash,decision_reward
from tests.test_direct_skills_v1 import observations,inputs
from tests.test_fresh_ground_30hz import force_goal


def shot_args():
    args = inputs(torch.full((1,2),2))
    for obs in args[:2]:
        obs[...,OBS_FIELD_NAMES.index('ball.position.y')]=2500/5120
        obs[...,OBS_FIELD_NAMES.index('ball.linear_velocity.y')]=1000/6000
    return args


def test_raw_projection_contact_retained_but_unpaid():
    old,new = SkillEvents(1,'cpu'),FinishingGoalEvents(1,'cpu')
    args=shot_args()
    prior=old.calculate(*args); current=new.calculate(*args)
    assert prior[0][0,0]==1.5
    assert current[1][0,0,0] and current[1][0,0,3]
    assert not current[0].any() and not current[2].any() and not current[3].any()
    assert not current[4].any()
    assert torch.equal(old.paid,new.paid)


def test_saved_projected_shot_times_out_as_failure_only_once():
    tracker=FinishingGoalEvents(1,'cpu');args=shot_args()
    tracker.calculate(*args)
    args[2].zero_();args[5].fill_(True)
    for expected in (-1.,0.):
        bonus,events,weighted,approach,success=tracker.calculate(*args)
        assert (bonus==expected).all()
        assert torch.equal(bonus,weighted.sum(-1)+approach)
        assert not success.any()
    tracker.reset(torch.tensor([True]))
    assert (tracker.calculate(*args)[0]==-1).all()


@pytest.mark.parametrize('scorer',[0,1])
def test_goal_and_concede_never_also_timeout_failure(scorer):
    tracker=FinishingGoalEvents(1,'cpu');args=shot_args()
    tracker.calculate(*args)
    args[4].fill_(True);args[5].fill_(True);args[6].fill_(scorer)
    bonus,events,weighted,approach,success=tracker.calculate(*args)
    assert not bonus.any() and not events.any() and not weighted.any() and not approach.any()
    assert success.tolist()==[[scorer==0,scorer==1]]


def test_non_finishing_outputs_and_shared_event_state_exact():
    n=10
    roles=torch.tensor([[i,j] for i in range(5) for j in range(2)])
    old,new=SkillEvents(n,'cpu'),FinishingGoalEvents(n,'cpu')
    non=roles!=2
    generator=torch.Generator().manual_seed(991)
    for tick in range(40):
        before=observations(n);after=before.clone()
        after[...,OBS_FIELD_NAMES.index('ball.position.y')]=2500/5120
        after[...,OBS_FIELD_NAMES.index('ball.linear_velocity.y')]=1000/6000
        contacts=torch.randint(0,3,(n,2),generator=generator)
        terminated=torch.rand(n,generator=generator)<.15
        truncated=torch.rand(n,generator=generator)<.2
        team=torch.randint(0,2,(n,),generator=generator)
        a=old.calculate(before,after,contacts,roles,terminated,truncated,team)
        b=new.calculate(before,after,contacts,roles,terminated,truncated,team)
        for x,y in zip(a,b,strict=True):
            assert torch.equal(x[non],y[non])
        for name in ('paid','own_last','control_ticks','clear_ticks','clear_pending','gain_y'):
            assert torch.equal(getattr(old,name)[non],getattr(new,name)[non])
        torch.testing.assert_close(b[0],b[2].sum(-1)+b[3],rtol=0,atol=0)
        old.reset(terminated|truncated);new.reset(terminated|truncated)


def test_approach_retired_only_in_finishing():
    args=inputs(torch.tensor([[2,1]]));args[2].zero_()
    args[0][...,OBS_FIELD_NAMES.index('self.linear_velocity.y')]=1000/2300
    old=SkillEvents(1,'cpu').calculate(*args)
    new=FinishingGoalEvents(1,'cpu').calculate(*args)
    assert old[3][0,0]>0 and new[3][0,0]==0
    assert old[3][0,1]==new[3][0,1]>0


@pytest.fixture(scope='module')
def env():
    root=Path('G:/dev/RLBot-Rival/bot/collision_meshes')
    if not torch.cuda.is_available() or not root.exists():
        pytest.skip('requires native CUDA and arena')
    torch.set_num_threads(4)
    return FinishingGoalEnv(32,root,device='cuda:0',seed=82,ssl_foundation_scenarios=scenarios(32))


def test_native_goal_at_every_hold_tick_and_reset(env):
    actions=torch.zeros((32,2,8),device=env.device)
    env.family[:4]=2
    before=env.observation.clone()
    def provider(tick):
        force_goal(env,tick,1 if tick%2==0 else -1)
        return actions
    tr=env.step_with_tick_actions(actions,provider)
    assert tr.terminated[:4].all() and not tr.truncated[:4].any()
    assert torch.equal(env.last_native['first_goal_tick'][:4],torch.arange(4,device=env.device))
    _,parts=decision_reward(before,tr.transition_observation,env.last_native['first_goal_tick'],env.last_native['scoring_team'])
    torch.testing.assert_close(tr.reward[:4],parts['terminal_goal'][:4],rtol=0,atol=0)
    assert not env.last_skill['events'][:4].any()
    assert not env.events.paid[:4].any()
    assert (env.bridge.views['rival2.episode_ticks'][:4]==0).all()
    assert env.contract_hashes['reward']==content_hash(reward_authority())


def test_native_projected_shot_does_not_excuse_timeout_bootstrap(env):
    env.family[6]=2;env.focal[6]=0
    env.events.paid[6,0,0]=True;env.events.paid[6,0,3]=True
    env.bridge.views['rival2.episode_ticks'][6]=1436
    tr=env.step(torch.zeros((32,2,8),device=env.device))
    assert tr.truncated[6] and not tr.terminated[6]
    assert tr.reward[6,0]==-1 and env.last_skill['events'][6,0,6]
    assert not env.last_skill['success'][6,0]
    assert not torch.equal(tr.transition_observation[6],tr.observation[6])
    assert not env.events.paid[6].any()


def test_native_actions_observations_and_non_finishing_transitions_exact():
    root=Path('G:/dev/RLBot-Rival/bot/collision_meshes')
    old=DirectSkillsEnv(32,root,device='cuda:0',seed=193,ssl_foundation_scenarios=scenarios(32))
    new=FinishingGoalEnv(32,root,device='cuda:0',seed=193,ssl_foundation_scenarios=scenarios(32))
    generator=torch.Generator().manual_seed(431)
    for tick in range(16):
        action=torch.rand((32,2,8),generator=generator).cuda()*2-1
        action[...,5:]=(action[...,5:]>0).float()
        assert torch.equal(old.observation,new.observation)
        a,b=old.step(action),new.step(action)
        for name in ('observation','transition_observation','emitted_action','terminated','truncated','reset_mask'):
            assert torch.equal(getattr(a,name),getattr(b,name)),name
        non=old.last_skill['roles']!=2
        assert torch.equal(a.reward[non],b.reward[non])
        for name in old.last_native:
            assert torch.equal(old.last_native[name],new.last_native[name]),name
        for name in old.last_skill:
            assert torch.equal(old.last_skill[name][non],new.last_skill[name][non]),name
