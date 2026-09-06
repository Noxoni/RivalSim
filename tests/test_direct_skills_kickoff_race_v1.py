from pathlib import Path

import numpy as np
import pytest
import torch
import warp as wp

from rivalsim.direct_skills_kickoff_race_v1 import (
    KickoffRaceEvents, KickoffRaceEnv, RaceState, observe_first_contact,
    scenarios, parent_scenarios, reward_authority,
)
from rivalsim.direct_skills_finishing_goal_v2 import FinishingGoalEvents, FinishingGoalEnv
from rivalsim.fresh_ground_30hz import scenario_hash, content_hash, decision_reward
from tests.test_direct_skills_v1 import inputs, observations
from tests.test_fresh_ground_30hz import force_goal


def test_physics_tick_order_tie_goal_and_reset():
    wp.init()
    state = RaceState(4, 'cpu')
    contacts = wp.array(np.array([1,0,0,1,1,1,1,0], np.int32), dtype=wp.int32, device='cpu')
    goal = wp.array(np.array([0,0,0,1], np.int32), dtype=wp.int32, device='cpu')
    def observe():
        wp.launch(observe_first_contact, dim=4, inputs=[contacts,goal,state.array], device='cpu')
    observe()
    assert state.winner.tolist() == [0,1,-2,-1]
    wp.to_torch(contacts).fill_(1)
    observe()
    assert state.winner.tolist() == [0,1,-2,-1]
    state.reset(torch.tensor([True,False,False,False]))
    wp.to_torch(contacts)[:2] = torch.tensor([0,1])
    observe()
    assert state.winner.tolist() == [1,1,-2,-1]


@pytest.mark.parametrize('winner', [0,1,-2])
def test_race_pays_only_once_and_only_winner(winner):
    race = RaceState(1,'cpu'); race.winner.fill_(winner)
    tracker = KickoffRaceEvents(1,'cpu',race)
    args = inputs(torch.full((1,2),4)); args[2].fill_(1)
    bonus,events,weighted,approach,_ = tracker.calculate(*args)
    expected = torch.tensor([[.5 if winner == 0 else 0., .5 if winner == 1 else 0.]])
    assert torch.equal(weighted[...,0], expected)
    assert torch.equal(bonus, weighted.sum(-1)+approach)
    assert torch.equal(bonus,expected)
    assert not tracker.calculate(*args)[2][...,0].any()


def test_later_loser_touch_is_unpaid_but_can_gain_control():
    race=RaceState(1,'cpu'); race.winner.fill_(1)
    tracker=KickoffRaceEvents(1,'cpu',race)
    args=inputs(torch.full((1,2),4)); args[2][:]=torch.tensor([[0,1]])
    tracker.calculate(*args)
    args[2][:]=torch.tensor([[1,0]])
    assert tracker.calculate(*args)[0][0,0] == 0
    args[2].zero_()
    for _ in range(6):
        tracker.calculate(*args)
    reward=tracker.calculate(*args)[0]
    assert reward[0,0] == 1.5
    assert reward[0,0] > .5  # Second-touch control is still more than first contact.


def test_terminal_goal_suppresses_race_bonus_and_clears_at_reset():
    race=RaceState(1,'cpu'); race.winner.fill_(0)
    tracker=KickoffRaceEvents(1,'cpu',race)
    args=inputs(torch.full((1,2),4)); args[4].fill_(True); args[6].fill_(0)
    bonus,events,*_=tracker.calculate(*args)
    assert not bonus.any() and not events.any()
    tracker.reset(torch.tensor([True]))
    assert race.winner.item() == -1 and not race.paid.item()


def test_other_reward_roles_and_state_exact():
    n=8
    roles=torch.tensor([[r,r] for r in range(4)]*2)
    old=FinishingGoalEvents(n,'cpu'); race=RaceState(n,'cpu'); new=KickoffRaceEvents(n,'cpu',race)
    generator=torch.Generator().manual_seed(872)
    for _ in range(24):
        args=(observations(n),observations(n),torch.randint(0,2,(n,2),generator=generator),
              roles,torch.rand(n,generator=generator)<.1,torch.rand(n,generator=generator)<.2,
              torch.randint(0,2,(n,),generator=generator))
        race.winner.copy_(torch.randint(-2,2,(n,),generator=generator))
        a,b=old.calculate(*args),new.calculate(*args)
        for x,y in zip(a,b,strict=True):
            assert torch.equal(x,y)
        for key in ('paid','own_last','control_ticks','clear_ticks','clear_pending','gain_y'):
            assert torch.equal(getattr(old,key),getattr(new,key))
        reset=args[4]|args[5]; old.reset(reset);new.reset(reset)


def test_start_bank_only_changes_half_kickoff_focal_velocities():
    old,new=parent_scenarios(1024),scenarios(1024)
    assert scenario_hash(new)==scenario_hash(scenarios(1024))
    changed=(old.state.car_vel!=new.state.car_vel).any(-1)
    assert int(changed.sum())==int((old.family==4).sum())//2
    rows,cars=np.nonzero(changed)
    assert (old.family[rows]==4).all() and np.array_equal(cars,old.focal_side[rows])
    for key in old.state.__dataclass_fields__:
        if key != 'car_vel':
            assert np.array_equal(getattr(old.state,key),getattr(new.state,key)),key
    for key in ('family','focal_side','kickoff_indicator','kickoff_layout','wall_aerial_variant'):
        assert np.array_equal(getattr(old,key),getattr(new,key))
    speed=np.linalg.norm(new.state.car_vel[rows,cars],axis=-1)
    assert ((speed>=250)&(speed<=750)).all()
    assert int((new.family==2).sum())==int(1024*.2)
    # Initial Nexto nearest-car admission cannot change when positions are exact.
    old_distance=np.linalg.norm(old.state.car_pos-old.state.ball_pos[:,None],axis=-1)
    new_distance=np.linalg.norm(new.state.car_pos-new.state.ball_pos[:,None],axis=-1)
    assert np.array_equal(old_distance,new_distance)


@pytest.fixture(scope='module')
def envs():
    root=Path('G:/dev/RLBot-Rival/bot/collision_meshes')
    assert torch.cuda.is_available() and root.exists()
    torch.set_num_threads(4)
    from benchmarks.direct_skills_eval_stream import owned_match_stream
    with owned_match_stream():
        yield (FinishingGoalEnv(32,root,device='cuda:0',seed=911,ssl_foundation_scenarios=parent_scenarios(32)),
               KickoffRaceEnv(32,root,device='cuda:0',seed=911,ssl_foundation_scenarios=parent_scenarios(32)))


def test_native_hook_does_not_change_physics_actions_or_other_rewards(envs):
    old,new=envs
    generator=torch.Generator().manual_seed(319)
    for _ in range(24):
        action=(torch.rand((32,2,8),generator=generator)*2-1).cuda()
        action[...,5:]=(action[...,5:]>0).float()
        assert torch.equal(old.observation,new.observation)
        a,b=old.step(action),new.step(action)
        for key in ('observation','transition_observation','emitted_action','terminated','truncated','reset_mask'):
            assert torch.equal(getattr(a,key),getattr(b,key)),key
        non=new.last_skill['roles']!=4
        assert torch.equal(a.reward[non],b.reward[non])
        for key in old.last_native:
            assert torch.equal(old.last_native[key],new.last_native[key]),key


def test_native_goal_on_each_tick_clears_race_and_preserves_reward(envs):
    _,env=envs
    env.family[:4]=4
    env.race.winner[:4]=0
    actions=torch.zeros((32,2,8),device=env.device)
    before=env.observation.clone()
    def provider(tick):
        force_goal(env,tick,1 if tick%2==0 else -1)
        return actions
    tr=env.step_with_tick_actions(actions,provider)
    assert tr.terminated[:4].all() and not tr.truncated[:4].any()
    assert torch.equal(env.last_native['first_goal_tick'][:4],torch.arange(4,device=env.device))
    _,parts=decision_reward(before,tr.transition_observation,env.last_native['first_goal_tick'],env.last_native['scoring_team'])
    assert torch.equal(tr.reward[:4],parts['terminal_goal'][:4])
    assert not env.last_skill['events'][:4].any()
    assert (env.race.winner[:4]==-1).all() and not env.race.paid[:4].any()
    assert env.contract_hashes['reward']==content_hash(reward_authority())
