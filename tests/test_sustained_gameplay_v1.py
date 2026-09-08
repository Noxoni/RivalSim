from pathlib import Path
from dataclasses import replace

import pytest
import torch
import warp as wp

from rivalsim.sustained_gameplay_v1 import (
    GAMMA, PHYSICS_GAMMA, GAE_LAMBDA, CONTROL_RATE, IDLE_TICKS,
    SustainedEnv, fresh_model, ppo_config, decision_reward, reward_authority,
)
from rivalsim.sustained_gameplay_training_v1 import SustainedCollector
from rivalsim.direct_skills_kickoff_race_v1 import scenarios
from rivalsim.fresh_ground_30hz import WEIGHTS, potentials
from rivalsim.ssl_entity_mixed_training import mixed_sequence_data, mixed_joint_ppo_update
from rivalsim.ssl_entity_training import fresh_entity_optimizer, joint_sequence_loss
from rivalsim.rival2_ppo import compute_gae_gpu
from tests.test_fresh_ground_30hz import observation, force_goal


@pytest.fixture(scope="module", autouse=True)
def threads():
    torch.set_num_threads(4)


def test_half_lives_and_goal_scale():
    assert GAMMA ** 900 == pytest.approx(.5)
    assert (GAMMA * GAE_LAMBDA) ** 90 == pytest.approx(.5)
    assert PHYSICS_GAMMA ** 4 == pytest.approx(GAMMA)
    assert 0 < GAE_LAMBDA < 1
    assert CONTROL_RATE / 120 / (1-PHYSICS_GAMMA) < 1
    assert reward_authority()["inactivity_penalty"] == -1


@pytest.mark.parametrize("goal_tick", [-1, 0, 1, 2, 3])
def test_goal_shaping_and_timeout_bootstrap_potential(goal_tick):
    before, after = observation(), observation()+.003
    terminal = torch.full((4,), goal_tick)
    truncated = torch.full((4,), goal_tick < 0)
    reward, parts = decision_reward(before, after, terminal, torch.zeros(4, dtype=torch.long),
                                    torch.zeros(4, 2), truncated)
    phi0 = sum(WEIGHTS[k]*v for k,v in potentials(before).items())
    phi1 = sum(WEIGHTS[k]*v for k,v in potentials(after).items())
    if goal_tick >= 0:
        expected = -phi0 + torch.tensor([10., -10.])*PHYSICS_GAMMA**goal_tick
    else:
        expected = GAMMA*phi1-phi0-PHYSICS_GAMMA**3
    torch.testing.assert_close(reward, expected)
    assert not parts["sustained_control"].any()


def test_goal_credit_reaches_earlier_decisions_and_timeout_uses_value():
    n=91
    reward=torch.zeros(n,1,1); reward[-1]=10
    value=torch.zeros_like(reward); end=torch.zeros_like(reward,dtype=torch.bool)
    end[-1]=True
    advantage,_=compute_gae_gpu(reward,value,value,end,torch.zeros_like(end),gamma=GAMMA,gae_lambda=GAE_LAMBDA)
    assert float(advantage[0]) == pytest.approx(5, abs=2e-5)
    reward=torch.tensor([[[-1.]]]); nxt=torch.tensor([[[4.]]]); zero=torch.zeros_like(nxt)
    yes=torch.ones_like(nxt,dtype=torch.bool); no=~yes
    a,_=compute_gae_gpu(reward,zero,nxt,no,yes,gamma=GAMMA,gae_lambda=GAE_LAMBDA)
    assert float(a[0]) == pytest.approx(-1+GAMMA*4)
    terminal,_=compute_gae_gpu(reward,zero,nxt,yes,no,gamma=GAMMA,gae_lambda=GAE_LAMBDA)
    assert float(terminal[0]) == -1


def test_fresh_seed_reproducible_independent_and_stateful():
    a,b=fresh_model(),fresh_model()
    assert all(torch.equal(v,b.state_dict()[k]) for k,v in a.state_dict().items())
    obs=torch.randn(3,7,182); h=a.initial_hidden(3)
    reset=torch.zeros(3,7,dtype=torch.bool);reset[1,3]=True
    full=a(obs,h,reset_before=reset)
    logits=[];values=[]
    for t in range(7):
        l,v,h=a(obs[:,t],h,reset_before=reset[:,t]);logits.append(l);values.append(v)
    torch.testing.assert_close(full[0],torch.stack(logits,1),atol=2e-6,rtol=2e-5)
    torch.testing.assert_close(full[1],torch.stack(values,1),atol=2e-6,rtol=2e-5)
    torch.testing.assert_close(full[2],h,atol=2e-6,rtol=2e-5)
    # Same instantaneous state, different real history => different value.
    x=obs[:,0]
    zero=a.bootstrap_value(x,a.initial_hidden(3))
    memory=a.bootstrap_value(x,h)
    assert not torch.equal(zero,memory)
    a.zero_grad(set_to_none=True)
    a(obs,a.initial_hidden(3))[1].square().mean().backward()
    assert all(p.grad is None for n,p in a.named_parameters() if not n.startswith("critic."))
    assert all(p.grad is not None for p in a.critic.parameters())
    with pytest.raises(RuntimeError,match="requires history"):
        a.isolated_value(x)


def test_bootstrap_peek_does_not_double_consume_state():
    model=fresh_model().eval();x=torch.randn(2,3,182)
    with torch.no_grad():
        _,_,h=model(x[:,0],model.initial_hidden(2))
        saved=h.clone()
        peek=model.bootstrap_value(x[:,1],h)
        _,actual,advanced=model(x[:,1],h)
    assert torch.equal(saved,h)
    torch.testing.assert_close(peek,actual,atol=0,rtol=0)
    assert not torch.equal(advanced[1],h[1])


@pytest.fixture(scope="module")
def env():
    root=Path("G:/dev/RLBot-Rival/bot/collision_meshes")
    if not torch.cuda.is_available() or not root.exists():pytest.skip("Native CUDA and arena required")
    return SustainedEnv(32,root,device="cuda:0",seed=45,ssl_foundation_scenarios=scenarios(32))


def test_native_goal_all_four_ticks_precedence_and_fresh_scenario(env):
    zero=torch.zeros(32,2,8,device=env.device)
    generation=wp.to_torch(env.world.ssl_foundation_reset.reset_generation)
    old=generation.clone()
    env.idle_ticks[:4]=IDLE_TICKS
    def provider(tick):
        force_goal(env,tick,1 if tick%2==0 else -1)
        return zero
    tr=env.step_with_tick_actions(zero,provider)
    assert tr.terminated[:4].all() and not tr.truncated[:4].any()
    assert torch.equal(env.last_native["first_goal_tick"][:4],torch.arange(4,device=env.device))
    assert torch.equal(generation[:4],old[:4]+1)
    assert not env.last_components["inactivity"][:4].any()
    assert (env.last_toucher[:4]==-1).all()
    assert (env.bridge.views["rival2.episode_ticks"][:4]==0).all()


def test_legacy_timeouts_cannot_end_sustained_play(env):
    zero=torch.zeros(32,2,8,device=env.device)
    rows=slice(12,16)
    env.bridge.views["rival2.episode_ticks"][rows]=torch.tensor([1440,3600,14400,72000],device=env.device)
    env.idle_ticks[rows]=1800
    tr=env.step(zero)
    assert not tr.reset_mask[rows].any()
    assert not tr.truncated[rows].any()
    assert (env.last_native["episode_ticks"][rows]>=1440).all()
    env.idle_ticks[20]=IDLE_TICKS-4
    tr=env.step(zero)
    assert tr.truncated[20] and not tr.terminated[20]
    assert (env.last_components["inactivity"][20] == -PHYSICS_GAMMA**3).all()
    assert not torch.equal(tr.transition_observation[20],tr.observation[20])
    # Episode reset clears idle clock: no repeated timeout penalty.
    env.step(zero)
    assert not env.last_components["inactivity"][20].any()


def test_native_collector_two_memories_masks_and_one_focused_update(env):
    model=fresh_model().to(env.device)
    collector=SustainedCollector(env,model)
    collector.config=replace(ppo_config(),rollout_horizon=8,epochs=1,minibatch_size=256)
    rollout=collector.collect()
    assert collector.hidden.shape==(2,32,2,256)
    assert collector.last_metrics["nexto_training_sample_count"]*3==collector.last_metrics["trainable_agent_samples"]
    assert torch.equal(model.action_table[rollout.action_indices][rollout.train_mask],rollout.actions[rollout.train_mask])
    data=mixed_sequence_data(rollout,collector.config)
    assert data["initial_hidden"].shape==(2,64,256)
    optimizer=fresh_entity_optimizer(model)
    report=mixed_joint_ppo_update(model,optimizer,rollout,collector.config,
                                torch.Generator(device=env.device).manual_seed(3))
    assert report["optimizer_steps"]>0 and report["kl_rejections"]==0
    assert torch.isfinite(torch.tensor(report["completed_update_mean_kl"]))
