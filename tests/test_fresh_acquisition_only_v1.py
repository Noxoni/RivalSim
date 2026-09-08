from dataclasses import replace
import numpy as np
import torch

from rivalsim.fresh_acquisition_only_v1 import training_bank, new_model, authority, SEED
from rivalsim.sustained_acquisition_v1 import AcquisitionEnv, AcquisitionCollector, FAMILY, acquisition_starts
from rivalsim.sustained_gameplay_v1 import ppo_config, IDLE_TICKS, reward_authority
from rivalsim.fresh_ground_30hz import scenario_hash
from rivalsim.ssl_entity_training import fresh_entity_optimizer
from rivalsim.ssl_entity_mixed_training import mixed_joint_ppo_update
from benchmarks.run_rival2_fresh_ground_30hz_v1 import tensor_hash
from tests.test_fresh_ground_30hz import force_goal


def test_only_acquisition_same_population_balanced_between_opponents():
    bank=training_bank(64);source=acquisition_starts(32)
    assert scenario_hash(bank)==scenario_hash(training_bank(64))
    assert (bank.family==FAMILY).all() and not bank.kickoff_indicator.any()
    # Uniquely map back through ball coordinates to assert whole-state exactness.
    for lane in (0,1):
        rows=np.arange(lane,64,2)
        found=[]
        for row in rows:
            match=np.flatnonzero((source.state.ball_pos==bank.state.ball_pos[row]).all(1))
            assert len(match)==1
            original=int(match[0]);found.append(original)
            for name in source.state.__dataclass_fields__:
                np.testing.assert_array_equal(getattr(bank.state,name)[row],getattr(source.state,name)[original])
        assert sorted(found)==list(range(32))


def test_fresh_random_initialization_and_unchanged_authorities():
    torch.set_num_threads(4)
    a=new_model();b=new_model()
    assert tensor_hash(a.state_dict())==tensor_hash(b.state_dict())
    assert not fresh_entity_optimizer(a).state
    assert authority()["parent"] is None and authority()["seed"]==SEED
    assert authority()["reward"]==reward_authority()
    assert authority()["ppo"]["epochs"]==2 and ppo_config().rollout_horizon==90


def test_actual_reset_bank_stays_acquisition_and_one_disposable_update():
    torch.set_num_threads(4)
    env=AcquisitionEnv(32,"G:/dev/RLBot-Rival/bot/collision_meshes",device="cuda:0",seed=SEED,
                       ssl_foundation_scenarios=training_bank(32))
    model=new_model().cuda();collector=AcquisitionCollector(env,model,seed=SEED)
    force_goal(env,0);env.idle_ticks[1]=IDLE_TICKS
    tr=env.step(torch.zeros((32,2,8),device=env.device))
    assert tr.terminated[0] and tr.truncated[1] and (env.family==FAMILY).all()
    collector.config=replace(ppo_config(),rollout_horizon=8,epochs=1,minibatch_size=256)
    before=tensor_hash(model.state_dict());rollout=collector.collect()
    assert collector.last_metrics["nexto_training_sample_count"]*3==collector.last_metrics["trainable_agent_samples"]
    assert all(v["world_seconds"]==0 for k,v in collector.last_metrics["by_start_family_and_opponent"].items() if not k.startswith("ball_acquisition_"))
    result=mixed_joint_ppo_update(model,fresh_entity_optimizer(model),rollout,collector.config,
                                torch.Generator(device=env.device).manual_seed(SEED+2))
    assert result["optimizer_steps"]>0 and result["kl_rejections"]==0
    assert tensor_hash(model.state_dict())!=before
