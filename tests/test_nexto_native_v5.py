import copy
import hashlib
import json
from pathlib import Path
import types

import numpy as np
import pytest
import torch

import third_party.nexto.native_v5 as native

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/"results/rival2/nexto_native_controller_v1"


def test_oracle_uses_bound_installed_methods_and_actual_literal():
    meta=json.loads((OUT/"oracle.json").read_text())
    assert all(meta["module_code_and_literal_checks"].values())
    assert hashlib.sha256((OUT/"oracle.npz").read_bytes()).hexdigest().upper()==meta["oracle_sha256"]
    assert hashlib.sha256((ROOT/"benchmarks/build_native_nexto_controller_oracle.py").read_bytes()).hexdigest().upper()==meta["generator_source_sha256"]
    data=np.load(OUT/"oracle.npz")
    np.testing.assert_array_equal(native.build_native_kickoff_sequence("cpu").numpy(),data["kickoff_table"])


def replay_oracle(device="cpu", resume_at=None):
    data=np.load(OUT/"oracle.npz")
    n=data["active"].shape[1]
    obj=native.NativeNextoController(n,device)
    count=0
    for step in range(len(data["active"])):
        if step==resume_at:
            saved=obj.state_dict()
            obj=native.NativeNextoController(n,device)
            obj.load_state_dict(saved)
        to=lambda key:torch.as_tensor(data[key][step],device=device)
        obj.set_player_index(to("side").long())
        obj.activate(to("activate"))
        obj.notify_kickoff(to("notify"))
        calls=[]
        def predict(compute,kickoff):
            np.testing.assert_array_equal(compute.cpu().numpy(),data["computed"][step])
            np.testing.assert_array_equal(obj.pending_action[compute].cpu().numpy(),data["previous"][step][data["computed"][step]])
            np.testing.assert_array_equal(torch.where(kickoff[compute],.5,1.).cpu().numpy(),data["beta"][step][data["computed"][step]])
            calls.append(True)
            return to("candidate_actions"),None
        state=types.SimpleNamespace(car_pos=to("car_pos"),ball_pos=to("ball_pos"))
        control,_=obj.step(state,to("kickoff"),predict,to("active"))
        assert bool(calls)==bool(data["computed"][step].any())
        for value,key in ((control,"emitted"),(obj.pending_action,"pending"),
                          (obj.ticks,"ticks"),(obj.update_action,"updates"),(obj.kickoff_index,"kickoff_index")):
            np.testing.assert_array_equal(value.cpu().numpy(),data[key][step],err_msg=f"step{step} {key}")
        count+=n
    return count


@pytest.mark.parametrize("resume_at",[None,17,87,255])
def test_batched_controller_matches_native_with_inactive_worlds_and_resume(resume_at):
    assert replay_oracle(resume_at=resume_at)==4096


def test_invalid_checkpoint_rejected_before_mutation():
    obj=native.NativeNextoController(2,"cpu")
    before=obj.state_dict()
    for name,value in (("ticks",torch.tensor([0,99])),("pending_action",torch.full((2,8),float("nan")))):
        corrupt=copy.deepcopy(before);corrupt["tensors"][name]=value
        with pytest.raises(ValueError):obj.load_state_dict(corrupt)
        for key,tensor in before["tensors"].items():torch.testing.assert_close(getattr(obj,key),tensor)
    corrupt=copy.deepcopy(before);corrupt["version"]="legacy"
    with pytest.raises(ValueError):obj.load_state_dict(corrupt)


def test_kickoff_notification_does_not_reset_neural_clock_or_action():
    obj=native.NativeNextoController(2,"cpu")
    obj.ticks.copy_(torch.tensor([3,5]));obj.pending_action.fill_(.5);obj.emitted_control.fill_(-.5)
    obj.notify_kickoff(torch.tensor([True,False]))
    torch.testing.assert_close(obj.ticks,torch.tensor([3,5]))
    assert torch.all(obj.pending_action==.5) and torch.all(obj.emitted_control==-.5)


def fake_adapter(monkeypatch,mode):
    data=np.load(OUT/"oracle.npz")
    obj=object.__new__(native.NextoNativeV5PolicyAdapter)
    obj.num_worlds=24;obj.device=torch.device("cpu");obj.sampling_mode=mode
    obj.controller=native.NativeNextoController(24,"cpu")
    obj.previous_action=obj.controller.pending_action;obj.player_index=obj.controller.player_index
    obj.constants=None;obj.action_table=torch.from_numpy(data["lookup"])
    obj.generator=torch.Generator().manual_seed(7331)
    obj.inference_calls=0;obj.observation_builds=0
    logits=torch.linspace(-2,2,90).repeat(24,1)
    def build(*args,**kwargs):
        return native.NextoObservation(torch.zeros(24,1,32),torch.zeros(24,37,24),torch.zeros(24,37))
    monkeypatch.setattr(native,"build_nexto_observation",build)
    obj.logits=lambda observation:logits[:len(observation.q)]
    return obj


@pytest.mark.parametrize("mode",native.SAMPLING_MODES)
def test_sampling_and_rng_checkpoint_reproduce_selected_subset(monkeypatch,mode):
    obj=fake_adapter(monkeypatch,mode)
    active=torch.arange(24)%3!=0; kickoff=torch.arange(24)%2==0
    saved=obj.checkpoint_state()
    a,i=obj._predict_native(None,active,kickoff)
    after=obj.generator.get_state().clone()
    obj.load_checkpoint_state(saved)
    b,j=obj._predict_native(None,active,kickoff)
    torch.testing.assert_close(a,b,rtol=0,atol=0);torch.testing.assert_close(i,j)
    torch.testing.assert_close(after,obj.generator.get_state())
    assert torch.all(i[~active]==-1)
    assert torch.all(i[active & ~kickoff]==89)
    if mode=="deterministic_argmax":
        assert torch.all(i[active]==89)
        torch.testing.assert_close(after,saved["rng_state"])
    else:
        expected=torch.multinomial(torch.linspace(-2,2,90).softmax(-1).repeat(int((active&kickoff).sum()),1),1,
                                   generator=torch.Generator().manual_seed(7331)).flatten()
        torch.testing.assert_close(i[active&kickoff],expected)
        assert torch.any(expected!=89)


def test_nonfinite_nexto_outputs_rejected(monkeypatch):
    obj=fake_adapter(monkeypatch,"native_v5")
    obj.logits=lambda observation:torch.full((24,90),float("nan"))
    with pytest.raises(RuntimeError,match="nonfinite"):
        obj._predict_native(None,torch.ones(24,dtype=torch.bool),torch.zeros(24,dtype=torch.bool))


def test_collector_uses_native_state_and_explicit_fresh_episode_resume(monkeypatch):
    import rivalsim.direct_skills_native_nexto as module
    adapter = fake_adapter(monkeypatch, "native_v5")
    received = []
    def factory(n, **kwargs):
        received.append((n, kwargs))
        return adapter
    monkeypatch.setattr(module, "NextoNativeV5PolicyAdapter", factory)
    monkeypatch.setattr(module.NextoStateTensors, "from_bridge", lambda bridge: None)
    env = types.SimpleNamespace(num_envs=24, device=torch.device("cpu"), bridge=None,
        focal=torch.arange(24) % 2, learner=torch.ones(24, 2, dtype=torch.bool),
        opponent_family=torch.zeros(24, dtype=torch.long))
    model = types.SimpleNamespace(to=lambda device: model,
        initial_hidden=lambda n: torch.zeros(1, n, 2))
    collector = module.DirectSkillNativeNextoCollector(env, model, seed=1,
        nexto_sampling_mode="native_v5", nexto_seed=2)
    assert received == [(24, dict(device=env.device, sampling_mode="native_v5", seed=2))]
    assert int(collector.training_mask().sum()) == 36
    assert torch.all(collector.training_mask()[~collector.is_nexto])
    assert torch.all(collector.training_mask()[collector.is_nexto].sum(1) == 1)
    adapter.controller.pending_action.fill_(.5)
    adapter.controller.emitted_control.fill_(-.5)
    adapter.controller.ticks.fill_(3)
    saved = collector.opponent_checkpoint_state()
    assert "native_nexto" in saved and "nexto_cache" not in saved
    wrong = dict(saved, version="legacy")
    with pytest.raises(ValueError):collector.restore_opponent_for_fresh_episodes(wrong)
    assert torch.all(adapter.controller.pending_action == .5)
    env.focal = 1 - env.focal
    collector.restore_opponent_for_fresh_episodes(saved)
    assert torch.equal(collector.side, env.focal)
    assert torch.equal(adapter.player_index, 1 - env.focal)
    assert not bool(adapter.controller.pending_action.any())
    assert not bool(adapter.controller.emitted_control.any())
    assert torch.all(adapter.controller.ticks == 8)
    assert torch.all(adapter.controller.update_action)
    assert torch.equal(adapter.generator.get_state(), saved["native_nexto"]["rng_state"])
