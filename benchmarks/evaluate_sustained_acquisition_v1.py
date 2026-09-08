"""Bounded, deterministic first-contact checks. Never an optimizer or reward gate."""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
import torch
from rivalsim.sustained_acquisition_v1 import AcquisitionEnv, acquisition_starts, specification, SEED
from rivalsim.sustained_gameplay_v1 import SustainedPolicy
from rivalsim.fresh_ground_30hz import scenario_hash,content_hash
from third_party.nexto.native_v5 import NextoNativeV5PolicyAdapter
from third_party.nexto.adapter import NextoStateTensors
from benchmarks.run_rival2_fresh_ground_30hz_v1 import sha,tensor_hash,utc,write_json


@torch.no_grad()
def evaluate(checkpoint,expected,output):
    checkpoint,output=Path(checkpoint),Path(output)
    assert sha(checkpoint)==expected
    if output.exists():
        prior=json.loads(output.read_text())
        assert prior["checkpoint_sha256"]==expected and prior["specification_sha256"]==content_hash(specification())
        assert prior["model_unchanged"] and prior["optimizer_steps"]==0
        return prior
    receipt=output.with_suffix(".started.json")
    if receipt.exists():raise RuntimeError("Interrupted probe needs audit before replay")
    write_json(receipt,dict(checkpoint=str(checkpoint),sha256=expected,utc=utc()))
    payload=torch.load(checkpoint,map_location="cpu",weights_only=False)
    model=SustainedPolicy().cuda().eval()
    model.load_state_dict(payload["model"],strict=True)
    before=tensor_hash(model.state_dict())
    n=specification()["probe"]["worlds"]
    batch=acquisition_starts(n,seed=SEED+100)
    env=AcquisitionEnv(n,"G:/dev/RLBot-Rival/bot/collision_meshes",device="cuda:0",seed=SEED+101,ssl_foundation_scenarios=batch)
    env._activate_torch_stream()
    device=env.device
    rows=torch.arange(n,device=device)
    side=torch.as_tensor(batch.focal_side.copy(),device=device).long()
    nexto=NextoNativeV5PolicyAdapter(n,device=device,sampling_mode="native_v5",seed=SEED+102)
    nexto.set_player_index(1-side)
    nexto_state=NextoStateTensors.from_bridge(env.bridge)
    nexto_hash=tensor_hash(nexto.actor.state_dict())
    alive=torch.ones(n,device=device,dtype=torch.bool)
    first=torch.full((n,),-1,device=device,dtype=torch.long)
    original_goal=torch.zeros_like(first)
    hidden=model.initial_hidden(n,device=device)
    for tick in range(240):
        obs=env.observation[rows,side]
        logits,value,hidden=model(obs,hidden)
        if not all(bool(torch.isfinite(x).all()) for x in (logits,value,hidden,obs)):
            raise RuntimeError("Nonfinite acquisition diagnostic")
        action=torch.zeros((n,2,8),device=device)
        action[rows,side]=model.action_table[logits.argmax(-1)]
        def provider(_):
            applied=action.clone()
            ball=nexto_state.ball_pos
            kickoff=(ball[:,0]==0)&(ball[:,1]==0)
            controls,_=nexto.tick_action(nexto_state,kickoff,active_mask=alive)
            applied[rows,1-side]=controls
            return applied
        tr=env.step_with_tick_actions(action,provider)
        native=env.last_native
        onset=native["first_touch_tick"][rows,side]
        first=torch.where(alive & (first<0) & (onset>=0),tick*4+onset+1,first)
        original_goal += (alive & tr.terminated).long()
        alive &= ~tr.reset_mask
        hidden.masked_fill_(~alive[None,:,None],0)
    first_host=first.cpu()
    result=dict(utc=utc(),accepted_updates=payload["accepted_updates"],checkpoint_sha256=expected,
        specification_sha256=content_hash(specification()),scenario_sha256=scenario_hash(batch),
        optimizer_steps=0,first_contact_ticks=first_host.tolist(),original_goal_terminations=int(original_goal.sum()),
        model_unchanged=before==tensor_hash(model.state_dict()),nexto_unchanged=nexto_hash==tensor_hash(nexto.actor.state_dict()),
        checkpoint_unchanged=sha(checkpoint)==expected)
    for parity,name,deadline in ((0,"easy",600),(1,"varied",960)):
        times=first_host[parity::2]
        success=(times>0)&(times<=deadline)
        result[name]=dict(worlds=len(times),success_count=int(success.sum()),success_fraction=float(success.double().mean()),
            deadline_seconds=deadline/120,median_seconds_if_success=float(times[success].double().median()/120) if success.any() else None,
            no_focal_contact_fraction=float((times<0).double().mean()))
    assert result["model_unchanged"] and result["nexto_unchanged"] and result["checkpoint_unchanged"]
    write_json(output,result)
    print("ACQUISITION_EVAL "+json.dumps({k:result[k] for k in ("accepted_updates","easy","varied")}),flush=True)
    return result


if __name__=="__main__":
    p=argparse.ArgumentParser();p.add_argument("--checkpoint",required=True);p.add_argument("--sha256",required=True);p.add_argument("--output",required=True)
    args=p.parse_args();torch.set_num_threads(4)
    evaluate(args.checkpoint,args.sha256.upper(),args.output)
