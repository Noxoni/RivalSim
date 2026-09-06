"""Parent-only training-state gradient calibration; no optimizer construction."""
from __future__ import annotations

import json
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))

import numpy as np
import torch

from benchmarks.analyze_direct_skills_learning_v1 import OUT,PARENT,IDENTITIES,save_json
from benchmarks.run_rival2_direct_skills_v1 import COLLISION,SEED,gpu_lease
from benchmarks.run_rival2_fresh_ground_30hz_v1 import sha,tensor_hash
from rivalsim.direct_skills_exploration_v1 import TrainingExplorationPolicy
from rivalsim.direct_skills_kickoff_race_v1 import KickoffRaceEnv,scenarios
from rivalsim.direct_skills_native_nexto import DirectSkillNativeNextoCollector
from rivalsim.direct_skills_log_barrier_v1 import uniform_log_barrier
from rivalsim.fresh_ground_30hz import ppo_config,scenario_hash
from rivalsim.ssl_entity_mixed_training import mixed_sequence_data


def main():
    assert not (OUT/'barrier_calibration.json').exists()
    assert sha(PARENT)==IDENTITIES[str(PARENT.relative_to(ROOT))]
    payload=torch.load(PARENT,map_location='cpu',weights_only=False)
    model=TrainingExplorationPolicy().cuda()
    model.load_state_dict(payload['model'],strict=True)
    before=tensor_hash(model.state_dict())
    bank=scenarios(1024)
    env=KickoffRaceEnv(1024,COLLISION,device='cuda:0',seed=SEED,ssl_foundation_scenarios=bank)
    collector=DirectSkillNativeNextoCollector(env,model,seed=SEED,nexto_sampling_mode='native_v5',nexto_seed=2026090691)
    rollout=collector.collect()
    data=mixed_sequence_data(rollout,ppo_config())
    # Collection selects eval mode; cuDNN GRU backward requires training mode.
    # This network has no dropout, so the policy distribution is unchanged.
    model.train()
    eligible=data['train_mask'].any(1).nonzero().flatten()
    generator=torch.Generator(device=env.device).manual_seed(SEED+83001)
    order=eligible[torch.randperm(len(eligible),device=env.device,generator=generator)]
    assert len(order)>=1024
    names,parameters=zip(*[(n,p) for n,p in model.named_parameters() if not n.startswith('critic.')])
    records=[]
    for start in range(0,1024,128):
        index=order[start:start+128]
        logits,_=model.forward_actor(data['observations'][index],data['initial_hidden'][:,index],reset_before=data['reset_before'][index])
        logp=logits.log_softmax(-1)
        mask=data['train_mask'][index]
        sampled=logp.gather(-1,data['action_indices'][index,:,None]).squeeze(-1)
        lr=sampled[mask]-data['old_log_probability'][index][mask]
        ratio=lr.exp()
        advantage=data['normalized_advantage'][index][mask]
        policy=-torch.minimum(ratio*advantage,ratio.clamp(.8,1.2)*advantage).mean()
        entropy=ppo_config().entropy_coefficient*(logp.exp()*logp).sum(-1)[mask].mean()
        barrier=uniform_log_barrier(logits)[mask].mean()
        vectors={}
        for key,loss in (('policy',policy),('existing_entropy',entropy),('unit_barrier',barrier)):
            grads=torch.autograd.grad(loss,parameters,retain_graph=True,allow_unused=True)
            vectors[key]=[torch.zeros_like(p) if g is None else g.detach() for p,g in zip(parameters,grads)]
        norms={key:float(torch.stack([x.square().sum() for x in values]).sum().sqrt()) for key,values in vectors.items()}
        assert all(np.isfinite(x) for x in norms.values()) and norms['policy']>0
        cosine=float(sum((a*b).sum() for a,b in zip(vectors['policy'],vectors['unit_barrier']))/(norms['policy']*norms['unit_barrier']))
        records.append(dict(sequence_start=start,sequence_count=128,trainable_decisions=int(mask.sum()),
            norms=norms,policy_barrier_cosine=cosine,unit_barrier=float(barrier.detach()),
            collection_logp_max_error=float(lr.detach().abs().max())))
        del vectors,logits,logp,policy,entropy,barrier,grads
    ratios=np.array([r['norms']['unit_barrier']/r['norms']['policy'] for r in records])
    candidates={str(beta):dict(median_gradient_ratio=float(np.median(beta*ratios)),
                              max_gradient_ratio=float(np.max(beta*ratios)),
                              eligible=bool(np.median(beta*ratios)<=.5 and np.max(beta*ratios)<=1))
                for beta in (.01,.03,.1)}
    eligible_beta=[float(beta) for beta,value in candidates.items() if value['eligible']]
    selected=max(eligible_beta) if eligible_beta else None
    assert tensor_hash(model.state_dict())==before
    assert sha(PARENT)==IDENTITIES[str(PARENT.relative_to(ROOT))]
    result=dict(version='RIVAL2_DIRECT_SKILLS_LOG_BARRIER_CALIBRATION_V1',parent_sha256=sha(PARENT),
        scenario_sha256=scenario_hash(bank),worlds=1024,horizon=90,sequence_sampling_seed=SEED+83001,
        model_unchanged=True,optimizer_steps=0,optimizer_constructed=False,
        records=records,candidates=candidates,selected_coefficient=selected,
        selection='Highest of0.01,0.03,0.1 with median added/policy gradient norm<=0.5 and maximum<=1.0 on eight128-sequence parent-only training microbatches. No test data or student outcomes.',
        caveat='Gradient norms are initialization diagnostics, not a guarantee about future optimization or gameplay.')
    save_json(OUT/'barrier_calibration.json',result)
    print(json.dumps(dict(candidates=candidates,selected=selected,model_unchanged=True,optimizer_steps=0)))


if __name__=='__main__':
    torch.set_num_threads(8)
    with gpu_lease():
        main()
