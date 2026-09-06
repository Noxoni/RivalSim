"""CPU-only entry/update audit and matched full-evaluation comparison."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
import torch
from benchmarks.audit_direct_skills_native_nexto_entry import equal, tensor_hash, sha, canonical
from benchmarks.report_direct_skills_native_nexto_v1 import training_prefix, compare_records, save_once
from benchmarks.report_rival2_ssl_entity_match_followup import reduce

OUT=ROOT/'results/rival2/direct_skills_kickoff_race_v1'
CKPTS=ROOT/'checkpoints/rival2/direct_skills_kickoff_race_v1'
RUN=Path('G:/dev/RivalSim-runs/direct-skills-kickoff-race-v1')
RNG=('policy_generator_state','shuffle_generator_state','torch_cpu_rng_state','torch_cuda_rng_state')


def audit(offset):
    spec=json.loads((OUT/'training_authority.json').read_text())
    if offset not in (1,*spec['evaluation_boundaries']):
        raise ValueError('Not a preserved audit boundary')
    parent_path=ROOT/spec['parent']['path']
    entry_path=CKPTS/'entry_000000.pt'
    target_path=CKPTS/('first_000001.pt' if offset==1 else f'child_{offset:06d}.pt')
    parent,entry,target=[torch.load(p,map_location='cpu',weights_only=False) for p in (parent_path,entry_path,target_path)]
    prefix,rows=training_prefix(RUN/'training_curve.jsonl',offset)
    for row in rows:
        assert row['native_nexto_authority_sha256']==canonical(spec)
        assert row['accepted_updates']==650+row['branch_updates']
        assert row['training']['trainable_agent_samples']==4423680
        assert row['training']['nexto_training_sample_count']==1474560
        assert row['ppo']['kl_rejections']==0
    steps=sum(r['ppo']['optimizer_steps'] for r in rows)
    samples=sum(r['training']['trainable_agent_samples'] for r in rows)
    ticks=sum(r['training']['physical_physics_ticks'] for r in rows)
    checks=dict(
        parent_file_exact=sha(parent_path)==spec['parent']['sha256'],
        entry_model_exact=equal(entry['model'],parent['model']),
        entry_optimizer_exact=equal(entry['optimizer'],parent['optimizer']),
        entry_four_rng_exact=all(equal(entry[k],parent[k]) for k in RNG),
        entry_counters_exact=all(entry[k]==parent[k] for k in
            ('accepted_updates','direct_skill_samples','direct_skill_physics_ticks','cumulative_optimizer_steps')),
        new_format=entry['format']==target['format']==spec['version']+'_CHECKPOINT',
        both_authority_bound=all(p['native_nexto_authority_sha256']==canonical(spec) for p in (entry,target)),
        both_parent_bound=all(p['native_nexto_parent_sha256']==spec['parent']['sha256'] for p in (entry,target)),
        target_offset=target['native_nexto_branch_updates']==offset and target['accepted_updates']==650+offset,
        model_changed=not equal(entry['model'],target['model']),
        finite_model=all(bool(torch.isfinite(t).all()) for t in target['model'].values()),
        finite_adam=all(bool(torch.isfinite(t).all()) for s in target['optimizer']['state'].values() for t in s.values()),
        adam_steps=target['cumulative_optimizer_steps']==parent['cumulative_optimizer_steps']+steps and steps>0,
        adam_steps_uniform={int(s['step']) for s in target['optimizer']['state'].values()}=={target['cumulative_optimizer_steps']},
        sample_count=target['direct_skill_samples']==parent['direct_skill_samples']+samples,
        physics_count=target['direct_skill_physics_ticks']==parent['direct_skill_physics_ticks']+ticks,
        unchanged_ppo=target['ppo_config_sha256']==parent['ppo_config_sha256'],
        unchanged_architecture=target['policy_config_sha256']==parent['policy_config_sha256'],
        changed_reward_exact=target['reward_authority']==entry['reward_authority']==spec['reward'],
        reward_parent_provenance=canonical(parent['reward_authority'])==spec['reward']['parent_sha256'],
        reward_runtime_exact=target['runtime_contract_hashes']['reward']==canonical(spec['reward']),
        corpus_bound=target['effective_training_scenario_sha256']==json.loads((OUT/'training_package.json').read_text())['scenario_sha256'],
        rng_saved=all(k in target for k in RNG),
        native_opponent=target['opponent_state']['native_nexto']['version']==spec['opponents']['controller'],
        native_mode=target['opponent_state']['native_nexto']['sampling_mode']=='native_v5',
        exploration_exact=target['training_distribution']==dict(temperature=2.,on_policy_sampling_and_likelihood=True))
    assert all(checks.values()),checks
    result=dict(branch_updates=offset,checks=checks,optimizer_steps_in_audit=0,audit_device='cpu',
        authority_sha256=canonical(spec),parent=spec['parent'],
        entry=dict(path=entry_path.relative_to(ROOT).as_posix(),sha256=sha(entry_path)),
        checkpoint=dict(path=target_path.relative_to(ROOT).as_posix(),sha256=sha(target_path)),
        model_sha256=tensor_hash(target['model']),
        training=dict(updates=offset,optimizer_steps=steps,learner_decisions=samples,physics_ticks=ticks,
            max_completed_update_mean_kl=max(r['ppo']['completed_update_mean_kl'] for r in rows),
            max_sample_kl_telemetry=max(r['ppo']['completed_update_sample_kl_max'] for r in rows),
            kl_rejections=0,first_row=rows[0],last_row=rows[-1]))
    if offset in spec['evaluation_boundaries']:
        baseline=ROOT/'results/rival2/direct_skills_native_nexto_v1/finishing650.json'
        match=OUT/f'child_{offset:06d}.json'
        before,after=[json.loads(p.read_text()) for p in (baseline,match)]
        assert after['context']==dict(native_nexto_authority_sha256=canonical(spec),branch_updates=offset)
        assert after['checkpoint']==result['checkpoint']
        reduction=reduce(match)
        assert all(reduction['integrity'].values())
        assert reduction==json.loads(match.with_suffix('.integrity.json').read_text())
        result.update(comparison=compare_records(before,after),contacts_before=reduce(baseline)['contacts'],
            contacts_after=reduction['contacts'],evaluation_hashes={p.name:sha(p) for p in (baseline,match,match.with_suffix('.integrity.json'))},
            interpretation='Completed corrected-controller development comparison, not SSL proof. Kickoff first contact is not kickoff possession. Repeat contact identity is not possession duration. No-touch resets are disabled by protocol.')
    else:
        result['interpretation']='Actual accepted learning and exact initialization, not gameplay improvement. First evaluation remains+5.'
    save_once(OUT/f'through_{offset:06d}.jsonl',prefix)
    save_once(OUT/f'progress_{offset:06d}.json',(json.dumps(result,indent=2,sort_keys=True,allow_nan=False)+'\n').encode())
    print(json.dumps(dict(offset=offset,checks=checks,checkpoint=result['checkpoint'],comparison=result.get('comparison',{}))))
    return result


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('offset',type=int)
    torch.set_num_threads(2)
    audit(p.parse_args().offset)
