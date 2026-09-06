"""CPU-only arm integrity and equal-offset gameplay comparison; no new matches."""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
import torch

from benchmarks import run_direct_skills_exploration_ablation_v1 as run
from benchmarks.audit_direct_skills_native_nexto_entry import equal, tensor_hash, sha, canonical
from benchmarks.report_direct_skills_native_nexto_v1 import training_prefix, compare_records, save_once
from benchmarks.report_direct_skills_exploration_continuation_v1 import validate_rows
from benchmarks.report_rival2_ssl_entity_match_followup import reduce

RNG = ('policy_generator_state','shuffle_generator_state','torch_cpu_rng_state','torch_cuda_rng_state')


def objective_checks(payload,rows,spec):
    objective = spec['exploration_objective']
    beta = objective['coefficient']
    keys = ('exploration_barrier','exploration_coefficient','weighted_exploration_barrier',
            'entropy','completed_update_mean_kl','completed_update_sample_kl_max')
    checks = dict(objective_bound=payload['native_nexto_package']['exploration_objective'] == objective,
        coefficient_frozen=beta == run.ARMS[spec['arm']],
        finite_telemetry=all(all(k in r['ppo'] and math.isfinite(r['ppo'][k]) for k in keys) for r in rows),
        coefficient_correct=all(abs(r['ppo'].get('exploration_coefficient',float('inf'))-beta)<1e-8 for r in rows),
        nonnegative_barrier=all(r['ppo'].get('exploration_barrier',-1)>=-1e-6 for r in rows),
        weighted_loss_correct=all(abs(r['ppo'].get('weighted_exploration_barrier',float('inf'))-
            beta*r['ppo'].get('exploration_barrier',0))<1e-5 for r in rows),
        zero_branch_exact=beta != 0 or all(r['ppo'].get('weighted_exploration_barrier') == 0 for r in rows))
    if not all(checks.values()):
        raise ValueError(checks)
    return checks


def encoded(obj):
    return (json.dumps(obj,indent=2,sort_keys=True,allow_nan=False)+'\n').encode()


def audit(arm,offset):
    run.arm_name(arm)
    out,ckpts,external = run.OUT/arm,run.CKPTS/arm,run.EXTERNAL/arm
    spec = json.loads((out/'training_authority.json').read_text())
    assert spec == run.authority(arm)
    if offset not in (1,*spec['evaluation_boundaries']):
        raise ValueError('Not a preserved boundary')
    parent_path = ROOT/spec['parent']['path']
    entry_path = ckpts/'entry_000000.pt'
    target_path = ckpts/('first_000001.pt' if offset == 1 else f'child_{offset:06d}.pt')
    parent,entry,target = [torch.load(p,map_location='cpu',weights_only=False)
                           for p in (parent_path,entry_path,target_path)]
    prefix,rows = training_prefix(external/'training_curve.jsonl',offset)
    validate_rows(rows,offset,spec)
    package = json.loads((out/'training_package.json').read_text())
    steps = sum(r['ppo']['optimizer_steps'] for r in rows)
    samples = sum(r['training']['trainable_agent_samples'] for r in rows)
    ticks = sum(r['training']['physical_physics_ticks'] for r in rows)
    checks = dict(parent_file_exact=sha(parent_path) == spec['parent']['sha256'],
        entry_model_exact=equal(entry['model'],parent['model']),
        entry_optimizer_exact=equal(entry['optimizer'],parent['optimizer']),
        entry_four_rng_exact=all(equal(entry[k],parent[k]) for k in RNG),
        entry_native_rng_exact=equal(entry['opponent_state']['native_nexto']['rng_state'],parent['opponent_state']['native_nexto']['rng_state']),
        entry_opponent_rng_exact=equal(entry['opponent_state']['generator_state'],parent['opponent_state']['generator_state']),
        entry_counters_exact=all(entry[k] == parent[k] for k in
            ('accepted_updates','direct_skill_samples','direct_skill_physics_ticks','cumulative_optimizer_steps')),
        new_format=entry['format'] == target['format'] == spec['version']+'_CHECKPOINT',
        authorities_bound=all(p['native_nexto_authority_sha256'] == canonical(spec) for p in (entry,target)),
        parents_bound=all(p['native_nexto_parent_sha256'] == spec['parent']['sha256'] for p in (entry,target)),
        lineage_bound=target['native_nexto_package']['lineage'] == package['lineage'],
        target_offset=target['native_nexto_branch_updates'] == offset and target['accepted_updates'] == 750+offset,
        model_changed=not equal(entry['model'],target['model']),
        finite_model=all(bool(torch.isfinite(t).all()) for t in target['model'].values()),
        finite_adam=all(bool(torch.isfinite(t).all()) for s in target['optimizer']['state'].values() for t in s.values()),
        adam_steps=target['cumulative_optimizer_steps'] == parent['cumulative_optimizer_steps']+steps and steps>0,
        adam_steps_uniform={int(s['step']) for s in target['optimizer']['state'].values()} == {target['cumulative_optimizer_steps']},
        sample_count=target['direct_skill_samples'] == parent['direct_skill_samples']+samples,
        physics_count=target['direct_skill_physics_ticks'] == parent['direct_skill_physics_ticks']+ticks,
        unchanged_ppo=target['ppo_config_sha256'] == parent['ppo_config_sha256'],
        unchanged_architecture=target['policy_config_sha256'] == parent['policy_config_sha256'],
        unchanged_reward=target['reward_authority'] == entry['reward_authority'] == parent['reward_authority'] == spec['reward'],
        runtime_reward=target['runtime_contract_hashes']['reward'] == canonical(spec['reward']),
        corpus_bound=target['effective_training_scenario_sha256'] == parent['effective_training_scenario_sha256'] == package['scenario_sha256'],
        rng_saved=all(k in target for k in RNG),
        native_opponent=target['opponent_state']['native_nexto']['version'] == spec['opponents']['controller'],
        native_mode=target['opponent_state']['native_nexto']['sampling_mode'] == 'native_v5',
        exploration_temperature=target['training_distribution'] == dict(temperature=2.,on_policy_sampling_and_likelihood=True))
    assert all(checks.values()),checks
    objective = objective_checks(target,rows,spec)
    result = dict(arm=arm,offset=offset,checks=checks,objective_checks=objective,
        optimizer_steps_in_audit=0,authority_sha256=canonical(spec),parent=spec['parent'],
        entry=dict(path=entry_path.relative_to(ROOT).as_posix(),sha256=sha(entry_path)),
        checkpoint=dict(path=target_path.relative_to(ROOT).as_posix(),sha256=sha(target_path)),
        model_sha256=tensor_hash(target['model']),
        training=dict(updates=offset,optimizer_steps=steps,learner_decisions=samples,physics_ticks=ticks,
            max_completed_update_mean_kl=max(r['ppo']['completed_update_mean_kl'] for r in rows),
            max_sample_kl_telemetry=max(r['ppo']['completed_update_sample_kl_max'] for r in rows),
            kl_rejections=0,first_row=rows[0],last_row=rows[-1]))
    if offset in spec['evaluation_boundaries']:
        match = out/f'child_{offset:06d}.json'
        after = json.loads(match.read_text())
        assert after['context'] == dict(native_nexto_authority_sha256=canonical(spec),branch_updates=offset)
        assert after['checkpoint'] == result['checkpoint']
        reduction = reduce(match)
        assert all(reduction['integrity'].values())
        assert reduction == json.loads(match.with_suffix('.integrity.json').read_text())
        baseline_path = run.prior.OUT/'child_000070.json'
        baseline = json.loads(baseline_path.read_text())
        assert baseline['checkpoint']['sha256'] == spec['parent']['sha256']
        assert all(reduce(baseline_path)['integrity'].values())
        result.update(parent_comparison=compare_records(baseline,after),contacts=reduction['contacts'],
            evaluation_hashes={p.relative_to(ROOT).as_posix():sha(p) for p in
                              (baseline_path,match,match.with_suffix('.integrity.json'))})
    result['interpretation'] = 'Integrity is not gameplay promotion. Base PPO is unchanged; added barrier differs prospectively by arm.'
    save_once(out/f'through_{offset:06d}.jsonl',prefix)
    save_once(out/f'progress_{offset:06d}.json',encoded(result))
    print(json.dumps(dict(arm=arm,offset=offset,checks=checks,objective_checks=objective,
        checkpoint=result['checkpoint'],comparison=result.get('parent_comparison',{}).get('summary'))),flush=True)
    return result


def pair_report(offset):
    if offset not in run.EVALUATIONS:
        raise ValueError('Expected equal frozen evaluation offsets')
    arms = {arm:audit(arm,offset) for arm in run.ARMS}
    entries = {arm:torch.load(ROOT/value['entry']['path'],map_location='cpu',weights_only=False)
               for arm,value in arms.items()}
    a,b = entries['retain'],entries['withdraw']
    checks = dict(common_parent=all(x['parent'] == run.PARENT for x in arms.values()),
        initial_models_identical=equal(a['model'],b['model']),
        initial_optimizers_identical=equal(a['optimizer'],b['optimizer']),
        initial_four_rng_identical=all(equal(a[k],b[k]) for k in RNG),
        initial_opponent_state_identical=equal(a['opponent_state'],b['opponent_state']),
        corpus_identical=a['effective_training_scenario_sha256'] == b['effective_training_scenario_sha256'],
        exposure_identical=arms['retain']['training']['learner_decisions'] == arms['withdraw']['training']['learner_decisions'] == offset*4423680,
        reward_identical=a['reward_authority'] == b['reward_authority'])
    assert all(checks.values()),checks
    matches = {arm:json.loads((run.OUT/arm/f'child_{offset:06d}.json').read_text()) for arm in run.ARMS}
    comparison = compare_records(matches['retain'],matches['withdraw'])
    result = dict(version=run.VERSION,offset=offset,checks=checks,arms=arms,
        comparison_authority_sha256=canonical(run.comparison_authority()),
        retain_to_withdraw=comparison,optimizer_steps_in_report=0,
        interpretation=run.comparison_authority()['interpretation'])
    save_once(run.OUT/f'comparison_{offset:06d}.json',encoded(result))
    print(json.dumps(dict(offset=offset,checks=checks,retain_to_withdraw=comparison['summary'])),flush=True)
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('offset',type=int)
    parser.add_argument('--arm',choices=tuple(run.ARMS))
    args = parser.parse_args()
    torch.set_num_threads(2)
    audit(args.arm,args.offset) if args.arm else pair_report(args.offset)
