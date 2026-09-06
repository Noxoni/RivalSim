"""Read-only checkpoint and matched-evaluation audit for continuation from +30."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import torch

from benchmarks.audit_direct_skills_native_nexto_entry import equal, tensor_hash, sha, canonical
from benchmarks.report_direct_skills_native_nexto_v1 import training_prefix, compare_records, save_once
from benchmarks.report_direct_skills_log_barrier_v1 import objective_checks
from benchmarks.report_rival2_ssl_entity_match_followup import reduce

OUT = ROOT / 'results/rival2/direct_skills_exploration_continuation_v1'
CKPTS = ROOT / 'checkpoints/rival2/direct_skills_exploration_continuation_v1'
RUN = Path('G:/dev/RivalSim-runs/direct-skills-exploration-continuation-v1')
RNG = ('policy_generator_state', 'shuffle_generator_state', 'torch_cpu_rng_state', 'torch_cuda_rng_state')


def validate_rows(rows, offset, spec):
    start = spec['parent']['accepted_updates']
    if len(rows) != offset or [r['branch_updates'] for r in rows] != list(range(1, offset+1)):
        raise ValueError('Noncontiguous continuation curve')
    for row in rows:
        if row['accepted_updates'] != start + row['branch_updates']:
            raise ValueError('Wrong parent-relative update count')
        if row['native_nexto_authority_sha256'] != canonical(spec):
            raise ValueError('Wrong continuation authority')
        if row['training']['trainable_agent_samples'] != 4423680 or row['training']['nexto_training_sample_count'] != 1474560:
            raise ValueError('Wrong trainable-sample allocation')
        if row['ppo']['kl_rejections'] != 0:
            raise ValueError('KL must remain telemetry')


def audit(offset):
    spec = json.loads((OUT / 'training_authority.json').read_text())
    if offset not in (1, *spec['evaluation_boundaries']):
        raise ValueError('Not a preserved boundary')
    parent_path = ROOT / spec['parent']['path']
    entry_path = CKPTS / 'entry_000000.pt'
    target_path = CKPTS / ('first_000001.pt' if offset == 1 else f'child_{offset:06d}.pt')
    parent, entry, target = [torch.load(p, map_location='cpu', weights_only=False)
                             for p in (parent_path, entry_path, target_path)]
    prefix, rows = training_prefix(RUN / 'training_curve.jsonl', offset)
    validate_rows(rows, offset, spec)
    package = json.loads((OUT / 'training_package.json').read_text())
    steps = sum(r['ppo']['optimizer_steps'] for r in rows)
    samples = sum(r['training']['trainable_agent_samples'] for r in rows)
    ticks = sum(r['training']['physical_physics_ticks'] for r in rows)
    checks = dict(
        parent_file_exact=sha(parent_path) == spec['parent']['sha256'],
        entry_model_exact=equal(entry['model'], parent['model']),
        entry_optimizer_exact=equal(entry['optimizer'], parent['optimizer']),
        entry_four_rng_exact=all(equal(entry[k], parent[k]) for k in RNG),
        entry_native_rng_exact=equal(entry['opponent_state']['native_nexto']['rng_state'],
                                    parent['opponent_state']['native_nexto']['rng_state']),
        entry_opponent_rng_exact=equal(entry['opponent_state']['generator_state'], parent['opponent_state']['generator_state']),
        entry_counters_exact=all(entry[k] == parent[k] for k in
            ('accepted_updates', 'direct_skill_samples', 'direct_skill_physics_ticks', 'cumulative_optimizer_steps')),
        new_format=entry['format'] == target['format'] == spec['version']+'_CHECKPOINT',
        authorities_bound=all(p['native_nexto_authority_sha256'] == canonical(spec) for p in (entry, target)),
        parents_bound=all(p['native_nexto_parent_sha256'] == spec['parent']['sha256'] for p in (entry, target)),
        lineage_bound=target['native_nexto_package']['lineage'] == package['lineage'],
        target_offset=target['native_nexto_branch_updates'] == offset and target['accepted_updates'] == spec['parent']['accepted_updates'] + offset,
        model_changed=not equal(entry['model'], target['model']),
        finite_model=all(bool(torch.isfinite(t).all()) for t in target['model'].values()),
        finite_adam=all(bool(torch.isfinite(t).all()) for s in target['optimizer']['state'].values() for t in s.values()),
        adam_steps=target['cumulative_optimizer_steps'] == parent['cumulative_optimizer_steps']+steps and steps > 0,
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
        exploration_exact=target['training_distribution'] == dict(temperature=2., on_policy_sampling_and_likelihood=True))
    assert all(checks.values()), checks
    objective = objective_checks(target, rows, spec)
    result = dict(branch_updates=offset, total_exploration_updates=30+offset, checks=checks,
        objective_checks=objective, optimizer_steps_in_audit=0, audit_device='cpu',
        authority_sha256=canonical(spec), parent=spec['parent'],
        entry=dict(path=entry_path.relative_to(ROOT).as_posix(), sha256=sha(entry_path)),
        checkpoint=dict(path=target_path.relative_to(ROOT).as_posix(), sha256=sha(target_path)),
        model_sha256=tensor_hash(target['model']),
        training=dict(updates=offset, optimizer_steps=steps, learner_decisions=samples, physics_ticks=ticks,
            max_completed_update_mean_kl=max(r['ppo']['completed_update_mean_kl'] for r in rows),
            max_sample_kl_telemetry=max(r['ppo']['completed_update_sample_kl_max'] for r in rows),
            kl_rejections=0, first_row=rows[0], last_row=rows[-1]))
    if offset in spec['evaluation_boundaries']:
        match = OUT / f'child_{offset:06d}.json'
        after = json.loads(match.read_text())
        assert after['context'] == dict(native_nexto_authority_sha256=canonical(spec), branch_updates=offset)
        assert after['checkpoint'] == result['checkpoint']
        reduction = reduce(match)
        assert all(reduction['integrity'].values())
        assert reduction == json.loads(match.with_suffix('.integrity.json').read_text())
        baseline_paths = dict(immediate_parent=ROOT/'results/rival2/direct_skills_log_barrier_v1/child_000030.json',
                              original_control=ROOT/'results/rival2/direct_skills_native_nexto_v1/finishing650.json')
        comparisons = {}
        for name, path in baseline_paths.items():
            before = json.loads(path.read_text())
            expected = spec['parent'] if name == 'immediate_parent' else spec['original_control_parent']
            assert before['checkpoint']['sha256'] == expected['sha256']
            assert all(reduce(path)['integrity'].values())
            comparisons[name] = compare_records(before, after)
        result.update(comparisons=comparisons, contacts=reduction['contacts'],
            evaluation_hashes={p.as_posix(): sha(p) for p in (*baseline_paths.values(), match, match.with_suffix('.integrity.json'))})
    result['interpretation'] = 'Resumable accepted learning and execution integrity, not automatic gameplay promotion or SSL.'
    save_once(OUT/f'through_{offset:06d}.jsonl', prefix)
    save_once(OUT/f'progress_{offset:06d}.json', (json.dumps(result, indent=2, sort_keys=True, allow_nan=False)+'\n').encode())
    print(json.dumps(dict(offset=offset, checks=checks, objective_checks=objective, checkpoint=result['checkpoint'],
        comparisons={k: v['summary'] for k, v in result.get('comparisons', {}).items()})))
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('offset', type=int)
    torch.set_num_threads(2)
    audit(parser.parse_args().offset)
