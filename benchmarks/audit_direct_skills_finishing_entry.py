"""CPU-only audit of the actual branch entry and first accepted PPO checkpoint."""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import torch
from benchmarks import run_direct_skills_finishing_goal_v2 as branch
from benchmarks.run_rival2_fresh_ground_30hz_v1 import sha, tensor_hash, write_json, utc


def equal(a, b):
    if isinstance(a, torch.Tensor):
        return isinstance(b, torch.Tensor) and torch.equal(a, b)
    if isinstance(a, dict):
        return isinstance(b, dict) and a.keys() == b.keys() and all(equal(a[k], b[k]) for k in a)
    if isinstance(a, (list, tuple)):
        return type(a) is type(b) and len(a) == len(b) and all(equal(x,y) for x,y in zip(a,b))
    return a == b


def run():
    package, fixed = branch.verify()
    paths = [branch.SOURCE, branch.CKPTS / 'entry_000000.pt', branch.CKPTS / 'first_000001.pt']
    parent, entry, first = [torch.load(p, map_location='cpu', weights_only=False) for p in paths]
    row = json.loads((branch.EXTERNAL / 'training_curve.jsonl').read_text().splitlines()[0])
    finishing_rows = [v for k,v in row['training']['by_reward_role_and_opponent'].items() if k.startswith('finishing_')]
    checks = dict(
        parent_file_exact=sha(paths[0]) == branch.SOURCE_SHA,
        entry_model_exact=equal(parent['model'], entry['model']),
        entry_optimizer_exact=equal(parent['optimizer'], entry['optimizer']),
        entry_four_rng_exact=all(equal(parent[k], entry[k]) for k in branch.RNG),
        entry_counters_exact=all(parent[k] == entry[k] for k in ('accepted_updates','direct_skill_samples','direct_skill_physics_ticks','cumulative_optimizer_steps')),
        branch_offsets=[entry['finishing_branch_updates'],first['finishing_branch_updates']] == [0,1],
        cumulative_ancestry=[entry['accepted_updates'],first['accepted_updates']] == [branch.START,branch.START+1],
        authority_bound=all(p['finishing_authority_sha256'] == branch.content_hash(branch.authority()) for p in (entry,first)),
        parent_bound=all(p['finishing_parent_sha256'] == branch.SOURCE_SHA for p in (entry,first)),
        runtime_bound=all(p['evaluation_runtime_authority_sha256'] == branch.runtime.digest(fixed) for p in (entry,first)),
        shooting_bank_bound=all(p['effective_training_scenario_sha256'] == package['scenario_sha256'] for p in (entry,first)),
        only_reward_contract_changes=all(p['runtime_contract_hashes'] == dict(parent['runtime_contract_hashes'],reward=branch.content_hash(branch.reward_authority())) for p in (entry,first)),
        new_reward_bound=all(p['reward_authority'] == branch.reward_authority() for p in (entry,first)),
        raw_events_unpaid=bool(finishing_rows) and all(v['direct_reward'] == -v['failed_attempt'] and v['approach_reward'] == 0 for v in finishing_rows),
        real_goal_success_only=all(v['success_endings'] == v['goals'] for v in finishing_rows),
        fresh_optimizer_false=not entry['fresh_optimizer'] and not first['fresh_optimizer'],
        exact_adam_increment=first['cumulative_optimizer_steps']-parent['cumulative_optimizer_steps'] == row['ppo']['optimizer_steps'],
        first_adam_counters=all(int(s['step']) == first['cumulative_optimizer_steps'] for s in first['optimizer']['state'].values()),
        expected_learner_samples=first['direct_skill_samples']-parent['direct_skill_samples'] == 4423680,
        expected_physics_ticks=first['direct_skill_physics_ticks']-parent['direct_skill_physics_ticks'] == 11796480,
        exact_nexto_share=row['training']['nexto_training_sample_count']*3 == row['training']['trainable_agent_samples'],
        model_changed=not equal(parent['model'], first['model']),
        model_finite=all(bool(torch.isfinite(t).all()) for t in first['model'].values()),
        optimizer_finite=all(bool(torch.isfinite(t).all()) for s in first['optimizer']['state'].values() for t in s.values()),
    )
    assert all(checks.values()), checks
    result = dict(utc=utc(), checks=checks, audit_optimizer_steps=0,
        actual_first_optimizer_steps=row['ppo']['optimizer_steps'],
        actual_cumulative_optimizer_steps=first['cumulative_optimizer_steps'],
        source_and_checkpoints=[dict(path=p.relative_to(ROOT).as_posix(),sha256=sha(p)) for p in paths],
        model_hashes=[tensor_hash(p['model']) for p in (parent,entry,first)],
        prospective_commit='3b7cae5c85d12c0a8cfbb50b7df5a48be9c8e955',
        source625_unchanged=True, stronger600_preserved=True,
        interpretation='Actual pre-step snapshot and first accepted finishing-goal update, not a dry-run. Child1 has cumulative direct ancestry626; old models preserved.',
    )
    write_json(branch.OUT / 'first_update_audit.json', result)
    write_json(branch.OUT / 'first_training_update.json', row)
    print(json.dumps(dict(checks=len(checks),passed=all(checks.values()), first_steps=row['ppo']['optimizer_steps'], cumulative_steps=first['cumulative_optimizer_steps'])))


if __name__ == '__main__':
    run()
