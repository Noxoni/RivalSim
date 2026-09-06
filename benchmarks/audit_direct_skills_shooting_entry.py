"""CPU-only verification of actual650 entry and first accepted651 snapshots."""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))

import torch  # noqa: E402
from benchmarks import run_direct_skills_shooting_progress_v1 as run  # noqa: E402
from rivalsim.direct_skills_shooting_curriculum_v1 import curriculum_authority  # noqa: E402


def exact(a,b):
    if isinstance(a,torch.Tensor):
        return (isinstance(b,torch.Tensor) and a.dtype==b.dtype and a.shape==b.shape
                and torch.equal(a.contiguous().reshape(-1).view(torch.uint8),
                                b.contiguous().reshape(-1).view(torch.uint8)))
    if isinstance(a,dict):
        return isinstance(b,dict) and a.keys()==b.keys() and all(exact(v,b[k]) for k,v in a.items())
    if isinstance(a,(tuple,list)):
        return type(a) is type(b) and len(a)==len(b) and all(exact(x,y) for x,y in zip(a,b,strict=True))
    return a==b


def finite(value):
    if isinstance(value,torch.Tensor):
        return bool(torch.isfinite(value).all())
    if isinstance(value,dict):
        return all(finite(v) for v in value.values())
    if isinstance(value,(tuple,list)):
        return all(finite(v) for v in value)
    return True


def build():
    _,package=run.verify()
    paths=[run.SOURCE,run.base.CHECKPOINTS/'shooting_entry_000650.pt',
           run.base.CHECKPOINTS/'shooting_resume_000651.pt']
    parent,entry,first=[torch.load(p,map_location='cpu',weights_only=False) for p in paths]
    rng=('policy_generator_state','shuffle_generator_state','torch_cpu_rng_state','torch_cuda_rng_state')
    checks=dict(
        source_file_unchanged=run.sha(paths[0])==run.SOURCE_SHA,
        entry_model_byte_exact=exact(parent['model'],entry['model']),
        entry_optimizer_byte_exact=exact(parent['optimizer'],entry['optimizer']),
        entry_rng_byte_exact=all(exact(parent[k],entry[k]) for k in rng),
        entry_counters_exact=all(parent[k]==entry[k] for k in ('accepted_updates','direct_skill_samples',
                                 'direct_skill_physics_ticks','cumulative_optimizer_steps')),
        new_authority_bound=all(p['exploration_amendment_sha256']==run.content_hash(run.amendment()) for p in (entry,first)),
        source650_bound=all(p['exploration_parent_sha256']==run.SOURCE_SHA for p in (entry,first)),
        effective_curriculum_bound=all(p['training_curriculum_authority_sha256']==run.content_hash(curriculum_authority())
                                      and p['effective_training_scenario_sha256']==package['scenario_sha256'] for p in (entry,first)),
        root_lineage_preserved=all(p['parent_sha256']==run.base.PARENT_SHA for p in (parent,entry,first)),
        contracts_unchanged=all(exact(parent['runtime_contract_hashes'],p['runtime_contract_hashes']) for p in (entry,first)),
        finite_model_adam=all(finite(p['model']) and finite(p['optimizer']) for p in (entry,first)),
        first_accepted651=first['accepted_updates']==651,
        learner_sample_increment=first['direct_skill_samples']-parent['direct_skill_samples']==4423680,
        physics_tick_increment=first['direct_skill_physics_ticks']-parent['direct_skill_physics_ticks']==11796480,
        no_fresh_optimizer=not entry['fresh_optimizer'] and not first['fresh_optimizer'],
    )
    lines=(run.base.EXTERNAL/'training_curve.jsonl').read_text().splitlines(keepends=True)
    curve=[json.loads(line) for line in lines if line.endswith('\n')]
    rows=[r for r in curve if r['accepted_updates']==651]
    assert len(rows)==1
    row=rows[0]
    checks.update(
        accepted_curve_continuous=[r['accepted_updates'] for r in curve if r['accepted_updates']<=651]==list(range(1,652)),
        adam_steps_recorded=first['cumulative_optimizer_steps']-144648==row['ppo']['optimizer_steps'],
        adam_tensor_counters_exact={int(s['step']) for s in first['optimizer']['state'].values()}=={first['cumulative_optimizer_steps']},
        full_on_policy_temperature=row['training_temperature']==2,
        exact_nexto_sample_share=row['training']['nexto_training_sample_count']*3==row['training']['trainable_agent_samples'],
    )
    assert all(checks.values()),checks
    return dict(checks=checks,optimizer_steps_in_audit=0,
                checkpoints=[dict(path=p.relative_to(ROOT).as_posix(),sha256=run.sha(p)) for p in paths],
                first_update=row,authority_sha256=run.content_hash(run.amendment()),
                meaning='Actual saved pre-step entry and accepted651; not a dry-run surrogate or capability proof')


if __name__=='__main__':
    report=build();path=run.OUT/'entry_and_first_update_audit.json'
    if path.exists():
        assert json.loads(path.read_text())==report
    else:
        path.write_text(json.dumps(report,sort_keys=True,indent=2)+'\n')
    print(json.dumps({'checks':report['checks'],'checkpoints':report['checkpoints'],
                      'first_update_steps':report['first_update']['ppo']['optimizer_steps']}))
