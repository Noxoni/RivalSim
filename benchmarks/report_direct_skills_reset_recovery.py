"""CPU reduction of immutable parent/child artifacts, not a new gameplay eval."""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np
import torch
from benchmarks import run_direct_skills_reset_recovery_v1 as run
from benchmarks.audit_direct_skills_reset_recovery_entry import equal
from benchmarks.audit_direct_skills_shooting_review import aggregate, freeze
from benchmarks.audit_direct_skills_shooting_entry import finite
from benchmarks.report_rival2_direct_skills import compare_documents, group_start_cases
from benchmarks.report_rival2_direct_skills_matches import compare, checkpoint_integrity
from benchmarks.report_rival2_ssl_entity_match_followup import reduce


def source_identity(path):
    return dict(path=path.relative_to(ROOT).as_posix(), text_sha256=run.runtime.text_sha(path))


def goal_intervals(doc):
    """Native event timings only. Short intervals are not a mechanic classifier."""
    raw = doc['raw']
    rows, elapsed, heights = [], [], []
    for world, count in enumerate(raw['match.goal_count']):
        ticks = raw['goal_tick'][world][:count]
        assert all(type(t) is int and 0 <= t < raw['match.total_ticks'][world] for t in ticks)
        assert all(b > a for a, b in zip(ticks, ticks[1:]))
        delta = [ticks[0]+1, *(b-a for a,b in zip(ticks,ticks[1:]))] if ticks else []
        for index, (tick, duration) in enumerate(zip(ticks,delta,strict=True)):
            scorer = raw['goal_scorer'][world][index]
            entry_height = raw['goal_entry_z'][world][index]
            assert raw['goal_entry_valid'][world][index] == 1
            assert math.isfinite(entry_height)
            rows.append(dict(match=world, goal_index=index, tick=tick,
                seconds_since_previous_goal_or_start=duration/120,
                scored_by_rival=scorer == raw['match.rival_side'][world],
                entry_x=raw['goal_entry_x'][world][index], entry_z=entry_height))
            elapsed.append(duration/120); heights.append(entry_height)
    def stats(values):
        return dict(count=len(values), minimum=min(values), median=float(np.median(values)), maximum=max(values)) if values else dict(count=0)
    return dict(interval_seconds=stats(elapsed), goal_entry_height=stats(heights), events=rows,
        meaning='Time between native goal events, with up to three ticks reset-boundary rounding. Entry height does not identify aerial control, shot intent or last touch.')


def skill_context(doc):
    compare_documents(doc, doc)
    bank = run.base.scenarios(64, run.base.SEED+1000, family_only=0)
    assert run.scenario_hash(bank) == doc['skills']['natural']['scenario_sha256']
    groups = group_start_cases(doc['skills']['natural'], *(
        getattr(bank,n).tolist() for n in ('kickoff_indicator','kickoff_layout','focal_side')))
    return dict(
        natural_start_groups=groups,
        families={name:{k:item[k] for k in ('worlds','goals_for','goals_against','truncation_endings','touches','touched_worlds','events')}
                  for name,item in doc['skills'].items()},
        meaning='Each fixed skill case ends at its first native goal or time limit. Natural includes kickoff AND ongoing ground; do not call aggregate kickoff gains general ground play.')


def parent_context():
    skill_path = run.base.RESULTS/'evaluation_000600.json'
    match_path = run.runtime.OUTPUT/'full_match_000600.json'
    skill, match = [json.loads(p.read_text()) for p in (skill_path,match_path)]
    for doc in (skill,match):
        checkpoint_integrity(doc)
        assert doc['checkpoint']['sha256'] == run.SOURCE_SHA
    return dict(parent_sha256=run.SOURCE_SHA,
        sources=[source_identity(p) for p in (skill_path,match_path)],
        skills=skill_context(skill), matches=reduce(match_path), timings=goal_intervals(match),
        optimizer_steps=0, new_policy_evaluations=0,
        conclusion='Stronger fixed kickoff-origin match result does not establish broad gameplay: keep original shooting/ongoing-start outcomes separate. No runtime or learner change.')


def validate(parent, child, rows, package):
    offset = child['recovery_branch_updates']
    checks = dict(
        bounded_child=type(offset) is int and 1 <= offset <= run.LIMIT,
        contiguous=[r['branch_updates'] for r in rows] == list(range(1,offset+1)),
        ancestor_offsets=[r['accepted_updates'] for r in rows] == list(range(601,601+offset)),
        child_offset=child['accepted_updates'] == run.START+offset,
        authority=child['recovery_authority_sha256'] == run.content_hash(run.authority()),
        parent600=child['recovery_parent_sha256'] == run.SOURCE_SHA,
        entity_lineage=child['parent_sha256'] == parent['parent_sha256'] == run.base.PARENT_SHA,
        package=child['recovery_package'] == package,
        root_package=child['package'] == parent['package'],
        contracts=all(equal(child[k],parent[k]) for k in ('runtime_contract_hashes','ppo_config_sha256','policy_config_sha256','action_contract','observation_schema_sha256','reward_authority')),
        bank=child['effective_training_scenario_sha256'] == package['scenario_sha256'],
        runtime=child['evaluation_runtime_authority_sha256'] == run.authority()['runtime_authority_sha256'],
        schema=child['model'].keys() == parent['model'].keys() and all(v.shape == parent['model'][k].shape and v.dtype == parent['model'][k].dtype for k,v in child['model'].items()),
        finite_model=finite(child['model']), finite_adam=finite(child['optimizer']),
        optimizer_groups=equal(child['optimizer']['param_groups'], parent['optimizer']['param_groups']),
        no_fresh_optimizer=not child['fresh_optimizer'],
        adam_steps=child['cumulative_optimizer_steps'] == parent['cumulative_optimizer_steps']+sum(r['ppo']['optimizer_steps'] for r in rows),
        adam_uniform={int(s['step']) for s in child['optimizer']['state'].values()} == {child['cumulative_optimizer_steps']},
        samples=child['direct_skill_samples'] == parent['direct_skill_samples']+offset*4423680,
        ticks=child['direct_skill_physics_ticks'] == parent['direct_skill_physics_ticks']+offset*11796480,
        curve_authority=all(r['recovery_authority_sha256'] == run.content_hash(run.authority()) for r in rows),
        temperature=child['training_distribution']['temperature'] == 2 and child['training_distribution']['on_policy_sampling_and_likelihood'],
        exact_nexto_third=all(r['training']['nexto_training_sample_count']*3 == r['training']['trainable_agent_samples'] == 4423680 for r in rows),
        finite_ppo=all(math.isfinite(v) for r in rows for v in r['ppo'].values() if isinstance(v,(int,float))),
        no_kl_rejection=all(r['ppo']['kl_rejections'] == 0 for r in rows),
        four_rng=all(torch.is_tensor(child[k]) and child[k].dtype == torch.uint8 and child[k].numel() > 0 for k in run.RNG),
    )
    assert all(checks.values()), {k:v for k,v in checks.items() if not v}
    return checks


def review():
    package,_ = run.verify()
    state = json.loads((run.EXTERNAL/'campaign_state.json').read_text())
    assert state['status'] == 'complete_review' and state['branch_updates'] == run.LIMIT
    child_path = run.CKPTS/f'child_{run.LIMIT:06d}.pt'
    child_sha = run.sha(child_path)
    parent = torch.load(run.SOURCE, map_location='cpu',weights_only=False)
    child = torch.load(child_path, map_location='cpu',weights_only=False)
    curve = (run.EXTERNAL/'training_curve.jsonl').read_text()
    rows = [json.loads(line) for line in curve.splitlines()]
    checks = validate(parent,child,rows,package)
    assert run.sha(run.SOURCE) == run.SOURCE_SHA and run.sha(child_path) == child_sha
    baseline_skill = json.loads((run.base.RESULTS/'evaluation_000600.json').read_text())
    child_skill_path = run.OUT/f'evaluation_child_{run.LIMIT:06d}.json'
    child_skill = json.loads(child_skill_path.read_text())
    child_match_path = run.OUT/f'full_match_child_{run.LIMIT:06d}.json'
    for doc in (child_skill,json.loads(child_match_path.read_text())):
        checkpoint_integrity(doc)
        assert doc['checkpoint']['sha256'] == child_sha
    comparisons = compare(run.runtime.OUTPUT/'full_match_000600.json',child_match_path)
    result = dict(checks=checks, checkpoint=dict(path=child_path.relative_to(ROOT).as_posix(),sha256=child_sha),
        child_updates=run.LIMIT, cumulative_direct_updates=child['accepted_updates'],
        direct_skill_samples=child['direct_skill_samples'], direct_skill_physics_ticks=child['direct_skill_physics_ticks'],
        cumulative_optimizer_steps=child['cumulative_optimizer_steps'], added_optimizer_steps=sum(r['ppo']['optimizer_steps'] for r in rows),
        max_completed_update_mean_kl=max(r['ppo']['completed_update_mean_kl'] for r in rows),
        max_completed_update_sample_kl=max(r['ppo']['completed_update_sample_kl_max'] for r in rows),
        peak_cuda_bytes=max(r['cuda_peak_allocated_bytes'] for r in rows),
        skill_comparison=compare_documents(baseline_skill,child_skill), child_skill_context=skill_context(child_skill),
        match_comparison=comparisons, child_match_timings=goal_intervals(json.loads(child_match_path.read_text())),
        training_all=aggregate(rows), training_early=aggregate(rows[:12]), training_late=aggregate(rows[12:]),
        training_note='All25updates included; early1..12 includes fresh-episode transient, late13..25. No outcome-dependent exclusion. Training proxy rates are not evaluation gains.',
        optimizer_steps_in_audit=0, new_policy_evaluations=0,
        sources=[source_identity(p) for p in (child_skill_path,child_match_path)],
    )
    curve_path = run.OUT/'training_curve_child_000001_to_000025.jsonl'
    encoded = ''.join(json.dumps(r,sort_keys=True,allow_nan=False)+'\n' for r in rows)
    if curve_path.exists():
        assert curve_path.read_text() == encoded
    else:
        curve_path.write_text(encoded)
    freeze(run.OUT/'review_child_000025.json',result)
    return result


if __name__ == '__main__':
    torch.set_num_threads(4)
    parser = argparse.ArgumentParser()
    parser.add_argument('mode',choices=('parent','review'))
    args = parser.parse_args()
    if args.mode == 'parent':
        result = parent_context()
        freeze(run.OUT/'parent_gameplay_context.json',result)
        print(json.dumps(dict(ongoing=result['skills']['natural_start_groups']['ongoing_ground'], intervals=result['timings']['interval_seconds'])))
    else:
        result = review()
        print(json.dumps(dict(checks=len(result['checks']), matches=result['match_comparison']['target']['summary'], skills={n:x['goals_for'] for n,x in result['skill_comparison'].items()})))
