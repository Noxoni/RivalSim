"""Reduce already committed training rows; no model, rollout, or learning code."""
from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'results/rival2/direct_skills_exploration_continuation_v1/task_progress_700.json'
SOURCES = (
    ('results/rival2/direct_skills_log_barrier_v1/through_000030.jsonl',
     '3A3C7A44C6ECABA71EAD9164CC63EFCF3919CC214CE5B22FCBFAC15C8BAAF721', 650, 30),
    ('results/rival2/direct_skills_exploration_continuation_v1/through_000020.jsonl',
     'B1316C94525FF8E6E2D9D29E37F8F2388528475DCCDC811A082EE8CBD32ACD89', 680, 20),
)
ROLES = ('natural', 'challenge', 'finishing', 'defense', 'kickoff')
COUNTS = ('samples', 'touches', 'goals', 'concedes', 'episode_endings', 'success_endings',
          'first_touch', 'control_gain', 'forward_control', 'on_target_touch', 'shot_clear',
          'lost_control', 'failed_attempt')


def aggregate(records):
    if not records:
        raise ValueError('No records')
    for record in records:
        if any(not math.isfinite(record[k]) or record[k] < 0 or int(record[k]) != record[k]
               for k in COUNTS):
            raise ValueError('Invalid count')
        if record['goals'] + record['concedes'] > record['episode_endings']:
            raise ValueError('More terminal goals than endings')
    totals = {k: sum(int(row[k]) for row in records) for k in COUNTS}
    if totals['samples'] <= 0:
        raise ValueError('No learner exposure')
    # One learner decision is 1/30 second, not one 120 Hz physics tick.
    minutes = totals['samples'] / 1800.0
    rates = {k: totals[k] / minutes for k in COUNTS if k != 'samples'}
    return dict(totals=totals, learner_minutes=minutes, per_learner_minute=rates,
        success_fraction_of_observed_endings=(totals['success_endings']/totals['episode_endings']
                                             if totals['episode_endings'] else None))


def summarize():
    windows = []
    for path, digest, parent, count in SOURCES:
        raw = (ROOT/path).read_bytes()
        if hashlib.sha256(raw).hexdigest().upper() != digest:
            raise ValueError('Changed immutable source: '+path)
        rows = [json.loads(line) for line in raw.splitlines()]
        if (len(rows) != count or [r['accepted_updates'] for r in rows] != list(range(parent+1, parent+count+1))
                or [r['branch_updates'] for r in rows] != list(range(1, count+1))):
            raise ValueError('Noncontiguous source')
        for row in rows:
            groups = row['training']['by_reward_role_and_opponent']
            if set(groups) != {r+'_'+o for r in ROLES for o in ('selfplay', 'nexto')}:
                raise ValueError('Missing role/opponent')
            if sum(g['samples'] for g in groups.values()) != 4423680:
                raise ValueError('Wrong learner allocation')
            if sum(g['samples'] for k, g in groups.items() if k.endswith('_nexto')) != 1474560:
                raise ValueError('Wrong Nexto allocation')
            for opponent in ('selfplay', 'nexto'):
                finishing = groups['finishing_'+opponent]
                if (finishing['success_endings'] != finishing['goals'] or finishing['approach_reward'] != 0
                        or finishing['direct_reward'] != -finishing['failed_attempt']):
                    raise ValueError('Finishing outcome/payment semantics differ')
        for start in range(0, count, 10):
            block = rows[start:start+10]
            windows.append(dict(first_update=block[0]['accepted_updates'], last_update=block[-1]['accepted_updates'],
                source_path=path, source_sha256=digest,
                roles={role: aggregate([r['training']['by_reward_role_and_opponent'][role] for r in block])
                       for role in block[0]['training']['by_reward_role_and_opponent']}))
    return dict(version='RIVAL2_TASK_PROGRESS_THROUGH_700_V1', windows=windows,
        optimizer_steps=0, new_rollouts=0, model_loaded=False,
        semantics=dict(exposure='30 Hz learner decisions; rates use pooled sums, not averages of rates.',
            outcomes='Goals/concedes native. Finishing success is exactly scored goals, not on-target projection.',
            proxies='Challenge/control and defensive clear are existing role-specific predicates, not independent possession/save proof.',
            kickoff='Training combines standing and momentum-assisted starts; race-first-contact is not possession.',
            comparison='Descriptive on-policy windows, not the same fixed states. Fresh physical entry at681 and episode-age/scenario visitation can confound trends.',
            censoring='Events and endings may straddle window edges. First touches divided by endings is NOT an attempt success rate.',
            evaluation='Do not substitute stochastic task rates for the completed deterministic full-match evaluation.'))


if __name__ == '__main__':
    result = summarize()
    data = (json.dumps(result, sort_keys=True, indent=2, allow_nan=False)+'\n').encode()
    if OUT.exists():
        if OUT.read_bytes() != data:
            raise ValueError('Existing immutable reduction differs')
    else:
        with OUT.open('xb') as stream:
            stream.write(data)
    print(json.dumps(dict(windows=len(result['windows']), optimizer_steps=0,
                         output=str(OUT), sha256=hashlib.sha256(data).hexdigest().upper())))
