"""Final task-window reduction from the closed, audited exploration arm only.

Standard library only. No model imports, rollouts, optimization or new matches.
The existing through700 output and default reduction remain unchanged.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from benchmarks.summarize_direct_skills_task_progress_700 import SOURCES, summarize

DIRECTORY = 'results/rival2/direct_skills_exploration_continuation_v1'
OUT = ROOT / DIRECTORY
VERSION = 'RIVAL2_TASK_PROGRESS_THROUGH_750_V1'


def final_sources(completion):
    required = dict(status='complete_review_not_SSL', accepted_updates=70,
                    total_exploration_updates=100, cumulative_updates=750,
                    optimizer_steps_in_finalization=0, new_evaluations_in_finalization=0)
    if any(completion.get(key) != value for key, value in required.items()):
        raise ValueError('Require the completed750 CPU closeout')
    checks = completion.get('checks', {})
    if (checks.get('no_active_worker') is not True or checks.get('frozen_boundary') is not True
            or not all(value is True for value in checks.values())):
        raise ValueError('Incomplete terminal checks')
    training = completion.get('training', {})
    if (training.get('updates') != 70 or training.get('learner_decisions') != 309657600
            or training.get('physics_ticks') != 825753600 or training.get('kl_rejections') != 0):
        raise ValueError('Wrong final training exposure')
    if set(completion.get('parent_comparisons', {})) != {'20', '45', '70'}:
        raise ValueError('Missing required final comparisons')
    digest = completion.get('training_curve_sha256', '')
    if len(digest) != 64 or any(c not in '0123456789abcdefABCDEF' for c in digest):
        raise ValueError('Invalid final curve hash')
    return (SOURCES[0], (DIRECTORY + '/through_000070.jsonl', digest.upper(), 680, 70))


def build():
    raw = (OUT / 'completion.json').read_bytes()
    completion = json.loads(raw)
    result = summarize(sources=final_sources(completion), version=VERSION)
    bounds = [(row['first_update'], row['last_update']) for row in result['windows']]
    if bounds != [(first, first + 9) for first in range(651, 751, 10)]:
        raise ValueError('Missing or overlapping exploration windows')
    result['completion_sha256'] = hashlib.sha256(raw).hexdigest().upper()
    result['training_included'] = dict(updates=100, learner_decisions=442368000,
                                       physical_world_ticks=1179648000)
    result['interpretation'] = (
        'On-policy task telemetry, not matched trials. A completed arm and finite '
        'optimizer do not establish skill or override full-match results.')
    return result


def markdown(result):
    lines = [
        '# Task progress through750', '',
        'CPU-only reduction of the committed30-update prefix and the closed,',
        'audited70-update continuation. No additional training or matches.', '',
        '## Shooting outcomes', '',
        'Fractions use observed episode endings, not an independently tracked',
        'attempt cohort. On-policy states, episode age and self-play change.', '',
        '| Updates | Nexto goals / endings | Nexto fraction | Self-play goals / endings | Self-play fraction |',
        '| --- | ---: | ---: | ---: | ---: |',
    ]
    for window in result['windows']:
        cells = []
        for opponent in ('nexto', 'selfplay'):
            data = window['roles']['finishing_' + opponent]
            totals = data['totals']
            fraction = data['success_fraction_of_observed_endings']
            cells.extend((f"{totals['goals']} / {totals['episode_endings']}",
                          'unavailable' if fraction is None else f'{fraction:.4%}'))
        lines.append(f"| {window['first_update']}-{window['last_update']} | " + ' | '.join(cells) + ' |')
    lines.extend([
        '', '## Task telemetry against Nexto', '',
        'Every rate below is per learner-minute (1,800 decisions at30Hz).',
        'These are existing event definitions, not newly defined skill detectors.', '',
        '| Updates | Natural touches | Natural goals | Challenge control gain | Defense clear | Kickoff race first contact | Kickoff control gain |',
        '| --- | ---: | ---: | ---: | ---: | ---: | ---: |',
    ])
    columns = [('natural', 'touches'), ('natural', 'goals'), ('challenge', 'control_gain'),
               ('defense', 'shot_clear'), ('kickoff', 'first_touch'), ('kickoff', 'control_gain')]
    for window in result['windows']:
        values = [window['roles'][role + '_nexto']['per_learner_minute'][metric]
                  for role, metric in columns]
        lines.append(f"| {window['first_update']}-{window['last_update']} | " +
                     ' | '.join(f'{value:.4f}' for value in values) + ' |')
    lines.extend(['', '## Interpretation limits', ''])
    lines.extend('- ' + value for value in result['semantics'].values())
    lines.extend(['', 'The JSON retains all five roles and both opponent families, raw counts,',
                  'exposure denominators, source hashes and final-closeout hash.',
                  'Use the final matched Nexto comparison for competitive conclusions.', ''])
    return '\n'.join(lines)


def save_once(path, raw):
    if path.exists():
        if path.read_bytes() != raw:
            raise ValueError('Existing immutable reduction differs: ' + str(path))
    else:
        with path.open('xb') as stream:
            stream.write(raw)


if __name__ == '__main__':
    result = build()
    raw = (json.dumps(result, sort_keys=True, indent=2, allow_nan=False) + '\n').encode()
    save_once(OUT / 'task_progress_750.json', raw)
    save_once(OUT / 'TASK_PROGRESS_750.md', markdown(result).encode())
    print(json.dumps(dict(windows=len(result['windows']), optimizer_steps=0,
                         sha256=hashlib.sha256(raw).hexdigest().upper())))
