"""CPU-only reduction of completed650 native shooting-pressure traces."""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np  # noqa: E402
import torch  # noqa: E402
from benchmarks.diagnose_direct_skills_shooting_000650 import OUT, MODES, SOURCE, SOURCE_SHA  # noqa: E402
from benchmarks.run_rival2_fresh_ground_30hz_v1 import sha  # noqa: E402
from rivalsim.direct_skills_v1 import projected_goal  # noqa: E402
from rivalsim.ssl_foundation_v1 import _physical_vectors  # noqa: E402


def build():
    assert sha(SOURCE) == SOURCE_SHA
    assert not (OUT / 'failure.json').exists()
    reports = {}
    original_actions = None
    for mode in MODES:
        path = OUT / (mode+'.json')
        d = json.loads(path.read_text())
        assert d['checkpoint_sha256'] == SOURCE_SHA
        assert sha(ROOT / d['trace']['path']) == d['trace']['sha256']
        a = np.load(ROOT / d['trace']['path'])
        alive = a['alive']; contact = a['touches'] * alive[...,None]
        own = np.array([np.inf if x is None else x for x in d['first_rival_contact_seconds']])
        opp = np.array([np.inf if x is None else x for x in d['first_opponent_contact_seconds']])
        gt = np.array([np.inf if x is None else x for x in d['goal_seconds']])
        goals = np.array(d['raw']['goals']).astype(bool)
        assert np.isfinite(own).all(), 'This first-contact reduction expects all64 current cases touched'
        first = np.argmax(contact[...,0]>0, axis=0)
        obs = torch.from_numpy(a['after'][first,np.arange(64)])
        physical = _physical_vectors(obs)
        actions = a['action'][first,np.arange(64)]
        if original_actions is None:
            original_actions = a['action'][0].copy()
        values = dict(
            goals=d['goals_for'], concedes=d['goals_against'], timeouts=d['timeouts'],
            goals_before_any_opponent_contact=d['goals_before_any_opponent_contact'],
            touches=d['touches'], first_contact_rival=int((own<opp).sum()),
            mean_first_rival_contact_s=float(own.mean()),
            opponent_contact_within1s=int((opp-own<=1).sum()),
            cases_with_opponent_contact=int(np.isfinite(opp).sum()),
            median_gap_to_opponent_contact_s=float(np.median((opp-own)[np.isfinite(opp)])),
            median_first_contact_to_goal_s=float(np.median((gt-own)[goals])),
            first_contact_projected_goal=int(projected_goal(physical['ball_position'],physical['ball_velocity'],1).sum()),
            any_projected_goal_event=d['events']['on_target_touch'],
            initial_contact_boost=int(actions[:,6].sum()),
            initial_contact_jump=int(actions[:,5].sum()),
            initial_contact_median_ball_speed=float(physical['ball_velocity'].norm(dim=-1).median()),
            initial_contact_median_ball_vy=float(physical['ball_velocity'][:,1].median()),
            changed_first_action_vs_original=int(np.any(a['action'][0]!=original_actions,axis=-1).sum()),
        )
        checks = dict(
            complete64=values['goals']+values['concedes']+values['timeouts']==64,
            focal_contact_trace_exact=np.array_equal(contact[...,0].sum(0),d['raw']['touches']),
            finite_observations=bool(np.isfinite(a['before']).all() and np.isfinite(a['after']).all()),
            finite_actions=bool(np.isfinite(a['action']).all()),
            exact_goals=int((alive & a['terminated'] & (a['scoring_team']==a['focal_side'][None])).sum())==values['goals'],
            model_nexto_unchanged=d['model_unchanged'] and d['nexto_unchanged'],
            zero_optimizer=d['optimizer_steps']==0,
            original_outcomes_reproduced=d['original_baseline_reproduced'] if mode==MODES[0] else True,
        )
        assert all(checks.values()), (mode,checks)
        reports[mode]=dict(values=values,checks=checks,source_sha256=sha(path),trace=d['trace'])
    return dict(checkpoint_sha256=SOURCE_SHA, modes=reports, optimizer_steps=0,
                conclusion='Accessible finishing exists, set-keeper conversion is much weaker; not SSL. '
                'All64 first contacts belong to Rival in every mode, so lack of reaching the ball is '
                'not the primary failure in these matched shooting starts. Changing defender state '
                'also changes policy actions; this is a closed-loop state intervention, not replaying '
                'identical shots into three defenders. No reward or inference change follows automatically.')


if __name__=='__main__':
    result=build()
    path=OUT/'report.json'
    if path.exists():
        assert json.loads(path.read_text())==result
    else:
        path.write_text(json.dumps(result,sort_keys=True,indent=2)+'\n')
    print(json.dumps({k:v['values'] for k,v in result['modes'].items()},indent=2))
