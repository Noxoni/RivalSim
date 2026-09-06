"""CPU-only same-layout/local-age reduction of immutable kickoff traces."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from rivalsim.rival2_contracts import OBS_FIELD_NAMES  # noqa: E402


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest().upper()


def pair_by_age(a):
    # lifecycle_layout is an event field and becomes -1 after age zero. Bind
    # once per physical episode, not from later ticks or the initial layout.
    episodes={}
    for d,w in zip(*np.where(a['eligible'] & (a['episode_ticks']==0))):
        goal=int(a['match_goal_count'][d,w]); key=(int(w),goal)
        assert key not in episodes, 'Duplicate episode origin'
        layout=int(a['lifecycle_layout'][d,w] if goal else a['match_starting_layout'][d,w])
        assert layout in range(5), 'Missing actual kickoff layout'
        episodes[key]=layout
    initial={}; later=[]; seen=set()
    for d,w in zip(*np.where(a['eligible'])):
        age=int(a['episode_ticks'][d,w]); side=int(a['rival_side'][d,w])
        goal=int(a['match_goal_count'][d,w]); episode=(int(w),goal)
        assert episode in episodes and age in range(0,32,4) and side in (0,1)
        identity=(*episode,age); assert identity not in seen, 'Duplicate local-age frame'
        seen.add(identity); key=(episodes[episode],side,age); index=(int(d),int(w),side)
        if goal==0:
            assert key not in initial, 'Duplicate initial layout/side/age'
            initial[key]=index
        else:
            later.append((key,index))
    assert len(initial)==80, 'Incomplete ten initial kickoff windows'
    assert later
    return [(key,initial[key],index) for key,index in later]


def reduce_arrays(a):
    assert all(np.isfinite(v).all() for v in a.values())
    pairs=pair_by_age(a); reports=[]
    for age in range(0,32,4):
        subset=[(x,y) for key,x,y in pairs if key[2]==age]
        assert subset
        before=np.stack([a['observation'][x] for x,_ in subset])
        after=np.stack([a['observation'][y] for _,y in subset]); diff=abs(before-after)
        actions=np.stack([a['rival_action'][x[:2]]!=a['rival_action'][y[:2]] for x,y in subset])
        native={}
        for name in ('car_pos','car_vel','car_quat','car_ang_vel','wheel_contact','boost','ball_pos','ball_vel'):
            delta=np.stack([abs(a[name][x[:2]]-a[name][y[:2]]) for x,y in subset])
            native[name]=dict(max_abs=float(delta.max()),changed_pairs=int((delta!=0).reshape(len(subset),-1).any(1).sum()))
        nexto={}
        for name in ('nexto_previous_action','nexto_neural_counter','nexto_kickoff_index','match_kickoff_active'):
            delta=np.stack([a[name][x[:2]]!=a[name][y[:2]] for x,y in subset])
            nexto[name]=int(delta.reshape(len(subset),-1).any(1).sum())
        if age==0:
            assert all(not a['hidden_before'][x[0],:,x[1]].any()
                       and not a['hidden_before'][y[0],:,y[1]].any() for x,y in subset)
        reports.append(dict(age_physics_ticks=age,pairs=len(subset),
            rival_action_changed_pairs=int(actions.any(1).sum()),per_control_changed=actions.sum(0).tolist(),
            observation_max_abs=float(diff.max()),native=native,nexto_predecision_state_changed_pairs=nexto,
            fields=[dict(field=OBS_FIELD_NAMES[i],max_abs=float(diff[:,i].max()),changed_pairs=int((diff[:,i]!=0).sum()))
                    for i in range(182) if diff[:,i].any()]))
    return dict(initial_kickoffs=10,postgoal_kickoff_origins=reports[0]['pairs'],
        matched_decisions=len(pairs),zero_hidden_at_all_origins=True,ages=reports,
        pair_indices=[dict(layout=k[0],side=k[1],age=k[2],initial=list(x),later=list(y)) for k,x,y in pairs])


def build(directory):
    manifest=json.loads((directory/'trace/manifest.json').read_text())
    archive=directory/'trace/kickoff_reset.npz'
    assert manifest['exact_saved_raw_summary_hidden_reset_parity'] and manifest['model_checkpoint_unchanged']
    assert manifest['new_optimizer_steps']==0 and sha(archive)==manifest['archive_sha256']
    source=ROOT/manifest['source']; assert sha(source)==manifest['source_sha256']
    checkpoint=ROOT/f"checkpoints/rival2/direct_skills_v1/plus_{manifest['accepted_updates']:06d}.pt"
    assert sha(checkpoint)==manifest['checkpoint_sha256']
    with np.load(archive,allow_pickle=False) as data:a={k:data[k] for k in data.files}
    result=reduce_arrays(a)
    return dict(**result,archive_sha256=sha(archive),checkpoint_sha256=sha(checkpoint),
        exact_complete_match_replay=True,optimizer_steps=0,policy_or_physics_interventions=0,
        source_text_sha256=hashlib.sha256(source.read_bytes().replace(b'\r\n',b'\n')).hexdigest().upper(),
        interpretation='Same-layout/side/local-age comparisons. Tiny quaternion differences remain explicit. '
        'Nexto previous action at age zero is its pre-reset history, not proof of an applied different control. '
        'Snapshots every four physics ticks cannot prove intervening controls identical. '
        'Physical divergence is evidence to isolate, not a proven new reset bug or learning improvement.')


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('directory',type=Path);args=parser.parse_args()
    report=build(args.directory);path=args.directory/'report.json'
    if path.exists():assert json.loads(path.read_text())==report
    else:path.write_text(json.dumps(report,sort_keys=True,indent=2)+'\n')
    print(json.dumps({k:v for k,v in report.items() if k not in ('pair_indices','ages')}))
    print(json.dumps([{k:v for k,v in r.items() if k in ('age_physics_ticks','pairs','rival_action_changed_pairs','observation_max_abs')} for r in report['ages']]))
