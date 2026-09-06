"""Preserve a closed 15-update arm, with honest early/final gameplay comparison."""
from __future__ import annotations

import json
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
import torch
from benchmarks.report_direct_skills_kickoff_race_v1 import OUT,RUN,audit
from benchmarks.report_direct_skills_native_nexto_v1 import sha,canonical,compare_records,save_once


def terminal_checks(state,latest,rows,spec):
    checks=dict(complete=state['status']=='complete_review',
        boundary=state['branch_updates']==latest['branch_updates']==spec['child_updates']==15,
        counters=state['accepted_updates']==latest['accepted_updates']==spec['parent']['accepted_updates']+15,
        authority=state['native_nexto_authority_sha256']==canonical(spec),
        fifteen_rows=len(rows)==15 and [r['branch_updates'] for r in rows]==list(range(1,16)),
        no_kl_rejection=all(r['ppo']['kl_rejections']==0 for r in rows),
        rows_bound=all(r['native_nexto_authority_sha256']==canonical(spec) for r in rows))
    if not all(checks.values()):
        raise ValueError(checks)
    return checks


def main():
    spec=json.loads((OUT/'training_authority.json').read_text())
    state=json.loads((RUN/'campaign_state.json').read_text())
    latest=json.loads((RUN/'latest.json').read_text())
    rows=[json.loads(line) for line in (RUN/'training_curve.jsonl').read_text().splitlines() if line]
    checks=terminal_checks(state,latest,rows,spec)
    if (RUN/'failure.json').exists():
        raise RuntimeError('Failure evidence requires explicit audit before ordinary finalization')
    package=json.loads((OUT/'training_package.json').read_text())
    import hashlib
    for name,digest in package['sources'].items():
        assert hashlib.sha256((ROOT/name).read_bytes().replace(b'\r\n',b'\n')).hexdigest().upper()==digest,name
    for name,digest in package['evidence'].items():
        assert sha(OUT/name)==digest,name
    early,final=audit(5),audit(15)
    assert sha(Path(latest['path']))==latest['sha256']==final['checkpoint']['sha256']
    assert state['cumulative_optimizer_steps']==rows[-1]['cumulative_optimizer_steps']
    comparison=compare_records(json.loads((OUT/'child_000005.json').read_text()),
                               json.loads((OUT/'child_000015.json').read_text()))
    archived={}
    for source,target in (('campaign_state.json','final_campaign_state.json'),('latest.json','final_latest.json'),
                          ('stdout.log','stdout.log'),('stderr.log','stderr.log')):
        data=(RUN/source).read_bytes()
        save_once(OUT/target,data)
        assert (RUN/source).read_bytes()==data,'Log changed during close-out'
        archived[target]=sha(OUT/target)
    result=dict(version='RIVAL2_KICKOFF_RACE_COMPLETION_V1',checks=checks,
        authority_sha256=canonical(spec),checkpoint=final['checkpoint'],
        accepted_updates=15,final_cumulative_update=665,
        training={k:v for k,v in final['training'].items() if k not in ('first_row','last_row')},
        parent_to_early=early['comparison'],parent_to_final=final['comparison'],early_to_final=comparison,
        contacts_parent=early['contacts_before'],contacts_early=early['contacts_after'],contacts_final=final['contacts_after'],
        archived=archived,optimizer_steps_in_finalization=0,
        interpretation='Training complete and evidence verified; actual match outcomes below determine usefulness. Not automatic SSL, learned kickoff strategy, or deployment approval.')
    save_once(OUT/'completion.json',(json.dumps(result,indent=2,sort_keys=True,allow_nan=False)+'\n').encode())
    print(json.dumps(dict(checkpoint=final['checkpoint'],parent_to_final=final['comparison']['summary'],early_to_final=comparison['summary'])))


if __name__=='__main__':
    torch.set_num_threads(2)
    main()
