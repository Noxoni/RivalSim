"""CPU-only checkpoint/evaluation audit including the actual new loss telemetry."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
import torch

from benchmarks import report_direct_skills_kickoff_race_v1 as core
from benchmarks.report_direct_skills_native_nexto_v1 import save_once

OUT=ROOT/'results/rival2/direct_skills_log_barrier_v1'
CKPTS=ROOT/'checkpoints/rival2/direct_skills_log_barrier_v1'
RUN=Path('G:/dev/RivalSim-runs/direct-skills-log-barrier-v1')


def objective_checks(payload,rows,spec):
    objective=spec['exploration_objective']
    beta=objective['coefficient']
    checks=dict(objective_checkpoint_bound=payload['native_nexto_package']['exploration_objective']==objective,
        coefficient_frozen=beta==.01,
        every_update_has_new_loss=all('exploration_barrier' in r['ppo'] for r in rows),
        every_update_coefficient_correct=all(abs(r['ppo']['exploration_coefficient']-beta)<1e-8 for r in rows),
        nonnegative_barrier=all(r['ppo']['exploration_barrier']>=-1e-6 for r in rows),
        weighted_loss_correct=all(abs(r['ppo']['weighted_exploration_barrier']-beta*r['ppo']['exploration_barrier'])<1e-5 for r in rows))
    assert all(checks.values()),checks
    return checks


def audit(offset):
    original=(core.OUT,core.CKPTS,core.RUN)
    try:
        core.OUT,core.CKPTS,core.RUN=OUT,CKPTS,RUN
        result=core.audit(offset)
    finally:
        core.OUT,core.CKPTS,core.RUN=original
    payload=torch.load(ROOT/result['checkpoint']['path'],map_location='cpu',weights_only=False)
    rows=[json.loads(line) for line in (OUT/f'through_{offset:06d}.jsonl').read_text().splitlines()]
    spec=json.loads((OUT/'training_authority.json').read_text())
    checks=objective_checks(payload,rows,spec)
    objective=dict(offset=offset,checks=checks,checkpoint=result['checkpoint'],
        optimizer_steps_in_audit=0,
        note='The inherited unchanged_ppo check refers to base PPO configuration, not the entire objective. This arm explicitly adds the prospectively bound uniform log barrier.',
        first_exploration_barrier=rows[0]['ppo']['exploration_barrier'],
        latest_exploration_barrier=rows[-1]['ppo']['exploration_barrier'],
        first_entropy=rows[0]['ppo']['entropy'],latest_entropy=rows[-1]['ppo']['entropy'])
    save_once(OUT/f'objective_audit_{offset:06d}.json',(json.dumps(objective,indent=2,sort_keys=True)+'\n').encode())
    print(json.dumps(objective))
    return result,objective


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('offset',type=int)
    torch.set_num_threads(2)
    audit(parser.parse_args().offset)
