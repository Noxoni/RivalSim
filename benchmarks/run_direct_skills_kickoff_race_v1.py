"""Prospectively frozen kickoff learning arm using the unchanged PPO engine.

The process-local binding below is explicit configuration of the preserved
native-Nexto runner, not a source edit to any historical campaign. It is restored
on exit. New directories and version prevent accidental old-lineage resumes.
"""
from __future__ import annotations

import argparse
from contextlib import contextmanager
from dataclasses import asdict
import json
from pathlib import Path
import subprocess
import sys
import xml.etree.ElementTree as ET

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
import torch

from benchmarks import run_direct_skills_native_nexto_v1 as engine
from benchmarks.evaluate_direct_skills_native_nexto_v1 import text_sha
from benchmarks.run_rival2_fresh_ground_30hz_v1 import sha, write_json
from rivalsim.direct_skills_kickoff_race_v1 import KickoffRaceEnv, scenarios, reward_authority, curriculum_authority
from rivalsim.fresh_ground_30hz import content_hash, scenario_hash, ppo_config

VERSION='RIVAL2_DIRECT_SKILLS_KICKOFF_RACE_CAMPAIGN_V1'
OUT=ROOT/'results/rival2/direct_skills_kickoff_race_v1'
CKPTS=ROOT/'checkpoints/rival2/direct_skills_kickoff_race_v1'
EXTERNAL=Path('G:/dev/RivalSim-runs/direct-skills-kickoff-race-v1')
LIMIT,EVALUATIONS=15,(5,15)
PARENT=dict(path='checkpoints/rival2/direct_skills_finishing_goal_v2/child_000025.pt',
    sha256='939213BF57FFC751C7C173FFCC8AC34DE84D2CAA66CFECB79C59792CBABB0F6D',accepted_updates=650)
DIAG=ROOT/'results/rival2/direct_skills_native_kickoff_v1/analysis.json'
SOURCES=tuple(dict.fromkeys((*engine.SOURCES,
    'benchmarks/run_direct_skills_kickoff_race_v1.py',
    'rivalsim/direct_skills_kickoff_race_v1.py',
    'tests/test_direct_skills_kickoff_race_v1.py',
    'tests/test_direct_skills_kickoff_race_runner_v1.py')))


def selected():
    assert sha(ROOT/PARENT['path'])==PARENT['sha256']
    return dict(checkpoint=PARENT)


def authority():
    selected()
    return dict(version=VERSION,parent=PARENT,worlds=32768,child_updates=LIMIT,
        evaluation_boundaries=list(EVALUATIONS),physics_hz=120,policy_hz=30,
        ppo=asdict(ppo_config()),critic_lr=3e-4,training_temperature=2.,
        reward=reward_authority(),reward_sha256=content_hash(reward_authority()),
        curriculum=curriculum_authority(),scenario_seed=engine.base.SEED,
        opponents=dict(collector=engine.COLLECTOR_VERSION,controller=engine.CONTROLLER,
            sampling_mode='native_v5',seed=2026090691,nexto_world_fraction=.5,nexto_learner_sample_fraction=1/3),
        diagnosis_sha256=sha(DIAG),architecture_change=False,fresh_optimizer=False,
        initialization='Exact parent650 model/Adam/groups/counters/four RNG. Fresh physical episodes and zero hidden. New native Nexto RNG from fixedseed. Not child10/25 continuation.',
        resume='Same new lineage, exact latest accepted checkpoint only; model/Adam/four RNG/native-opponent RNG restored with fresh physical episodes/hidden and explicit controller-cache reset. Not exact physical replay.',
        evaluation_spec_sha256=content_hash(engine.evaluation.specification()),
        evaluation='At+5 and+15: original ten corrected-controller full matches from standard standing kickoffs, no momentum assistance. Compare preserved parent650; first-contact counts, control continuation and scoring remain distinct. No easy-start substitution or checkpoint reselection.',
        checkpoints='Entry0,first1,permanent5/15 and alternating durable latest every accepted update; preserve all historical lineages.',
        budget='Exactly15 accepted updates maximum:66,355,200 trainable decisions. No automatic extension. Review outcomes before another arm.',
        safety='Unchanged finite model/gradient/Adam and transactional corruption protection. KL telemetry only; no KL gate or preservation loss.',
        success='Directional improvement in real standard kickoff and subsequent control/scoring, without confusing easy curriculum wins with deployment. No SSL claim from this short experiment.',
        end='Stop at15 for review; user STOP respected. The broader development goal remains active.')


def freeze():
    engine.evaluation.verify()
    assert not (OUT/'training_package.json').exists()
    pre=json.loads((OUT/'training_preflight.json').read_text())
    assert all(pre['checks'].values()) and pre['optimizer_steps']==0
    for name in ('reward_tests_fixed.xml','runner_tests.xml'):
        suites=ET.parse(OUT/name).getroot().findall('testsuite')
        assert suites and all(int(s.get(k,0))==0 for s in suites for k in ('failures','errors','skipped'))
    bank=scenarios(32768)
    speed=(bank.state.car_vel**2).sum(-1)**.5
    assisted=(bank.family==4)&(speed.max(-1)>0)
    write_json(OUT/'scenario_audit.json',dict(worlds=32768,scenario_sha256=scenario_hash(bank),
        family_counts={str(k):int((bank.family==k).sum()) for k in range(5)},
        momentum_assisted_kickoffs=int(assisted.sum()),
        standing_dedicated_kickoffs=int(((bank.family==4)&~assisted).sum()),
        interpretation='kickoff_indicator is preserved as controller admission context; assisted starts are not described as standard standing kickoffs'))
    spec=authority()
    write_json(OUT/'training_authority.json',spec)
    write_json(OUT/'training_package.json',dict(authority_sha256=content_hash(spec),
        scenario_sha256=scenario_hash(bank),sources={p:text_sha(ROOT/p) for p in SOURCES},
        evidence={n:sha(OUT/n) for n in ('training_preflight.json','reward_tests_fixed.xml','runner_tests.xml','scenario_audit.json')}))


def verify():
    engine.evaluation.verify()
    package=json.loads((OUT/'training_package.json').read_text())
    assert json.loads((OUT/'training_authority.json').read_text())==authority()
    assert package['authority_sha256']==content_hash(authority())
    for p,h in package['sources'].items():
        assert text_sha(ROOT/p)==h,p
    for p,h in package['evidence'].items():
        assert sha(OUT/p)==h,p
    for p in (*SOURCES,*[(OUT/n).relative_to(ROOT).as_posix() for n in
            ('training_authority.json','training_package.json',*package['evidence'])]):
        remote=subprocess.check_output(['git','show','origin/main:'+p],cwd=ROOT)
        assert remote.replace(b'\r\n',b'\n')==(ROOT/p).read_bytes().replace(b'\r\n',b'\n'),p
    return package


@contextmanager
def configured_engine():
    # validate_resume's default external was bound at original function definition;
    # pass the new directory explicitly rather than relying on global replacement.
    validate=engine.validate_resume
    changes=dict(VERSION=VERSION,OUT=OUT,CKPTS=CKPTS,EXTERNAL=EXTERNAL,LIMIT=LIMIT,
        EVALUATIONS=EVALUATIONS,FinishingGoalEnv=KickoffRaceEnv,scenarios=scenarios,
        reward_authority=reward_authority,authority=authority,verify=verify,selection=selected,
        validate_resume=lambda p,h,s:validate(p,h,s,external=EXTERNAL))
    old={k:getattr(engine,k) for k in changes}
    try:
        for k,v in changes.items():
            setattr(engine,k,v)
        yield engine
    finally:
        for k,v in old.items():
            setattr(engine,k,v)


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('mode',choices=('preflight','freeze','verify','run'))
    parser.add_argument('--resume',type=Path)
    parser.add_argument('--resume-sha256')
    args=parser.parse_args();torch.set_num_threads(8)
    if args.mode=='freeze':
        freeze()
    elif args.mode=='verify':
        verify()
    else:
        with engine.base.gpu_lease(),configured_engine() as configured:
            configured.preflight() if args.mode=='preflight' else configured.run(args)
