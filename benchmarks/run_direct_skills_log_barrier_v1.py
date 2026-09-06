"""Opt-in, bounded PPO exploration-only intervention from preserved parent650."""
from __future__ import annotations

import argparse
from contextlib import contextmanager
import json
from pathlib import Path
import subprocess
import sys
import xml.etree.ElementTree as ET

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
import torch

from benchmarks import run_direct_skills_kickoff_race_v1 as prior
from benchmarks import run_direct_skills_native_nexto_v1 as engine
from benchmarks.evaluate_direct_skills_native_nexto_v1 import text_sha
from benchmarks.run_rival2_fresh_ground_30hz_v1 import sha,write_json
from rivalsim import ssl_entity_mixed_training as mixed
from rivalsim.direct_skills_log_barrier_v1 import VERSION as OBJECTIVE_VERSION,sequence_loss
from rivalsim.fresh_ground_30hz import content_hash,scenario_hash

VERSION='RIVAL2_DIRECT_SKILLS_LOG_BARRIER_CAMPAIGN_V1'
OUT=ROOT/'results/rival2/direct_skills_log_barrier_v1'
CKPTS=ROOT/'checkpoints/rival2/direct_skills_log_barrier_v1'
EXTERNAL=Path('G:/dev/RivalSim-runs/direct-skills-log-barrier-v1')
CALIBRATION=ROOT/'results/rival2/direct_skills_learning_diagnostic_v1/barrier_calibration.json'
DIAG=ROOT/'results/rival2/direct_skills_learning_diagnostic_v1/report.json'
LIMIT,EVALUATIONS,COEFFICIENT=30,(5,15,30),.01
SOURCES=tuple(dict.fromkeys((*prior.SOURCES,
    'benchmarks/run_direct_skills_log_barrier_v1.py','rivalsim/direct_skills_log_barrier_v1.py',
    'benchmarks/analyze_direct_skills_learning_v1.py','benchmarks/calibrate_direct_skills_log_barrier_v1.py',
    'tests/test_direct_skills_log_barrier_v1.py','tests/test_direct_skills_log_barrier_runner_v1.py')))


def authority():
    calibration=json.loads(CALIBRATION.read_text())
    assert calibration['selected_coefficient']==COEFFICIENT
    assert calibration['optimizer_steps']==0 and calibration['model_unchanged']
    spec=prior.authority()
    spec.update(version=VERSION,child_updates=LIMIT,evaluation_boundaries=list(EVALUATIONS),
        diagnosis_sha256=sha(DIAG),calibration_sha256=sha(CALIBRATION),
        previous_arm_authority_sha256=content_hash(prior.authority()),
        exploration_objective=dict(version=OBJECTIVE_VERSION,coefficient=COEFFICIENT,
            formula='PPO loss + beta*(-mean_action(log_softmax(logits_T2))-log(90))',
            reference='Uniform across all90 joint actions, NOT parent-policy retention',
            unchanged='Same on-policy T2 sampling/likelihood, raw argmax evaluation, existing entropy0.001, no action injection or reward change',
            gradient='For each state: beta*(pi_i-1/90); no vanishing probability factor on suppressed-action term'),
        evaluation='Same ten complete native-v5 Nexto matches at+5,+15,+30; unassisted standard starts. Compare exactparent650 and frozen prior arm; no new benchmark or easy-start acceptance.',
        checkpoints='Entry0,first1,permanent5/15/30,alternating durable latest every accepted update.',
        budget='Exactly30accepted updates maximum,132710400learnerdecisions. Review, no automatic extension.',
        changes='Only added actor exploration regularizer. Preserve reward/curriculum/model/Adam from controlled parent650; do not initialize from the negative child15.',
        end='Stop at30 for review, respect user STOP. The broad SSL-development goal remains active.')
    return spec


def frozen_loss(model,data,index,config):
    return sequence_loss(model,data,index,config,coefficient=COEFFICIENT)


@contextmanager
def configured_engine():
    # Reuse the already validated kickoff environment/configuration and native
    # controller. Only this explicit process receives the new loss and paths.
    with prior.configured_engine():
        original_validate=engine.validate_resume
        # Prior's wrapper closes over its directory; use the preserved original
        # implementation, captured from the module-level reference below.
        changes=dict(VERSION=VERSION,OUT=OUT,CKPTS=CKPTS,EXTERNAL=EXTERNAL,LIMIT=LIMIT,
            EVALUATIONS=EVALUATIONS,authority=authority,verify=verify,
            validate_resume=lambda p,h,s:VALIDATE_RESUME(p,h,s,external=EXTERNAL))
        old={k:getattr(engine,k) for k in changes}
        old_mixed_loss=mixed.joint_sequence_loss
        old_base_loss=engine.base.joint_sequence_loss
        try:
            for k,v in changes.items():setattr(engine,k,v)
            mixed.joint_sequence_loss=frozen_loss
            engine.base.joint_sequence_loss=frozen_loss
            yield engine
        finally:
            mixed.joint_sequence_loss=old_mixed_loss
            engine.base.joint_sequence_loss=old_base_loss
            for k,v in old.items():setattr(engine,k,v)
            assert engine.validate_resume is original_validate


VALIDATE_RESUME=engine.validate_resume


def freeze():
    assert not (OUT/'training_package.json').exists()
    pre=json.loads((OUT/'training_preflight.json').read_text())
    assert all(pre['checks'].values()) and pre['optimizer_steps']==0
    suites=ET.parse(OUT/'tests.xml').getroot().findall('testsuite')
    assert suites and all(int(s.get(k,0))==0 for s in suites for k in ('failures','errors','skipped'))
    spec=authority()
    write_json(OUT/'training_authority.json',spec)
    evidence={str(path.relative_to(ROOT).as_posix()):sha(path) for path in
        (DIAG,CALIBRATION,DIAG.parent/'sample_evidence.npz',DIAG.parent/'EXPLORATION_PROPOSAL.md')}
    write_json(OUT/'training_package.json',dict(authority_sha256=content_hash(spec),
        scenario_sha256=scenario_hash(prior.scenarios(32768)),
        exploration_objective=spec['exploration_objective'],
        sources={p:text_sha(ROOT/p) for p in SOURCES},
        evidence={n:sha(OUT/n) for n in ('training_preflight.json','tests.xml')},
        diagnosis_evidence=evidence))


def verify():
    engine.evaluation.verify()
    package=json.loads((OUT/'training_package.json').read_text())
    assert json.loads((OUT/'training_authority.json').read_text())==authority()
    assert package['authority_sha256']==content_hash(authority())
    assert package['exploration_objective']==authority()['exploration_objective']
    for p,h in package['sources'].items():assert text_sha(ROOT/p)==h,p
    for p,h in package['diagnosis_evidence'].items():assert sha(ROOT/p)==h,p
    for p,h in package['evidence'].items():assert sha(OUT/p)==h,p
    paths=(*SOURCES,*package['diagnosis_evidence'],*[(OUT/n).relative_to(ROOT).as_posix() for n in
        ('training_authority.json','training_package.json',*package['evidence'])])
    for path in paths:
        blob=subprocess.check_output(['git','show','origin/main:'+path],cwd=ROOT)
        if path.endswith('.npz'):
            assert blob==(ROOT/path).read_bytes(),path
        else:
            assert blob.replace(b'\r\n',b'\n')==(ROOT/path).read_bytes().replace(b'\r\n',b'\n'),path
    return package


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('mode',choices=('preflight','freeze','verify','run'))
    parser.add_argument('--resume',type=Path)
    parser.add_argument('--resume-sha256')
    args=parser.parse_args();torch.set_num_threads(8)
    if args.mode=='freeze':freeze()
    elif args.mode=='verify':verify()
    else:
        with engine.base.gpu_lease(),configured_engine() as configured:
            configured.preflight() if args.mode=='preflight' else configured.run(args)
