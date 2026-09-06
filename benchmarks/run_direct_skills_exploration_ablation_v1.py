"""Prospective matched retain/withdraw experiment from the preserved750 policy.

Two finite arms, independent processes, identical parent and training setup.
Only the uniform exploration-barrier coefficient differs. No automatic extension.
"""
from __future__ import annotations

import argparse
from contextlib import contextmanager
from copy import deepcopy
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import xml.etree.ElementTree as ET

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import torch

from benchmarks import run_direct_skills_exploration_continuation_v1 as prior
from benchmarks import run_direct_skills_native_nexto_v1 as engine
from benchmarks.evaluate_direct_skills_native_nexto_v1 import text_sha
from benchmarks.run_rival2_fresh_ground_30hz_v1 import sha, utc, write_json
from rivalsim import ssl_entity_mixed_training as mixed
from rivalsim.direct_skills_log_barrier_v1 import sequence_loss
from rivalsim.fresh_ground_30hz import content_hash, scenario_hash

VERSION = 'RIVAL2_DIRECT_SKILLS_EXPLORATION_ABLATION_V1'
OUT = ROOT / 'results/rival2/direct_skills_exploration_ablation_v1'
CKPTS = ROOT / 'checkpoints/rival2/direct_skills_exploration_ablation_v1'
EXTERNAL = Path('G:/dev/RivalSim-runs/direct-skills-exploration-ablation-v1')
PARENT = dict(path='checkpoints/rival2/direct_skills_exploration_continuation_v1/child_000070.pt',
              sha256='517A2217CBDF4124612B518E10F30BC4D5F00F5C1DC77A5119D10D2B1DD89613',
              accepted_updates=750)
ARMS = {'retain': .01, 'withdraw': 0.}
LIMIT, EVALUATIONS = 30, (10, 30)
DECISION_FILES = tuple('results/rival2/direct_skills_exploration_continuation_v1/' + name
    for name in ('completion.json', 'child_000070.json', 'task_progress_750.json', 'FINAL_REVIEW.md'))
SOURCES = tuple(dict.fromkeys((*prior.SOURCES,
    'benchmarks/run_direct_skills_exploration_ablation_v1.py',
    'benchmarks/report_direct_skills_exploration_ablation_v1.py',
    'tests/test_direct_skills_exploration_ablation_v1.py')))


def arm_name(name):
    if name not in ARMS:
        raise ValueError('Expected retain or withdraw')
    return name


def selected():
    assert sha(ROOT / PARENT['path']) == PARENT['sha256']
    return dict(checkpoint=dict(PARENT))


def comparison_authority():
    return dict(version=VERSION, parent=PARENT, arms=ARMS, execution_order=list(ARMS),
        accepted_updates_per_arm=LIMIT, evaluation_offsets=list(EVALUATIONS),
        maximum_total_learner_decisions=2 * LIMIT * 4423680,
        matching='Same750 model/Adam/groups/counters/four learner RNG/native-opponent RNG, '
                 'same deterministic fresh physical scenario bank/seed and zero recurrent hidden. '
                 'Independent processes; not continuation of750 physical episodes. '
                 'Entry identities are audited; no assertion of bit-identical subsequent trajectories.',
        variable='Only added uniform exploration loss coefficient: retain0.01 versus withdraw0.0. '
                 'Ordinary entropy0.001 and temperature2 remain in both. No reward change.',
        endpoint='Primary: total and paired-world goal difference at+30; also report scored and conceded '
                 'separately, wins, contacts, next-contact identity, standing-kickoff first contacts. '
                 '+10 is an intermediate report, not an adaptive stopping/selection point.',
        interpretation='A better final scoreline in one arm is descriptive evidence only: one training seed '
                       'and ten reused development matches, not statistical significance or an SSL claim. '
                       'Contact/entropy improvement alone is not success. Report ties and tradeoffs; '
                       'no automatic promotion, deployment, restart, hyperparameter change or extension.',
        stopping='Each arm stops after30 accepted updates. User STOP and numerical/corruption failures '
                 'stop work at the latest accepted checkpoint. KL remains telemetry only. '
                 'Failure prevents automatic launch of the other arm; investigate before any recovery.')


def authority(arm):
    arm_name(arm)
    selected()
    spec = deepcopy(prior.authority())
    for key in ('preceding_exploration_updates', 'calibration_sha256', 'diagnosis_sha256',
                'selection_sha256', 'parent_choice'):
        spec.pop(key, None)
    spec.update(version=VERSION + '_' + arm.upper(), parent=dict(PARENT), arm=arm,
        child_updates=LIMIT, evaluation_boundaries=list(EVALUATIONS),
        previous_arm_authority_sha256=content_hash(prior.authority()),
        comparison_authority_sha256=content_hash(comparison_authority()),
        decision_evidence={p: (text_sha(ROOT/p) if p.endswith('.md') else sha(ROOT/p))
                           for p in DECISION_FILES},
        hypothesis='Ongoing uniform-barrier pressure may impede consolidation of useful actions. '
                   'Compare continued pressure versus withdrawal from the same750 parent; cause unproven.',
        initialization=comparison_authority()['matching'],
        changes=comparison_authority()['variable'],
        budget='Maximum30 accepted updates per arm,132710400 learner decisions each; two arms only.',
        checkpoints='Entry0,first1,permanent10/30; alternating durable rolling every accepted update.',
        evaluation='Same ten complete native-v5 Nexto matches at+10/+30; raw argmax and standing kickoffs. '
                   'Compare both arms at equal offsets and both against the completed750 evaluation.',
        end=comparison_authority()['stopping'], success=comparison_authority()['interpretation'])
    spec['exploration_objective']['coefficient'] = ARMS[arm]
    return spec


def loss_for(arm):
    beta = ARMS[arm_name(arm)]

    def loss(model, data, index, config):
        return sequence_loss(model, data, index, config, coefficient=beta)
    return loss


def make_collector(env, model):
    collector = engine.DirectSkillNativeNextoCollector(env, model, seed=engine.base.SEED,
        nexto_sampling_mode='native_v5', nexto_seed=engine.NEXTO_SEED)
    selected()
    payload = torch.load(ROOT/PARENT['path'], map_location='cpu', weights_only=False)
    prior.restore_new_episode_rng(collector, payload['opponent_state'])
    return collector


@contextmanager
def configured_engine(arm):
    arm_name(arm)
    with prior.configured_engine():
        changes = dict(VERSION=VERSION+'_'+arm.upper(), OUT=OUT/arm, CKPTS=CKPTS/arm,
            EXTERNAL=EXTERNAL/arm, LIMIT=LIMIT, EVALUATIONS=EVALUATIONS,
            authority=lambda: authority(arm), verify=lambda: verify(arm), selection=selected,
            make_collector=make_collector,
            validate_resume=lambda p,h,s: prior.VALIDATE_RESUME(p,h,s,external=EXTERNAL/arm))
        old = {k: getattr(engine,k) for k in changes}
        old_loss, old_base = mixed.joint_sequence_loss, engine.base.joint_sequence_loss
        try:
            for k,v in changes.items():
                setattr(engine,k,v)
            mixed.joint_sequence_loss = engine.base.joint_sequence_loss = loss_for(arm)
            yield engine
        finally:
            mixed.joint_sequence_loss, engine.base.joint_sequence_loss = old_loss, old_base
            for k,v in old.items():
                setattr(engine,k,v)


def freeze():
    assert not (OUT/'comparison_package.json').exists()
    suites = ET.parse(OUT/'tests.xml').getroot().findall('testsuite')
    assert suites and sum(int(s.get('tests',0)) for s in suites) > 0
    assert all(int(s.get(k,0)) == 0 for s in suites for k in ('failures','errors','skipped'))
    for arm in ARMS:
        pre = json.loads((OUT/arm/'training_preflight.json').read_text())
        assert pre['parent'] == PARENT and pre['optimizer_steps'] == 0 and all(pre['checks'].values())
        assert not (OUT/arm/'training_package.json').exists()
    common = comparison_authority()
    write_json(OUT/'comparison_authority.json', common)
    for arm in ARMS:
        spec = authority(arm)
        write_json(OUT/arm/'training_authority.json', spec)
        write_json(OUT/arm/'training_package.json', dict(authority_sha256=content_hash(spec),
            comparison_authority_sha256=content_hash(common),
            scenario_sha256=scenario_hash(prior.prior.prior.scenarios(32768)),
            exploration_objective=spec['exploration_objective'],
            lineage=dict(parent=PARENT, arm=arm, comparison_authority_sha256=content_hash(common)),
            sources={p:text_sha(ROOT/p) for p in SOURCES},
            evidence={p.relative_to(ROOT).as_posix():sha(p) for p in
                      (OUT/'tests.xml', OUT/arm/'training_preflight.json')},
            decision_evidence=spec['decision_evidence']))
    write_json(OUT/'comparison_package.json', dict(authority_sha256=content_hash(common),
        arm_packages={arm:sha(OUT/arm/'training_package.json') for arm in ARMS},
        optimizer_steps_before_freeze=0))


def verify(arm):
    arm_name(arm)
    prior.verify()
    spec = authority(arm)
    package = json.loads((OUT/arm/'training_package.json').read_text())
    pair = json.loads((OUT/'comparison_package.json').read_text())
    assert json.loads((OUT/'comparison_authority.json').read_text()) == comparison_authority()
    assert pair['authority_sha256'] == package['comparison_authority_sha256'] == content_hash(comparison_authority())
    for name in ARMS:
        assert pair['arm_packages'][name] == sha(OUT/name/'training_package.json')
    assert json.loads((OUT/arm/'training_authority.json').read_text()) == spec
    assert package['authority_sha256'] == content_hash(spec)
    assert package['exploration_objective'] == spec['exploration_objective']
    for p,h in package['sources'].items():
        assert text_sha(ROOT/p) == h,p
    for p,h in package['decision_evidence'].items():
        assert (text_sha(ROOT/p) if p.endswith('.md') else sha(ROOT/p)) == h,p
    for p,h in package['evidence'].items():
        assert sha(ROOT/p) == h,p
    paths = set((*SOURCES,*DECISION_FILES,*package['evidence']))
    paths.update((OUT/n).relative_to(ROOT).as_posix() for n in ('comparison_authority.json','comparison_package.json'))
    paths.update((OUT/a/n).relative_to(ROOT).as_posix() for a in ARMS
                 for n in ('training_authority.json','training_package.json'))
    for p in sorted(paths):
        remote = subprocess.check_output(['git','show','origin/main:'+p],cwd=ROOT)
        assert remote.replace(b'\r\n',b'\n') == (ROOT/p).read_bytes().replace(b'\r\n',b'\n'),p
    return package


def pair_run():
    """Sequential workers; never resume an interrupted arm without explicit audit."""
    from benchmarks.report_direct_skills_exploration_ablation_v1 import audit, pair_report
    for arm in ARMS:
        verify(arm)
    for arm in ARMS:
        if (EXTERNAL/'STOP').exists():
            raise RuntimeError('Respect pair STOP')
        directory = EXTERNAL/arm
        state_path = directory/'campaign_state.json'
        if state_path.exists():
            state = json.loads(state_path.read_text())
            if state['status'] != 'complete_review' or state['branch_updates'] != LIMIT:
                raise RuntimeError('Existing unfinished arm requires explicit resume/audit: '+arm)
        else:
            directory.mkdir(parents=True, exist_ok=True)
            command = [sys.executable,str(Path(__file__).resolve()),'run','--arm',arm]
            with (directory/'stdout.log').open('ab') as stdout, (directory/'stderr.log').open('ab') as stderr:
                child = subprocess.Popen(command,cwd=ROOT,stdout=stdout,stderr=stderr)
                write_json(EXTERNAL/'pair_state.json',dict(utc=utc(),pid=os.getpid(),child_pid=child.pid,
                    arm=arm,status='running',comparison_authority_sha256=content_hash(comparison_authority())))
                while child.poll() is None:
                    if (EXTERNAL/'STOP').exists() and not (directory/'STOP').exists():
                        (directory/'STOP').write_text('Pair STOP requested; preserve accepted checkpoint.\n')
                    time.sleep(2)
                if child.returncode:
                    write_json(EXTERNAL/'pair_state.json',dict(utc=utc(),pid=os.getpid(),arm=arm,
                        status='worker_failed',exit_code=child.returncode))
                    raise RuntimeError('Arm failed; next arm not launched: '+arm)
            state = json.loads(state_path.read_text())
            if state['status'] != 'complete_review' or state['branch_updates'] != LIMIT:
                raise RuntimeError('Arm stopped before frozen completion: '+arm)
        for offset in (1,*EVALUATIONS):
            audit(arm,offset)
    result = pair_report(LIMIT)
    write_json(EXTERNAL/'pair_state.json',dict(utc=utc(),pid=os.getpid(),status='complete_review',
        comparison_authority_sha256=content_hash(comparison_authority()),
        report_sha256=sha(OUT/f'comparison_{LIMIT:06d}.json'),training_updates=2*LIMIT,
        final_checkpoint_hashes={arm:result['arms'][arm]['checkpoint']['sha256'] for arm in ARMS}))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('mode',choices=('preflight','freeze','verify','run','pair'))
    parser.add_argument('--arm',choices=tuple(ARMS))
    parser.add_argument('--resume',type=Path)
    parser.add_argument('--resume-sha256')
    args = parser.parse_args()
    torch.set_num_threads(8)
    if args.mode == 'freeze':
        freeze()
    elif args.mode == 'verify':
        for name in ([args.arm] if args.arm else ARMS):
            verify(name)
    elif args.mode == 'pair':
        if args.resume or args.resume_sha256 or args.arm:
            parser.error('Use run --arm for explicit recovery; pair never implicitly resumes')
        pair_run()
    else:
        if not args.arm:
            parser.error('--arm required')
        if (EXTERNAL/'STOP').exists():
            raise RuntimeError('Respect pair STOP')
        with engine.base.gpu_lease(), configured_engine(args.arm) as configured:
            configured.preflight() if args.mode == 'preflight' else configured.run(args)
