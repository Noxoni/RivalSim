"""Reviewed, bounded continuation from the completed exploration +30 checkpoint.

No PPO, reward, architecture, exploration or scenario change. New lineage
metadata and directories keep the completed original authority immutable.
"""
from __future__ import annotations

import argparse
from contextlib import contextmanager
import json
from pathlib import Path
import subprocess
import sys
import xml.etree.ElementTree as ET

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import torch

from benchmarks import run_direct_skills_log_barrier_v1 as prior
from benchmarks import run_direct_skills_native_nexto_v1 as engine
from benchmarks.run_rival2_fresh_ground_30hz_v1 import sha, write_json
from benchmarks.evaluate_direct_skills_native_nexto_v1 import text_sha
from rivalsim.fresh_ground_30hz import content_hash, scenario_hash

VERSION = 'RIVAL2_DIRECT_SKILLS_EXPLORATION_CONTINUATION_V1'
OUT = ROOT / 'results/rival2/direct_skills_exploration_continuation_v1'
CKPTS = ROOT / 'checkpoints/rival2/direct_skills_exploration_continuation_v1'
EXTERNAL = Path('G:/dev/RivalSim-runs/direct-skills-exploration-continuation-v1')
PARENT = dict(path='checkpoints/rival2/direct_skills_log_barrier_v1/child_000030.pt',
              sha256='16BF2B904785D49E0363B869AF953CFCCC62FF754286A307D386729AC691DDAD',
              accepted_updates=680)
LIMIT, EVALUATIONS = 70, (20, 45, 70)
SOURCES = tuple(dict.fromkeys((*prior.SOURCES,
    'benchmarks/run_direct_skills_exploration_continuation_v1.py',
    'benchmarks/report_direct_skills_exploration_continuation_v1.py',
    'tests/test_direct_skills_exploration_continuation_v1.py')))
DECISION_FILES = (
    'results/rival2/direct_skills_log_barrier_v1/completion.json',
    'results/rival2/direct_skills_log_barrier_v1/child_000030.json',
    'results/rival2/direct_skills_post_exploration_probe_v1/summary.json',
    'results/rival2/direct_skills_post_exploration_probe_v1/barrier_calibration.json',
)
VALIDATE_RESUME = prior.VALIDATE_RESUME


def selected():
    assert sha(ROOT / PARENT['path']) == PARENT['sha256']
    return dict(checkpoint=PARENT)


def authority():
    selected()
    previous = json.loads((prior.OUT / 'training_authority.json').read_text())
    completion = json.loads((prior.OUT / 'completion.json').read_text())
    assert completion['checkpoint']['sha256'] == PARENT['sha256']
    assert completion['between_evaluations']['15_to_30']['goal_difference_worlds']['improved'] == 10
    spec = dict(previous)
    spec.update(version=VERSION, parent=dict(PARENT), child_updates=LIMIT,
        evaluation_boundaries=list(EVALUATIONS), preceding_exploration_updates=30,
        original_control_parent=previous['parent'],
        previous_arm_authority_sha256=content_hash(previous),
        decision_evidence={p: sha(ROOT / p) for p in DECISION_FILES},
        hypothesis='Late recovery may continue now that exploration has broadened. '
                   'This is unproven; retain exact settings rather than attribute failure '
                   'to unsupported critic/exploration-gradient explanations.',
        initialization='Exact completed exploration +30 (cumulative680) model/Adam/groups/counters/four RNG '
                       'and native-opponent RNG. Fresh physical episodes and zero recurrent hidden; '
                       'controller caches explicitly reset. Not exact physical-world continuation.',
        changes='Only continued accepted learning and explicit parent/authority metadata. '
                'All PPO/loss/reward/scenario/cadence/controller/architecture settings unchanged.',
        budget='Maximum70 additional accepted updates,309657600 learner decisions. '
               'Original exploration offsets50/75/100; cumulative700/725/750. No automatic extension.',
        checkpoints='Entry0,first1,permanent20/45/70; alternating durable rolling each update.',
        evaluation='Same ten full native-v5 Nexto matches at child20/45/70. Compare both '
                   'immediate parent680 and original control650. No new benchmark or easier starts.',
        end='Stop after70 additional accepted updates for review; respect user STOP and numerical safety. '
            'A positive late trend is not deployment approval or demonstrated SSL.',
        success='Improved actual scoring/conceding and ball acquisition versus immediate parent, '
                'with progress toward surpassing original control650. No promotion from entropy or finite checks.')
    return spec


def make_collector(env, model):
    collector = engine.DirectSkillNativeNextoCollector(env, model, seed=engine.base.SEED,
        nexto_sampling_mode='native_v5', nexto_seed=engine.NEXTO_SEED)
    # The original engine restores this on a same-arm resume only. At this new
    # arm's entry the parent already has the correct native-v5 RNG lineage too.
    selected()
    payload = torch.load(ROOT / PARENT['path'], map_location='cpu', weights_only=False)
    restore_new_episode_rng(collector, payload['opponent_state'])
    return collector


def restore_new_episode_rng(collector, state):
    """Keep RNG/counters without loading obsolete per-world physical caches.

    This also supports the smaller no-update preflight. Same-arm production
    resumes still use the original exact-size validator in the base runner.
    """
    if (state.get('version') != engine.COLLECTOR_VERSION
            or state.get('controller_version') != engine.CONTROLLER
            or state.get('nexto_sampling_mode') != collector.nexto_sampling_mode
            or state.get('nexto_seed') != collector.nexto_seed
            or state.get('nexto_probability') != .5):
        raise ValueError('Wrong native opponent RNG lineage')
    native = state['native_nexto']
    current = collector.nexto.checkpoint_state()
    if any(native.get(k) != current[k] for k in ('version', 'model_sha256', 'sampling_mode')):
        raise ValueError('Wrong native controller identity')
    counts = [native[k] for k in ('inference_calls', 'observation_builds')]
    if any(type(x) is not int or x < 0 for x in counts):
        raise ValueError('Invalid native controller counters')
    opponent_rng = torch.Generator(device=collector.env.device)
    native_rng = torch.Generator(device=collector.env.device)
    opponent_rng.set_state(state['generator_state'].cpu())
    native_rng.set_state(native['rng_state'].cpu())
    # Both RNG payloads validated before mutating either live generator.
    collector.opponent_generator.set_state(opponent_rng.get_state())
    collector.nexto.generator.set_state(native_rng.get_state())
    collector.nexto.inference_calls, collector.nexto.observation_builds = counts
    collector.nexto.activate(torch.ones_like(collector.is_nexto))
    collector.assign(torch.ones_like(collector.is_nexto))


@contextmanager
def configured_engine():
    with prior.configured_engine():
        changes = dict(VERSION=VERSION, OUT=OUT, CKPTS=CKPTS, EXTERNAL=EXTERNAL,
            LIMIT=LIMIT, EVALUATIONS=EVALUATIONS, authority=authority, verify=verify,
            selection=selected, make_collector=make_collector,
            validate_resume=lambda p, h, s: VALIDATE_RESUME(p, h, s, external=EXTERNAL))
        old = {name: getattr(engine, name) for name in changes}
        try:
            for name, value in changes.items():
                setattr(engine, name, value)
            yield engine
        finally:
            for name, value in old.items():
                setattr(engine, name, value)


def freeze():
    assert not (OUT / 'training_package.json').exists()
    pre = json.loads((OUT / 'training_preflight.json').read_text())
    assert pre['parent'] == PARENT and pre['optimizer_steps'] == 0 and all(pre['checks'].values())
    suites = ET.parse(OUT / 'tests.xml').getroot().findall('testsuite')
    assert suites and all(int(s.get(k, 0)) == 0 for s in suites for k in ('failures', 'errors', 'skipped'))
    spec = authority()
    write_json(OUT / 'training_authority.json', spec)
    write_json(OUT / 'training_package.json', dict(authority_sha256=content_hash(spec),
        scenario_sha256=scenario_hash(prior.prior.scenarios(32768)),
        exploration_objective=spec['exploration_objective'],
        lineage=dict(parent=PARENT, original_control_parent=spec['original_control_parent'],
                     preceding_authority=spec['previous_arm_authority_sha256']),
        sources={p: text_sha(ROOT / p) for p in SOURCES},
        evidence={name: sha(OUT / name) for name in ('training_preflight.json', 'tests.xml')},
        decision_evidence=spec['decision_evidence']))


def verify():
    prior.verify()
    package = json.loads((OUT / 'training_package.json').read_text())
    assert json.loads((OUT / 'training_authority.json').read_text()) == authority()
    assert package['authority_sha256'] == content_hash(authority())
    assert package['exploration_objective'] == authority()['exploration_objective']
    for path, digest in package['sources'].items():
        assert text_sha(ROOT / path) == digest, path
    for path, digest in package['decision_evidence'].items():
        assert sha(ROOT / path) == digest, path
    for name, digest in package['evidence'].items():
        assert sha(OUT / name) == digest, name
    paths = (*SOURCES, *DECISION_FILES, *[(OUT / name).relative_to(ROOT).as_posix()
        for name in ('training_authority.json', 'training_package.json', *package['evidence'])])
    for path in paths:
        remote = subprocess.check_output(['git', 'show', 'origin/main:' + path], cwd=ROOT)
        assert remote.replace(b'\r\n', b'\n') == (ROOT / path).read_bytes().replace(b'\r\n', b'\n'), path
    return package


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('mode', choices=('preflight', 'freeze', 'verify', 'run'))
    parser.add_argument('--resume', type=Path)
    parser.add_argument('--resume-sha256')
    args = parser.parse_args()
    torch.set_num_threads(8)
    if args.mode == 'freeze':
        freeze()
    elif args.mode == 'verify':
        verify()
    else:
        with engine.base.gpu_lease(), configured_engine() as configured:
            configured.preflight() if args.mode == 'preflight' else configured.run(args)
