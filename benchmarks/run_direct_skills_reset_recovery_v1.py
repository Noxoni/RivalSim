"""Finite shooting-pressure branch from measured stronger +600; no old overwrite."""
from __future__ import annotations

import argparse
import gc
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import traceback
from dataclasses import asdict

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import torch
from benchmarks import evaluate_direct_skills_handbrake_v3 as runtime
from benchmarks import run_rival2_direct_skills_v1 as base
from benchmarks.run_direct_skills_shooting_progress_v1 import load, adam_hash
from benchmarks.run_rival2_fresh_ground_30hz_v1 import append_json, sha, tensor_hash, utc, write_json
from benchmarks.report_rival2_ssl_entity_match_followup import reduce
from rivalsim.direct_skills_exploration_v1 import TEMPERATURE
from rivalsim.direct_skills_shooting_curriculum_v1 import scenarios, curriculum_authority
from rivalsim.fresh_ground_30hz import content_hash, scenario_hash, ppo_config

VERSION = 'RIVAL2_DIRECT_SKILLS_RESET_RECOVERY_V1'
OUT = ROOT / 'results/rival2/direct_skills_reset_recovery_v1'
CKPTS = ROOT / 'checkpoints/rival2/direct_skills_reset_recovery_v1'
EXTERNAL = Path('G:/dev/RivalSim-runs/direct-skills-reset-recovery-v1')
SOURCE = base.CHECKPOINTS / 'plus_000600.pt'
SOURCE_SHA = '8AC192B4154223DB7CE82B41C77DC475D63B6AFB260EF0B9B9AC4455F6C6F6B2'
START, LIMIT = 600, 25
EXTRA = (
    'benchmarks/run_direct_skills_reset_recovery_v1.py',
    'tests/test_direct_skills_reset_recovery.py',
    'benchmarks/run_direct_skills_shooting_progress_v1.py',
    'rivalsim/direct_skills_shooting_curriculum_v1.py',
    'rivalsim/direct_skills_exploration_v1.py',
)
RNG = ('policy_generator_state', 'shuffle_generator_state', 'torch_cpu_rng_state', 'torch_cuda_rng_state')


def authority():
    return dict(
        version=VERSION, parent=SOURCE.relative_to(ROOT).as_posix(), parent_sha256=SOURCE_SHA,
        parent_direct_updates=START, child_updates=LIMIT,
        selection='Prospective branch after corrected V3 comparison:600 wins8/10 versus675 wins1/10; nine cases regress. Preserve all old models. Not SSL proof.',
        evidence='results/rival2/direct_skills_v1/handbrake_reset_v3/RESULTS.md',
        initialization='Exact600 weights, Adam moments/counters, four RNG; fresh physical episodes and zero recurrent hidden. Not fresh optimizer or675 continuation.',
        root_authority_sha256=content_hash(base.authority()),
        runtime_authority_sha256=runtime.digest(json.loads((runtime.OUTPUT / 'authority.json').read_text())),
        method=runtime.METHOD, curriculum=curriculum_authority(),
        training_change='Already tested intermediate keeper pressure on half finishing starts, now from600; no other learning change.',
        ppo=asdict(ppo_config()), training_temperature=TEMPERATURE,
        worlds=32768, physics_hz=120, policy_hz=30, nexto_hz=15,
        reward_change=False, architecture_change=False, fresh_optimizer=False,
        source_mix='30%natural/20%challenge/20%finishing/15%defense/15%kickoff starts; unchanged base direct/PBRS reward per role.',
        opponents='50%Nexto/50%selfplay worlds; exactly one-third learner samples against Nexto.',
        safety='Finite model/gradient/Adam and corruption rollback unchanged; KL telemetry ONLY.',
        checkpoints='Separate branch: entry0, first1, every rolling accepted update, permanent child25; cumulative ancestry625 at boundary.',
        evaluation='Original64cases per skill and original ten V3-method Nexto matches at child25; no easier evaluation bank. Same raw argmax; no-touch resets disabled in fullmatch.',
        end='Stop at25 for actual goals/match review. Retain parent if regression; no automatic extension or deployment.',
    )


def freeze():
    runtime.verify()
    assert sha(SOURCE) == SOURCE_SHA
    assert not (OUT / 'package.json').exists()
    bank = scenarios(32768)
    pre = json.loads((OUT / 'preflight.json').read_text())
    assert all(pre['checks'].values()) and pre['optimizer_steps'] == 0
    import xml.etree.ElementTree as ET
    suites = ET.parse(OUT / 'tests.xml').getroot().findall('testsuite')
    assert suites and all(int(s.get(k, 0)) == 0 for s in suites for k in ('failures', 'errors', 'skipped'))
    write_json(OUT / 'authority.json', authority())
    evidence = ['preflight.json', 'tests.xml', 'AUTHORITY.md']
    decision = 'results/rival2/direct_skills_v1/handbrake_reset_v3/RESULTS.md'
    write_json(OUT / 'package.json', dict(
        authority_sha256=content_hash(authority()), scenario_sha256=scenario_hash(bank),
        sources={p: runtime.text_sha(ROOT / p) for p in EXTRA},
        evidence={str((OUT / n).relative_to(ROOT).as_posix()): runtime.text_sha(OUT / n) for n in evidence},
        decision={decision: runtime.text_sha(ROOT / decision)},
    ))
    print(json.dumps(dict(authority_sha256=content_hash(authority()), scenario_sha256=scenario_hash(bank))))


def verify(published=True):
    fixed = runtime.verify(published=published)
    assert sha(SOURCE) == SOURCE_SHA
    spec = authority()
    assert json.loads((OUT / 'authority.json').read_text()) == spec
    package = json.loads((OUT / 'package.json').read_text())
    assert package['authority_sha256'] == content_hash(spec)
    paths = {}
    for key in ('sources', 'evidence', 'decision'):
        paths.update(package[key])
    for p, digest in paths.items():
        assert runtime.text_sha(ROOT / p) == digest, p
    if published:
        for p in (*paths, *((OUT / n).relative_to(ROOT).as_posix() for n in ('package.json', 'authority.json'))):
            remote = subprocess.check_output(['git', 'show', f'origin/main:{p}'], cwd=ROOT)
            assert remote.replace(b'\r\n', b'\n') == (ROOT / p).read_bytes().replace(b'\r\n', b'\n'), p
    return package, fixed


def preflight():
    runtime.verify()
    assert sha(SOURCE) == SOURCE_SHA
    payload = torch.load(SOURCE, map_location='cpu', weights_only=False)
    model, optimizer = load(payload)
    before, adam_before = tensor_hash(model.state_dict()), adam_hash(optimizer)
    env = base.DirectSkillsEnv(1024, base.COLLISION, device='cuda:0', seed=base.SEED,
                              ssl_foundation_scenarios=scenarios(1024))
    collector = base.DirectSkillCollector(env, model, seed=base.SEED)
    rollout = collector.collect()
    data = base.mixed_sequence_data(rollout, ppo_config())
    indices = data['train_mask'].any(1).nonzero().flatten()[:728]
    model.train()
    loss, _ = base.joint_sequence_loss(model, data, indices, ppo_config())
    loss.backward()
    grad = torch.nn.utils.clip_grad_norm_(model.parameters(), .5, error_if_nonfinite=True)
    optimizer.zero_grad(set_to_none=True)
    model.isolated_value(data['observations'][indices]).sum().backward()
    isolated = all(p.grad is None or not bool(p.grad.any()) for n, p in model.named_parameters()
                   if not n.startswith('critic.'))
    optimizer.zero_grad(set_to_none=True)
    checks = dict(
        exact600_parent=payload['accepted_updates'] == START and sha(SOURCE) == SOURCE_SHA,
        unchanged_model=before == tensor_hash(model.state_dict()),
        unchanged_adam=adam_before == adam_hash(optimizer),
        no_optimizer_step={int(s['step']) for s in optimizer.state.values()} == {payload['cumulative_optimizer_steps']},
        finite_model_adam=base.finite_model_and_optimizer(model, optimizer),
        finite_loss_gradient=bool(torch.isfinite(loss)) and bool(torch.isfinite(grad)),
        critic_isolated=isolated,
        exact_action_targets=torch.equal(model.action_table[rollout.action_indices][rollout.train_mask], rollout.actions[rollout.train_mask]),
        nexto_one_third=collector.last_metrics['nexto_training_sample_count'] * 3 == collector.last_metrics['trainable_agent_samples'],
        no_mechanics_hotpath=env.world.gameplay_v3 is None and env.world.gameplay_120 is None,
    )
    result = dict(utc=utc(), checks=checks, optimizer_steps=0, worlds=1024, rollout_decisions=90,
                  gradient_norm=float(grad), parent_sha256=SOURCE_SHA, model_sha256=before,
                  adam_sha256=adam_before, adam_steps=payload['cumulative_optimizer_steps'],
                  training=collector.last_metrics)
    assert all(checks.values()), checks
    write_json(OUT / 'preflight.json', result)
    print(json.dumps(dict(checks=checks, adam_steps=result['adam_steps'])), flush=True)


def validate_resume(path, digest, external=EXTERNAL):
    if (external / 'STOP').exists():
        raise RuntimeError('Respect STOP; no automatic removal')
    latest_path = external / 'latest.json'
    if not latest_path.exists():
        if path is not None or digest is not None:
            raise RuntimeError('Fresh branch must use frozen600 parent, not arbitrary resume')
        if (external / 'campaign_state.json').exists():
            raise RuntimeError('State without checkpoint requires operational audit')
        return SOURCE, SOURCE_SHA, 0
    latest = json.loads(latest_path.read_text())
    step = latest['branch_updates']
    if type(step) is not int or not 0 <= step <= LIMIT:
        raise RuntimeError('Outside finite branch')
    if path is None or not digest or digest.upper() != latest['sha256']:
        raise RuntimeError('Explicit latest accepted resume required; no older fallback')
    if Path(path).resolve() != Path(latest['path']).resolve() or sha(Path(path)) != digest.upper():
        raise RuntimeError('Resume path/hash mismatch')
    state = json.loads((external / 'campaign_state.json').read_text())
    if state['branch_updates'] not in (step, step-1) or state['recovery_authority_sha256'] != content_hash(authority()):
        raise RuntimeError('State/checkpoint authority discrepancy')
    if state['status'] in ('complete_review', 'nonfinite_or_runtime_failure'):
        raise RuntimeError('Intentional review or failure requires audit, not blind retry')
    return Path(path), digest.upper(), step


def run(args):
    package, fixed = verify()
    source, digest, child = validate_resume(args.resume, args.resume_sha256)
    assert sha(source) == digest
    payload = torch.load(source, map_location='cpu', weights_only=False)
    assert payload['accepted_updates'] == START + child
    assert payload['parent_sha256'] == base.PARENT_SHA
    assert payload['authority_sha256'] == content_hash(base.authority())
    if source != SOURCE:
        assert payload['recovery_authority_sha256'] == content_hash(authority())
        assert payload['recovery_parent_sha256'] == SOURCE_SHA
    model, optimizer = load(payload)
    samples, ticks, adam_steps = (payload[k] for k in ('direct_skill_samples', 'direct_skill_physics_ticks', 'cumulative_optimizer_steps'))
    bank = scenarios(32768)
    assert scenario_hash(bank) == package['scenario_sha256']
    env = base.DirectSkillsEnv(32768, base.COLLISION, device='cuda:0', seed=base.SEED, ssl_foundation_scenarios=bank)
    collector = base.DirectSkillCollector(env, model, seed=base.SEED)
    collector.generator.set_state(payload[RNG[0]].cpu())
    shuffle = torch.Generator(device='cuda:0')
    shuffle.set_state(payload[RNG[1]].cpu())
    torch.set_rng_state(payload[RNG[2]].cpu())
    torch.cuda.set_rng_state(payload[RNG[3]].cpu())
    latest = None

    def state(status, **kwargs):
        write_json(EXTERNAL / 'campaign_state.json', dict(utc=utc(), pid=os.getpid(), status=status,
            branch_updates=child, accepted_updates=START+child, cumulative_optimizer_steps=adam_steps,
            latest_checkpoint=latest, recovery_authority_sha256=content_hash(authority()), review_boundary=LIMIT, **kwargs))

    def save(path, immutable=False):
        p = dict(payload)
        p.update(model=model.state_dict(), optimizer=optimizer.state_dict(),
            recovery_authority_sha256=content_hash(authority()), recovery_parent_sha256=SOURCE_SHA,
            recovery_package=package, recovery_branch_updates=child,
            prior_exploration_package=payload.get('exploration_package'),
            accepted_updates=START+child, direct_skill_samples=samples, direct_skill_physics_ticks=ticks,
            new_agent_samples=1676472162+samples, new_physics_ticks=3456368640+ticks,
            cumulative_optimizer_steps=adam_steps, fresh_optimizer=False,
            runtime_contract_hashes=env.contract_hashes, evaluation_runtime_authority_sha256=runtime.digest(fixed),
            policy_generator_state=collector.generator.get_state(), shuffle_generator_state=shuffle.get_state(),
            torch_cpu_rng_state=torch.get_rng_state(), torch_cuda_rng_state=torch.cuda.get_rng_state(),
            opponent_state=collector.opponent_checkpoint_state(), last_training_metrics=collector.last_metrics,
            resume_count=payload.get('resume_count', 0)+1,
            scenario_reset_note=authority()['initialization'], effective_training_scenario_sha256=package['scenario_sha256'],
            training_curriculum_authority_sha256=content_hash(curriculum_authority()),
            training_distribution=dict(temperature=TEMPERATURE, on_policy_sampling_and_likelihood=True, deterministic_evaluation='raw argmax'))
        if immutable and path.exists():
            stored = torch.load(path, map_location='cpu', weights_only=False)
            assert stored['recovery_authority_sha256'] == content_hash(authority())
            assert stored['recovery_branch_updates'] == child
            assert tensor_hash(stored['model']) == tensor_hash(p['model'])
        else:
            path.parent.mkdir(parents=True, exist_ok=True)
            temp = path.with_suffix('.pt.tmp')
            with temp.open('wb') as stream:
                torch.save(p, stream); stream.flush(); os.fsync(stream.fileno())
            os.replace(temp, path)
        return dict(path=str(path), sha256=sha(path), accepted_updates=START+child, branch_updates=child)

    def checkpoint():
        result = save(EXTERNAL / f'rolling_{child % 2}.pt')
        write_json(EXTERNAL / 'latest.json', result)
        return result

    def evaluate():
        state('evaluating')
        identity = save(CKPTS / f'child_{child:06d}.pt', immutable=True)
        cpu, cuda = torch.get_rng_state(), torch.cuda.get_rng_state()
        common = dict(utc=utc(), accepted_updates=START+child, branch_updates=child, checkpoint=identity,
            optimizer_steps=0, authority_sha256=content_hash(base.authority()), recovery_authority_sha256=content_hash(authority()))
        skill_path = OUT / f'evaluation_child_{child:06d}.json'
        if not skill_path.exists():
            write_json(skill_path, dict(common, skills=base.skill_evaluation(model)))
        match_path = OUT / f'full_match_child_{child:06d}.json'
        if not match_path.exists():
            with base.owned_match_stream():
                runner = base.CandidateMatchRunner(Path(identity['path']), identity['sha256'], entity=True)
                elapsed = runner.run_ticks(base.REGULATION_TICKS).seconds
                for _ in range(base.OVERTIME_CAP_TICKS // 600):
                    if bool(runner.phase_status()['done'].all()):
                        break
                    elapsed += runner.run_ticks(600).seconds
                raw = runner.export()['raw']
                assert not raw['goal_overflow'].any()
                assert tensor_hash(runner.rival_policy.state_dict()) == runner.model_hash_before
                assert sha(Path(identity['path'])) == identity['sha256']
                write_json(match_path, dict(common, match_reset_version=runtime.METHOD,
                    runtime_package_sha256=runtime.digest(fixed), summary=base.summarize(raw),
                    raw={k: v.tolist() for k,v in raw.items()}, wall_seconds=elapsed,
                    hidden_resets=runner.hidden_reset_count.cpu().tolist(), model_unchanged=True, checkpoint_unchanged=True))
                print('MATCH_EVAL '+json.dumps(base.summarize(raw)), flush=True)
                del runner
        saved = json.loads(match_path.read_text())
        assert saved['checkpoint']['sha256'] == identity['sha256']
        integrity = reduce(match_path)
        assert all(integrity['integrity'].values())
        write_json(OUT / f'integrity_child_{child:06d}.json', integrity)
        torch.set_rng_state(cpu); torch.cuda.set_rng_state(cuda)
        gc.collect(); torch.cuda.empty_cache()

    try:
        state('initializing')
        if child == 0:
            entry = save(CKPTS / 'entry_000000.pt', immutable=True)
            stored = torch.load(entry['path'], map_location='cpu', weights_only=False)
            checks = dict(exact_model=tensor_hash(stored['model']) == tensor_hash(payload['model']),
                exact_optimizer=all(torch.equal(stored['optimizer']['state'][i][k], v.cpu())
                    for i,s in payload['optimizer']['state'].items() for k,v in s.items()),
                exact_optimizer_groups=stored['optimizer']['param_groups'] == payload['optimizer']['param_groups'],
                exact_four_rng=all(torch.equal(stored[k], payload[k].cpu()) for k in RNG),
                exact_counters=all(stored[k] == payload[k] for k in ('accepted_updates','direct_skill_samples','direct_skill_physics_ticks','cumulative_optimizer_steps')),
                source_unchanged=sha(SOURCE) == SOURCE_SHA)
            assert all(checks.values()), checks
            write_json(OUT / 'entry_integrity.json', dict(utc=utc(), checks=checks, optimizer_steps=0, checkpoint=entry))
            del stored
        latest = checkpoint()
        while child < LIMIT and not (EXTERNAL / 'STOP').exists():
            state('rollout')
            torch.cuda.reset_peak_memory_stats()
            start = time.monotonic(); rollout = collector.collect(); rollout_seconds = time.monotonic()-start
            state('optimizing')
            start = time.monotonic()
            ppo = base.mixed_joint_ppo_update(model, optimizer, rollout, ppo_config(), shuffle)
            ppo_seconds = time.monotonic()-start
            child += 1
            adam_steps += ppo['optimizer_steps']
            samples += collector.last_metrics['trainable_agent_samples']
            ticks += collector.last_metrics['physical_physics_ticks']
            del rollout
            latest = checkpoint()
            if child == 1:
                save(CKPTS / 'first_000001.pt', immutable=True)
            row = dict(utc=utc(), branch_updates=child, accepted_updates=START+child,
                cumulative_optimizer_steps=adam_steps, samples=samples, physical_ticks=ticks,
                ppo=ppo, training=collector.last_metrics, rollout_seconds=rollout_seconds, ppo_seconds=ppo_seconds,
                cuda_peak_allocated_bytes=torch.cuda.max_memory_allocated(), recovery_authority_sha256=content_hash(authority()))
            append_json(EXTERNAL / 'training_curve.jsonl', row)
            print('ACCEPTED '+json.dumps(row), flush=True)
        if child == LIMIT:
            evaluate()
            state('complete_review')
        else:
            latest = save(CKPTS / f'paused_child_{child:06d}.pt', immutable=True)
            state('stopped_at_accepted_boundary')
    except Exception as exc:
        failure = dict(utc=utc(), exception=repr(exc), traceback=traceback.format_exc(), latest_checkpoint=latest, branch_updates=child)
        write_json(EXTERNAL / 'failure.json', failure)
        state('nonfinite_or_runtime_failure', failure=failure)
        raise


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('mode', choices=('preflight','freeze','verify','run'))
    parser.add_argument('--resume', type=Path)
    parser.add_argument('--resume-sha256')
    args = parser.parse_args()
    torch.set_num_threads(8)
    if args.mode == 'freeze':
        freeze()
    elif args.mode == 'verify':
        verify()
    else:
        with base.gpu_lease():
            preflight() if args.mode == 'preflight' else run(args)
