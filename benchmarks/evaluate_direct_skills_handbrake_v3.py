"""Prospectively frozen600/675 comparison after handbrake reset correction."""
from __future__ import annotations

import argparse
import gc
import hashlib
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
BASE = ROOT / 'results/rival2/direct_skills_v1'
OUTPUT = BASE / 'handbrake_reset_v3'
METHOD = 'RIVAL2_STANDARD_KICKOFF_HANDBRAKE_RESET_V3'
MODIFIED = ('rivalsim/kernels/rival2.py', 'rivalsim/rival2_env.py')
EXTRA_SOURCES = ('benchmarks/evaluate_direct_skills_handbrake_v3.py',
                 'benchmarks/verify_direct_skills_handbrake_reset.py',
                 'benchmarks/report_rival2_direct_skills_matches.py',
                 'tests/test_direct_skills_handbrake_v3.py')


def text_sha(path):
    return hashlib.sha256(path.read_bytes().replace(b'\r\n', b'\n')).hexdigest().upper()


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest().upper()


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':')).encode()).hexdigest().upper()


def write(path, value):
    assert not path.exists(), path
    path.write_text(json.dumps(value, sort_keys=True, indent=2, allow_nan=False) + '\n')


def freeze():
    import xml.etree.ElementTree as ET
    old = json.loads((BASE / 'package.json').read_text())
    assert sorted(p for p, h in old['sources'].items() if text_sha(ROOT / p) != h) == sorted(MODIFIED)
    before = json.loads((OUTPUT / 'before.json').read_text())
    after = json.loads((OUTPUT / 'after.json').read_text())
    assert after['exact_before_after'] and before['archive_sha256'] == after['archive_sha256']
    assert before['arrays'] == after['arrays'] == 4368
    suites = ET.parse(OUTPUT / 'focused_tests.xml').getroot().findall('testsuite')
    assert suites and all(int(s.get(k, 0)) == 0 for s in suites for k in ('failures', 'errors', 'skipped'))
    checkpoints = {}
    for offset in (600, 675):
        path = ROOT / f'checkpoints/rival2/direct_skills_v1/plus_{offset:06d}.pt'
        saved = json.loads((BASE / f'full_match_{offset:06d}.json').read_text())
        assert sha(path) == saved['checkpoint']['sha256']
        checkpoints[str(offset)] = dict(path=path.relative_to(ROOT).as_posix(), sha256=sha(path))
    authority = dict(
        version=METHOD, parent_runtime_package_sha256=digest(old),
        parent_authority_sha256=old['authority_sha256'],
        prospective_plan_commit='732c008fb8548e35e7d31c87e3453250f450a6b4',
        prior_sources=old['sources'], sources={p: text_sha(ROOT / p) for p in (*old['sources'], *EXTRA_SOURCES)},
        changed_runtime_sources=list(MODIFIED), checkpoints=checkpoints,
        semantics='Selected standard-kickoff car handbrake smoothing set to0. No other physics or timing edit.',
        training_reset_parity=dict(arrays=4368, exact=True, original_and_shooting_banks=True,
                                   archive_sha256=after['archive_sha256']),
        evidence={p: text_sha(OUTPUT / p) for p in ('before.json', 'after.json', 'focused_tests.xml', 'AUTHORITY.md')},
        method=dict(matches_per_checkpoint=10, regulation_ticks=36000, overtime_cap_ticks=14400,
                    rival_hz=30, physics_hz=120, nexto_hz=15, initial_layouts=[0,1,2,3,4,0,1,2,3,4],
                    rival_sides=[0,0,0,0,0,1,1,1,1,1], no_touch_resets=False,
                    deterministic=True, native_goal_resets=True, recurrent_hidden_reset=True),
        optimizer_steps=0, checkpoint_selection=False, reward_changes=False,
        legacy_package_and_results_unchanged=True,
    )
    write(OUTPUT / 'authority.json', authority)
    print(json.dumps(dict(authority_sha256=digest(authority), changed=MODIFIED)))


def verify(published=True):
    authority = json.loads((OUTPUT / 'authority.json').read_text())
    old = json.loads((BASE / 'package.json').read_text())
    assert authority['version'] == METHOD and digest(old) == authority['parent_runtime_package_sha256']
    assert old['sources'] == authority['prior_sources']
    for p, h in authority['sources'].items():
        assert text_sha(ROOT / p) == h, p
    for p, h in authority['evidence'].items():
        assert text_sha(OUTPUT / p) == h, p
    assert sha(OUTPUT / 'native_reference.npz') == authority['training_reset_parity']['archive_sha256']
    for item in authority['checkpoints'].values():
        assert sha(ROOT / item['path']) == item['sha256']
    if published:
        for p in (*authority['sources'], *(str((OUTPUT / p).relative_to(ROOT).as_posix()) for p in authority['evidence']),
                  (OUTPUT / 'authority.json').relative_to(ROOT).as_posix()):
            remote = subprocess.check_output(['git', 'show', f'origin/main:{p}'], cwd=ROOT)
            assert remote.replace(b'\r\n', b'\n') == (ROOT / p).read_bytes().replace(b'\r\n', b'\n'), p
    return authority


def run():
    import torch
    from benchmarks.direct_skills_eval_stream import owned_match_stream
    from benchmarks.evaluate_rival2_ssl_entity_full_match import CandidateMatchRunner, summarize
    from benchmarks.run_rival2_fresh_ground_30hz_v1 import tensor_hash, utc
    from benchmarks.run_rival2_ssl_entity_continuation import gpu_lease
    from benchmarks.report_rival2_ssl_entity_match_followup import reduce
    from benchmarks.report_rival2_direct_skills_matches import compare
    authority = verify()
    with gpu_lease(), owned_match_stream():
        for offset in (675, 600):
            output = OUTPUT / f'full_match_{offset:06d}.json'
            identity = authority['checkpoints'][str(offset)]
            if output.exists():
                saved = json.loads(output.read_text())
                assert saved['checkpoint']['sha256'] == identity['sha256']
                assert saved['runtime_package_sha256'] == digest(authority)
                assert all(reduce(output)['integrity'].values())
                continue
            runner = CandidateMatchRunner(ROOT / identity['path'], identity['sha256'], entity=True)
            elapsed = runner.run_ticks(36000).seconds
            for _ in range(24):
                if bool(runner.phase_status()['done'].all()):
                    break
                elapsed += runner.run_ticks(600).seconds
            raw = runner.export()['raw']
            assert not raw['goal_overflow'].any()
            assert tensor_hash(runner.rival_policy.state_dict()) == runner.model_hash_before
            assert sha(ROOT / identity['path']) == identity['sha256']
            write(output, dict(
                utc=utc(), accepted_updates=offset, checkpoint=dict(**identity, accepted_updates=offset),
                authority_sha256=authority['parent_authority_sha256'], runtime_package_sha256=digest(authority),
                match_reset_version=METHOD, summary=summarize(raw), raw={k:v.tolist() for k,v in raw.items()},
                wall_seconds=elapsed, hidden_resets=runner.hidden_reset_count.cpu().tolist(),
                optimizer_steps=0, model_unchanged=True, checkpoint_unchanged=True,
                interpretation='Same handbrake-corrected method for600/675. Old-vs-new method at a fixed checkpoint is runtime effect, not learning.',
            ))
            reduced = reduce(output)
            assert all(reduced['integrity'].values())
            write(OUTPUT / f'full_match_{offset:06d}_integrity.json', reduced)
            print(json.dumps(dict(offset=offset, summary=summarize(raw))), flush=True)
            del runner
            gc.collect()
            torch.cuda.empty_cache()
    verify(published=False)
    result = compare(OUTPUT / 'full_match_000600.json', OUTPUT / 'full_match_000675.json')
    destination = OUTPUT / 'comparison_000600_to_000675.json'
    if destination.exists():
        assert json.loads(destination.read_text()) == result
    else:
        write(destination, result)
    print(json.dumps(result['goal_difference_cases']))


if __name__ == '__main__':
    import torch
    torch.set_num_threads(8)
    parser = argparse.ArgumentParser()
    parser.add_argument('mode', choices=('freeze', 'run'))
    args = parser.parse_args()
    freeze() if args.mode == 'freeze' else run()
