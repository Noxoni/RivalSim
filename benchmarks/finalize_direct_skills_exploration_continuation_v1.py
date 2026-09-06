"""CPU-only closeout of the 70-update continuation, never a training entrypoint."""
from __future__ import annotations

import json
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import torch

from benchmarks import report_direct_skills_exploration_continuation_v1 as report
from benchmarks.finalize_direct_skills_native_nexto_v1 import validate_terminal
from benchmarks.report_direct_skills_native_nexto_v1 import canonical, compare_records, save_once, sha


def active_campaign_pids():
    command = (
        "@(Get-CimInstance Win32_Process | Where-Object { "
        "$_.Name -match '^python(w)?\\.exe$' -and "
        "$_.CommandLine -match 'run_direct_skills_exploration_continuation_v1\\.py' "
        "} | Select-Object -ExpandProperty ProcessId) | ConvertTo-Json -Compress"
    )
    output = subprocess.check_output(
        ['powershell', '-NoProfile', '-NonInteractive', '-Command', command], text=True
    ).strip()
    value = json.loads(output) if output else []
    return value if isinstance(value, list) else [value]


def terminal_checks(state, latest, rows, spec, active_pids):
    validate_terminal(state, latest, rows, spec)
    report.validate_rows(rows, 70, spec)
    checks = dict(
        no_active_worker=not active_pids,
        correct_version=spec['version'] == 'RIVAL2_DIRECT_SKILLS_EXPLORATION_CONTINUATION_V1',
        frozen_boundary=spec['child_updates'] == 70,
        all_evaluations_required=spec['evaluation_boundaries'] == [20, 45, 70],
        parent_680=spec['parent']['accepted_updates'] == 680,
        original_control_650=spec['original_control_parent']['accepted_updates'] == 650,
        cumulative_updates=state['accepted_updates'] == 750,
        preceding_exploration=spec['preceding_exploration_updates'] == 30,
        state_authority=state['native_nexto_authority_sha256'] == canonical(spec),
    )
    if not all(checks.values()):
        raise ValueError(checks)
    return checks


def verify_package():
    # Only source/hash verification; no environment, policy or optimizer created.
    from benchmarks.run_direct_skills_exploration_continuation_v1 import verify
    return verify()


def main():
    out, run = report.OUT, report.RUN
    spec = json.loads((out / 'training_authority.json').read_text())
    captured = {name: (run / name).read_bytes() for name in
                ('campaign_state.json', 'latest.json', 'training_curve.jsonl', 'stdout.log', 'stderr.log')}
    state = json.loads(captured['campaign_state.json'])
    latest = json.loads(captured['latest.json'])
    rows = [json.loads(line) for line in captured['training_curve.jsonl'].splitlines() if line]
    checks = terminal_checks(state, latest, rows, spec, active_campaign_pids())
    if (run / 'failure.json').exists():
        raise RuntimeError('Failure evidence requires investigation, not ordinary finalization')
    package = verify_package()
    reports = {offset: report.audit(offset) for offset in (1, *spec['evaluation_boundaries'])}
    final = reports[70]
    if set(final['comparisons']) != {'immediate_parent', 'original_control'}:
        raise ValueError('Both parent comparisons are required')
    latest_path = Path(latest['path'])
    if latest_path.resolve().parent != run.resolve():
        raise ValueError('Checkpoint outside campaign')
    if sha(latest_path) != latest['sha256'] or latest['sha256'] != final['checkpoint']['sha256']:
        raise ValueError('Final rolling/permanent checkpoint mismatch')
    if captured['training_curve.jsonl'] != (out / 'through_000070.jsonl').read_bytes():
        raise ValueError('Final curve differs from audited prefix')
    progression = {}
    for before, after in ((20, 45), (45, 70), (20, 70)):
        progression[f'{before}_to_{after}'] = compare_records(
            json.loads((out / f'child_{before:06d}.json').read_text()),
            json.loads((out / f'child_{after:06d}.json').read_text()))
    if active_campaign_pids():
        raise RuntimeError('Worker appeared during finalization')
    for name, data in captured.items():
        if (run / name).read_bytes() != data:
            raise RuntimeError(f'External evidence changed: {name}')
    archives = {}
    for source, target in (('campaign_state.json', 'final_campaign_state.json'),
                           ('latest.json', 'final_latest.json'),
                           ('stdout.log', 'stdout.log'), ('stderr.log', 'stderr.log')):
        save_once(out / target, captured[source])
        archives[target] = sha(out / target)
    result = dict(
        version='RIVAL2_DIRECT_SKILLS_EXPLORATION_CONTINUATION_COMPLETION_V1',
        status='complete_review_not_SSL', checks=checks,
        authority_sha256=canonical(spec), frozen_package=package,
        accepted_updates=70, total_exploration_updates=100, cumulative_updates=750,
        checkpoint=final['checkpoint'], rolling_checkpoint=latest,
        training={k: v for k, v in final['training'].items() if k not in ('first_row', 'last_row')},
        parent_comparisons={str(i): reports[i]['comparisons'] for i in spec['evaluation_boundaries']},
        between_evaluations=progression,
        objective_checks={str(i): audit['objective_checks'] for i, audit in reports.items()},
        progress_hashes={str(i): sha(out / f'progress_{i:06d}.json') for i in reports},
        final_contacts=final['contacts'], archived=archives,
        training_curve_sha256=sha(out / 'through_000070.jsonl'),
        optimizer_steps_in_finalization=0, new_evaluations_in_finalization=0,
        interpretation='Execution integrity is not competitive improvement or SSL. '
                       'Review both parent comparisons. No automatic promotion or extension.')
    save_once(out / 'completion.json', (json.dumps(result, indent=2, sort_keys=True, allow_nan=False)+'\n').encode())
    print(json.dumps(dict(status=result['status'], checkpoint=result['checkpoint'],
        comparisons={name: value['summary'] for name, value in final['comparisons'].items()})))
    return result


if __name__ == '__main__':
    torch.set_num_threads(2)
    main()
