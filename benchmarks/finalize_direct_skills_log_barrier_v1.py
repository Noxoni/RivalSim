"""CPU-only closeout of the frozen 30-update exploration experiment.

Never calls a collector, backward, optimizer, or evaluator. Training must have
exited; already completed evaluations are independently reduced by the auditor.
"""
from __future__ import annotations

import json
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import torch

from benchmarks import report_direct_skills_log_barrier_v1 as report
from benchmarks.finalize_direct_skills_native_nexto_v1 import validate_terminal
from benchmarks.report_direct_skills_native_nexto_v1 import canonical, compare_records, save_once, sha


def active_campaign_pids():
    command = (
        "@(Get-CimInstance Win32_Process | Where-Object { "
        "$_.Name -match '^python(w)?\\.exe$' -and "
        "$_.CommandLine -match 'run_direct_skills_log_barrier_v1\\.py' "
        "} | Select-Object -ExpandProperty ProcessId) | ConvertTo-Json -Compress"
    )
    output = subprocess.check_output(
        ["powershell", "-NoProfile", "-NonInteractive", "-Command", command], text=True
    ).strip()
    value = json.loads(output) if output else []
    return value if isinstance(value, list) else [value]


def terminal_checks(state, latest, rows, spec, active_pids):
    validate_terminal(state, latest, rows, spec)
    digest = canonical(spec)
    checks = dict(
        no_active_worker=not active_pids,
        correct_version=spec['version'] == 'RIVAL2_DIRECT_SKILLS_LOG_BARRIER_CAMPAIGN_V1',
        frozen_boundary=spec['child_updates'] == 30,
        all_evaluations_required=spec['evaluation_boundaries'] == [5, 15, 30],
        state_authority=state['native_nexto_authority_sha256'] == digest,
        every_row_authority=all(row['native_nexto_authority_sha256'] == digest for row in rows),
        cumulative_updates=state['accepted_updates'] == spec['parent']['accepted_updates'] + 30,
        exact_curve=[row['accepted_updates'] for row in rows] == list(
            range(spec['parent']['accepted_updates'] + 1, spec['parent']['accepted_updates'] + 31)),
    )
    if not all(checks.values()):
        raise ValueError(checks)
    return checks


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
        raise RuntimeError('Failure evidence needs explicit investigation, not ordinary finalization')
    from benchmarks.run_direct_skills_log_barrier_v1 import verify
    package = verify()
    reports = {offset: report.audit(offset) for offset in spec['evaluation_boundaries']}
    final, objective = reports[30]
    latest_path = Path(latest['path'])
    assert latest_path.resolve().parent == run.resolve(), 'Checkpoint outside campaign'
    assert sha(latest_path) == latest['sha256'] == final['checkpoint']['sha256']
    assert captured['training_curve.jsonl'] == (out / 'through_000030.jsonl').read_bytes()
    progression = {}
    for before, after in ((5, 15), (15, 30), (5, 30)):
        progression[f'{before}_to_{after}'] = compare_records(
            json.loads((out / f'child_{before:06d}.json').read_text()),
            json.loads((out / f'child_{after:06d}.json').read_text()))
    assert not active_campaign_pids(), 'Worker appeared during finalization'
    for name, data in captured.items():
        assert (run / name).read_bytes() == data, f'External evidence changed: {name}'
    archives = {}
    for source, target in (('campaign_state.json', 'final_campaign_state.json'),
                           ('latest.json', 'final_latest.json'),
                           ('stdout.log', 'stdout.log'), ('stderr.log', 'stderr.log')):
        save_once(out / target, captured[source])
        archives[target] = sha(out / target)
    result = dict(
        version='RIVAL2_DIRECT_SKILLS_LOG_BARRIER_COMPLETION_V1',
        status='complete_review_not_SSL', checks=checks,
        authority_sha256=canonical(spec), frozen_package=package,
        accepted_updates=30, cumulative_updates=state['accepted_updates'],
        checkpoint=final['checkpoint'], rolling_checkpoint=latest,
        training={k: v for k, v in final['training'].items() if k not in ('first_row', 'last_row')},
        parent_comparisons={str(offset): audit['comparison'] for offset, (audit, _) in reports.items()},
        between_evaluations=progression,
        objective_audits={str(offset): audit for offset, (_, audit) in reports.items()},
        final_contacts=final['contacts_after'], archived=archives,
        training_curve_sha256=sha(out / 'through_000030.jsonl'),
        optimizer_steps_in_finalization=0, new_evaluations_in_finalization=0,
        interpretation='Execution integrity and exploration telemetry are not gameplay competence. '
                       'Use matched match results to determine next work; no automatic promotion or extension.')
    save_once(out / 'completion.json', (json.dumps(result, indent=2, sort_keys=True, allow_nan=False)+'\n').encode())
    print(json.dumps(dict(status=result['status'], checkpoint=result['checkpoint'],
                         parent_to_final=final['comparison']['summary'],
                         early_to_final=progression['5_to_30']['summary'])))
    return result


if __name__ == '__main__':
    torch.set_num_threads(2)
    main()
