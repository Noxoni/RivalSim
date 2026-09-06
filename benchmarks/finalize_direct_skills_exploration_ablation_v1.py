"""Read-only CPU closeout after both matched-arm workers and supervisor exit."""
from __future__ import annotations

import json
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
import torch

from benchmarks import run_direct_skills_exploration_ablation_v1 as run
from benchmarks import report_direct_skills_exploration_ablation_v1 as report
from benchmarks.finalize_direct_skills_native_nexto_v1 import validate_terminal
from benchmarks.report_direct_skills_native_nexto_v1 import canonical,compare_records,save_once,sha


def active_campaign_pids():
    command = (
        "@(Get-CimInstance Win32_Process | Where-Object { "
        "$_.Name -match '^python(w)?\\.exe$' -and "
        "$_.CommandLine -match 'run_direct_skills_exploration_ablation_v1\\.py' "
        "} | Select-Object -ExpandProperty ProcessId) | ConvertTo-Json -Compress"
    )
    output = subprocess.check_output(['powershell','-NoProfile','-NonInteractive',
        '-Command',command],text=True).strip()
    value = json.loads(output) if output else []
    return value if isinstance(value,list) else [value]


def terminal_checks(pair_state,states,latest,rows,specs,active_pids):
    if active_pids:
        raise ValueError('Live worker or supervisor; do not finalize')
    common = run.comparison_authority()
    if pair_state.get('status') != 'complete_review' or pair_state.get('training_updates') != 60:
        raise ValueError('Both arms must finish their frozen budgets')
    if pair_state.get('comparison_authority_sha256') != canonical(common):
        raise ValueError('Wrong pair authority')
    for arm in run.ARMS:
        spec = specs[arm]
        if (spec['arm'] != arm or spec['parent'] != run.PARENT or spec['child_updates'] != 30
                or spec['evaluation_boundaries'] != [10,30]
                or spec['comparison_authority_sha256'] != canonical(common)
                or spec['exploration_objective']['coefficient'] != run.ARMS[arm]):
            raise ValueError('Wrong frozen arm contract')
        validate_terminal(states[arm],latest[arm],rows[arm],spec)
        report.validate_rows(rows[arm],30,spec)
        if (states[arm]['accepted_updates'] != 780
                or states[arm]['native_nexto_authority_sha256'] != canonical(spec)):
            raise ValueError('Wrong completed state identity')
    samples = sum(r['training']['trainable_agent_samples'] for arm in run.ARMS for r in rows[arm])
    if samples != common['maximum_total_learner_decisions']:
        raise ValueError('Unexpected exposure')
    return dict(no_active_workers=True,both_arms_complete=True,frozen_contracts=True,
        common_parent=True,paired_budget=True,continuous_curves=True,kl_telemetry_only=True)


def main():
    if active_campaign_pids():
        raise RuntimeError('Campaign is still running; no finalization performed')
    names = ('campaign_state.json','latest.json','training_curve.jsonl','stdout.log','stderr.log')
    captured = {arm:{name:(run.EXTERNAL/arm/name).read_bytes() for name in names} for arm in run.ARMS}
    root_names = ('pair_state.json','supervisor_stdout.log','supervisor_stderr.log')
    root_captured = {name:(run.EXTERNAL/name).read_bytes() for name in root_names}
    states = {arm:json.loads(v['campaign_state.json']) for arm,v in captured.items()}
    latest = {arm:json.loads(v['latest.json']) for arm,v in captured.items()}
    rows = {arm:[json.loads(line) for line in v['training_curve.jsonl'].splitlines() if line]
            for arm,v in captured.items()}
    specs = {arm:json.loads((run.OUT/arm/'training_authority.json').read_text()) for arm in run.ARMS}
    pair_state = json.loads(root_captured['pair_state.json'])
    checks = terminal_checks(pair_state,states,latest,rows,specs,active_campaign_pids())
    if any((run.EXTERNAL/arm/'failure.json').exists() for arm in run.ARMS):
        raise RuntimeError('Failure evidence requires investigation, not ordinary closeout')
    packages = {arm:run.verify(arm) for arm in run.ARMS}
    audits = {arm:{offset:report.audit(arm,offset) for offset in (1,*run.EVALUATIONS)} for arm in run.ARMS}
    comparisons = {offset:report.pair_report(offset) for offset in run.EVALUATIONS}
    if sha(run.OUT/'comparison_000030.json') != pair_state['report_sha256']:
        raise ValueError('Supervisor completion report changed')
    for arm in run.ARMS:
        target = Path(latest[arm]['path'])
        if target.resolve().parent != (run.EXTERNAL/arm).resolve():
            raise ValueError('Rolling checkpoint outside its arm')
        if sha(target) != latest[arm]['sha256'] or latest[arm]['sha256'] != audits[arm][30]['checkpoint']['sha256']:
            raise ValueError('Final rolling/permanent mismatch')
        if pair_state['final_checkpoint_hashes'][arm] != latest[arm]['sha256']:
            raise ValueError('Supervisor checkpoint identity mismatch')
        if captured[arm]['training_curve.jsonl'] != (run.OUT/arm/'through_000030.jsonl').read_bytes():
            raise ValueError('Full closed curve differs from audited prefix')
    if active_campaign_pids():
        raise RuntimeError('Worker appeared during closeout')
    for arm,files in captured.items():
        for name,data in files.items():
            if (run.EXTERNAL/arm/name).read_bytes() != data:
                raise RuntimeError('External arm evidence changed')
    for name,data in root_captured.items():
        if (run.EXTERNAL/name).read_bytes() != data:
            raise RuntimeError('External supervisor evidence changed')
    archives = {}
    for arm,files in captured.items():
        for name,data in files.items():
            target = run.OUT/arm/('final_'+name if name in ('campaign_state.json','latest.json') else name)
            save_once(target,data)
            archives[target.relative_to(ROOT).as_posix()] = sha(target)
    for name,data in root_captured.items():
        target = run.OUT/('final_'+name if name == 'pair_state.json' else name)
        save_once(target,data)
        archives[target.relative_to(ROOT).as_posix()] = sha(target)
    progression = {arm:compare_records(
        json.loads((run.OUT/arm/'child_000010.json').read_text()),
        json.loads((run.OUT/arm/'child_000030.json').read_text())) for arm in run.ARMS}
    result = dict(version=run.VERSION+'_COMPLETION',status='complete_review_not_SSL',checks=checks,
        comparison_authority_sha256=canonical(run.comparison_authority()),packages=packages,
        total_accepted_updates=60,total_learner_decisions=265420800,
        final_checkpoints={arm:audits[arm][30]['checkpoint'] for arm in run.ARMS},
        rolling_checkpoints=latest,between_evaluations=progression,
        final_arm_comparison=comparisons[30]['retain_to_withdraw'],
        parent_comparisons={arm:audits[arm][30]['parent_comparison'] for arm in run.ARMS},
        comparison_hashes={str(offset):sha(run.OUT/f'comparison_{offset:06d}.json') for offset in comparisons},
        audit_hashes={arm:{str(offset):sha(run.OUT/arm/f'progress_{offset:06d}.json') for offset in audits[arm]} for arm in run.ARMS},
        archives=archives,optimizer_steps_in_finalization=0,new_matches_in_finalization=0,
        interpretation=run.comparison_authority()['interpretation'])
    save_once(run.OUT/'completion.json',report.encoded(result))
    print(json.dumps(dict(status=result['status'],checkpoints=result['final_checkpoints'],
        final_comparison=result['final_arm_comparison']['summary'])))
    return result


if __name__ == '__main__':
    torch.set_num_threads(2)
    main()
