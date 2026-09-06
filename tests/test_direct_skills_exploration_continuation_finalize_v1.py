import copy
import json

import pytest

from benchmarks import finalize_direct_skills_exploration_continuation_v1 as closeout
from benchmarks.report_direct_skills_native_nexto_v1 import canonical


def fixture():
    spec = dict(version='RIVAL2_DIRECT_SKILLS_EXPLORATION_CONTINUATION_V1',
        child_updates=70, evaluation_boundaries=[20, 45, 70], parent=dict(accepted_updates=680),
        original_control_parent=dict(accepted_updates=650), preceding_exploration_updates=30)
    digest = canonical(spec)
    latest = dict(branch_updates=70, accepted_updates=750, path='rolling_0.pt', sha256='fixture')
    rows = [dict(branch_updates=i, accepted_updates=680+i, cumulative_optimizer_steps=100+i,
        native_nexto_authority_sha256=digest,
        training=dict(trainable_agent_samples=4423680, nexto_training_sample_count=1474560),
        ppo=dict(kl_rejections=0, completed_update_sample_kl_max=1000000.0)) for i in range(1, 71)]
    state = dict(status='complete_review', branch_updates=70, accepted_updates=750,
        latest_checkpoint=copy.deepcopy(latest), cumulative_optimizer_steps=170,
        native_nexto_authority_sha256=digest)
    return state, latest, rows, spec


def test_exact_70_boundary_with_large_finite_kl_is_allowed():
    assert all(closeout.terminal_checks(*fixture(), active_pids=()).values())


@pytest.mark.parametrize('fault', ['live', 'old30', 'old_parent', 'missing_eval', 'authority',
    'gap', 'extra', 'kl_rejection', 'adam', 'latest', 'running', 'cumulative', 'samples', 'nexto'])
def test_reject_invalid_final_evidence(fault):
    state, latest, rows, spec = fixture()
    pids = ()
    if fault == 'live': pids = (123,)
    elif fault == 'old30': spec['child_updates'] = 30
    elif fault == 'old_parent': spec['parent']['accepted_updates'] = 650
    elif fault == 'missing_eval': spec['evaluation_boundaries'] = [20, 70]
    elif fault == 'authority': rows[8]['native_nexto_authority_sha256'] = 'wrong'
    elif fault == 'gap': rows[5]['branch_updates'] = 5
    elif fault == 'extra': rows.append(copy.deepcopy(rows[-1]))
    elif fault == 'kl_rejection': rows[2]['ppo']['kl_rejections'] = 1
    elif fault == 'adam': state['cumulative_optimizer_steps'] += 1
    elif fault == 'latest': latest['sha256'] = 'different'
    elif fault == 'running': state['status'] = 'optimizing'
    elif fault == 'cumulative': rows[6]['accepted_updates'] = 6870
    elif fault == 'samples': rows[6]['training']['trainable_agent_samples'] = 1
    elif fault == 'nexto': rows[6]['training']['nexto_training_sample_count'] = 0
    with pytest.raises(ValueError):
        closeout.terminal_checks(state, latest, rows, spec, pids)


def test_running_campaign_refused_before_any_audit_or_write(tmp_path, monkeypatch):
    state, latest, rows, spec = fixture()
    state['status'] = 'rollout'
    out, run = tmp_path/'out', tmp_path/'run'
    out.mkdir()
    run.mkdir()
    (out/'training_authority.json').write_text(json.dumps(spec))
    (run/'campaign_state.json').write_text(json.dumps(state))
    (run/'latest.json').write_text(json.dumps(latest))
    (run/'training_curve.jsonl').write_text(''.join(json.dumps(row)+'\n' for row in rows))
    (run/'stdout.log').write_text('')
    (run/'stderr.log').write_text('')
    monkeypatch.setattr(closeout.report, 'OUT', out)
    monkeypatch.setattr(closeout.report, 'RUN', run)
    monkeypatch.setattr(closeout, 'active_campaign_pids', lambda: [123])
    def forbidden(*args):
        pytest.fail('Reached audit or write while live')
    monkeypatch.setattr(closeout, 'verify_package', forbidden)
    monkeypatch.setattr(closeout.report, 'audit', forbidden)
    monkeypatch.setattr(closeout, 'save_once', forbidden)
    with pytest.raises(ValueError):
        closeout.main()
    assert list(out.iterdir()) == [out/'training_authority.json']
