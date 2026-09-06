import copy

import pytest

from benchmarks.finalize_direct_skills_log_barrier_v1 import terminal_checks
from benchmarks.report_direct_skills_native_nexto_v1 import canonical


def fixture():
    spec = dict(version='RIVAL2_DIRECT_SKILLS_LOG_BARRIER_CAMPAIGN_V1',
                child_updates=30, evaluation_boundaries=[5, 15, 30], parent=dict(accepted_updates=650))
    digest = canonical(spec)
    latest = dict(branch_updates=30, accepted_updates=680, path='rolling_0.pt', sha256='fixture')
    rows = [dict(branch_updates=i, accepted_updates=650+i, cumulative_optimizer_steps=100+i,
                 native_nexto_authority_sha256=digest,
                 ppo=dict(kl_rejections=0, completed_update_sample_kl_max=1000000.0)) for i in range(1, 31)]
    state = dict(status='complete_review', branch_updates=30, accepted_updates=680,
                 latest_checkpoint=copy.deepcopy(latest), cumulative_optimizer_steps=130,
                 native_nexto_authority_sha256=digest)
    return state, latest, rows, spec


def test_exact_30_boundary_with_large_finite_kl_is_allowed():
    assert all(terminal_checks(*fixture(), active_pids=()).values())


def test_live_worker_blocks_closeout():
    with pytest.raises(ValueError, match='no_active_worker'):
        terminal_checks(*fixture(), active_pids=(123,))


@pytest.mark.parametrize('fault', ['old15', 'missing_eval', 'wrong_authority', 'curve_gap',
                                    'extra_update', 'kl_rejection', 'optimizer_counter',
                                    'latest_mismatch', 'running', 'cumulative_counter'])
def test_reject_invalid_final_evidence(fault):
    state, latest, rows, spec = fixture()
    if fault == 'old15':
        spec['child_updates'] = 15
    elif fault == 'missing_eval':
        spec['evaluation_boundaries'] = [5, 30]
    elif fault == 'wrong_authority':
        rows[8]['native_nexto_authority_sha256'] = 'wrong'
    elif fault == 'curve_gap':
        rows[5]['branch_updates'] = 5
    elif fault == 'extra_update':
        rows.append(copy.deepcopy(rows[-1]))
    elif fault == 'kl_rejection':
        rows[2]['ppo']['kl_rejections'] = 1
    elif fault == 'optimizer_counter':
        state['cumulative_optimizer_steps'] += 1
    elif fault == 'latest_mismatch':
        latest['sha256'] = 'different'
    elif fault == 'running':
        state['status'] = 'optimizing'
    elif fault == 'cumulative_counter':
        rows[6]['accepted_updates'] = 6570
    with pytest.raises(ValueError):
        terminal_checks(state, latest, rows, spec, ())
