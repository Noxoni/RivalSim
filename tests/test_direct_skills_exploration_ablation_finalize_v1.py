import copy

import pytest

from benchmarks import run_direct_skills_exploration_ablation_v1 as run
from benchmarks import finalize_direct_skills_exploration_ablation_v1 as closeout
from rivalsim.fresh_ground_30hz import content_hash


def fixture():
    specs = {arm:run.authority(arm) for arm in run.ARMS}
    pair = dict(status='complete_review',training_updates=60,
        comparison_authority_sha256=content_hash(run.comparison_authority()))
    latest = {arm:dict(branch_updates=30,accepted_updates=780,path=arm,sha256=arm) for arm in run.ARMS}
    states = {arm:dict(status='complete_review',branch_updates=30,accepted_updates=780,
        cumulative_optimizer_steps=160000,latest_checkpoint=latest[arm],
        native_nexto_authority_sha256=content_hash(specs[arm])) for arm in run.ARMS}
    rows = {arm:[dict(branch_updates=i,accepted_updates=750+i,cumulative_optimizer_steps=160000,
        native_nexto_authority_sha256=content_hash(specs[arm]),
        training=dict(trainable_agent_samples=4423680,nexto_training_sample_count=1474560),
        ppo=dict(kl_rejections=0,completed_update_mean_kl=1e9)) for i in range(1,31)] for arm in run.ARMS}
    return pair,states,latest,rows,specs


def test_complete_pair_large_finite_kl_allowed():
    assert all(closeout.terminal_checks(*fixture(),[]).values())


@pytest.mark.parametrize('fault',['live','supervisor_incomplete','wrong_budget','arm_incomplete',
    'old_parent','coefficient','wrong_authority','wrong_latest','gap','kl_rejection','extra_update'])
def test_premature_or_inconsistent_finalization_rejected(fault):
    pair,states,latest,rows,specs = copy.deepcopy(fixture())
    pids = []
    if fault == 'live': pids = [22812]
    elif fault == 'supervisor_incomplete': pair['status'] = 'running'
    elif fault == 'wrong_budget': pair['training_updates'] = 120
    elif fault == 'arm_incomplete': states['withdraw']['status'] = 'optimizing'
    elif fault == 'old_parent': specs['withdraw']['parent']['accepted_updates'] = 680
    elif fault == 'coefficient': specs['withdraw']['exploration_objective']['coefficient'] = .01
    elif fault == 'wrong_authority': states['withdraw']['native_nexto_authority_sha256'] = 'wrong'
    elif fault == 'wrong_latest': latest['withdraw'] = dict(latest['withdraw'],sha256='wrong')
    elif fault == 'gap': rows['withdraw'].pop(12)
    elif fault == 'kl_rejection': rows['withdraw'][0]['ppo']['kl_rejections'] = 1
    elif fault == 'extra_update': rows['withdraw'].append(dict(rows['withdraw'][-1],branch_updates=31))
    with pytest.raises(ValueError):
        closeout.terminal_checks(pair,states,latest,rows,specs,pids)


def test_main_checks_actual_worker_before_reading_or_writing(monkeypatch):
    monkeypatch.setattr(closeout,'active_campaign_pids',lambda: [22812])
    with pytest.raises(RuntimeError,match='still running'):
        closeout.main()
