import copy
import json

import pytest
import torch

from benchmarks import report_direct_skills_finishing_goal as report


@pytest.fixture(scope='module')
def saved():
    run = report.run
    parent = torch.load(run.SOURCE,map_location='cpu',weights_only=False)
    first = torch.load(run.CKPTS/'first_000001.pt',map_location='cpu',weights_only=False)
    rows = [json.loads((run.OUT/'first_training_update.json').read_text())]
    package = json.loads((run.OUT/'package.json').read_text())
    return parent,first,rows,package


def test_actual_first_update_integrity(saved):
    checks = report.validate(*saved)
    assert len(checks)==31 and all(checks.values())


@pytest.mark.parametrize('field,value',[
    ('finishing_branch_updates',26),('accepted_updates',675),('fresh_optimizer',True),
    ('finishing_parent_sha256','wrong'),('cumulative_optimizer_steps',0),
    ('direct_skill_samples',0),('direct_skill_physics_ticks',0),
    ('effective_training_scenario_sha256','wrong'),('evaluation_runtime_authority_sha256','wrong')])
def test_misbound_checkpoints_rejected(saved,field,value):
    parent,first,rows,package = saved
    altered = dict(first);altered[field]=value
    with pytest.raises(AssertionError):
        report.validate(parent,altered,rows,package)


def test_nonfinite_and_kl_semantics(saved):
    parent,first,rows,package = saved
    altered=copy.deepcopy(rows)
    altered[0]['ppo']['completed_update_mean_kl']=10000.
    assert all(report.validate(parent,first,altered,package).values())
    altered[0]['ppo']['completed_update_mean_kl']=float('nan')
    with pytest.raises(AssertionError):
        report.validate(parent,first,altered,package)
    altered=copy.deepcopy(rows);altered[0]['ppo']['kl_rejections']=1
    with pytest.raises(AssertionError):
        report.validate(parent,first,altered,package)



@pytest.mark.parametrize('change',['paid_touch','approach','proxy_success','old_reward'])
def test_finishing_semantics_fail_if_legacy_payments_return(saved,change):
    parent,first,rows,package = saved
    altered=copy.deepcopy(rows); child=dict(first)
    stats=altered[0]['training']['by_reward_role_and_opponent']['finishing_nexto']
    if change=='paid_touch':
        stats['direct_reward']+=.5
    elif change=='approach':
        stats['approach_reward']=.1
    elif change=='proxy_success':
        stats['success_endings']+=1
    else:
        child['reward_authority']=parent['reward_authority']
    with pytest.raises(AssertionError):
        report.validate(parent,child,altered,package)
