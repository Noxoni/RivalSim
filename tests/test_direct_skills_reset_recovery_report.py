import copy
import json

import pytest
import torch

from benchmarks import report_direct_skills_reset_recovery as report


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
    assert len(checks)==27 and all(checks.values())


@pytest.mark.parametrize('field,value',[
    ('recovery_branch_updates',26),('accepted_updates',675),('fresh_optimizer',True),
    ('recovery_parent_sha256','wrong'),('cumulative_optimizer_steps',0),
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


def test_parent_context_separates_kickoff_and_ongoing():
    a=report.parent_context(); b=report.parent_context()
    assert a==b
    ongoing=a['skills']['natural_start_groups']['ongoing_ground']
    assert (ongoing['cases'],ongoing['goals_for'],ongoing['goals_against'])==(29,3,25)
    assert a['skills']['families']['finishing']['goals_for']==6
    assert a['matches']['summary']['wins']==8
    assert a['optimizer_steps']==a['new_policy_evaluations']==0
    assert len(a['timings']['events'])==291


def test_goal_times_are_native_and_ordered():
    path=report.run.runtime.OUTPUT/'full_match_000600.json'
    doc=json.loads(path.read_text())
    result=report.goal_intervals(doc)
    assert sum(r['scored_by_rival'] for r in result['events'])==169
    assert result['interval_seconds']['minimum']>0
    doc['raw']['goal_tick'][0][1]=doc['raw']['goal_tick'][0][0]
    with pytest.raises(AssertionError):
        report.goal_intervals(doc)
