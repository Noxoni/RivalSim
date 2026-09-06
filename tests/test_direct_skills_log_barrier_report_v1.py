import copy

import pytest

from benchmarks.report_direct_skills_log_barrier_v1 import objective_checks


def fixture():
    spec=dict(exploration_objective=dict(coefficient=.01,version='test'))
    payload=dict(native_nexto_package=copy.deepcopy(spec))
    rows=[dict(ppo=dict(exploration_barrier=15.,exploration_coefficient=.01,weighted_exploration_barrier=.15))]
    return payload,rows,spec


def test_actual_added_loss_and_checkpoint_objective_required():
    assert all(objective_checks(*fixture()).values())


@pytest.mark.parametrize('fault',['old_loss','wrong_coefficient','wrong_weighted_loss','wrong_checkpoint'])
def test_not_enough_to_only_pass_historical_ppo_checks(fault):
    payload,rows,spec=fixture()
    if fault=='old_loss':rows[0]['ppo']['exploration_barrier']=-99
    if fault=='wrong_coefficient':rows[0]['ppo']['exploration_coefficient']=.03
    if fault=='wrong_weighted_loss':rows[0]['ppo']['weighted_exploration_barrier']=0
    if fault=='wrong_checkpoint':payload['native_nexto_package']['exploration_objective']['version']='other'
    with pytest.raises(AssertionError):objective_checks(payload,rows,spec)
