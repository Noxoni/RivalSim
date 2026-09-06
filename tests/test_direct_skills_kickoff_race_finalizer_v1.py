import copy

import pytest

from benchmarks.finalize_direct_skills_kickoff_race_v1 import terminal_checks,canonical


def fixture():
    spec=dict(parent=dict(accepted_updates=650),child_updates=15)
    state=dict(status='complete_review',branch_updates=15,accepted_updates=665,native_nexto_authority_sha256=canonical(spec))
    latest=dict(branch_updates=15,accepted_updates=665)
    rows=[dict(branch_updates=i,native_nexto_authority_sha256=canonical(spec),ppo=dict(kl_rejections=0)) for i in range(1,16)]
    return state,latest,rows,spec


def test_only_closed_frozen_arm_accepted():
    assert all(terminal_checks(*fixture()).values())


@pytest.mark.parametrize('fault',['running','wrong_counter','wrong_authority','extra_update','missing_row','kl_rejection','foreign_row'])
def test_failures_are_not_hidden(fault):
    state,latest,rows,spec=copy.deepcopy(fixture())
    if fault=='running':state['status']='optimizing'
    if fault=='wrong_counter':latest['accepted_updates']=675
    if fault=='wrong_authority':state['native_nexto_authority_sha256']='other'
    if fault=='extra_update':rows.append(copy.deepcopy(rows[-1]))
    if fault=='missing_row':rows.pop(3)
    if fault=='kl_rejection':rows[0]['ppo']['kl_rejections']=1
    if fault=='foreign_row':rows[0]['native_nexto_authority_sha256']='other'
    with pytest.raises(ValueError):terminal_checks(state,latest,rows,spec)
