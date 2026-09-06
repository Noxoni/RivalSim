import copy
import json

import pytest
import torch

from benchmarks import audit_direct_skills_shooting_review as audit


@pytest.fixture(scope='module')
def artifacts():
    run = audit.run
    parent = torch.load(run.SOURCE, map_location='cpu', weights_only=False)
    first = torch.load(run.base.CHECKPOINTS/'shooting_resume_000651.pt', map_location='cpu', weights_only=False)
    row = json.loads((run.OUT/'entry_and_first_update_audit.json').read_text())['first_update']
    package = json.loads((run.OUT/'package.json').read_text())
    return parent, first, [row], run.content_hash(run.amendment()), package


def test_actual_first_accepted_artifacts(artifacts):
    assert all(audit.validate(*artifacts).values())


@pytest.mark.parametrize('field,value', [
    ('effective_training_scenario_sha256', 'wrong'),
    ('exploration_amendment_sha256', 'wrong'),
    ('fresh_optimizer', True),
    ('direct_skill_samples', 0),
    ('cumulative_optimizer_steps', 0),
])
def test_rejects_checkpoint_corruption(artifacts, field, value):
    parent, first, rows, identity, package = artifacts
    corrupted = dict(first, **{field:value})
    with pytest.raises(AssertionError):
        audit.validate(parent, corrupted, rows, identity, package)


def test_rejects_curve_gap_and_wrong_likelihood_temperature(artifacts):
    parent, first, rows, identity, package = artifacts
    for field, value in [('accepted_updates',652), ('training_temperature',1)]:
        changed=copy.deepcopy(rows)
        changed[0][field]=value
        with pytest.raises(AssertionError):
            audit.validate(parent,first,changed,identity,package)


def test_role_rates_use_actual_samples(artifacts):
    rows=artifacts[2]
    result=audit.aggregate(rows)
    role=rows[0]['training']['by_reward_role_and_opponent']['finishing_nexto']
    assert result['roles']['finishing_nexto']['per_million_role_decisions']['goals'] == role['goals']/role['samples']*1e6
    assert result['samples']==4423680
    assert result==audit.aggregate(rows)


def test_kl_magnitude_is_telemetry_but_nonfinite_fails(artifacts):
    parent, first, rows, identity, package = artifacts
    changed=copy.deepcopy(rows)
    changed[0]['ppo']['completed_update_mean_kl']=1e6
    assert all(audit.validate(parent,first,changed,identity,package).values())
    changed[0]['ppo']['completed_update_mean_kl']=float('inf')
    with pytest.raises(AssertionError):
        audit.validate(parent,first,changed,identity,package)


def test_missing_rng_and_nonfinite_model_fail(artifacts):
    parent, first, rows, identity, package = artifacts
    changed=dict(first,torch_cpu_rng_state=torch.tensor([],dtype=torch.uint8))
    with pytest.raises(AssertionError):
        audit.validate(parent,changed,rows,identity,package)
    model=dict(first['model'])
    key=next(k for k,v in model.items() if v.is_floating_point() and v.numel())
    model[key]=model[key].clone()
    model[key].view(-1)[0]=float('nan')
    with pytest.raises(AssertionError):
        audit.validate(parent,dict(first,model=model),rows,identity,package)
