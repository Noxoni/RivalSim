import torch
from benchmarks.audit_direct_skills_shooting_entry import build, exact


def test_exact_checks_bits_not_only_equal_floats():
    assert torch.equal(torch.tensor([0.]),torch.tensor([-0.]))
    assert not exact(torch.tensor([0.]),torch.tensor([-0.]))
    assert exact({'a':[torch.tensor(3)]},{'a':[torch.tensor(3)]})


def test_actual_entry_and_first_update_rebuild_exactly():
    a=build()
    assert a==build()
    assert all(a['checks'].values())
    assert a['optimizer_steps_in_audit']==0
