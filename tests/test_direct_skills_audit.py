import copy

import torch

from benchmarks.audit_rival2_direct_skills import equal


def test_recursive_exact_optimizer_equality_and_corruption():
    source = {
        "state": {7: {"step": torch.tensor(52150.0), "exp_avg": torch.arange(5.0)}},
        "param_groups": [{"lr": 1e-4, "params": [7]}],
    }
    target = copy.deepcopy(source)
    assert equal(source, target)
    target["state"][7]["exp_avg"][0] = 1
    assert not equal(source, target)
    assert not equal(torch.ones(2), torch.ones(2, dtype=torch.float64))
    assert not equal(torch.tensor(float("nan")), torch.tensor(float("nan")))
    assert not equal([1], (1,))
    assert not equal({"x": 1}, {"y": 1})
