import torch

from benchmarks.audit_rival2_ssl_entity_resume import identical


def test_recursive_state_equality_and_corruption():
    state = {"adam": [{"step": torch.tensor(123.0), "exp_avg": torch.arange(4.0)}], "lr": 1e-4}
    equal = {"adam": [{"step": torch.tensor(123.0), "exp_avg": torch.arange(4.0)}], "lr": 1e-4}
    assert identical(state, equal)
    equal["adam"][0]["exp_avg"][0] = 1
    assert not identical(state, equal)
    assert not identical(torch.tensor(float("nan")), torch.tensor(float("nan")))
    assert not identical(torch.ones(2), torch.ones(2, dtype=torch.float64))
    assert not identical([1], (1,))
    assert not identical({"x": 1}, {"y": 1})
