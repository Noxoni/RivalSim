from __future__ import annotations

import copy

import pytest
import torch

from benchmarks.benchmark_rival2_reset_execution import CANDIDATES, legacy, masks


@pytest.fixture(autouse=True)
def limit_threads():
    before = torch.get_num_threads()
    torch.set_num_threads(2)
    yield
    torch.set_num_threads(before)


@pytest.mark.parametrize(
    "device",
    [
        "cpu",
        pytest.param(
            "cuda",
            marks=pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA unavailable"),
        ),
    ],
)
@pytest.mark.parametrize("pattern", ["none", "first", "mixed", "staggered", "dense", "every"])
@pytest.mark.parametrize("method", ["spans"])
def test_equivalent_outputs_hidden_gradients_and_adam(device, pattern, method):
    torch.manual_seed(431)
    dtype = torch.float64 if device == "cpu" else torch.float32
    reference = torch.nn.GRU(8, 8, batch_first=True).to(device=device, dtype=dtype)
    candidate = copy.deepcopy(reference)
    candidate.flatten_parameters()
    inputs = torch.randn(4, 17, 8, device=device, dtype=dtype)
    hidden = torch.randn(1, 4, 8, device=device, dtype=dtype)
    reset = masks(pattern, 4, 17, device)
    results, opts = [], []
    for model, fn in ((reference, legacy), (candidate, CANDIDATES[method])):
        x, h = inputs.clone().requires_grad_(), hidden.clone().requires_grad_()
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
        y, final = fn(model, x, h, reset)
        (y.square().mean() + final.square().mean()).backward()
        results.append([y, final, x.grad, h.grad] + [p.grad.clone() for p in model.parameters()])
        optimizer.step()
        opts.append(optimizer)
    atol, rtol = (1e-12, 1e-10) if device == "cpu" else (3e-6, 1e-4)
    for a, b in zip(*results, strict=True):
        torch.testing.assert_close(a, b, atol=atol, rtol=rtol)
    for a, b in zip(reference.parameters(), candidate.parameters(), strict=True):
        torch.testing.assert_close(a, b, atol=atol, rtol=rtol)
    for a, b in zip(opts[0].state.values(), opts[1].state.values(), strict=True):
        for key in a:
            torch.testing.assert_close(a[key], b[key], atol=atol, rtol=rtol)


@pytest.mark.parametrize("method", ["spans", "packed_fp32"])
def test_reset_cuts_only_reset_rows_gradient_and_single_tick(method):
    torch.manual_seed(19)
    gru = torch.nn.GRU(4, 4, batch_first=True).double()
    x = torch.randn(2, 12, 4, dtype=torch.float64, requires_grad=True)
    h = torch.randn(1, 2, 4, dtype=torch.float64, requires_grad=True)
    reset = torch.zeros(2, 12, dtype=torch.bool)
    reset[0, 5] = True
    fn = CANDIDATES[method]
    _output, final = fn(gru, x, h, reset)
    final.sum().backward()
    assert torch.count_nonzero(x.grad[0, :5]) == 0
    assert torch.count_nonzero(h.grad[:, 0]) == 0
    assert torch.count_nonzero(x.grad[1, :5]) > 0
    assert torch.count_nonzero(h.grad[:, 1]) > 0
    for one_reset in (None, reset[:, :1], torch.ones(2, 1, dtype=torch.bool)):
        a = legacy(gru, x[:, :1], h, one_reset)
        b = fn(gru, x[:, :1], h, one_reset)
        for left, right in zip(a, b, strict=True):
            torch.testing.assert_close(left, right)
    with pytest.raises(ValueError, match="shape"):
        fn(gru, x, h, torch.zeros(2, 13, dtype=torch.bool))


def test_single_initial_reset_uses_one_gru_call_and_mixed_resets_use_only_boundaries():
    from rivalsim.recurrent_execution import gru_reset_spans

    gru = torch.nn.GRU(4, 4, batch_first=True)
    x, h = torch.randn(3, 360, 4), torch.randn(1, 3, 4)
    reset = torch.zeros(3, 360, dtype=torch.bool)
    reset[:, 0] = True
    calls = []
    handle = gru.register_forward_hook(lambda *args: calls.append(1))
    gru_reset_spans(gru, x, h, reset)
    assert len(calls) == 1
    calls.clear()
    reset[0, 25] = reset[2, 25] = reset[1, 250] = True
    gru_reset_spans(gru, x, h, reset)
    assert len(calls) == 3
    handle.remove()
