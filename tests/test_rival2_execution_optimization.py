import copy
from types import SimpleNamespace

import pytest
import torch

from rivalsim.recurrent_execution import ResetMetadata
from rivalsim.rival2_exploration import fresh_human_seed_exploration
from rivalsim.rival2_independent_critic import IndependentCriticActorCritic
from rivalsim.rival2_policy import deterministic_hybrid_action, sample_hybrid_action
from rivalsim.rival2_ppo import Rival2PPOConfig
from rivalsim.rival2_recurrent_ppo import (
    Rival2RecurrentPPOCorruption,
    _finite_parameters,
    _finite_parameters_grouped,
    recurrent_minibatch_step,
)
from rivalsim.rival2_ssl_foundation_training import Rival2SslFoundationTrainer


@pytest.fixture(autouse=True)
def threads():
    before = torch.get_num_threads()
    torch.set_num_threads(2)
    yield
    torch.set_num_threads(before)


@pytest.fixture(
    params=[
        "cpu",
        pytest.param(
            "cuda", marks=pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA")
        ),
    ]
)
def device(request):
    return request.param


def model_on(device):
    torch.manual_seed(9421)
    model = IndependentCriticActorCritic().to(device)
    with torch.no_grad():
        model.context_actor.weight.normal_(std=0.02)
    return model


def test_actor_value_hidden_and_gradients(device):
    model = model_on(device)
    obs = torch.randn(4, 19, 182, device=device, requires_grad=True)
    h = torch.randn(1, 4, 256, device=device, requires_grad=True)
    reset = torch.zeros(4, 19, dtype=torch.bool, device=device)
    reset[0, 0] = True
    reset[1, 7] = True
    reset[2, 12] = True
    metadata = ResetMetadata.from_mask(reset)
    a, v, nh = model(obs, h, reset_before=reset)
    loss = a.square().sum() + nh.square().sum()
    loss.backward()
    grads = {n: p.grad.clone() for n, p in model.named_parameters() if p.grad is not None}
    input_grads = obs.grad.clone(), h.grad.clone()
    model.zero_grad(set_to_none=True)
    obs.grad = h.grad = None
    b, nh2 = model.forward_actor(obs, h, reset_before=reset, reset_metadata=metadata)
    (b.square().sum() + nh2.square().sum()).backward()
    for x, y in ((a, b), (nh, nh2), (input_grads[0], obs.grad), (input_grads[1], h.grad)):
        torch.testing.assert_close(x, y, atol=3e-6, rtol=1e-4)
    for n, p in model.named_parameters():
        if n in grads:
            torch.testing.assert_close(p.grad, grads[n], atol=3e-6, rtol=1e-4)
        elif n.startswith("critic."):
            assert p.grad is None
    torch.testing.assert_close(v, model.isolated_value(obs), atol=0, rtol=0)
    for resets in (None, reset[:, 0]):
        one = model(obs[:, 0], h, reset_before=resets)
        actor, hidden = model.forward_actor(obs[:, 0], h, reset_before=resets)
        torch.testing.assert_close(actor, one[0], atol=0, rtol=0)
        torch.testing.assert_close(hidden, one[2], atol=0, rtol=0)
    with pytest.raises(ValueError, match="metadata"):
        model.forward_actor(obs, h, reset_before=reset.clone(), reset_metadata=metadata)
    reset[0, 1] = True
    with pytest.raises(ValueError, match="metadata"):
        model.forward_actor(obs, h, reset_before=reset, reset_metadata=metadata)


@pytest.mark.parametrize("microbatch", [2, 5])
def test_optimizer_step_and_telemetry_parity(device, microbatch):
    model = model_on(device)
    obs = torch.randn(5, 17, 182, device=device)
    h = torch.randn(1, 5, 256, device=device)
    reset = torch.zeros(5, 17, dtype=torch.bool, device=device)
    reset[0, 3] = reset[2, 9] = True
    mask = torch.ones(5, 17, dtype=torch.bool, device=device)
    mask[:2] = False  # zero-trainable microbatch must be skipped identically
    mask[3, 8:] = False
    override = fresh_human_seed_exploration(1).distribution_override
    with torch.no_grad():
        actor, _, _ = model(obs, h, reset_before=reset)
        sample = sample_hybrid_action(
            actor.reshape(-1, 13),
            generator=torch.Generator(device=device).manual_seed(6),
            config=model.config,
            distribution_override=override,
        )
    kwargs = dict(
        observation=obs,
        initial_hidden=h,
        reset_before=reset,
        action=sample.action.reshape(5, 17, 8),
        pre_tanh=sample.pre_tanh.reshape(5, 17, 5),
        old_log_probability=sample.log_probability.reshape(5, 17),
        normalized_advantage=torch.randn(5, 17, device=device),
        returns=torch.randn(5, 17, device=device),
        train_mask=mask,
        sequence_index=torch.arange(5, device=device),
        sequence_microbatch_size=microbatch,
    )
    outputs = []
    for optimized in (False, True):
        candidate = copy.deepcopy(model)
        optimizer = torch.optim.Adam(candidate.parameters(), lr=1e-4)
        rng = torch.random.get_rng_state().clone()
        cuda_rng = torch.cuda.get_rng_state().clone() if device == "cuda" else None
        metrics = recurrent_minibatch_step(
            candidate,
            optimizer,
            Rival2PPOConfig(),
            override,
            **kwargs,
            optimize_execution=optimized,
        )
        assert torch.equal(rng, torch.random.get_rng_state())
        if cuda_rng is not None:
            assert torch.equal(cuda_rng, torch.cuda.get_rng_state())
        outputs.append((candidate, optimizer, metrics))
    a, oa, ma = outputs[0]
    b, ob, mb = outputs[1]
    for pa, pb in zip(a.parameters(), b.parameters(), strict=True):
        torch.testing.assert_close(pa, pb, atol=3e-6, rtol=1e-4)
        torch.testing.assert_close(pa.grad, pb.grad, atol=3e-6, rtol=1e-4)
    for sa, sb in zip(oa.state.values(), ob.state.values(), strict=True):
        for key in sa:
            torch.testing.assert_close(sa[key], sb[key], atol=3e-6, rtol=1e-4)
    for key in ma:
        torch.testing.assert_close(ma[key], mb[key], atol=3e-6, rtol=1e-4)
    for candidate, optimizer, optimized in ((a, oa, False), (b, ob, True)):
        original = optimizer.step

        def corrupt(*args, _step=original, _model=candidate, **kw):
            result = _step(*args, **kw)
            with torch.no_grad():
                next(_model.parameters()).flatten()[0] = float("nan")
            return result

        optimizer.step = corrupt
        with pytest.raises(Rival2RecurrentPPOCorruption, match="nonfinite_parameter"):
            recurrent_minibatch_step(
                candidate,
                optimizer,
                Rival2PPOConfig(),
                override,
                **kwargs,
                optimize_execution=optimized,
            )


def test_every_parameter_nonfinite_detection(device):
    model = model_on(device)
    assert _finite_parameters_grouped(model)
    with torch.no_grad():
        for p in model.parameters():
            value = p.flatten()[0].clone()
            p.flatten()[0] = float("nan")
            assert not _finite_parameters_grouped(model)
            assert not _finite_parameters(model)
            p.flatten()[0] = value
    assert _finite_parameters_grouped(model)


def test_active_opponent_hidden_reset_and_actions(device):
    model = model_on(device).eval()
    worlds = 13
    family = torch.arange(worlds, device=device) % 3
    sides = torch.arange(worlds, device=device) % 2
    trainer = SimpleNamespace(
        world_rows=torch.arange(worlds, device=device),
        opponent_family=family,
        rival_side=sides,
        env=SimpleNamespace(num_envs=worlds),
        policy_config=model.config,
        frozen_v5=model,
    )
    trainer.frozen_hidden = model.initial_hidden(worlds * 2).reshape(1, worlds, 2, 256)
    trainer._flat_frozen_hidden = lambda: trainer.frozen_hidden.reshape(1, -1, 256)
    dense_hidden = trainer._flat_frozen_hidden().clone()
    with torch.no_grad():
        for step in range(18):
            reset = torch.zeros(worlds, 2, dtype=torch.bool, device=device)
            if step % 3 == 0:
                row = step // 3
                reset[row] = True
                family[row] = (family[row] + 1) % 3
                sides[row] = 1 - sides[row]
                trainer.frozen_hidden[:, row] = 0
                dense_hidden[:, row * 2 : row * 2 + 2] = 0
            obs = torch.randn(worlds, 2, 182, device=device)
            full, _, dense_hidden = model(
                obs.reshape(-1, 182), dense_hidden, reset_before=reset.flatten()
            )
            before = torch.random.get_rng_state().clone()
            selected, h = Rival2SslFoundationTrainer.active_frozen_forward(trainer, obs, reset)
            assert torch.equal(before, torch.random.get_rng_state())
            rows = trainer.world_rows[family == 2]
            idx = rows * 2 + 1 - sides[rows]
            torch.testing.assert_close(selected[idx], full[idx], atol=3e-6, rtol=1e-4)
            torch.testing.assert_close(h[:, idx], dense_hidden[:, idx], atol=3e-6, rtol=1e-4)
            assert torch.equal(
                deterministic_hybrid_action(selected[idx])[:, 5:],
                deterministic_hybrid_action(full[idx])[:, 5:],
            )
            trainer.frozen_hidden = h.reshape_as(trainer.frozen_hidden)
        family.zero_()
        actor, h = Rival2SslFoundationTrainer.active_frozen_forward(trainer, obs, reset)
        assert not actor.any()
        assert torch.equal(h, trainer._flat_frozen_hidden())


def test_execution_rebinding_changes_only_authority_metadata():
    from benchmarks.prepare_rival2_execution_optimization import rebind
    from benchmarks.run_rival2_ssl_foundation_v5_long_trace_v1 import tree_sha256

    payload = {
        "model": {"x": torch.randn(3)},
        "optimizer": {"state": {0: {"step": torch.tensor(45), "exp_avg": torch.randn(3)}}},
        "rng": torch.get_rng_state(),
        "accepted_updates_total": 50,
        "source": {
            "sha256": "root",
            "authority_sha256": "old",
            "schedule_authority_sha256": "old_schedule",
        },
        "phase_transition": {
            "credit_assignment_amendment": {"authority_sha256": "old", "optimizer_reset": False}
        },
    }
    before = tree_sha256(payload)
    result = rebind(payload, "new", "new_schedule")
    assert tree_sha256(payload) == before
    for key in payload:
        if key not in {"source", "phase_transition"}:
            assert tree_sha256(payload[key]) == tree_sha256(result[key])
    assert result["source"]["sha256"] == "root"
