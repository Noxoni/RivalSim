"""CPU checks for the observational kickoff tap, not native physics proof."""

import hashlib
import json
from contextlib import contextmanager
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from benchmarks.trace_rival2_direct_skills import BoundedFrames, temporary_attribute
from benchmarks.trace_rival2_direct_skills_kickoff_reset import capture_decision


def fake_runner(ages=(0, 0), done=(0, 0)):
    n = 2
    views = {"rival2.episode_ticks": torch.tensor(ages)}
    for name, width in (
        ("car_pos", 6),
        ("car_vel", 6),
        ("car_quat", 8),
        ("car_ang_vel", 6),
        ("wheel_contact", 8),
        ("boost", 2),
        ("ball_pos", 3),
        ("ball_vel", 3),
    ):
        views[name] = torch.arange(n * width, dtype=torch.float32)
    return SimpleNamespace(
        num_worlds=n,
        host_tick=0,
        bridge=SimpleNamespace(views=views),
        match_views={
            "done": torch.tensor(done),
            "kickoff_active": torch.ones(n),
            "goal_count": torch.zeros(n),
            "starting_layout": torch.tensor([0, 1]),
        },
        rival_observation=torch.arange(n * 2 * 182).reshape(n, 2, 182).float(),
        hidden=torch.zeros(1, n, 3),
        rival_side=torch.tensor([0, 1]),
        hidden_reset_count=torch.zeros(n),
        rival_action=torch.zeros(n, 8),
        nexto=SimpleNamespace(
            previous_action=torch.zeros(n, 8),
            neural_counter=torch.tensor([0, 4]),
            kickoff_index=torch.zeros(n),
        ),
    )


def lifecycle():
    return {"kickoff_layout": torch.tensor([0, 1]), "kickoff_selector": torch.tensor([1, 2])}


def test_tap_preserves_original_call_and_copies_observation_before_output():
    runner, frames = fake_runner(), BoundedFrames(10)
    before = runner.rival_observation.clone()
    calls = []

    def original():
        calls.append(True)
        runner.rival_action.fill_(1)
        runner.hidden.fill_(2)
        return "untouched-return"

    assert capture_decision(runner, original, frames, lifecycle()) == "untouched-return"
    assert len(calls) == 1
    assert torch.equal(runner.rival_observation, before)
    row = frames.rows[0]
    assert np.array_equal(row["observation"], before.numpy())
    assert not row["hidden_before"].any()
    assert np.all(row["hidden_after"] == 2)
    assert np.all(row["rival_action"] == 1)
    assert row["wheel_contact"].shape == (2, 2, 4)
    runner.hidden.zero_()
    runner.rival_observation.zero_()
    assert np.all(row["hidden_after"] == 2)
    assert np.array_equal(row["observation"], before.numpy())


@pytest.mark.parametrize(
    "ages,done,eligible", [((31, 32), (0, 0), [True, False]), ((0, 0), (1, 0), [False, True])]
)
def test_exact_window_and_completed_world_mask(ages, done, eligible):
    runner, frames = fake_runner(ages, done), BoundedFrames(10)
    capture_decision(runner, lambda: None, frames, lifecycle())
    assert frames.rows[0]["eligible"].tolist() == eligible


def test_noneligible_decision_still_calls_original_without_recording():
    runner, frames, calls = fake_runner((32, 40)), BoundedFrames(10), []
    capture_decision(runner, lambda: calls.append(1), frames, lifecycle())
    assert calls == [1] and frames.rows == []


def test_original_failure_propagates_and_temporary_method_restores():
    runner, frames = fake_runner(), BoundedFrames(10)

    def original():
        raise RuntimeError("native failure")

    runner.action_method = original
    with (
        pytest.raises(RuntimeError, match="native failure"),
        temporary_attribute(
            runner, "action_method", lambda: capture_decision(runner, original, frames, lifecycle())
        ),
    ):
        runner.action_method()
    assert runner.action_method is original and frames.rows == []


def test_busy_lease_prevents_stream_runner_and_output(monkeypatch, tmp_path):
    import benchmarks.direct_skills_eval_stream as streams
    import benchmarks.evaluate_rival2_ssl_entity_full_match as evaluator
    import benchmarks.run_rival2_ssl_entity_continuation as campaign
    import benchmarks.trace_rival2_direct_skills_kickoff_reset as trace

    monkeypatch.setattr(trace, "ROOT", tmp_path)
    result_dir = tmp_path / "results/rival2/direct_skills_v1"
    result_dir.mkdir(parents=True)
    checkpoint = tmp_path / "checkpoints/rival2/direct_skills_v1/plus_000300.pt"
    checkpoint.parent.mkdir(parents=True)
    checkpoint.write_bytes(b"never loaded as a policy")
    saved = dict(
        accepted_updates=300,
        optimizer_steps=0,
        model_unchanged=True,
        checkpoint_unchanged=True,
        checkpoint=dict(sha256=hashlib.sha256(checkpoint.read_bytes()).hexdigest().upper()),
    )
    authority_hash = hashlib.sha256(b"{}").hexdigest().upper()
    saved["authority_sha256"] = authority_hash
    (result_dir / "full_match_000300.json").write_text(json.dumps(saved))
    (result_dir / "package.json").write_text(
        json.dumps(dict(sources={}, authority_sha256=authority_hash))
    )
    (result_dir / "authority.json").write_text("{}")

    @contextmanager
    def busy():
        raise RuntimeError("learner owns GPU lease")
        yield  # pragma: no cover

    def forbidden(*args, **kwargs):
        pytest.fail("CUDA stream or evaluator created despite busy lease")

    monkeypatch.setattr(campaign, "gpu_lease", busy)
    monkeypatch.setattr(streams, "owned_match_stream", forbidden)
    monkeypatch.setattr(evaluator, "CandidateMatchRunner", forbidden)
    output = result_dir / "trace"
    with pytest.raises(RuntimeError, match="learner owns GPU lease"):
        trace.run(300, output)
    assert not output.exists()
