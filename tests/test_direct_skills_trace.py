"""CPU-only trace integrity tests; these do not claim live replay parity."""

import json
from contextlib import contextmanager
from types import SimpleNamespace

import numpy as np
import pytest

from benchmarks.trace_rival2_direct_skills import (
    BoundedFrames,
    differences,
    install_finishing_tap,
    temporary_attribute,
    verify_trace_arrays,
)


@pytest.mark.parametrize("raises", [False, True])
@pytest.mark.parametrize("instance_override", [False, True])
def test_method_delegation_and_binding_restoration(raises, instance_override):
    class World:
        def step(self, ticks):
            return ticks + 1

    world = World()
    if instance_override:
        world.step = lambda ticks: ticks + 2
    original = world.step
    calls = []

    def tap(ticks):
        calls.append(("before", ticks))
        result = original(ticks)
        calls.append(("after", result))
        if raises:
            raise RuntimeError("tap failure")
        return result

    try:
        with temporary_attribute(world, "step", tap):
            assert world.step(1) == (3 if instance_override else 2)
    except RuntimeError:
        assert raises
    assert calls == [("before", 1), ("after", 3 if instance_override else 2)]
    assert ("step" in vars(world)) is instance_override
    assert world.step == original


def test_archive_roundtrip_copies_native_reused_buffer(tmp_path):
    frames = BoundedFrames(2)
    native = np.array([1.0, -0.0, np.nextafter(np.float32(1), np.float32(2))], np.float32)
    frames.append(dict(value=native, index=np.int64(0)))
    native[0] = 5
    frames.append(dict(value=native, index=np.int64(1)))
    with pytest.raises(RuntimeError, match="bound"):
        frames.append(dict(value=native, index=np.int64(2)))
    arrays = frames.arrays("physics.")
    assert arrays["physics.value"][0, 0] == 1
    assert np.signbit(arrays["physics.value"][0, 1])
    np.savez_compressed(tmp_path / "frames.npz", **arrays)
    with np.load(tmp_path / "frames.npz", allow_pickle=False) as restored:
        for key, value in arrays.items():
            assert restored[key].dtype == value.dtype
            assert restored[key].tobytes() == value.tobytes()


@pytest.mark.parametrize("different", [np.ones(2, np.float64), np.ones(3, np.float32)])
def test_schema_drift_rejected(different):
    frames = BoundedFrames(2)
    frames.append(dict(x=np.ones(2, np.float32)))
    with pytest.raises(RuntimeError, match="schema"):
        frames.append(dict(x=different))


def test_object_arrays_and_empty_trace_rejected():
    with pytest.raises(ValueError, match="object"):
        BoundedFrames(1).append(dict(x=np.array([{}], dtype=object)))
    with pytest.raises(RuntimeError, match="no trace"):
        BoundedFrames(1).arrays("")


def test_exact_replay_comparison_does_not_tolerate_float_drift():
    value = dict(finishing=dict(seconds=[12.000046730041504], goals=9))
    assert differences(value, json.loads(json.dumps(value))) == []
    changed = json.loads(json.dumps(value))
    changed["finishing"]["seconds"][0] += 1e-12
    assert differences(value, changed)[0].startswith("$.finishing.seconds[0]")
    assert differences([1], [1.0]) == ["$[0]: type differs"]
    assert differences(dict(a=1), dict(b=1)) == ["$: keys differ"]
    assert differences([1], []) == ["$: length differs"]
    assert len(differences(list(range(100)), [0] * 100, limit=4)) == 4


def trace_fixture():
    # Case0 scores during decision0/tick1; case1 times out in decision1.
    active = np.array([[True, True], [False, True]])
    cumulative = np.zeros((2, 4, 2, 2), dtype=np.int64)
    cumulative[0, 0:, 0, 0] = 1
    cumulative[0, 2:, 0, 0] = 2  # Absorbing-remainder contact must not count.
    cumulative[1, 2:, 1, 1] = 1
    valid = np.repeat(active[:, None], 4, axis=1)
    valid[0, 2:, 0] = False
    touches = np.zeros((2, 2, 2), dtype=np.int64)
    touches[0, 0, 0] = touches[1, 1, 1] = 1
    arrays = {
        "decision.alive": active,
        "decision.reset_mask": np.array([[True, False], [False, True]]),
        "decision.terminated": np.array([[True, False], [False, False]]),
        "decision.truncated": np.array([[False, False], [False, True]]),
        "decision.native.touch_count": touches,
        "decision.native.scoring_team": np.array([[0, -1], [-1, -1]]),
        "physics.decision": np.repeat(np.arange(2), 4),
        "physics.tick": np.tile(np.arange(4), 2),
        "physics.valid": valid.reshape(8, 2),
        "physics.post.interval_touch_count": cumulative.reshape(8, 2, 2),
    }
    expected = dict(
        raw=dict(touches=[1.0, 1.0], goals=[1.0, 0.0], concedes=[0.0, 0.0], endings=[1, 2])
    )
    return arrays, expected, np.array([0, 1])


def test_original_case_masks_and_contact_accounting():
    arrays, expected, sides = trace_fixture()
    assert verify_trace_arrays(arrays, expected, sides)["native_contact_accounting"]


@pytest.mark.parametrize(
    "fault", ["missing_tick", "duplicate_tick", "reactivation", "extra_contact", "nonfinite"]
)
def test_corrupt_trace_rejected(fault):
    arrays, expected, sides = trace_fixture()
    if fault == "missing_tick":
        arrays["physics.decision"] = arrays["physics.decision"][:-1]
    elif fault == "duplicate_tick":
        arrays["physics.tick"][1] = 0
    elif fault == "reactivation":
        arrays["decision.alive"][1, 0] = True
    elif fault == "extra_contact":
        arrays["physics.post.interval_touch_count"][-1, 1, 1] = 2
    else:
        arrays["extra"] = np.array([np.nan])
    with pytest.raises((ValueError, AssertionError)):
        verify_trace_arrays(arrays, expected, sides)


def test_exclusive_lease_refusal_precedes_cuda_and_output_creation(tmp_path, monkeypatch):
    from benchmarks import run_rival2_direct_skills_v1 as campaign
    from benchmarks.trace_rival2_direct_skills import run

    monkeypatch.setattr(campaign, "verify", lambda **kwargs: dict(authority_sha256="authority"))
    monkeypatch.setattr(campaign, "CHECKPOINTS", tmp_path)
    monkeypatch.setattr(campaign, "RESULTS", tmp_path)
    monkeypatch.setattr(campaign, "sha", lambda path: "hash")
    (tmp_path / "evaluation_000200.json").write_text(
        json.dumps(
            dict(
                checkpoint=dict(sha256="hash"),
                accepted_updates=200,
                authority_sha256="authority",
                optimizer_steps=0,
            )
        )
    )

    @contextmanager
    def locked():
        raise OSError("training owns GPU lease")
        yield

    monkeypatch.setattr(campaign, "gpu_lease", locked)
    monkeypatch.setattr(
        campaign, "EntityJointControlActorCritic", lambda: pytest.fail("model instantiated")
    )
    target = tmp_path / "trace"
    with pytest.raises(OSError, match="lease"):
        run(200, target)
    assert not target.exists()


def test_environment_patch_restored_after_failure():
    namespace = SimpleNamespace(DirectSkillsEnv=object())
    original = namespace.DirectSkillsEnv
    with pytest.raises(RuntimeError), temporary_attribute(namespace, "DirectSkillsEnv", object()):
        raise RuntimeError("diagnostic failed")
    assert namespace.DirectSkillsEnv is original


@pytest.mark.parametrize("fail_at", [None, 2])
def test_tapped_environment_delegates_and_keeps_pre_reset_state(monkeypatch, fail_at):
    import torch
    import warp as wp

    from benchmarks.trace_rival2_direct_skills import BALL_FIELDS, CAR_FIELDS

    monkeypatch.setattr(wp, "to_torch", lambda value: value)

    class World:
        def __init__(self, env):
            self.env = env
            self.calls = 0
            self.car_ball = SimpleNamespace(hit_this_tick=torch.zeros(2))
            self.car_ball_b = SimpleNamespace(hit_this_tick=torch.zeros(2))

        def step(self, ticks):
            assert ticks == 1
            self.calls += 1
            if self.calls == fail_at:
                raise RuntimeError("physical step failed")
            self.env.bridge.views["ball_pos"].add_(1)

    class Base:
        def __init__(self):
            self.num_envs, self.device = 2, "cpu"
            self.family, self.focal = torch.tensor([2, 2]), torch.tensor([0, 1])
            self.observation = torch.zeros(2, 2, 182)
            self.goal_latched = torch.zeros(2, dtype=torch.int32)
            views = {name: torch.zeros(4) for name in CAR_FIELDS}
            views.update({name: torch.zeros(2, 3) for name in BALL_FIELDS})
            views["rival2.previous_action"] = torch.zeros(2, 2, 8)
            views["rival2.touch_count"] = torch.zeros(4, dtype=torch.int32)
            views["rival2.scoring_team_latched"] = torch.full((2,), -1)
            self.bridge = SimpleNamespace(views=views)
            self.world = World(self)

        def _step_impl(self, action, markers=None, tick_action_provider=None):
            for tick in range(4):
                applied = tick_action_provider(tick)
                self.bridge.views["rival2.previous_action"].copy_(applied)
                self.world.step(1)
            self.last_native = dict(touch_count=torch.zeros(2, 2))
            self.last_skill = dict(events=torch.zeros(2, 2, 7))
            transition = SimpleNamespace(
                transition_observation=torch.full((2, 2, 182), 4.0),
                reward=torch.zeros(2, 2),
                terminated=torch.tensor([False, False]),
                truncated=torch.tensor([True, True]),
                reset_mask=torch.tensor([True, True]),
            )
            # Simulate automatic reset before returning; physical tap must precede it.
            self.bridge.views["ball_pos"].zero_()
            self.observation.fill_(9)
            return transition

    physics, decisions, identity = BoundedFrames(8), BoundedFrames(2), {}
    env = install_finishing_tap(
        SimpleNamespace(DirectSkillsEnv=Base), physics, decisions, identity
    )()
    original = env.world.step
    action = torch.zeros(2, 2, 8)

    def provider(tick):
        applied = action.clone()
        applied[:, 1, 0] = 0.25 * (tick + 1)
        return applied

    if fail_at:
        with pytest.raises(RuntimeError, match="physical step"):
            env._step_impl(action, tick_action_provider=provider)
        assert not decisions.rows
    else:
        transition = env._step_impl(action, tick_action_provider=provider)
        assert transition.reset_mask.all()
        assert env.world.calls == 4
        assert len(physics.rows) == 4 and len(decisions.rows) == 1
        assert identity == dict(focal_side=[0, 1], worlds=2)
        assert decisions.rows[0]["observation"].sum() == 0
        assert decisions.rows[0]["transition_observation"].min() == 4
        assert not env.trace_alive.any()
        for tick, row in enumerate(physics.rows):
            assert row["pre.ball_pos"].min() == tick
            assert row["post.ball_pos"].min() == tick + 1
            assert row["pre.applied_action"][0, 1, 0] == 0.25 * (tick + 1)
        assert not env.bridge.views["ball_pos"].any()
    assert env.world.step == original and "step" not in vars(env.world)
