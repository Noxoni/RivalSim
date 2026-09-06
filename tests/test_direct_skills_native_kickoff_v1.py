"""Focused CPU tests for observational copying and bounded physical reduction."""
import json
from types import SimpleNamespace

import numpy as np
import pytest
import torch

import benchmarks.trace_direct_skills_native_kickoff_v1 as trace


def fixture(ticks=720):
    data = {}
    for prefix in ("pre", "post"):
        for name, shape in trace.SHAPES.items():
            data[prefix + "." + name] = np.zeros((ticks, 1, *shape), dtype=np.float32)
        data[prefix + ".car_quat"][..., 3] = 1
        data[prefix + ".on_ground"][:] = 1
        data[prefix + ".goal_count"] = np.zeros((ticks, 1), dtype=np.int32)
        data[prefix + ".touch_count"] = np.zeros((ticks, 1, 2), dtype=np.int32)
        data[prefix + ".ball_pos"][..., 0] = 1000
    data["applied_controls"] = np.zeros((ticks, 1, 2, 8), dtype=np.float32)
    data["applied_controls"][..., 0] = 1
    return data


def test_native_xyzw_heading_and_velocity_projection():
    d = fixture()
    d["pre.car_vel"][..., 0] = 300
    row = trace.physical_summary(d, [0], [3])[0]
    assert row["cars"][0]["race_heading_dot_mean"] == 1
    assert row["cars"][0]["race_closing_velocity_mean"] == 300
    assert row["cars"][0]["race_speed_mean"] == 300
    np.testing.assert_allclose(trace.forward(np.array([0, 0, 1, 0.])), [-1, 0, 0])


def test_race_stops_before_either_contact_and_excludes_postgoal():
    d = fixture()
    d["post.touch_count"][100:, 0, 1] = 1
    d["pre.touch_count"][101:, 0, 1] = 1
    d["pre.goal_count"][150:] = 1
    d["pre.car_vel"][100:, 0, :, 0] = 999
    d["post.touch_count"][200:, 0, 0] = 1
    d["pre.touch_count"][201:, 0, 0] = 1
    d["applied_controls"][160:, 0, 0, 5] = 1
    row = trace.physical_summary(d, [0], [0])[0]
    assert row["pre_contact_ticks"] == 100
    assert row["initial_ticks"] == 150
    assert row["first_contact_physics_tick"] == [None, 101]
    assert row["cars"][0]["race_speed_mean"] == 0
    assert row["cars"][0]["first_jump_tick"] is None
    assert row["cars"][0]["initial_contacts"] == 0
    assert [s["tick"] for s in row["cars"][0]["samples"]] == [0, 60, 120]


def test_tie_and_initial_contact_have_no_fabricated_race_mean():
    d = fixture()
    d["post.touch_count"][:] = 1
    d["pre.touch_count"][1:] = 1
    row = trace.physical_summary(d, [1], [4])[0]
    assert row["first_contact_physics_tick"] == [1, 1]
    assert row["pre_contact_ticks"] == 0
    assert row["cars"][1]["race_speed_mean"] is None


def goal_data(ticks):
    data = {"match.goal_count": [len(ticks)], "goal_tick": [ticks]}
    for key in ("goal_scorer", "goal_overtime", "goal_kickoff", "goal_entry_valid", "goal_entry_x", "goal_entry_z"):
        data[key] = [[0] * len(ticks)]
    return data


def test_goal_prefix_ignores_later_goals_and_includes_last_tick():
    assert trace.goal_prefix(goal_data([720]), goal_data([720, 800]))[0]["exact"]
    assert not trace.goal_prefix(goal_data([719]), goal_data([720, 800]))[0]["exact"]


def test_goal_prefix_checks_side_and_entry_not_only_count():
    data = goal_data([100])
    data["goal_scorer"] = [[1]]
    assert not trace.goal_prefix(data, goal_data([100]))[0]["exact"]
    data["goal_scorer"] = [[0]]
    data["goal_entry_x"] = [[0.01]]
    assert not trace.goal_prefix(data, goal_data([100]))[0]["exact"]


def test_buffers_do_not_alias_or_mutate_source(monkeypatch):
    n = 2
    views = {k: torch.zeros((n, *s)) for k, s in trace.SHAPES.items()}
    runner = SimpleNamespace(num_worlds=n, bridge=SimpleNamespace(views=views),
        telemetry=SimpleNamespace(touch_count=torch.zeros(n, 2, dtype=torch.int32)),
        match_views={k: torch.zeros(n, dtype=torch.int32) for k in ("goal_count", "kickoff_active", "pending_reset")},
        actions=torch.zeros(n, 2, 8), rival_observation=torch.zeros(n, 2, 182),
        hidden=torch.zeros(1, n, 256), rival_action=torch.zeros(n, 8),
        host_tick=0, batch_index=torch.arange(n), rival_side=torch.tensor([0, 1]))
    monkeypatch.setattr(trace.wp, "to_torch", lambda x: x)
    buffers = trace.TraceBuffers(runner, ticks=4)
    buffers.state("pre", 0)
    buffers.action_input(runner)
    runner.hidden.fill_(2)
    runner.rival_action.fill_(1)
    buffers.action_output(runner)
    views["car_pos"].fill_(123)
    assert torch.count_nonzero(buffers.data["pre.car_pos"][0]) == 0
    assert torch.count_nonzero(buffers.data["hidden_before"]) == 0
    assert torch.all(buffers.data["hidden_after"] == 2)
    assert torch.all(buffers.data["policy_action"] == 1)
    assert buffers.action_calls == 1


def test_json_immutable_and_no_nan(tmp_path):
    path = tmp_path / "test.json"
    trace.save_json(path, {"exact": True})
    assert b"\r" not in path.read_bytes()
    assert json.loads(path.read_text()) == {"exact": True}
    with pytest.raises(FileExistsError):
        trace.save_json(path, {})
    with pytest.raises(ValueError):
        trace.save_json(tmp_path / "bad.json", {"bad": float("nan")})


def test_source_freezes_corrected_three_checkpoints_and_finite_scope():
    spec = trace.protocol()
    assert spec["ticks"] == 720 and spec["worlds"] == 10
    assert set(spec["cases"]) == {"parent650", "child10", "child25"}
    assert spec["evaluation"]["mode"] == "native_v5"
    assert spec["evaluation"]["nexto_seed"] == 2026090693
    assert spec["evaluation"]["optimizer_steps"] == 0
    assert len(spec["full_evaluation_hashes"]) == 3


def test_readonly_subclass_delegates_exactly_once():
    import inspect
    tick = inspect.getsource(trace.TracedRunner.tick)
    action = inspect.getsource(trace.TracedRunner._update_rival_action)
    assert tick.count("super().tick()") == 1
    assert action.count("super()._update_rival_action()") == 1
    assert "step_graph" not in tick and "set_actions" not in tick
    assert "forward_actor" not in action
