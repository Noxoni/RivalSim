from types import SimpleNamespace

import pytest
import torch
import warp as wp

from benchmarks.validate_rival2_ssl_entity_match_interface import (
    check_idle_status,
    set_goal_fixture,
)


def test_dynamic_cuda_check_requires_completed_pilot():
    check_idle_status({"accepted_updates": 100, "status": "completed"})
    for status in ("rollout", "optimizing", "evaluating", "failed", "stopped_at_accepted_boundary"):
        with pytest.raises(RuntimeError, match="do not interrupt training"):
            check_idle_status({"accepted_updates": 100, "status": status})
    for count in (0, 50, 99, 101):
        with pytest.raises(RuntimeError):
            check_idle_status({"accepted_updates": count, "status": "completed"})


def test_goal_fixture_updates_authoritative_state_not_just_exported_telemetry():
    world = SimpleNamespace(ball_world=SimpleNamespace(
        position_bt=wp.zeros(2, dtype=wp.vec3, device="cpu"),
        velocity_bt=wp.zeros(2, dtype=wp.vec3, device="cpu"),
    ))
    bridge = SimpleNamespace(views={
        "ball_pos": torch.zeros(2, 3), "ball_vel": torch.zeros(2, 3),
    })
    set_goal_fixture(world, bridge, "cpu")
    torch.testing.assert_close(bridge.views["ball_pos"][0], torch.tensor([0., 5100., 120.]))
    torch.testing.assert_close(bridge.views["ball_vel"][0], torch.tensor([0., 6000., 0.]))
    for public, native in (("ball_pos", "position_bt"), ("ball_vel", "velocity_bt")):
        torch.testing.assert_close(
            wp.to_torch(getattr(world.ball_world, native)), bridge.views[public] * .02,
            rtol=0, atol=0,
        )
        assert bridge.views[public][1].count_nonzero() == 0
