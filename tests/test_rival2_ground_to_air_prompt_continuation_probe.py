from __future__ import annotations

import pytest
import torch

from rivalsim.rival2_contracts import (
    BALL_LINEAR_SPEED_SCALE,
    CAR_LINEAR_SPEED_SCALE,
    POSITION_SCALE,
)
from rivalsim.rival2_ground_to_air_option import FIELD
from rivalsim.rival2_ground_to_air_prompt_continuation_probe import (
    PromptContinuationProbe,
)


def _observation() -> torch.Tensor:
    value = torch.zeros((1, 2, 182), dtype=torch.float32)
    value[:, :, FIELD["self.on_ground"]] = 1.0
    return value


def _set_geometry(
    value: torch.Tensor,
    *,
    car_z: float,
    ball_z: float,
    relative: tuple[float, float, float],
    on_ground: bool,
    touch: bool,
    ball_velocity: tuple[float, float, float] = (0.0, 0.0, 0.0),
    car_vertical_speed: float = 0.0,
) -> None:
    value[:, 0, FIELD["self.position.z"]] = car_z / POSITION_SCALE[2]
    value[:, 0, FIELD["ball.position.z"]] = ball_z / POSITION_SCALE[2]
    for axis, component, scale in zip("xyz", relative, POSITION_SCALE, strict=True):
        value[:, 0, FIELD[f"relative.ball_position.{axis}"]] = component / scale
    value[:, 0, FIELD["self.on_ground"]] = float(on_ground)
    value[:, 0, FIELD["lifecycle.self_touch_event"]] = float(touch)
    for axis, component in zip("xyz", ball_velocity, strict=True):
        value[:, 0, FIELD[f"ball.linear_velocity.{axis}"]] = (
            component / BALL_LINEAR_SPEED_SCALE
        )
    value[:, 0, FIELD["self.linear_velocity.z"]] = (
        car_vertical_speed / CAR_LINEAR_SPEED_SCALE
    )


def test_probe_tracks_prompt_contact_and_continuation_without_mutation() -> None:
    probe = PromptContinuationProbe(1, attacker_side=0)
    active = torch.ones(1, dtype=torch.bool)

    before = _observation()
    after = before.clone()
    _set_geometry(
        before,
        car_z=17.0,
        ball_z=190.0,
        relative=(0.0, 80.0, 173.0),
        on_ground=True,
        touch=False,
        ball_velocity=(0.0, 400.0, -50.0),
    )
    _set_geometry(
        after,
        car_z=17.0,
        ball_z=195.0,
        relative=(0.0, 80.0, 178.0),
        on_ground=True,
        touch=True,
        ball_velocity=(0.0, 500.0, 100.0),
    )
    original = after.clone()
    probe.step(before, after, tick=0, active=active)
    torch.testing.assert_close(after, original, rtol=0.0, atol=0.0)

    before = after.clone()
    before[:, 0, FIELD["lifecycle.self_touch_event"]] = 0.0
    after = before.clone()
    _set_geometry(
        before,
        car_z=70.0,
        ball_z=185.0,
        relative=(0.0, 90.0, 115.0),
        on_ground=False,
        touch=False,
        ball_velocity=(0.0, 500.0, 100.0),
        car_vertical_speed=350.0,
    )
    _set_geometry(
        after,
        car_z=75.0,
        ball_z=195.0,
        relative=(0.0, 90.0, 120.0),
        on_ground=False,
        touch=True,
        ball_velocity=(0.0, 700.0, 450.0),
        car_vertical_speed=400.0,
    )
    probe.step(before, after, tick=5, active=active)

    before = after.clone()
    before[:, 0, FIELD["lifecycle.self_touch_event"]] = 0.0
    after = before.clone()
    _set_geometry(
        before,
        car_z=145.0,
        ball_z=245.0,
        relative=(0.0, 75.0, 100.0),
        on_ground=False,
        touch=False,
        ball_velocity=(0.0, 720.0, 350.0),
        car_vertical_speed=380.0,
    )
    _set_geometry(
        after,
        car_z=160.0,
        ball_z=260.0,
        relative=(0.0, 75.0, 100.0),
        on_ground=False,
        touch=False,
        ball_velocity=(0.0, 740.0, 300.0),
        car_vertical_speed=360.0,
    )
    probe.step(before, after, tick=6, active=active)

    before = after.clone()
    after = before.clone()
    _set_geometry(
        before,
        car_z=260.0,
        ball_z=305.0,
        relative=(0.0, 70.0, 45.0),
        on_ground=False,
        touch=False,
        ball_velocity=(0.0, 800.0, 250.0),
        car_vertical_speed=300.0,
    )
    _set_geometry(
        after,
        car_z=270.0,
        ball_z=315.0,
        relative=(0.0, 70.0, 45.0),
        on_ground=False,
        touch=True,
        ball_velocity=(0.0, 900.0, 300.0),
        car_vertical_speed=290.0,
    )
    probe.step(before, after, tick=10, active=active)

    telemetry = probe.telemetry()
    assert telemetry["prompt_airborne_follow_fraction"] == pytest.approx(1.0)
    assert telemetry["second_recontact_fraction"] == pytest.approx(1.0)
    assert telemetry["bridge_elevated_fraction"] == pytest.approx(1.0)
    assert telemetry["bridge_high_fraction"] == pytest.approx(1.0)
    assert telemetry["prompt_contact"][
        "ball_vertical_transfer_uu_per_second"
    ]["p50"] == pytest.approx(350.0)
    assert telemetry["prompt_contact"][
        "ball_goalward_transfer_uu_per_second"
    ]["p50"] == pytest.approx(200.0)
    assert telemetry["continuation"]["maximum_ball_height_uu"][
        "maximum"
    ] == pytest.approx(315.0)
    assert telemetry["continuation"]["maximum_close_streak_ticks"][
        "maximum"
    ] == pytest.approx(2.0)


def test_probe_rejects_misaligned_shapes_and_empty_telemetry() -> None:
    probe = PromptContinuationProbe(1, attacker_side=0)
    with pytest.raises(RuntimeError, match="no observations"):
        probe.telemetry()
    with pytest.raises(ValueError, match=r"must be \[N,2,182\]"):
        probe.step(
            torch.zeros((1, 182)),
            torch.zeros((1, 182)),
            tick=0,
            active=torch.ones(1, dtype=torch.bool),
        )
