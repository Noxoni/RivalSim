from __future__ import annotations

import torch

from rivalsim.rival2_contracts import POSITION_SCALE
from rivalsim.rival2_ground_to_air_option import FIELD
from rivalsim.rival2_ground_to_air_touch_geometry import (
    NaturalAerialTouchGeometryProbe,
)


def _observation(*, car_z: float, ball_z: float, touch: bool, grounded: bool) -> torch.Tensor:
    observation = torch.zeros((1, 2, 182), dtype=torch.float32)
    observation[:, 0, FIELD["self.position.z"]] = car_z / POSITION_SCALE[2]
    observation[:, 0, FIELD["ball.position.z"]] = ball_z / POSITION_SCALE[2]
    observation[:, 0, FIELD["relative.ball_position.z"]] = (
        ball_z - car_z
    ) / POSITION_SCALE[2]
    observation[:, 0, FIELD["lifecycle.self_touch_event"]] = float(touch)
    observation[:, 0, FIELD["self.on_ground"]] = float(grounded)
    return observation


def test_probe_separates_low_airborne_follow_from_strict_threshold() -> None:
    probe = NaturalAerialTouchGeometryProbe(1, attacker_side=0)
    active = torch.ones(1, dtype=torch.bool)
    before = _observation(car_z=17.0, ball_z=120.0, touch=False, grounded=True)
    setup = _observation(car_z=20.0, ball_z=130.0, touch=True, grounded=True)
    probe.step(before, setup, tick=2, active=active)

    before_follow = _observation(
        car_z=130.0, ball_z=225.0, touch=False, grounded=False
    )
    low_follow = _observation(
        car_z=142.0, ball_z=238.0, touch=True, grounded=False
    )
    probe.step(before_follow, low_follow, tick=30, active=active)
    telemetry = probe.telemetry()

    categories = telemetry["categories"]
    assert categories["first_distinct_follow"]["attempt_fraction"] == 1.0
    assert categories["first_airborne_follow"]["attempt_fraction"] == 1.0
    assert categories["first_strict_elevated_follow"]["attempt_fraction"] == 0.0
    assert (
        categories["first_airborne_follow"]["measurements"]["car_height_uu"][
            "p50"
        ]
        == 142.0
    )


def test_probe_counts_strict_follow_and_ignores_continuous_duplicate() -> None:
    probe = NaturalAerialTouchGeometryProbe(1, attacker_side=0)
    active = torch.ones(1, dtype=torch.bool)
    setup_before = _observation(
        car_z=17.0, ball_z=140.0, touch=False, grounded=True
    )
    setup = _observation(car_z=25.0, ball_z=145.0, touch=True, grounded=False)
    probe.step(setup_before, setup, tick=4, active=active)
    duplicate = _observation(
        car_z=155.0, ball_z=260.0, touch=True, grounded=False
    )
    probe.step(setup, duplicate, tick=6, active=active)
    assert (
        probe.telemetry()["categories"]["first_distinct_follow"][
            "attempt_fraction"
        ]
        == 0.0
    )
    probe.step(setup, duplicate, tick=12, active=active)
    categories = probe.telemetry()["categories"]
    assert categories["first_strict_elevated_follow"]["attempt_fraction"] == 1.0
    assert categories["first_distinct_follow"]["attempt_fraction"] == 1.0
