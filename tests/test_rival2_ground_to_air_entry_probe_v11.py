from __future__ import annotations

import torch

from rivalsim.rival2_contracts import (
    CAR_LINEAR_SPEED_SCALE,
    OBS_DIM,
    POSITION_SCALE,
)
from rivalsim.rival2_ground_to_air_entry_probe_v11 import (
    GroundToAirEntryProbeV11,
)
from rivalsim.rival2_ground_to_air_entry_v11 import (
    SETUP_ASSISTED_LOW_BOUNCE,
    SETUP_RISING_DOUBLE_JUMP,
)
from rivalsim.rival2_ground_to_air_human_bridge_v11 import (
    HumanAerialEnvelopeConfig,
)
from rivalsim.rival2_ground_to_air_option import FIELD


def _config() -> HumanAerialEnvelopeConfig:
    return HumanAerialEnvelopeConfig(
        target_car_height_uu=141.0,
        target_ball_height_uu=274.0,
        target_car_vertical_speed_uu_per_second=443.0,
        target_distance_uu=140.0,
        target_vertical_standoff_uu=133.0,
        distance_tolerance_uu=40.0,
        vertical_standoff_tolerance_uu=60.0,
        minimum_event_car_height_uu=141.0,
        minimum_event_ball_height_uu=274.0,
        minimum_event_car_vertical_speed_uu_per_second=265.0,
        maximum_event_distance_uu=157.0,
        maximum_bridge_ticks=180,
        car_height_weight=1.0,
        ball_height_weight=1.0,
        car_vertical_speed_weight=2.0,
        distance_weight=1.0,
        vertical_standoff_weight=1.0,
    )


def _observation(
    *,
    car_height: float,
    ball_height: float,
    car_vertical_speed: float,
    relative_y: float,
    relative_z: float,
    on_ground: bool,
    touch: bool = False,
    ball_vertical_speed: float = 0.0,
    ball_goalward_speed: float = 0.0,
) -> torch.Tensor:
    result = torch.zeros((1, 2, OBS_DIM), dtype=torch.float32)
    for side in (0, 1):
        result[0, side, FIELD["self.position.z"]] = car_height / POSITION_SCALE[2]
        result[0, side, FIELD["ball.position.z"]] = ball_height / POSITION_SCALE[2]
        result[0, side, FIELD["self.linear_velocity.z"]] = (
            car_vertical_speed / CAR_LINEAR_SPEED_SCALE
        )
        result[0, side, FIELD["relative.ball_position.y"]] = (
            relative_y / POSITION_SCALE[1]
        )
        result[0, side, FIELD["relative.ball_position.z"]] = (
            relative_z / POSITION_SCALE[2]
        )
        result[0, side, FIELD["ball.linear_velocity.z"]] = (
            ball_vertical_speed / 6000.0
        )
        result[0, side, FIELD["ball.linear_velocity.y"]] = (
            ball_goalward_speed / 6000.0
        )
        result[0, side, FIELD["self.on_ground"]] = float(on_ground)
        result[0, side, FIELD["lifecycle.self_touch_event"]] = float(touch)
    return result


def test_assisted_bounce_requires_a_separated_airborne_recontact() -> None:
    probe = GroundToAirEntryProbeV11(
        torch.tensor([SETUP_ASSISTED_LOW_BOUNCE]),
        attacker_side=0,
        envelope_config=_config(),
    )
    active = torch.tensor([True])
    before = _observation(
        car_height=17.0,
        ball_height=140.0,
        car_vertical_speed=0.0,
        relative_y=160.0,
        relative_z=123.0,
        on_ground=True,
        ball_vertical_speed=100.0,
    )
    setup = _observation(
        car_height=17.0,
        ball_height=145.0,
        car_vertical_speed=0.0,
        relative_y=140.0,
        relative_z=128.0,
        on_ground=True,
        touch=True,
        ball_vertical_speed=350.0,
        ball_goalward_speed=500.0,
    )
    first = probe.step(before, setup, tick=0, active=active)
    assert bool(first.first_contact.item())
    assert not bool(first.entry_airborne_contact.item())

    too_soon = _observation(
        car_height=60.0,
        ball_height=190.0,
        car_vertical_speed=150.0,
        relative_y=90.0,
        relative_z=130.0,
        on_ground=False,
        touch=True,
        ball_vertical_speed=300.0,
        ball_goalward_speed=500.0,
    )
    early = probe.step(setup, too_soon, tick=3, active=active)
    assert not bool(early.entry_airborne_contact.item())

    entry = too_soon.clone()
    entered = probe.step(too_soon, entry, tick=4, active=active)
    assert bool(entered.entry_airborne_contact.item())


def test_rising_double_jump_accepts_the_first_airborne_contact_as_entry() -> None:
    probe = GroundToAirEntryProbeV11(
        torch.tensor([SETUP_RISING_DOUBLE_JUMP]),
        attacker_side=1,
        envelope_config=_config(),
    )
    active = torch.tensor([True])
    before = _observation(
        car_height=80.0,
        ball_height=210.0,
        car_vertical_speed=250.0,
        relative_y=80.0,
        relative_z=130.0,
        on_ground=False,
        ball_vertical_speed=300.0,
    )
    touch = _observation(
        car_height=95.0,
        ball_height=225.0,
        car_vertical_speed=350.0,
        relative_y=70.0,
        relative_z=130.0,
        on_ground=False,
        touch=True,
        ball_vertical_speed=500.0,
        ball_goalward_speed=700.0,
    )
    events = probe.step(before, touch, tick=10, active=active)
    assert bool(events.first_contact.item())
    assert bool(events.entry_airborne_contact.item())


def test_envelope_second_contact_and_bounded_goal_are_independent_events() -> None:
    probe = GroundToAirEntryProbeV11(
        torch.tensor([SETUP_ASSISTED_LOW_BOUNCE]),
        attacker_side=0,
        envelope_config=_config(),
        separation_ticks=4,
        maximum_contacts=6,
    )
    active = torch.tensor([True])
    base = _observation(
        car_height=17.0,
        ball_height=140.0,
        car_vertical_speed=0.0,
        relative_y=150.0,
        relative_z=123.0,
        on_ground=True,
    )
    setup = base.clone()
    setup[:, :, FIELD["lifecycle.self_touch_event"]] = 1.0
    probe.step(base, setup, tick=0, active=active)
    entry = _observation(
        car_height=80.0,
        ball_height=195.0,
        car_vertical_speed=150.0,
        relative_y=80.0,
        relative_z=115.0,
        on_ground=False,
        touch=True,
    )
    probe.step(setup, entry, tick=4, active=active)
    target_y = (140.0**2 - 133.0**2) ** 0.5
    target = _observation(
        car_height=141.0,
        ball_height=274.0,
        car_vertical_speed=443.0,
        relative_y=target_y,
        relative_z=133.0,
        on_ground=False,
    )
    envelope = probe.step(entry, target, tick=5, active=active)
    assert bool(envelope.human_envelope_reached.item())
    second = target.clone()
    second[:, :, FIELD["lifecycle.self_touch_event"]] = 1.0
    second_event = probe.step(target, second, tick=8, active=active)
    assert bool(second_event.second_airborne_contact.item())
    goal = probe.step(
        second,
        target,
        tick=9,
        active=active,
        goal_for_attacker=torch.tensor([True]),
    )
    assert bool(goal.goal_within_contact_budget.item())
    telemetry = probe.telemetry()
    assert telemetry["fractions"]["entry_airborne_contact"] == 1.0
    assert telemetry["fractions"]["human_envelope_reached"] == 1.0
    assert telemetry["fractions"]["second_airborne_contact"] == 1.0
    assert telemetry["fractions"]["goal_within_contact_budget"] == 1.0


def test_seventh_contact_exceeds_budget_and_prevents_later_goal_credit() -> None:
    probe = GroundToAirEntryProbeV11(
        torch.tensor([SETUP_RISING_DOUBLE_JUMP]),
        attacker_side=0,
        envelope_config=_config(),
        separation_ticks=1,
        maximum_contacts=6,
    )
    active = torch.tensor([True])
    before = _observation(
        car_height=100.0,
        ball_height=230.0,
        car_vertical_speed=300.0,
        relative_y=70.0,
        relative_z=130.0,
        on_ground=False,
    )
    after = before.clone()
    after[:, :, FIELD["lifecycle.self_touch_event"]] = 1.0
    probe.step(before, after, tick=0, active=active)
    quiet = before.clone()
    budget = None
    for tick in range(1, 7):
        touch = quiet.clone()
        touch[:, :, FIELD["lifecycle.self_touch_event"]] = 1.0
        budget = probe.step(quiet, touch, tick=tick, active=active)
    assert budget is not None
    assert bool(budget.contact_budget_exceeded.item())
    goal = probe.step(
        quiet,
        quiet,
        tick=7,
        active=active,
        goal_for_attacker=torch.tensor([True]),
    )
    assert not bool(goal.goal_within_contact_budget.item())


def test_invalid_probe_fails_closed() -> None:
    try:
        GroundToAirEntryProbeV11(
            torch.tensor([[0]]),
            attacker_side=0,
            envelope_config=_config(),
        )
    except ValueError:
        pass
    else:
        raise AssertionError("two-dimensional setup was accepted")
    try:
        GroundToAirEntryProbeV11(
            torch.tensor([99]),
            attacker_side=0,
            envelope_config=_config(),
        )
    except ValueError:
        pass
    else:
        raise AssertionError("unknown setup was accepted")
