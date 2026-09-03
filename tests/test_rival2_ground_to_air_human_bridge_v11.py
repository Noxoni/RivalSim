from __future__ import annotations

import json
from dataclasses import asdict, replace
from pathlib import Path

import torch

from rivalsim.rival2_contracts import (
    CAR_LINEAR_SPEED_SCALE,
    OBS_DIM,
    POSITION_SCALE,
)
from rivalsim.rival2_ground_to_air_human_bridge_v11 import (
    HumanAerialEnvelopeConfig,
    HumanEnvelopeBridgeTrainingTracker,
    human_envelope_features,
)
from rivalsim.rival2_ground_to_air_option import FIELD

ROOT = Path(__file__).resolve().parents[1]


def config() -> HumanAerialEnvelopeConfig:
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


def observation(
    *,
    car_height: float,
    ball_height: float,
    car_vertical_speed: float,
    relative_y: float,
    relative_z: float,
) -> torch.Tensor:
    result = torch.zeros((1, 2, OBS_DIM), dtype=torch.float32)
    for side in (0, 1):
        result[:, side, FIELD["self.position.z"]] = (
            car_height / POSITION_SCALE[2]
        )
        result[:, side, FIELD["ball.position.z"]] = (
            ball_height / POSITION_SCALE[2]
        )
        result[:, side, FIELD["self.linear_velocity.z"]] = (
            car_vertical_speed / CAR_LINEAR_SPEED_SCALE
        )
        result[:, side, FIELD["relative.ball_position.y"]] = (
            relative_y / POSITION_SCALE[1]
        )
        result[:, side, FIELD["relative.ball_position.z"]] = (
            relative_z / POSITION_SCALE[2]
        )
    return result


def test_measured_human_envelope_reaches_full_potential_and_event() -> None:
    measured = observation(
        car_height=141.0,
        ball_height=274.0,
        car_vertical_speed=443.0,
        relative_y=(140.0**2 - 133.0**2) ** 0.5,
        relative_z=133.0,
    )
    features = human_envelope_features(measured, side=0, config=config())
    assert torch.allclose(features["potential"], torch.ones(1), atol=1.0e-5)
    assert bool(features["envelope"].item())


def test_low_v10_like_touch_has_room_for_causal_progress() -> None:
    low = observation(
        car_height=80.0,
        ball_height=196.0,
        car_vertical_speed=120.0,
        relative_y=80.0,
        relative_z=120.0,
    )
    higher = observation(
        car_height=115.0,
        ball_height=235.0,
        car_vertical_speed=300.0,
        relative_y=65.0,
        relative_z=125.0,
    )
    low_features = human_envelope_features(low, side=1, config=config())
    high_features = human_envelope_features(higher, side=1, config=config())
    assert float(high_features["potential"]) > float(low_features["potential"])
    assert not bool(low_features["envelope"].item())
    assert not bool(high_features["envelope"].item())


def test_falling_or_distant_states_do_not_satisfy_event() -> None:
    falling = observation(
        car_height=180.0,
        ball_height=320.0,
        car_vertical_speed=-100.0,
        relative_y=40.0,
        relative_z=140.0,
    )
    distant = observation(
        car_height=180.0,
        ball_height=320.0,
        car_vertical_speed=400.0,
        relative_y=300.0,
        relative_z=140.0,
    )
    assert not bool(
        human_envelope_features(falling, side=0, config=config())["envelope"].item()
    )
    assert not bool(
        human_envelope_features(distant, side=0, config=config())["envelope"].item()
    )


def test_invalid_envelope_fails_closed() -> None:
    invalid = replace(config(), maximum_bridge_ticks=0)
    try:
        invalid.validate()
    except ValueError as error:
        assert "positive" in str(error)
    else:
        raise AssertionError("invalid human bridge configuration was accepted")


def test_tracker_pays_only_after_native_prompt_transition() -> None:
    authority = json.loads(
        (
            ROOT / "results/rival2/ground_to_air_natural_v10/authority.json"
        ).read_text(encoding="utf-8")
    )
    authority["human_bridge"] = asdict(config())
    authority["reward"]["human_bridge_progress_per_potential_unit"] = 1.0
    authority["reward"]["human_bridge_envelope_event"] = 2.0
    tracker = HumanEnvelopeBridgeTrainingTracker(
        1,
        attacker_side=0,
        horizon=600,
        authority=authority,
    )
    active = torch.ones(1, dtype=torch.bool)
    no_goal = torch.zeros(1, dtype=torch.bool)

    setup_before = observation(
        car_height=17.0,
        ball_height=150.0,
        car_vertical_speed=0.0,
        relative_y=40.0,
        relative_z=133.0,
    )
    setup_after = setup_before.clone()
    setup_after[:, 0, FIELD["lifecycle.self_touch_event"]] = 1.0
    tracker.step(
        setup_before,
        setup_after,
        tick=0,
        goal_for_attacker=no_goal,
        any_goal=no_goal,
        active=active,
    )
    assert tracker.human_bridge_active_ticks == 0

    prompt_before = observation(
        car_height=70.0,
        ball_height=185.0,
        car_vertical_speed=100.0,
        relative_y=80.0,
        relative_z=115.0,
    )
    prompt_after = prompt_before.clone()
    prompt_after[:, 0, FIELD["lifecycle.self_touch_event"]] = 1.0
    tracker.step(
        prompt_before,
        prompt_after,
        tick=5,
        goal_for_attacker=no_goal,
        any_goal=no_goal,
        active=active,
    )
    assert tracker.human_bridge_active_ticks == 1
    assert tracker.human_bridge_progress_reward_sum == 0.0

    bridge_after = observation(
        car_height=115.0,
        ball_height=235.0,
        car_vertical_speed=300.0,
        relative_y=65.0,
        relative_z=125.0,
    )
    tracker.step(
        prompt_after,
        bridge_after,
        tick=6,
        goal_for_attacker=no_goal,
        any_goal=no_goal,
        active=active,
    )
    assert tracker.human_bridge_progress_reward_sum > 0.0
    assert tracker.human_bridge_envelope_reached == 0

    target = observation(
        car_height=141.0,
        ball_height=274.0,
        car_vertical_speed=443.0,
        relative_y=(140.0**2 - 133.0**2) ** 0.5,
        relative_z=133.0,
    )
    tracker.step(
        bridge_after,
        target,
        tick=7,
        goal_for_attacker=no_goal,
        any_goal=no_goal,
        active=active,
    )
    assert tracker.human_bridge_envelope_reached == 1
    assert tracker.human_bridge_event_reward_sum == 2.0
    assert tracker.telemetry()["human_bridge_envelope_fraction"] == 1.0
