from __future__ import annotations

import numpy as np

from rivalsim.nexto_short_eval import ShortEvalTelemetry, classify_dash_events


def _empty_raw(*, worlds: int = 1, capacity: int = 4) -> dict[str, np.ndarray]:
    car_shape = (worlds, 2)
    event_shape = (*car_shape, capacity)
    raw: dict[str, np.ndarray] = {
        "event_count": np.zeros(car_shape, dtype=np.int32),
        "event_overflow": np.zeros(car_shape, dtype=np.int32),
    }
    for name in ShortEvalTelemetry._EVENT_INT_FIELDS:
        initial = (
            -1
            if name
            in {
                "event_tick",
                "event_last_wheel_contact_tick",
                "event_last_takeoff_tick",
                "event_last_landing_tick",
                "event_last_jump_rise_tick",
                "event_last_first_jump_tick",
                "event_landing_tick",
            }
            else 0
        )
        raw[name] = np.full(event_shape, initial, dtype=np.int32)
    for name in ShortEvalTelemetry._EVENT_FLOAT_FIELDS:
        raw[name] = np.zeros(event_shape, dtype=np.float32)
    for name in ShortEvalTelemetry._EVENT_VEC_FIELDS:
        raw[name] = np.zeros((*event_shape, 3), dtype=np.float32)
    raw["event_action"] = np.zeros((*event_shape, 8), dtype=np.float32)
    for name in (
        "event_suspension_length_before",
        "event_suspension_velocity_before",
        "event_landing_suspension_length",
        "event_landing_suspension_velocity",
    ):
        raw[name] = np.zeros((*event_shape, 4), dtype=np.float32)
    return raw


def _add_flip(
    raw: dict[str, np.ndarray],
    *,
    side: int = 0,
    ordinal: int = 0,
    tick: int,
    landing_tick: int,
    air_time: float = 0.05,
    last_landing_tick: int = -1,
    first_jump_tick: int = 96,
    wheel_before: int = 0,
    wheel_after: int = 0,
    speed_before: float = 900.0,
    speed_after_landing: float = 1200.0,
) -> None:
    raw["event_count"][0, side] = max(int(raw["event_count"][0, side]), ordinal + 1)
    index = (0, side, ordinal)
    raw["event_tick"][index] = tick
    raw["event_on_ground_before"][index] = int(wheel_before != 0)
    raw["event_on_ground_after"][index] = int(wheel_after != 0)
    raw["event_wheel_mask_before"][index] = wheel_before
    raw["event_wheel_mask_after"][index] = wheel_after
    raw["event_last_wheel_contact_tick"][index] = tick - 6
    raw["event_last_takeoff_tick"][index] = tick - 5
    raw["event_last_landing_tick"][index] = last_landing_tick
    if last_landing_tick >= 0:
        raw["event_prior_landing_wheel_mask"][index] = 0b0011
        raw["event_prior_landing_normal"][index] = (0.0, 0.0, 1.0)
    raw["event_last_jump_rise_tick"][index] = first_jump_tick
    raw["event_last_first_jump_tick"][index] = first_jump_tick
    raw["event_landing_tick"][index] = landing_tick
    raw["event_landing_wheel_mask"][index] = 0b0011
    raw["event_first_jump_wheel_mask_before"][index] = 0b0111
    raw["event_first_jump_wheel_mask_after"][index] = 0b0011
    raw["event_air_time_before"][index] = air_time
    raw["event_air_time_since_jump_before"][index] = max(0.0, (tick - first_jump_tick) / 120.0)
    raw["event_velocity_before"][index] = (speed_before, 0.0, -100.0)
    raw["event_velocity_after"][index] = (speed_before + 500.0, 0.0, -80.0)
    raw["event_landing_velocity"][index] = (
        speed_after_landing,
        0.0,
        0.0,
    )
    raw["event_post_landing_velocity"][index] = (
        speed_after_landing,
        0.0,
        0.0,
    )
    raw["event_landing_normal"][index] = (0.0, 0.0, 1.0)
    raw["event_action"][index] = (1.0, 0.0, -1.0, 0.0, 0.0, 1.0, 0.0, 0.0)


def _classify(raw: dict[str, np.ndarray]):
    return classify_dash_events(
        raw,
        rival_side=np.array([0], dtype=np.int32),
        starting_layout=np.array([0], dtype=np.int32),
        checkpoint_label="synthetic",
    )


def test_wavedash_requires_actual_airborne_flip_followed_by_fast_landing() -> None:
    raw = _empty_raw()
    _add_flip(raw, tick=100, landing_tick=106)

    events, summary = _classify(raw)

    assert len(events) == 1
    assert events[0]["policy"] == "Rival"
    assert "wavedash_candidate" in events[0]["candidate_labels"]
    assert "speed_increasing_wavedash_candidate" in events[0]["candidate_labels"]
    assert events[0]["flip_to_landing_ticks"] == 6
    assert summary["actual_flip_events_retained"] == 1


def test_grounded_dodge_is_not_called_a_wavedash() -> None:
    raw = _empty_raw()
    _add_flip(
        raw,
        tick=100,
        landing_tick=100,
        wheel_before=0b1111,
        wheel_after=0b1111,
    )

    events, _summary = _classify(raw)

    assert len(events) == 1
    assert events[0]["candidate_labels"] == ["ground_contact_dodge_candidate"]


def test_zapdash_requires_recent_landing_jump_and_wavedash_sequence() -> None:
    raw = _empty_raw()
    _add_flip(
        raw,
        tick=100,
        landing_tick=106,
        last_landing_tick=90,
        first_jump_tick=96,
    )

    events, _summary = _classify(raw)

    assert "zapdash_candidate" in events[0]["candidate_labels"]
    evidence = events[0]["classification_evidence"]["zapdash_candidate"]
    assert evidence["prior_landing_to_first_jump_ticks"] == 6
    assert evidence["first_jump_to_flip_ticks"] == 4


def test_two_landing_dashes_are_retained_as_one_double_dash_sequence() -> None:
    raw = _empty_raw()
    _add_flip(raw, ordinal=0, tick=100, landing_tick=106, first_jump_tick=96)
    _add_flip(
        raw,
        ordinal=1,
        tick=155,
        landing_tick=161,
        last_landing_tick=106,
        first_jump_tick=150,
    )

    events, summary = _classify(raw)

    assert len(events) == 2
    assert all("double_dash_candidate" in event["candidate_labels"] for event in events)
    assert summary["candidate_event_counts_by_policy"]["Rival"]["double_dash_candidate"] == 2
