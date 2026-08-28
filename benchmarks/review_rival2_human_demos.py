"""Review native Rival 2.0 human demonstrations without labeling mechanics.

This tool is intentionally an evidence/indexing pass, not a mechanic detector.  It:

* validates every native recorder session and binds the review to source hashes;
* classifies only the user-declared workflow (gameplay, mechanic practice, smoke);
* segments freeplay practice with generic 120 Hz activity/recovery evidence; and
* records descriptive physical outcomes without rewards or success labels.

The relative outcome buckets are triage aids.  They do not assert that the session's
named mechanic occurred, and they are not suitable as reward or training labels.
"""

from __future__ import annotations

import argparse
import bisect
import hashlib
import json
import math
import os
import statistics
from collections import Counter
from collections.abc import Iterable, Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from rivalsim.human_demo import SessionReader
from rivalsim.human_demo.format import ACTION_NAMES

FORMAT = "RIVALRL_HUMAN_DEMO_REVIEW_V1"
SEGMENTATION_VERSION = "GENERIC_ACTIVITY_RECOVERY_V1"
OUTCOME_GROUPING_VERSION = "DESCRIPTIVE_RELATIVE_OUTCOME_V1"
PRE_ROLL_TICKS = 60
RECOVERY_RUN_TICKS = 24
MIN_RECOVERY_WHEEL_CONTACTS = 2
MIN_POST_ACTIVITY_TICKS = 24
MAX_ATTEMPT_TICKS = 960
TOUCH_OUTCOME_TICKS = 12
POST_ATTEMPT_EVENT_HORIZON_TICKS = 360
HARD_BOUNDARY_KINDS = {
    "kickoff_or_round_reset",
    "freeplay_reset",
    "local_car_rebind",
    "respawn",
}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest().upper()


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True, ensure_ascii=True) + "\n",
        encoding="utf-8",
        newline="\n",
    )


def _write_jsonl(path: Path, values: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as stream:
        for value in values:
            stream.write(json.dumps(value, sort_keys=True, ensure_ascii=True) + "\n")


def _finite(value: float | int | None) -> float | None:
    if value is None:
        return None
    result = float(value)
    return result if math.isfinite(result) else None


def _vec(value: Sequence[float | int] | None) -> list[float] | None:
    if value is None:
        return None
    result = [float(item) for item in value]
    return result if all(math.isfinite(item) for item in result) else None


def _sub(a: Sequence[float], b: Sequence[float]) -> list[float]:
    return [float(x) - float(y) for x, y in zip(a, b, strict=True)]


def _dot(a: Sequence[float], b: Sequence[float]) -> float:
    return sum(float(x) * float(y) for x, y in zip(a, b, strict=True))


def _norm(value: Sequence[float]) -> float:
    return math.sqrt(_dot(value, value))


def _unit(value: Sequence[float]) -> list[float] | None:
    magnitude = _norm(value)
    if magnitude <= 1e-9:
        return None
    return [float(item) / magnitude for item in value]


def _direction_change_degrees(before: Sequence[float], after: Sequence[float]) -> float | None:
    before_unit = _unit(before)
    after_unit = _unit(after)
    if before_unit is None or after_unit is None:
        return None
    return math.degrees(math.acos(max(-1.0, min(1.0, _dot(before_unit, after_unit)))))


def _rotator_degrees(rotation: Sequence[float | int]) -> list[float]:
    return [float(value) * 360.0 / 65536.0 for value in rotation]


def _wrapped_degrees_delta(before: float, after: float) -> float:
    return (after - before + 180.0) % 360.0 - 180.0


def _orientation_basis(rotation: Sequence[float | int]) -> dict[str, list[float]]:
    pitch, yaw, roll = [float(value) * 2.0 * math.pi / 65536.0 for value in rotation]
    sp, cp = math.sin(pitch), math.cos(pitch)
    sy, cy = math.sin(yaw), math.cos(yaw)
    sr, cr = math.sin(roll), math.cos(roll)
    return {
        "forward": [cp * cy, cp * sy, sp],
        "right": [sr * sp * cy - cr * sy, sr * sp * sy + cr * cy, -sr * cp],
        "up": [-(cr * sp * cy + sr * sy), cy * sr - cr * sp * sy, cr * cp],
    }


def _human(frame: dict[str, Any]) -> dict[str, Any] | None:
    cars = [car for car in frame.get("cars", []) if car.get("flags", {}).get("is_local_human")]
    return cars[0] if len(cars) == 1 else None


def _speed(record: dict[str, Any], key: str = "linear_velocity") -> float:
    return _norm(record.get(key, [0.0, 0.0, 0.0]))


def _planar_speed(record: dict[str, Any]) -> float:
    velocity = record.get("linear_velocity", [0.0, 0.0, 0.0])
    return math.hypot(float(velocity[0]), float(velocity[1]))


def _stable_recovery(frame: dict[str, Any]) -> bool:
    car = _human(frame)
    if car is None:
        return False
    return bool(
        car.get("flags", {}).get("on_ground")
        and int(car.get("num_wheel_world_contacts", 0)) >= MIN_RECOVERY_WHEEL_CONTACTS
        and not car.get("jump_component", {}).get("active")
        and not car.get("dodge_component", {}).get("active")
        and not car.get("flip_component", {}).get("active")
        and not car.get("flags", {}).get("demolished")
    )


def _runs_at_least(
    values: Sequence[bool], start: int, stop: int, length: int
) -> list[tuple[int, int]]:
    runs: list[tuple[int, int]] = []
    run_start: int | None = None
    for index in range(max(0, start), min(len(values), stop)):
        if values[index]:
            if run_start is None:
                run_start = index
        elif run_start is not None:
            if index - run_start >= length:
                runs.append((run_start, index - 1))
            run_start = None
    if run_start is not None and min(len(values), stop) - run_start >= length:
        runs.append((run_start, min(len(values), stop) - 1))
    return runs


def _event_index(event: dict[str, Any], physics_frames: Sequence[int]) -> int | None:
    physics = int(event.get("physics_frame", -1))
    if physics < 0 or not physics_frames:
        return None
    index = bisect.bisect_left(physics_frames, physics)
    if index >= len(physics_frames):
        return len(physics_frames) - 1
    if index > 0 and abs(physics_frames[index - 1] - physics) < abs(
        physics_frames[index] - physics
    ):
        return index - 1
    return index


def _hard_boundaries(
    frames: Sequence[dict[str, Any]], events: Sequence[dict[str, Any]], physics: Sequence[int]
) -> set[int]:
    boundaries: set[int] = {0, len(frames)}
    for index in range(1, len(frames)):
        if (
            int(frames[index]["sequence"]) != int(frames[index - 1]["sequence"]) + 1
            or int(frames[index]["physics_frame"]) != int(frames[index - 1]["physics_frame"]) + 1
        ):
            boundaries.add(index)
    for event in events:
        if str(event.get("kind", "")) in HARD_BOUNDARY_KINDS:
            index = _event_index(event, physics)
            if index is not None:
                boundaries.add(index)
    return boundaries


def _bounded_region(index: int, boundaries: Sequence[int]) -> tuple[int, int]:
    position = bisect.bisect_right(boundaries, index)
    left = boundaries[max(0, position - 1)]
    right = boundaries[min(len(boundaries) - 1, position)]
    if right <= index and position + 1 < len(boundaries):
        right = boundaries[position + 1]
    return left, right


def _human_jump_anchors(
    events: Sequence[dict[str, Any]], physics: Sequence[int], human_id: str
) -> list[tuple[int, dict[str, Any]]]:
    result = []
    for event in events:
        if event.get("kind") != "jump_onset":
            continue
        actor = str(event.get("actor_id", ""))
        if human_id and actor and actor != human_id:
            continue
        index = _event_index(event, physics)
        if index is not None:
            result.append((index, event))
    result.sort(key=lambda item: (item[0], int(item[1].get("monotonic_ns", 0))))
    return result


def _segment_attempts(
    frames: Sequence[dict[str, Any]], events: Sequence[dict[str, Any]], human_id: str
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    physics = [int(frame["physics_frame"]) for frame in frames]
    stable = [_stable_recovery(frame) for frame in frames]
    boundaries = sorted(_hard_boundaries(frames, events, physics))
    anchors = _human_jump_anchors(events, physics, human_id)
    groups: list[list[tuple[int, dict[str, Any]]]] = []
    for anchor in anchors:
        if not groups:
            groups.append([anchor])
            continue
        previous_index = groups[-1][-1][0]
        current_index = anchor[0]
        hard_between = any(previous_index < boundary <= current_index for boundary in boundaries)
        recovered_between = bool(
            _runs_at_least(stable, previous_index + 1, current_index, RECOVERY_RUN_TICKS)
        )
        if (
            hard_between
            or recovered_between
            or current_index - groups[-1][0][0] > MAX_ATTEMPT_TICKS
        ):
            groups.append([anchor])
        else:
            groups[-1].append(anchor)

    spans: list[dict[str, Any]] = []
    for group_index, group in enumerate(groups):
        first_anchor = group[0][0]
        last_anchor = group[-1][0]
        region_start, region_stop = _bounded_region(first_anchor, boundaries)
        start = max(region_start, first_anchor - PRE_ROLL_TICKS)
        search_start = min(region_stop, last_anchor + MIN_POST_ACTIVITY_TICKS)
        recovery_runs = _runs_at_least(
            stable,
            search_start,
            min(region_stop, last_anchor + MAX_ATTEMPT_TICKS + 1),
            RECOVERY_RUN_TICKS,
        )
        recovered = bool(recovery_runs)
        recovery_start = recovery_runs[0][0] if recovery_runs else None
        recovery_end = recovery_runs[0][1] if recovery_runs else None
        end = (
            recovery_end
            if recovery_end is not None
            else min(region_stop - 1, last_anchor + MAX_ATTEMPT_TICKS)
        )
        if group_index + 1 < len(groups):
            next_anchor = groups[group_index + 1][0][0]
            midpoint = (last_anchor + next_anchor) // 2
            end = min(end, midpoint, next_anchor - 1)
        spans.append(
            {
                "start_index": start,
                "end_index": max(start, end),
                "anchor_indices": [item[0] for item in group],
                "anchor_events": [item[1] for item in group],
                "recovered": recovered and recovery_start is not None and recovery_start <= end,
                "recovery_start_index": recovery_start,
            }
        )

    for index in range(1, len(spans)):
        if spans[index]["start_index"] <= spans[index - 1]["end_index"]:
            previous_anchor = spans[index - 1]["anchor_indices"][-1]
            current_anchor = spans[index]["anchor_indices"][0]
            split = (previous_anchor + current_anchor) // 2
            spans[index - 1]["end_index"] = max(spans[index - 1]["start_index"], split)
            spans[index]["start_index"] = min(spans[index]["end_index"], split + 1)

    audit = {
        "version": SEGMENTATION_VERSION,
        "policy": (
            "Label-agnostic activity segmentation: local-human native jump-onset anchors; "
            "physics/reset/respawn/rebind hard boundaries; sustained two-wheel grounded "
            "recovery separates attempts. No mechanic detector is used."
        ),
        "pre_roll_ticks": PRE_ROLL_TICKS,
        "recovery_run_ticks": RECOVERY_RUN_TICKS,
        "minimum_recovery_world_contact_wheels": MIN_RECOVERY_WHEEL_CONTACTS,
        "minimum_post_activity_ticks": MIN_POST_ACTIVITY_TICKS,
        "maximum_attempt_ticks": MAX_ATTEMPT_TICKS,
        "anchor_count": len(anchors),
        "attempt_count": len(spans),
        "hard_boundary_indices": boundaries,
        "unsegmented_policy": "No synthetic attempt is created when no native jump onset exists.",
    }
    return spans, audit


def _find_exact_physics(physics_to_index: dict[int, int], physics: int) -> int | None:
    return physics_to_index.get(physics)


def _frame_car_state(frame: dict[str, Any]) -> dict[str, Any] | None:
    car = _human(frame)
    if car is None:
        return None
    rotation = car.get("rotation", [0, 0, 0])
    return {
        "position": _vec(car.get("position")),
        "rotation_unreal_units": [int(value) for value in rotation],
        "rotation_degrees": _rotator_degrees(rotation),
        "orientation_basis": _orientation_basis(rotation),
        "linear_velocity": _vec(car.get("linear_velocity")),
        "speed_uu_per_s": _speed(car),
        "planar_speed_uu_per_s": _planar_speed(car),
        "angular_velocity": _vec(car.get("angular_velocity")),
        "angular_speed_rad_per_s": _speed(car, "angular_velocity"),
        "boost": _finite(car.get("boost")),
        "on_ground": bool(car.get("flags", {}).get("on_ground")),
        "can_jump": bool(car.get("flags", {}).get("can_jump")),
        "has_flip": bool(car.get("flags", {}).get("has_flip")),
        "jumped": bool(car.get("flags", {}).get("jumped")),
        "double_jumped": bool(car.get("flags", {}).get("double_jumped")),
        "wheel_world_contact_count": int(car.get("num_wheel_world_contacts", 0)),
        "jump_component_active": bool(car.get("jump_component", {}).get("active")),
        "dodge_component_active": bool(car.get("dodge_component", {}).get("active")),
        "dodge_direction": _vec(car.get("dodge_component", {}).get("direction")),
        "flip_component_active": bool(car.get("flip_component", {}).get("active")),
        "flip_time": _finite(car.get("flip_component", {}).get("flip_time")),
        "individual_wheels": [
            {
                "index": int(wheel.get("index", wheel_index)),
                "has_contact": bool(wheel.get("has_contact")),
                "has_world_contact": bool(wheel.get("has_world_contact")),
                "contact_location": _vec(wheel.get("contact_location")),
                "contact_normal": _vec(wheel.get("contact_normal")),
            }
            for wheel_index, wheel in enumerate(car.get("wheels", []))
        ],
    }


def _frame_ball_state(frame: dict[str, Any]) -> dict[str, Any]:
    ball = frame["ball"]
    return {
        "position": _vec(ball.get("position")),
        "rotation_unreal_units": [int(value) for value in ball.get("rotation", [0, 0, 0])],
        "linear_velocity": _vec(ball.get("linear_velocity")),
        "speed_uu_per_s": _speed(ball),
        "angular_velocity": _vec(ball.get("angular_velocity")),
        "angular_speed_rad_per_s": _speed(ball, "angular_velocity"),
    }


def _touch_episodes(events: Sequence[dict[str, Any]], human_id: str) -> list[list[dict[str, Any]]]:
    touches = []
    for event in events:
        if event.get("kind") != "ball_touch":
            continue
        actor = str(event.get("contacting_car_id") or event.get("actor_id") or "")
        if human_id and actor != human_id:
            continue
        touches.append(event)
    touches.sort(
        key=lambda event: (int(event.get("physics_frame", -1)), int(event.get("monotonic_ns", 0)))
    )
    episodes: list[list[dict[str, Any]]] = []
    for touch in touches:
        if (
            not episodes
            or int(touch.get("physics_frame", -1)) - int(episodes[-1][-1].get("physics_frame", -1))
            > 2
        ):
            episodes.append([touch])
        else:
            episodes[-1].append(touch)
    return episodes


def _contact_telemetry(
    episode: Sequence[dict[str, Any]],
    frames: Sequence[dict[str, Any]],
    physics_to_index: dict[int, int],
) -> dict[str, Any]:
    first = episode[0]
    event_physics = int(first.get("physics_frame", -1))
    contact_index = _find_exact_physics(physics_to_index, event_physics)
    pre_index = _find_exact_physics(physics_to_index, event_physics - 1)
    post_index = _find_exact_physics(physics_to_index, event_physics + TOUCH_OUTCOME_TICKS)
    contact_frame = frames[contact_index] if contact_index is not None else None
    pre_frame = frames[pre_index] if pre_index is not None else None
    post_frame = frames[post_index] if post_index is not None else None
    before_velocity = _vec(pre_frame["ball"]["linear_velocity"]) if pre_frame else None
    after_velocity = _vec(post_frame["ball"]["linear_velocity"]) if post_frame else None
    delta_velocity = (
        _sub(after_velocity, before_velocity)
        if before_velocity is not None and after_velocity is not None
        else None
    )
    car_state = _frame_car_state(contact_frame) if contact_frame else None
    ball_state = _frame_ball_state(contact_frame) if contact_frame else None
    geometry: dict[str, Any] | None = None
    if car_state and ball_state and car_state["position"] and ball_state["position"]:
        car_to_ball = _sub(ball_state["position"], car_state["position"])
        car_to_ball_unit = _unit(car_to_ball)
        basis = car_state["orientation_basis"]
        normal = _vec(first.get("hit_normal"))
        geometry = {
            "hit_location": _vec(first.get("hit_location")),
            "hit_normal": normal,
            "car_to_ball_vector": car_to_ball,
            "car_to_ball_distance_uu": _norm(car_to_ball),
            "car_to_ball_forward_alignment": (
                _dot(car_to_ball_unit, basis["forward"]) if car_to_ball_unit else None
            ),
            "car_to_ball_up_alignment": (
                _dot(car_to_ball_unit, basis["up"]) if car_to_ball_unit else None
            ),
            "hit_normal_forward_alignment": (
                _dot(normal, basis["forward"]) if normal is not None else None
            ),
            "hit_normal_up_alignment": (_dot(normal, basis["up"]) if normal is not None else None),
        }
    return {
        "event_count": len(episode),
        "first_event": first,
        "last_event": episode[-1],
        "contact_physics_frame": event_physics,
        "contact_sequence_boundary": int(first.get("sequence_boundary", -1)),
        "exact_pre_frame_available": pre_frame is not None,
        "exact_contact_frame_available": contact_frame is not None,
        "exact_post_12_frame_available": post_frame is not None,
        "ball_before": _frame_ball_state(pre_frame) if pre_frame else None,
        "ball_at_contact": ball_state,
        "ball_after_12_ticks": _frame_ball_state(post_frame) if post_frame else None,
        "ball_delta_velocity_12_ticks": delta_velocity,
        "ball_delta_velocity_magnitude_12_ticks": _norm(delta_velocity) if delta_velocity else None,
        "unit_mass_momentum_change_proxy_12_ticks": delta_velocity,
        "ball_speed_change_12_ticks": (
            _norm(after_velocity) - _norm(before_velocity)
            if before_velocity is not None and after_velocity is not None
            else None
        ),
        "ball_direction_change_degrees_12_ticks": (
            _direction_change_degrees(before_velocity, after_velocity)
            if before_velocity is not None and after_velocity is not None
            else None
        ),
        "car_at_contact": car_state,
        "contact_geometry": geometry,
        "momentum_note": (
            "Native recording contains velocity but no authoritative ball mass. The unit-mass "
            "momentum-change proxy equals delta velocity; no mass is invented."
        ),
    }


def _orientation_delta(start: dict[str, Any], end: dict[str, Any]) -> list[float] | None:
    start_rotation = start.get("rotation_degrees")
    end_rotation = end.get("rotation_degrees")
    if start_rotation is None or end_rotation is None:
        return None
    return [
        _wrapped_degrees_delta(before, after)
        for before, after in zip(start_rotation, end_rotation, strict=True)
    ]


def _attempt_telemetry(
    session_uuid: str,
    label: str,
    attempt_number: int,
    span: dict[str, Any],
    frames: Sequence[dict[str, Any]],
    events: Sequence[dict[str, Any]],
    human_id: str,
) -> dict[str, Any]:
    start_index = int(span["start_index"])
    end_index = int(span["end_index"])
    selected = frames[start_index : end_index + 1]
    start_frame, end_frame = selected[0], selected[-1]
    start_physics = int(start_frame["physics_frame"])
    end_physics = int(end_frame["physics_frame"])
    physics_to_index = {int(frame["physics_frame"]): index for index, frame in enumerate(frames)}
    selected_events = [
        event
        for event in events
        if start_physics <= int(event.get("physics_frame", -1)) <= end_physics
    ]
    episode_records = [
        episode
        for episode in _touch_episodes(selected_events, human_id)
        if start_physics <= int(episode[0].get("physics_frame", -1)) <= end_physics
    ]
    contacts = [
        _contact_telemetry(episode, frames, physics_to_index) for episode in episode_records
    ]
    primary_contact_index = None
    primary_value = -1.0
    for index, contact in enumerate(contacts):
        value = contact.get("ball_delta_velocity_magnitude_12_ticks")
        if value is not None and float(value) > primary_value:
            primary_value = float(value)
            primary_contact_index = index

    cars = [_human(frame) for frame in selected]
    valid_cars = [car for car in cars if car is not None]
    start_car = _frame_car_state(start_frame)
    end_car = _frame_car_state(end_frame)
    start_ball = _frame_ball_state(start_frame)
    end_ball = _frame_ball_state(end_frame)
    max_planar_speed = max((_planar_speed(car) for car in valid_cars), default=0.0)
    max_speed = max((_speed(car) for car in valid_cars), default=0.0)
    max_angular_speed = max((_speed(car, "angular_velocity") for car in valid_cars), default=0.0)
    max_ball_speed = max((_speed(frame["ball"]) for frame in selected), default=0.0)
    event_counts = Counter(str(event.get("kind", "unknown")) for event in selected_events)

    def no_intervening_activity(candidate: dict[str, Any]) -> bool:
        candidate_physics = int(candidate.get("physics_frame", -1))
        for event in events:
            event_physics = int(event.get("physics_frame", -1))
            if not end_physics < event_physics < candidate_physics:
                continue
            kind = str(event.get("kind", ""))
            actor = str(event.get("actor_id", ""))
            if kind in HARD_BOUNDARY_KINDS or (
                kind == "jump_onset" and (not human_id or not actor or actor == human_id)
            ):
                return False
        return True

    goal_events_after = [
        event
        for event in events
        if event.get("kind") == "goal"
        and end_physics
        < int(event.get("physics_frame", -1))
        <= end_physics + POST_ATTEMPT_EVENT_HORIZON_TICKS
        and no_intervening_activity(event)
    ]
    reset_events_after = [
        event
        for event in events
        if event.get("kind") in HARD_BOUNDARY_KINDS
        and end_physics
        < int(event.get("physics_frame", -1))
        <= end_physics + POST_ATTEMPT_EVENT_HORIZON_TICKS
    ]
    timeline_events = [
        {
            **event,
            "ticks_from_attempt_start": int(event.get("physics_frame", -1)) - start_physics,
            "seconds_from_attempt_start": float(event.get("engine_physics_time", 0.0))
            - float(start_frame["engine_physics_time"]),
        }
        for event in selected_events
    ]
    recovery_index = span.get("recovery_start_index")
    recovery_delay = None
    if span.get("recovered") and recovery_index is not None:
        recovery_delay = float(frames[int(recovery_index)]["engine_physics_time"]) - float(
            frames[int(span["anchor_indices"][-1])]["engine_physics_time"]
        )
    invalid_actions = 0
    for frame in selected:
        action = frame.get("rival_action", {})
        if not all(-1.0 <= float(action.get(name, 0.0)) <= 1.0 for name in ACTION_NAMES[:5]):
            invalid_actions += 1
    action_summary = {}
    for name in ACTION_NAMES:
        values = [frame["rival_action"][name] for frame in selected]
        if name in ACTION_NAMES[:5]:
            numeric = [float(value) for value in values]
            action_summary[name] = {
                "minimum": min(numeric),
                "maximum": max(numeric),
                "range": max(numeric) - min(numeric),
                "mean": statistics.fmean(numeric),
            }
        else:
            active_count = sum(bool(value) for value in values)
            action_summary[name] = {
                "active_tick_count": active_count,
                "active_fraction": active_count / len(values),
            }
    native_dodge_forward = [float(frame["native_input"]["dodge_forward"]) for frame in selected]
    native_dodge_strafe = [float(frame["native_input"]["dodge_strafe"]) for frame in selected]
    orientation_delta = _orientation_delta(start_car, end_car) if start_car and end_car else None
    return {
        "format": FORMAT,
        "session_uuid": session_uuid,
        "declared_label": label,
        "attempt_id": f"{session_uuid}:{attempt_number:04d}",
        "attempt_number": attempt_number,
        "segmentation": {
            "version": SEGMENTATION_VERSION,
            "start_sequence": int(start_frame["sequence"]),
            "end_sequence": int(end_frame["sequence"]),
            "start_physics_frame": start_physics,
            "end_physics_frame": end_physics,
            "anchor_physics_frames": [
                int(frames[index]["physics_frame"]) for index in span["anchor_indices"]
            ],
            "anchor_event_count": len(span["anchor_indices"]),
            "frame_count": len(selected),
            "duration_seconds": float(end_frame["engine_physics_time"])
            - float(start_frame["engine_physics_time"]),
            "generic_recovery_observed": bool(span.get("recovered")),
            "recovery_delay_from_last_anchor_seconds": recovery_delay,
        },
        "data_quality": {
            "missing_physics_frame_count": sum(
                int(frame.get("missing_physics_frames", 0)) for frame in selected
            ),
            "noncontiguous_adjacent_frame_count": sum(
                int(
                    int(selected[index]["physics_frame"])
                    != int(selected[index - 1]["physics_frame"]) + 1
                )
                for index in range(1, len(selected))
            ),
            "invalid_rival_action_frame_count": invalid_actions,
            "unique_human_car_frame_count": len(valid_cars),
            "complete_human_car_coverage": len(valid_cars) == len(selected),
        },
        "event_counts": dict(sorted(event_counts.items())),
        "timeline_events": timeline_events,
        "jump_onset_events": [
            event for event in selected_events if event.get("kind") == "jump_onset"
        ],
        "action": {
            "rival_projection": action_summary,
            "native_dodge_forward_minimum": min(native_dodge_forward),
            "native_dodge_forward_maximum": max(native_dodge_forward),
            "native_dodge_strafe_minimum": min(native_dodge_strafe),
            "native_dodge_strafe_maximum": max(native_dodge_strafe),
        },
        "car_motion": {
            "start": start_car,
            "end": end_car,
            "orientation_delta_degrees_wrapped": orientation_delta,
            "maximum_speed_uu_per_s": max_speed,
            "maximum_planar_speed_uu_per_s": max_planar_speed,
            "planar_speed_gain_from_start_to_peak_uu_per_s": (
                max_planar_speed - float(start_car["planar_speed_uu_per_s"]) if start_car else None
            ),
            "planar_speed_change_start_to_end_uu_per_s": (
                float(end_car["planar_speed_uu_per_s"]) - float(start_car["planar_speed_uu_per_s"])
                if start_car and end_car
                else None
            ),
            "maximum_angular_speed_rad_per_s": max_angular_speed,
            "maximum_height_uu": max(
                (float(car["position"][2]) for car in valid_cars), default=None
            ),
            "airborne_tick_count": sum(
                not bool(car.get("flags", {}).get("on_ground")) for car in valid_cars
            ),
            "minimum_wheel_world_contact_count": min(
                (int(car.get("num_wheel_world_contacts", 0)) for car in valid_cars),
                default=None,
            ),
            "maximum_wheel_world_contact_count": max(
                (int(car.get("num_wheel_world_contacts", 0)) for car in valid_cars),
                default=None,
            ),
            "jump_component_active_tick_count": sum(
                bool(car.get("jump_component", {}).get("active")) for car in valid_cars
            ),
            "dodge_component_active_tick_count": sum(
                bool(car.get("dodge_component", {}).get("active")) for car in valid_cars
            ),
            "flip_component_active_tick_count": sum(
                bool(car.get("flip_component", {}).get("active")) for car in valid_cars
            ),
            "has_flip_tick_count": sum(
                bool(car.get("flags", {}).get("has_flip")) for car in valid_cars
            ),
            "stable_recovery_tick_count": sum(_stable_recovery(frame) for frame in selected),
        },
        "ball_outcome": {
            "start": start_ball,
            "end": end_ball,
            "maximum_speed_uu_per_s": max_ball_speed,
            "net_velocity_change": _sub(end_ball["linear_velocity"], start_ball["linear_velocity"]),
            "net_velocity_change_magnitude": _norm(
                _sub(end_ball["linear_velocity"], start_ball["linear_velocity"])
            ),
            "net_direction_change_degrees": _direction_change_degrees(
                start_ball["linear_velocity"], end_ball["linear_velocity"]
            ),
            "human_touch_episode_count": len(contacts),
            "primary_contact_index": primary_contact_index,
            "contacts": contacts,
        },
        "post_attempt_events_within_3_seconds": {
            "goals": goal_events_after,
            "resets_or_respawns": reset_events_after,
        },
        "outcome_group": None,
        "interpretation_policy": (
            "Physical evidence only. This record does not assert mechanic correctness, assign "
            "reward, or create a training label."
        ),
    }


def _percentile(values: Sequence[float], fraction: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    position = (len(ordered) - 1) * fraction
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    weight = position - lower
    return ordered[lower] * (1.0 - weight) + ordered[upper] * weight


def _assign_outcome_groups(attempts: list[dict[str, Any]]) -> dict[str, Any]:
    contact_attempts = [
        attempt
        for attempt in attempts
        if int(attempt["ball_outcome"]["human_touch_episode_count"]) > 0
    ]
    contact_oriented = bool(attempts) and len(contact_attempts) / len(attempts) >= 0.25
    transfer_values = []
    for attempt in contact_attempts:
        primary_index = attempt["ball_outcome"]["primary_contact_index"]
        if primary_index is None:
            continue
        value = attempt["ball_outcome"]["contacts"][primary_index].get(
            "ball_delta_velocity_magnitude_12_ticks"
        )
        if value is not None:
            transfer_values.append(float(value))
    speed_gains = [
        float(attempt["car_motion"]["planar_speed_gain_from_start_to_peak_uu_per_s"])
        for attempt in attempts
        if attempt["car_motion"]["planar_speed_gain_from_start_to_peak_uu_per_s"] is not None
    ]
    comparison_values = transfer_values if contact_oriented else speed_gains
    comparison_available = len(comparison_values) >= 3
    q25 = _percentile(comparison_values, 0.25) if comparison_available else None
    q75 = _percentile(comparison_values, 0.75) if comparison_available else None
    counts: Counter[str] = Counter()
    for attempt in attempts:
        recovered = bool(attempt["segmentation"]["generic_recovery_observed"])
        goals = int(attempt["event_counts"].get("goal", 0)) + len(
            attempt["post_attempt_events_within_3_seconds"]["goals"]
        )
        if contact_oriented:
            primary_index = attempt["ball_outcome"]["primary_contact_index"]
            transfer = None
            if primary_index is not None:
                transfer = attempt["ball_outcome"]["contacts"][primary_index].get(
                    "ball_delta_velocity_magnitude_12_ticks"
                )
            if transfer is None:
                group = "limited_evidence_no_measured_human_contact"
                reasons = ["no exact human touch outcome was measurable in this segment"]
            elif recovered and (goals > 0 or (q75 is not None and float(transfer) >= q75)):
                group = "stronger_physical_outcome_candidate"
                reasons = [
                    "human contact measured",
                    "sustained recovery observed",
                    "goal timing or upper-quartile within-session velocity transfer",
                ]
            elif not recovered or (q25 is not None and float(transfer) <= q25):
                group = "limited_or_failed_physical_outcome_candidate"
                reasons = [
                    "no sustained recovery"
                    if not recovered
                    else "lower-quartile within-session velocity transfer"
                ]
            else:
                group = "ambiguous_middle_physical_outcome"
                reasons = ["contact outcome falls between the descriptive extremes"]
        else:
            gain = attempt["car_motion"]["planar_speed_gain_from_start_to_peak_uu_per_s"]
            if gain is not None and recovered and q75 is not None and float(gain) >= q75:
                group = "stronger_physical_outcome_candidate"
                reasons = ["sustained recovery", "upper-quartile within-session planar speed gain"]
            elif not recovered or (gain is not None and q25 is not None and float(gain) <= q25):
                group = "limited_or_failed_physical_outcome_candidate"
                reasons = [
                    "no sustained recovery"
                    if not recovered
                    else "lower-quartile within-session planar speed gain"
                ]
            else:
                group = "ambiguous_middle_physical_outcome"
                reasons = ["locomotion outcome falls between the descriptive extremes"]
        attempt["outcome_group"] = {"name": group, "reasons": reasons}
        counts[group] += 1
    return {
        "version": OUTCOME_GROUPING_VERSION,
        "contact_oriented_by_evidence": contact_oriented,
        "contact_attempt_fraction": len(contact_attempts) / len(attempts) if attempts else 0.0,
        "comparison_metric": (
            "primary_contact_ball_delta_velocity_magnitude_12_ticks"
            if contact_oriented
            else "planar_speed_gain_from_start_to_peak_uu_per_s"
        ),
        "comparison_value_count": len(comparison_values),
        "relative_quartile_comparison_available": comparison_available,
        "within_session_q25": q25,
        "within_session_q75": q75,
        "counts": dict(sorted(counts.items())),
        "policy": (
            "Relative descriptive triage only. Buckets are not mechanic detectors, success "
            "labels, rewards, or training targets; all attempts remain preserved."
        ),
    }


def _source_files(session_dir: Path) -> list[dict[str, Any]]:
    files = [path for path in session_dir.rglob("*") if path.is_file()]
    return [
        {
            "path": str(path.relative_to(session_dir)).replace("\\", "/"),
            "bytes": path.stat().st_size,
            "sha256": _sha256(path),
        }
        for path in sorted(files, key=lambda value: str(value.relative_to(session_dir)))
    ]


def _session_class(manifest: dict[str, Any]) -> tuple[str, str]:
    session_type = str(manifest.get("session_type", "")).lower()
    label = str(manifest.get("mechanic_label") or manifest.get("label") or "").lower()
    if "smoke" in label:
        return "recorder_smoke", "infrastructure_smoke"
    if session_type == "match":
        return "gameplay", "new_demonstration"
    if session_type == "freeplay":
        return "freeplay_mechanic_practice", "new_demonstration"
    return "unclassified", "new_demonstration"


def _match_summary(
    frames: Sequence[dict[str, Any]], events: Sequence[dict[str, Any]], human_id: str
) -> dict[str, Any]:
    event_counts = Counter(str(event.get("kind", "unknown")) for event in events)
    human_touches = sum(
        1
        for event in events
        if event.get("kind") == "ball_touch"
        and str(event.get("contacting_car_id") or event.get("actor_id") or "") == human_id
    )
    opponent_names = sorted(
        {
            str(car.get("player_name", ""))
            for frame in frames
            for car in frame.get("cars", [])
            if not car.get("flags", {}).get("is_local_human") and car.get("player_name")
        }
    )
    return {
        "continuous_gameplay_review_only": True,
        "attempt_segmentation_applied": False,
        "frame_count": len(frames),
        "duration_seconds": (
            float(frames[-1]["engine_physics_time"]) - float(frames[0]["engine_physics_time"])
            if frames
            else 0.0
        ),
        "event_counts": dict(sorted(event_counts.items())),
        "human_ball_touch_event_count": human_touches,
        "opponent_player_names_observed": opponent_names,
        "start_match_state": frames[0].get("match") if frames else None,
        "end_match_state": frames[-1].get("match") if frames else None,
    }


def _review_session(
    session_dir: Path, output_dir: Path
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    reader = SessionReader(session_dir)
    validation = reader.validate().as_dict()
    frames = list(reader.iter_frames())
    events = list(reader.iter_events())
    markers = list(reader.iter_markers())
    manifest = reader.manifest
    classification, scope = _session_class(manifest)
    human = manifest.get("local_player", {})
    human_id = str(human.get("stable_id", ""))
    label = str(
        manifest.get("mechanic_label")
        or manifest.get("opponent_label")
        or manifest.get("label")
        or "unlabeled"
    )
    event_counts = Counter(str(event.get("kind", "unknown")) for event in events)
    attempts: list[dict[str, Any]] = []
    segmentation_audit = None
    outcome_grouping = None
    if classification == "freeplay_mechanic_practice":
        spans, segmentation_audit = _segment_attempts(frames, events, human_id)
        attempts = [
            _attempt_telemetry(
                str(manifest.get("session_uuid", session_dir.name)),
                label,
                attempt_number,
                span,
                frames,
                events,
                human_id,
            )
            for attempt_number, span in enumerate(spans, 1)
        ]
        outcome_grouping = _assign_outcome_groups(attempts)
        _write_jsonl(output_dir / "attempts" / f"{session_dir.name}.jsonl", attempts)
    gameplay_summary = (
        _match_summary(frames, events, human_id) if classification == "gameplay" else None
    )
    attempt_quality_summary = None
    if attempts:
        attempt_quality_summary = {
            "attempt_count": len(attempts),
            "generic_recovery_observed_count": sum(
                bool(attempt["segmentation"]["generic_recovery_observed"]) for attempt in attempts
            ),
            "human_contact_attempt_count": sum(
                int(attempt["ball_outcome"]["human_touch_episode_count"]) > 0
                for attempt in attempts
            ),
            "noncontiguous_attempt_count": sum(
                int(attempt["data_quality"]["noncontiguous_adjacent_frame_count"]) > 0
                for attempt in attempts
            ),
            "missing_tick_attempt_count": sum(
                int(attempt["data_quality"]["missing_physics_frame_count"]) > 0
                for attempt in attempts
            ),
            "invalid_action_attempt_count": sum(
                int(attempt["data_quality"]["invalid_rival_action_frame_count"]) > 0
                for attempt in attempts
            ),
            "complete_human_car_coverage_attempt_count": sum(
                bool(attempt["data_quality"]["complete_human_car_coverage"]) for attempt in attempts
            ),
        }
    source_files = _source_files(session_dir)
    source_digest = (
        hashlib.sha256(
            "".join(f"{row['path']}:{row['sha256']}\n" for row in source_files).encode("utf-8")
        )
        .hexdigest()
        .upper()
    )
    report = {
        "format": FORMAT,
        "session_uuid": str(manifest.get("session_uuid", session_dir.name)),
        "source_directory": str(session_dir.resolve()),
        "source_file_set_sha256": source_digest,
        "source_files": source_files,
        "classification": classification,
        "review_scope": scope,
        "declared_session_type": manifest.get("session_type"),
        "declared_label": label,
        "declared_mechanic_label": manifest.get("mechanic_label"),
        "declared_opponent_label": manifest.get("opponent_label"),
        "capture_start_utc": manifest.get("capture_start_utc"),
        "capture_end_utc": manifest.get("capture_end_utc"),
        "map": manifest.get("map"),
        "rocket_league_build": manifest.get("rocket_league_build"),
        "recorder_git_sha": manifest.get("recorder_git_sha"),
        "plugin_build": manifest.get("plugin_build"),
        "bakkesmod_sdk_revision": manifest.get("bakkesmod_sdk_revision"),
        "local_player": human,
        "validation": validation,
        "ingestion_eligible_under_current_validator": bool(validation["valid"]),
        "frame_count": len(frames),
        "event_count": len(events),
        "marker_count": len(markers),
        "event_counts": dict(sorted(event_counts.items())),
        "markers": markers,
        "segmentation_audit": segmentation_audit,
        "attempt_count": len(attempts),
        "attempt_quality_summary": attempt_quality_summary,
        "outcome_grouping": outcome_grouping,
        "gameplay_summary": gameplay_summary,
        "authority_boundary": {
            "training_performed": False,
            "behavior_cloning_performed": False,
            "new_mechanic_detector_defined": False,
            "mechanic_rewards_assigned": False,
            "all_attempts_preserved": True,
        },
    }
    _write_json(output_dir / "sessions" / f"{session_dir.name}.json", report)
    return report, attempts


def _markdown_report(index: dict[str, Any], groupings: dict[str, Any]) -> str:
    lines = [
        "# Rival 2.0 human-demo review V1",
        "",
        f"Generated: `{index['generated_utc']}`",
        "",
        (
            "This is an inventory, integrity review, generic activity segmentation, and "
            "descriptive physical-outcome pass. It does not train, behavior-clone, detect "
            "named mechanics, judge mechanic correctness, or assign rewards. Relative "
            "outcome buckets are review candidates only; successful and failed evidence "
            "is preserved together."
        ),
        "",
        "## Inventory and validation",
        "",
        (
            "| Session | Class | Declared label | Frames | Attempts | Ingestion-safe | "
            "Principal validation issue |"
        ),
        "|---|---|---:|---:|---:|---:|---|",
    ]
    for session in index["sessions"]:
        issue = "; ".join(session["validation_errors"] + session["completeness_errors"]) or "none"
        lines.append(
            f"| `{session['session_uuid']}` | {session['classification']} | "
            f"{session['declared_label']} | {session['frame_count']} | "
            f"{session['attempt_count']} | {str(session['ingestion_eligible']).lower()} | {issue} |"
        )
    lines.extend(
        [
            "",
            "## Generic segmentation authority",
            "",
            (
                f"Version: `{SEGMENTATION_VERSION}`. Local-human native jump-onset events "
                "are activity anchors. Recorder sequence/physics discontinuities and native "
                "reset, respawn, and local-car rebind events are hard boundaries. A sustained "
                f"{RECOVERY_RUN_TICKS}-tick period with at least "
                f"{MIN_RECOVERY_WHEEL_CONTACTS} world-contact wheels, "
                "grounded state, and inactive jump/dodge/flip components separates attempts. "
                "The rule never branches on the declared mechanic label."
            ),
            "",
            "## Descriptive outcome groupings",
            "",
            (
                "Contact-oriented sessions are selected only when at least 25% of generic "
                "segments contain a native local-human ball-touch episode. Their review "
                "extremes use exact 12-tick ball velocity change, goal timing, and sustained "
                "recovery. Other sessions use within-session planar speed gain and sustained "
                "recovery. Quartiles are computed independently inside each recording. "
                "These are not training labels or named-mechanic detectors."
            ),
            "",
            (
                "| Label | Attempts | Stronger candidate | Limited/failed candidate | "
                "No measured human contact | Ambiguous |"
            ),
            "|---|---:|---:|---:|---:|---:|",
        ]
    )
    for row in groupings["sessions"]:
        counts = row["counts"]
        lines.append(
            f"| {row['declared_label']} | {row['attempt_count']} | "
            f"{counts.get('stronger_physical_outcome_candidate', 0)} | "
            f"{counts.get('limited_or_failed_physical_outcome_candidate', 0)} | "
            f"{counts.get('limited_evidence_no_measured_human_contact', 0)} | "
            f"{counts.get('ambiguous_middle_physical_outcome', 0)} |"
        )
    lines.extend(
        [
            "",
            "## Evidence files",
            "",
            (
                "- `source_inventory.json` binds every native manifest, chunk, event stream, "
                "and marker stream by byte count and SHA-256."
            ),
            "- `validation.json` preserves every validator verdict and diagnostic.",
            (
                "- `sessions/*.json` contains session metadata, classification, validation, "
                "event inventory, segmentation audit, and grouping criteria."
            ),
            (
                "- `attempts/*.jsonl` preserves each generic attempt with raw timing "
                "references and physical outcome telemetry."
            ),
            "- `groupings.json` is a compact descriptive grouping index.",
            "- `artifact_manifest.json` hashes the generated review package.",
            "- Re-run `python benchmarks/review_rival2_human_demos.py --verify-only` "
            "to verify source and review hashes plus attempt/index invariants.",
            "",
            "## Known evidence boundary",
            "",
            (
                "A readable source prefix is still reviewed when the current validator "
                "rejects the complete demonstration. Those sessions remain "
                "`ingestion_eligible_under_current_validator=false`. Missing physics ticks "
                "are never synthesized, invalid native action values are not clamped, and "
                "incomplete termination is not upgraded to success."
            ),
            "",
        ]
    )
    return "\n".join(lines)


def run(source_root: Path, output_dir: Path) -> dict[str, Any]:
    session_dirs = sorted(
        [path for path in source_root.iterdir() if (path / "manifest.json").is_file()],
        key=lambda path: json.loads((path / "manifest.json").read_text(encoding="utf-8")).get(
            "capture_start_utc", ""
        ),
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    reports: list[dict[str, Any]] = []
    attempts_by_session: dict[str, list[dict[str, Any]]] = {}
    for session_dir in session_dirs:
        report, attempts = _review_session(session_dir, output_dir)
        reports.append(report)
        attempts_by_session[report["session_uuid"]] = attempts

    generated_utc = datetime.now(UTC).isoformat().replace("+00:00", "Z")
    script_path = Path(__file__).resolve()
    index_rows = []
    validation_rows = []
    inventory_rows = []
    grouping_rows = []
    for report in reports:
        validation = report["validation"]
        row = {
            "session_uuid": report["session_uuid"],
            "classification": report["classification"],
            "review_scope": report["review_scope"],
            "declared_label": report["declared_label"],
            "capture_start_utc": report["capture_start_utc"],
            "frame_count": report["frame_count"],
            "attempt_count": report["attempt_count"],
            "ingestion_eligible": report["ingestion_eligible_under_current_validator"],
            "container_valid": validation["container_valid"],
            "clean_termination": validation["clean_termination"],
            "capture_complete": validation["capture_complete"],
            "missing_physics_frame_count": validation["missing_physics_frame_count"],
            "validation_errors": validation["errors"],
            "completeness_errors": validation["completeness_errors"],
        }
        index_rows.append(row)
        validation_rows.append(
            {
                "session_uuid": report["session_uuid"],
                "declared_label": report["declared_label"],
                "validation": validation,
            }
        )
        inventory_rows.append(
            {
                "session_uuid": report["session_uuid"],
                "source_directory": report["source_directory"],
                "source_file_set_sha256": report["source_file_set_sha256"],
                "files": report["source_files"],
            }
        )
        if report["outcome_grouping"] is not None:
            grouping_rows.append(
                {
                    "session_uuid": report["session_uuid"],
                    "declared_label": report["declared_label"],
                    "attempt_count": report["attempt_count"],
                    **report["outcome_grouping"],
                }
            )
    demo_rows = [row for row in index_rows if row["review_scope"] == "new_demonstration"]
    index = {
        "format": FORMAT,
        "generated_utc": generated_utc,
        "source_root": str(source_root.resolve()),
        "review_script": str(script_path),
        "review_script_sha256": _sha256(script_path),
        "session_count": len(index_rows),
        "new_demonstration_session_count": len(demo_rows),
        "infrastructure_smoke_session_count": len(index_rows) - len(demo_rows),
        "gameplay_session_count": sum(row["classification"] == "gameplay" for row in demo_rows),
        "mechanic_practice_session_count": sum(
            row["classification"] == "freeplay_mechanic_practice" for row in demo_rows
        ),
        "mechanic_attempt_count": sum(row["attempt_count"] for row in demo_rows),
        "invalid_new_demonstration_session_count": sum(
            not row["ingestion_eligible"] for row in demo_rows
        ),
        "sessions": index_rows,
        "authority_boundary": {
            "training_performed": False,
            "behavior_cloning_performed": False,
            "new_mechanic_detector_defined": False,
            "mechanic_rewards_assigned": False,
            "raw_success_and_failure_evidence_preserved": True,
        },
    }
    groupings = {
        "format": FORMAT,
        "version": OUTCOME_GROUPING_VERSION,
        "sessions": grouping_rows,
        "all_attempts_preserved": True,
        "use_restriction": (
            "Descriptive reviewer triage only; not a mechanic detector, correctness label, "
            "reward, or training target."
        ),
    }
    _write_json(output_dir / "index.json", index)
    _write_json(
        output_dir / "source_inventory.json",
        {"format": FORMAT, "sessions": inventory_rows},
    )
    _write_json(
        output_dir / "validation.json",
        {"format": FORMAT, "sessions": validation_rows},
    )
    _write_json(output_dir / "groupings.json", groupings)
    (output_dir / "REVIEW.md").write_text(
        _markdown_report(index, groupings), encoding="utf-8", newline="\n"
    )
    manifest_rows = []
    for path in sorted(
        (
            path
            for path in output_dir.rglob("*")
            if path.is_file() and path.name != "artifact_manifest.json"
        ),
        key=lambda value: str(value.relative_to(output_dir)),
    ):
        manifest_rows.append(
            {
                "path": str(path.relative_to(output_dir)).replace("\\", "/"),
                "bytes": path.stat().st_size,
                "sha256": _sha256(path),
            }
        )
    _write_json(
        output_dir / "artifact_manifest.json",
        {
            "format": FORMAT,
            "generated_utc": generated_utc,
            "files": manifest_rows,
            "file_count": len(manifest_rows),
        },
    )
    return index


def verify_review(output_dir: Path) -> dict[str, Any]:
    """Verify source bindings, generated hashes, and attempt/index invariants."""

    errors: list[str] = []
    artifact_manifest = json.loads(
        (output_dir / "artifact_manifest.json").read_text(encoding="utf-8")
    )
    for row in artifact_manifest.get("files", []):
        path = output_dir / str(row["path"])
        if not path.is_file():
            errors.append(f"missing generated artifact: {row['path']}")
            continue
        if path.stat().st_size != int(row["bytes"]):
            errors.append(f"generated byte-size mismatch: {row['path']}")
        if _sha256(path) != str(row["sha256"]):
            errors.append(f"generated SHA-256 mismatch: {row['path']}")

    index = json.loads((output_dir / "index.json").read_text(encoding="utf-8"))
    inventory = json.loads((output_dir / "source_inventory.json").read_text(encoding="utf-8"))
    inventory_by_uuid = {str(row["session_uuid"]): row for row in inventory.get("sessions", [])}
    grouping = json.loads((output_dir / "groupings.json").read_text(encoding="utf-8"))
    grouping_by_uuid = {str(row["session_uuid"]): row for row in grouping.get("sessions", [])}
    attempt_total = 0
    seen_attempt_ids: set[str] = set()
    for session in index.get("sessions", []):
        session_uuid = str(session["session_uuid"])
        session_report_path = output_dir / "sessions" / f"{session_uuid}.json"
        session_report = json.loads(session_report_path.read_text(encoding="utf-8"))
        if session_report["authority_boundary"] != {
            "all_attempts_preserved": True,
            "behavior_cloning_performed": False,
            "mechanic_rewards_assigned": False,
            "new_mechanic_detector_defined": False,
            "training_performed": False,
        }:
            errors.append(f"authority-boundary mismatch: {session_uuid}")

        inventory_row = inventory_by_uuid.get(session_uuid)
        if inventory_row is None:
            errors.append(f"source inventory row missing: {session_uuid}")
        else:
            source_dir = Path(str(inventory_row["source_directory"]))
            source_digest = hashlib.sha256()
            for source_file in inventory_row.get("files", []):
                source_path = source_dir / str(source_file["path"])
                if not source_path.is_file():
                    errors.append(f"source artifact missing: {source_path}")
                    continue
                actual_hash = _sha256(source_path)
                if source_path.stat().st_size != int(source_file["bytes"]):
                    errors.append(f"source byte-size mismatch: {source_path}")
                if actual_hash != str(source_file["sha256"]):
                    errors.append(f"source SHA-256 mismatch: {source_path}")
                source_digest.update(f"{source_file['path']}:{actual_hash}\n".encode())
            if source_digest.hexdigest().upper() != str(inventory_row["source_file_set_sha256"]):
                errors.append(f"source file-set digest mismatch: {session_uuid}")

        expected_attempts = int(session["attempt_count"])
        attempt_path = output_dir / "attempts" / f"{session_uuid}.jsonl"
        attempts = []
        if expected_attempts:
            attempts = [
                json.loads(line)
                for line in attempt_path.read_text(encoding="utf-8").splitlines()
                if line.strip()
            ]
        elif attempt_path.exists() and attempt_path.stat().st_size:
            errors.append(f"unexpected attempts for non-segmented session: {session_uuid}")
        if len(attempts) != expected_attempts:
            errors.append(
                f"attempt count mismatch for {session_uuid}: "
                f"expected {expected_attempts}, decoded {len(attempts)}"
            )
        previous_end: int | None = None
        for attempt in attempts:
            attempt_id = str(attempt["attempt_id"])
            if attempt_id in seen_attempt_ids:
                errors.append(f"duplicate attempt id: {attempt_id}")
            seen_attempt_ids.add(attempt_id)
            segmentation = attempt["segmentation"]
            start = int(segmentation["start_physics_frame"])
            end = int(segmentation["end_physics_frame"])
            anchors = [int(value) for value in segmentation["anchor_physics_frames"]]
            if start > end or not anchors or any(not start <= value <= end for value in anchors):
                errors.append(f"invalid attempt span: {attempt_id}")
            if previous_end is not None and start <= previous_end:
                errors.append(f"overlapping attempt span: {attempt_id}")
            previous_end = end
            if attempt["outcome_group"] is None:
                errors.append(f"missing descriptive outcome group: {attempt_id}")
        attempt_total += len(attempts)
        if expected_attempts:
            grouping_row = grouping_by_uuid.get(session_uuid)
            if grouping_row is None:
                errors.append(f"outcome grouping missing: {session_uuid}")
            elif sum(int(value) for value in grouping_row["counts"].values()) != len(attempts):
                errors.append(f"outcome grouping count mismatch: {session_uuid}")

    if attempt_total != int(index["mechanic_attempt_count"]):
        errors.append(
            f"global attempt count mismatch: {attempt_total} != {index['mechanic_attempt_count']}"
        )
    return {
        "format": FORMAT,
        "valid": not errors,
        "session_count": len(index.get("sessions", [])),
        "attempt_count": attempt_total,
        "generated_artifact_count": len(artifact_manifest.get("files", [])),
        "source_file_count": sum(
            len(row.get("files", [])) for row in inventory.get("sessions", [])
        ),
        "errors": errors,
    }


def build_parser() -> argparse.ArgumentParser:
    default_source = (
        Path(os.environ["APPDATA"]) / "bakkesmod" / "bakkesmod" / "data" / "rival2" / "human_demos"
    )
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", type=Path, default=default_source)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("results/rival2/human_demo_review_v1"),
    )
    parser.add_argument("--verify-only", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.verify_only:
        verification = verify_review(args.output_dir.resolve())
        print(json.dumps(verification, sort_keys=True))
        return 0 if verification["valid"] else 1
    index = run(args.source_root.resolve(), args.output_dir.resolve())
    print(
        json.dumps(
            {
                "output_dir": str(args.output_dir.resolve()),
                "sessions": index["session_count"],
                "new_demonstrations": index["new_demonstration_session_count"],
                "mechanic_attempts": index["mechanic_attempt_count"],
                "invalid_new_demonstrations": index["invalid_new_demonstration_session_count"],
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
