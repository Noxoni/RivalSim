"""Review native Rival 2.0 human demonstrations for later behavior cloning.

The review keeps the recorder's strict completeness diagnostics, but separately decides
whether the paired state/action stream is usable.  Native reset, respawn, and local-car
rebind discontinuities are segmentation boundaries rather than invented missing samples;
events after the final captured action are unpaired tail metadata and are excluded.

Freeplay attempts are segmented from native 120 Hz evidence.  A source-bound offline
adjudication then identifies high-confidence successful demonstrations for later behavior
cloning while preserving failures and ambiguous attempts.  This is not a production
mechanic detector, reward definition, behavior-cloning run, or training run.
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

FORMAT = "RIVALRL_HUMAN_DEMO_REVIEW_V2"
SEGMENTATION_VERSION = "RESET_AWARE_PAIRED_STATE_ACTION_V2"
OUTCOME_GROUPING_VERSION = "DESCRIPTIVE_RELATIVE_OUTCOME_V1"
MECHANIC_ASSESSMENT_VERSION = "SOURCE_BOUND_MECHANIC_BC_ADJUDICATION_V1"
ADJUDICATION_PATH = Path(__file__).with_name("human_demo_mechanic_adjudication_v1.json")
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
LIFECYCLE_GAP_KINDS = {
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


def _cross(a: Sequence[float], b: Sequence[float]) -> list[float]:
    return [
        float(a[1]) * float(b[2]) - float(a[2]) * float(b[1]),
        float(a[2]) * float(b[0]) - float(a[0]) * float(b[2]),
        float(a[0]) * float(b[1]) - float(a[1]) * float(b[0]),
    ]


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
    if physics < physics_frames[0] or physics > physics_frames[-1]:
        return None
    index = bisect.bisect_left(physics_frames, physics)
    if index >= len(physics_frames):
        return None
    if index > 0 and abs(physics_frames[index - 1] - physics) < abs(
        physics_frames[index] - physics
    ):
        return index - 1
    return index


def _paired_records(
    records: Sequence[dict[str, Any]], physics_frames: Sequence[int]
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Split records into frame-paired evidence and unpaired pre/post-capture metadata."""

    if not physics_frames:
        return [], list(records)
    first, last = physics_frames[0], physics_frames[-1]
    paired = [
        record
        for record in records
        if first <= int(record.get("physics_frame", -1)) <= last
    ]
    unpaired = [record for record in records if record not in paired]
    return paired, unpaired


def _paired_stream_assessment(
    manifest: dict[str, Any],
    validation: dict[str, Any],
    frames: Sequence[dict[str, Any]],
    events: Sequence[dict[str, Any]],
    markers: Sequence[dict[str, Any]],
) -> dict[str, Any]:
    """Assess usable paired samples without weakening strict recorder diagnostics."""

    physics = [int(frame["physics_frame"]) for frame in frames]
    paired_events, unpaired_events = _paired_records(events, physics)
    paired_markers, unpaired_markers = _paired_records(markers, physics)
    gaps: list[dict[str, Any]] = []
    unexplained_ticks = 0
    for index in range(1, len(frames)):
        previous = int(frames[index - 1]["physics_frame"])
        current = int(frames[index]["physics_frame"])
        missing = max(0, current - previous - 1)
        if not missing:
            continue
        lifecycle = sorted(
            {
                str(event.get("kind", ""))
                for event in paired_events
                if previous < int(event.get("physics_frame", -1)) <= current
                and str(event.get("kind", "")) in LIFECYCLE_GAP_KINDS
            }
        )
        explained = bool(lifecycle)
        if not explained:
            unexplained_ticks += missing
        gaps.append(
            {
                "previous_physics_frame": previous,
                "resumed_physics_frame": current,
                "global_engine_frame_ids_skipped": missing,
                "explained_by_lifecycle_transition": explained,
                "lifecycle_event_kinds": lifecycle,
                "segmentation_boundary_index": index,
            }
        )

    match_ended_in_paired_frames = any(
        bool(frame.get("match", {}).get("flags", {}).get("match_ended")) for frame in frames
    )
    queue_drops = int(manifest.get("queue_dropped_frame_count", 0) or 0)
    structural_checks = {
        "container_valid": bool(validation.get("container_valid")),
        "manifest_hashes_valid": bool(validation.get("manifest_hashes_valid")),
        "complete_chunk_prefix": int(validation.get("partial_chunks", 0)) == 0,
        "recorder_sequence_contiguous": (
            int(validation.get("missing_sequence_count", 0)) == 0
            and int(validation.get("sequence_gap_count", 0)) == 0
            and int(validation.get("duplicate_sequence_count", 0)) == 0
            and int(validation.get("out_of_order_sequence_count", 0)) == 0
        ),
        "authoritative_action_values_valid": int(validation.get("invalid_action_frames", 0)) == 0,
        "unique_human_car_on_every_frame": (
            int(validation.get("invalid_human_car_frames", 0)) == 0
        ),
        "recorder_queue_drops_zero": queue_drops == 0,
        "all_global_frame_id_gaps_lifecycle_explained": unexplained_ticks == 0,
        "at_least_one_paired_frame": bool(frames),
    }
    review_usable = all(structural_checks.values())
    post_match_identity_shutdown = bool(
        review_usable
        and match_ended_in_paired_frames
        and not bool(validation.get("clean_termination"))
        and str(manifest.get("termination_reason", "")).startswith("local_human_identity_lost")
    )
    return {
        "version": SEGMENTATION_VERSION,
        "review_usable": review_usable,
        "paired_state_action_frame_count": len(frames),
        "strict_recorder_validation_valid": bool(validation.get("valid")),
        "strict_capture_complete": bool(validation.get("capture_complete")),
        "structural_checks": structural_checks,
        "lifecycle_gap_episode_count": len(gaps),
        "lifecycle_explained_global_frame_id_gap_count": sum(
            row["global_engine_frame_ids_skipped"]
            for row in gaps
            if row["explained_by_lifecycle_transition"]
        ),
        "unexplained_global_frame_id_gap_count": unexplained_ticks,
        "lifecycle_gaps": gaps,
        "paired_event_count": len(paired_events),
        "unpaired_event_count": len(unpaired_events),
        "paired_marker_count": len(paired_markers),
        "unpaired_marker_count": len(unpaired_markers),
        "unpaired_tail_policy": (
            "Events and markers outside the first-to-last captured action frame are retained "
            "as metadata but excluded from attempt anchors, boundaries, and paired telemetry."
        ),
        "match_ended_in_paired_frames": match_ended_in_paired_frames,
        "post_match_identity_shutdown": post_match_identity_shutdown,
        "interpretation": (
            "Global engine physics-frame identifiers may jump across native lifecycle resets. "
            "No missing action tick is synthesized. Each such jump is a hard segment boundary."
        ),
    }


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
            "Local-human native jump-onset activity anchors are grouped only within a "
            "continuous paired state/action region. Native reset, respawn, local-car rebind, "
            "and global physics-frame discontinuities are hard boundaries. Sustained "
            "two-wheel grounded recovery separates attempts. The declared mechanic does not "
            "change segmentation."
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


def _dodge_onsets(frames: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    onsets: list[dict[str, Any]] = []
    for index, frame in enumerate(frames):
        car = _human(frame)
        previous = _human(frames[index - 1]) if index else None
        if car is None:
            continue
        dodge_active = bool(car.get("dodge_component", {}).get("active"))
        flip_active = bool(car.get("flip_component", {}).get("active"))
        double_jumped = bool(car.get("flags", {}).get("double_jumped"))
        jump_pressed = bool(frame.get("native_input", {}).get("jump"))
        started = bool(
            (
                dodge_active
                and not bool(previous and previous.get("dodge_component", {}).get("active"))
            )
            or (
                double_jumped
                and not bool(previous and previous.get("flags", {}).get("double_jumped"))
            )
            or (
                flip_active
                and not bool(previous and previous.get("flip_component", {}).get("active"))
            )
            or (
                jump_pressed
                and not bool(
                    index and frames[index - 1].get("native_input", {}).get("jump")
                )
                and int(car.get("num_wheel_world_contacts", 0)) <= 2
            )
        )
        if not started:
            continue
        native = frame.get("native_input", {})
        onsets.append(
            {
                "index": index,
                "physics_frame": int(frame["physics_frame"]),
                "dodge_direction": _vec(car.get("dodge_component", {}).get("direction")),
                "native_dodge_forward": _finite(native.get("dodge_forward")),
                "native_dodge_strafe": _finite(native.get("dodge_strafe")),
                "world_contact_wheel_count": int(car.get("num_wheel_world_contacts", 0)),
                "on_ground": bool(car.get("flags", {}).get("on_ground")),
                "planar_speed_uu_per_s": _planar_speed(car),
            }
        )
    return onsets


def _closest_soccar_surface(ball_position: Sequence[float]) -> dict[str, Any]:
    ball_radius = 92.75
    x, y, z = (float(value) for value in ball_position)
    gaps = {
        "ground": abs(z - ball_radius),
        "ceiling": abs((2044.0 - z) - ball_radius),
        "side_wall": abs((4096.0 - abs(x)) - ball_radius),
        "back_wall": abs((5120.0 - abs(y)) - ball_radius),
    }
    surface = min(gaps, key=gaps.__getitem__)
    return {
        "surface": surface,
        "surface_gap_uu": gaps[surface],
        "all_surface_gaps_uu": gaps,
        "geometry_assumption": (
            "Standard Soccar center planes: x=+-4096, y=+-5120, z=0/2044; "
            "native ball radius 92.75 uu. Curved corners are not approximated."
        ),
    }


def _mechanic_physical_evidence(
    attempt: dict[str, Any],
    span: dict[str, Any],
    frames: Sequence[dict[str, Any]],
) -> dict[str, Any]:
    """Extract auditable native evidence; this does not decide the mechanic verdict."""

    selected = frames[int(span["start_index"]) : int(span["end_index"]) + 1]
    physics_to_local = {int(frame["physics_frame"]): index for index, frame in enumerate(selected)}
    onsets = _dodge_onsets(selected)
    contact_rows: list[dict[str, Any]] = []
    airborne_touch_frames: list[int] = []
    grounded_touch_frames: list[int] = []
    for contact_index, contact in enumerate(attempt["ball_outcome"]["contacts"]):
        physics = int(contact["contact_physics_frame"])
        local_index = physics_to_local.get(physics)
        frame = selected[local_index] if local_index is not None else None
        car = _human(frame) if frame else None
        ball = frame.get("ball") if frame else None
        latest_onset = next(
            (onset for onset in reversed(onsets) if onset["physics_frame"] <= physics), None
        )
        surface = _closest_soccar_surface(ball["position"]) if ball else None
        rotational_closing = None
        rotational_share = None
        yaw_from_dodge = None
        if car and ball:
            geometry = contact.get("contact_geometry") or {}
            hit_location = geometry.get("hit_location") or ball.get("position")
            lever = _sub(hit_location, car["position"])
            direction = _unit(_sub(ball["position"], car["position"]))
            if direction is not None:
                rotational_velocity = _cross(car.get("angular_velocity", [0.0, 0.0, 0.0]), lever)
                rotational_closing = _dot(rotational_velocity, direction)
                linear_closing = _dot(
                    _sub(
                        car.get("linear_velocity", [0.0, 0.0, 0.0]),
                        ball.get("linear_velocity", [0.0, 0.0, 0.0]),
                    ),
                    direction,
                )
                positive_total = max(0.0, rotational_closing) + max(0.0, linear_closing)
                rotational_share = (
                    max(0.0, rotational_closing) / positive_total if positive_total > 1e-9 else 0.0
                )
            if latest_onset is not None:
                onset_frame = selected[int(latest_onset["index"])]
                onset_car = _human(onset_frame)
                if onset_car is not None:
                    onset_yaw = _rotator_degrees(onset_car.get("rotation", [0, 0, 0]))[1]
                    contact_yaw = _rotator_degrees(car.get("rotation", [0, 0, 0]))[1]
                    yaw_from_dodge = _wrapped_degrees_delta(onset_yaw, contact_yaw)
        world_contacts = int(car.get("num_wheel_world_contacts", 0)) if car else None
        airborne = bool(car and not car.get("flags", {}).get("on_ground") and world_contacts == 0)
        if airborne:
            airborne_touch_frames.append(physics)
        elif car and (car.get("flags", {}).get("on_ground") or (world_contacts or 0) > 0):
            grounded_touch_frames.append(physics)
        delta_velocity = contact.get("ball_delta_velocity_12_ticks")
        row = {
            "contact_index": contact_index,
            "physics_frame": physics,
            "airborne_zero_world_wheels": airborne,
            "world_contact_wheel_count": world_contacts,
            "ball_velocity_delta_12_ticks_uu_per_s": delta_velocity,
            "ball_velocity_delta_magnitude_12_ticks_uu_per_s": contact.get(
                "ball_delta_velocity_magnitude_12_ticks"
            ),
            "ball_vertical_velocity_delta_12_ticks_uu_per_s": (
                float(delta_velocity[2]) if delta_velocity is not None else None
            ),
            "ball_outgoing_vertical_velocity_12_ticks_uu_per_s": (
                contact.get("ball_after_12_ticks", {}).get("linear_velocity", [None, None, None])[2]
                if contact.get("ball_after_12_ticks")
                else None
            ),
            "closest_soccar_surface": surface,
            "latest_dodge_onset": latest_onset,
            "dodge_age_ticks_at_contact": (
                physics - int(latest_onset["physics_frame"]) if latest_onset else None
            ),
            "yaw_change_from_dodge_to_contact_degrees": yaw_from_dodge,
            "rotational_closing_contribution_uu_per_s": rotational_closing,
            "rotational_closing_fraction": rotational_share,
        }
        contact_rows.append(row)

    reset_acquisitions: list[dict[str, Any]] = []
    for index in range(1, len(selected)):
        previous_car = _human(selected[index - 1])
        car = _human(selected[index])
        if previous_car is None or car is None:
            continue
        if previous_car.get("flags", {}).get("has_flip") or not car.get("flags", {}).get(
            "has_flip"
        ):
            continue
        car_to_ball = _sub(selected[index]["ball"]["position"], car["position"])
        world_contacts = int(car.get("num_wheel_world_contacts", 0))
        all_contacts = int(car.get("num_wheel_contacts", 0))
        distance = _norm(car_to_ball)
        if (
            world_contacts != 0
            or all_contacts < 3
            or distance > 140.0
            or car.get("flags", {}).get("on_ground")
        ):
            continue
        reset_acquisitions.append(
            {
                "physics_frame": int(selected[index]["physics_frame"]),
                "has_flip_false_to_true": True,
                "world_contact_wheel_count": world_contacts,
                "all_wheel_contact_count": all_contacts,
                "car_ball_center_distance_uu": distance,
                "airborne": not bool(car.get("flags", {}).get("on_ground")),
                "jumped_after_transition": bool(car.get("flags", {}).get("jumped")),
                "double_jumped_after_transition": bool(
                    car.get("flags", {}).get("double_jumped")
                ),
            }
        )

    dash_outcomes: list[dict[str, Any]] = []
    for onset in onsets:
        onset_index = int(onset["index"])
        landing_index = next(
            (
                index
                for index in range(onset_index, min(len(selected), onset_index + 7))
                if (_human(selected[index]) or {}).get("num_wheel_world_contacts", 0) >= 1
            ),
            None,
        )
        if landing_index is None:
            continue
        before_car = _human(selected[max(0, onset_index - 1)])
        after_car = _human(selected[min(len(selected) - 1, landing_index + 6)])
        if before_car is None or after_car is None:
            continue
        landing_car = _human(selected[landing_index])
        world_normals = [
            _vec(wheel.get("contact_normal"))
            for wheel in (landing_car or {}).get("wheels", [])
            if wheel.get("has_world_contact") and _vec(wheel.get("contact_normal")) is not None
        ]
        normal = (
            _unit([sum(values) for values in zip(*world_normals, strict=True)])
            if world_normals
            else None
        )

        def tangent_speed(
            car: dict[str, Any], contact_normal: list[float] | None = normal
        ) -> float | None:
            if contact_normal is None:
                return None
            velocity = _vec(car.get("linear_velocity"))
            if velocity is None:
                return None
            normal_speed = _dot(velocity, contact_normal)
            tangent = _sub(
                velocity, [normal_speed * value for value in contact_normal]
            )
            return _norm(tangent)

        before_tangent = tangent_speed(before_car)
        after_tangent = tangent_speed(after_car)
        dash_outcomes.append(
            {
                "dodge_onset_physics_frame": onset["physics_frame"],
                "landing_physics_frame": int(selected[landing_index]["physics_frame"]),
                "ticks_to_stable_landing": landing_index - onset_index,
                "world_contact_wheels_at_onset": onset["world_contact_wheel_count"],
                "planar_speed_change_through_landing_uu_per_s": (
                    _planar_speed(after_car) - _planar_speed(before_car)
                ),
                "surface_contact_normal": normal,
                "surface_tangent_speed_change_uu_per_s": (
                    after_tangent - before_tangent
                    if before_tangent is not None and after_tangent is not None
                    else None
                ),
            }
        )

    ground_to_air_pairs: list[dict[str, Any]] = []
    for index, contact in enumerate(contact_rows):
        if contact["physics_frame"] not in grounded_touch_frames:
            continue
        vertical_delta = contact["ball_vertical_velocity_delta_12_ticks_uu_per_s"]
        outgoing_vertical = contact["ball_outgoing_vertical_velocity_12_ticks_uu_per_s"]
        if vertical_delta is None or outgoing_vertical is None:
            continue
        if float(vertical_delta) < 100.0 or float(outgoing_vertical) <= 0.0:
            continue
        later_air = next(
            (row for row in contact_rows[index + 1 :] if row["airborne_zero_world_wheels"]),
            None,
        )
        if later_air:
            ground_to_air_pairs.append(
                {
                    "ground_pop_physics_frame": contact["physics_frame"],
                    "ground_pop_vertical_delta_uu_per_s": vertical_delta,
                    "first_later_air_touch_physics_frame": later_air["physics_frame"],
                    "ticks_from_pop_to_air_touch": (
                        later_air["physics_frame"] - contact["physics_frame"]
                    ),
                }
            )

    return {
        "version": MECHANIC_ASSESSMENT_VERSION,
        "native_dodge_onsets": onsets,
        "dash_landing_outcomes": dash_outcomes,
        "contacts": contact_rows,
        "airborne_zero_world_wheel_touch_episode_count": len(airborne_touch_frames),
        "airborne_zero_world_wheel_touch_physics_frames": airborne_touch_frames,
        "grounded_touch_physics_frames": grounded_touch_frames,
        "ground_to_air_pop_followup_pairs": ground_to_air_pairs,
        "flip_resource_reacquisitions": reset_acquisitions,
    }


def _load_adjudication() -> dict[str, Any]:
    document = json.loads(ADJUDICATION_PATH.read_text(encoding="utf-8"))
    if document.get("format") != MECHANIC_ASSESSMENT_VERSION:
        raise ValueError(f"unexpected mechanic adjudication format: {document.get('format')}")
    return document


def _attach_mechanic_assessments(
    attempts: list[dict[str, Any]],
    spans: Sequence[dict[str, Any]],
    frames: Sequence[dict[str, Any]],
    *,
    session_uuid: str,
    label: str,
    source_file_set_sha256: str,
    adjudication: dict[str, Any],
) -> dict[str, Any]:
    session = adjudication.get("sessions", {}).get(session_uuid)
    if session is None:
        raise ValueError(f"missing source-bound mechanic adjudication: {session_uuid}")
    if str(session.get("label")) != label:
        raise ValueError(f"adjudication label mismatch for {session_uuid}")
    if str(session.get("source_file_set_sha256")) != source_file_set_sha256:
        raise ValueError(f"adjudication source hash mismatch for {session_uuid}")
    if int(session.get("attempt_count", -1)) != len(attempts):
        raise ValueError(f"adjudication attempt-count mismatch for {session_uuid}")
    successes = {int(value) for value in session.get("success_attempts", [])}
    ambiguous = {int(value) for value in session.get("ambiguous_attempts", [])}
    if successes & ambiguous:
        raise ValueError(f"overlapping success/ambiguous adjudication for {session_uuid}")
    expected = set(range(1, len(attempts) + 1))
    if not successes | ambiguous <= expected:
        raise ValueError(f"out-of-range adjudication attempt for {session_uuid}")

    notes = session.get("attempt_notes", {})
    counts: Counter[str] = Counter()
    candidates = []
    for attempt, span in zip(attempts, spans, strict=True):
        number = int(attempt["attempt_number"])
        verdict = (
            "success"
            if number in successes
            else "ambiguous"
            if number in ambiguous
            else "failure"
        )
        evidence = _mechanic_physical_evidence(attempt, span, frames)
        completion_physics = None
        label_key = str(attempt["declared_label"]).lower()
        if label_key == "flipreset" and evidence["flip_resource_reacquisitions"]:
            completion_physics = evidence["flip_resource_reacquisitions"][0]["physics_frame"]
        elif label_key == "groundtoairdribble" and evidence["ground_to_air_pop_followup_pairs"]:
            completion_physics = evidence["ground_to_air_pop_followup_pairs"][0][
                "first_later_air_touch_physics_frame"
            ]
        elif label_key in {"wavedash", "walldash", "zapdash"} and evidence[
            "dash_landing_outcomes"
        ]:
            completion_physics = evidence["dash_landing_outcomes"][0][
                "landing_physics_frame"
            ]
        elif evidence["contacts"]:
            completion_physics = max(
                evidence["contacts"],
                key=lambda row: float(
                    row.get("ball_velocity_delta_magnitude_12_ticks_uu_per_s") or -1.0
                ),
            )["physics_frame"]
        note = str(notes.get(str(number), notes.get("*", "")))
        assessment = {
            "version": MECHANIC_ASSESSMENT_VERSION,
            "verdict": verdict,
            "behavior_cloning_eligible": verdict == "success",
            "criterion": str(session["criterion"]),
            "adjudication_note": note,
            "completion_physics_frame": completion_physics if verdict == "success" else None,
            "representative_evidence_physics_frame": completion_physics,
            "evidence": evidence,
            "policy": (
                "Source-bound offline adjudication for this exact recording. It selects "
                "high-confidence examples for later curation; it is not a production detector."
            ),
        }
        attempt["mechanic_assessment"] = assessment
        counts[verdict] += 1
        if verdict == "success":
            candidates.append(
                {
                    "attempt_id": attempt["attempt_id"],
                    "attempt_number": number,
                    "start_sequence": attempt["segmentation"]["start_sequence"],
                    "end_sequence": attempt["segmentation"]["end_sequence"],
                    "start_physics_frame": attempt["segmentation"]["start_physics_frame"],
                    "end_physics_frame": attempt["segmentation"]["end_physics_frame"],
                    "completion_physics_frame": completion_physics,
                }
            )
    return {
        "version": MECHANIC_ASSESSMENT_VERSION,
        "criterion": str(session["criterion"]),
        "counts": dict(sorted(counts.items())),
        "behavior_cloning_candidate_count": len(candidates),
        "behavior_cloning_candidates": candidates,
        "source_binding": {
            "session_uuid": session_uuid,
            "source_file_set_sha256": source_file_set_sha256,
            "adjudication_file": str(ADJUDICATION_PATH.resolve()),
            "adjudication_file_sha256": _sha256(ADJUDICATION_PATH),
        },
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
    physics = [int(frame["physics_frame"]) for frame in frames]
    paired_events, unpaired_events = _paired_records(events, physics)
    paired_markers, unpaired_markers = _paired_records(markers, physics)
    paired_stream = _paired_stream_assessment(manifest, validation, frames, events, markers)
    classification, scope = _session_class(manifest)
    human = manifest.get("local_player", {})
    human_id = str(human.get("stable_id", ""))
    label = str(
        manifest.get("mechanic_label")
        or manifest.get("opponent_label")
        or manifest.get("label")
        or "unlabeled"
    )
    event_counts = Counter(str(event.get("kind", "unknown")) for event in paired_events)
    source_files = _source_files(session_dir)
    source_digest = (
        hashlib.sha256(
            "".join(f"{row['path']}:{row['sha256']}\n" for row in source_files).encode("utf-8")
        )
        .hexdigest()
        .upper()
    )
    attempts: list[dict[str, Any]] = []
    segmentation_audit = None
    outcome_grouping = None
    mechanic_assessment_summary = None
    if classification == "freeplay_mechanic_practice":
        spans, segmentation_audit = _segment_attempts(frames, paired_events, human_id)
        attempts = [
            _attempt_telemetry(
                str(manifest.get("session_uuid", session_dir.name)),
                label,
                attempt_number,
                span,
                frames,
                paired_events,
                human_id,
            )
            for attempt_number, span in enumerate(spans, 1)
        ]
        mechanic_assessment_summary = _attach_mechanic_assessments(
            attempts,
            spans,
            frames,
            session_uuid=str(manifest.get("session_uuid", session_dir.name)),
            label=label,
            source_file_set_sha256=source_digest,
            adjudication=_load_adjudication(),
        )
        outcome_grouping = _assign_outcome_groups(attempts)
        _write_jsonl(output_dir / "attempts" / f"{session_dir.name}.jsonl", attempts)
    gameplay_summary = (
        _match_summary(frames, paired_events, human_id) if classification == "gameplay" else None
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
            "mechanic_success_count": sum(
                attempt["mechanic_assessment"]["verdict"] == "success" for attempt in attempts
            ),
            "mechanic_ambiguous_count": sum(
                attempt["mechanic_assessment"]["verdict"] == "ambiguous" for attempt in attempts
            ),
            "mechanic_failure_count": sum(
                attempt["mechanic_assessment"]["verdict"] == "failure" for attempt in attempts
            ),
        }
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
        "strict_recorder_validation_valid": bool(validation["valid"]),
        "paired_stream_assessment": paired_stream,
        "review_usable": bool(paired_stream["review_usable"]),
        "frame_count": len(frames),
        "event_count": len(paired_events),
        "marker_count": len(paired_markers),
        "event_counts": dict(sorted(event_counts.items())),
        "markers": paired_markers,
        "unpaired_tail": {
            "events": unpaired_events,
            "markers": unpaired_markers,
            "excluded_from_segmentation_and_attempt_telemetry": True,
        },
        "segmentation_audit": segmentation_audit,
        "attempt_count": len(attempts),
        "attempt_quality_summary": attempt_quality_summary,
        "outcome_grouping": outcome_grouping,
        "mechanic_assessment_summary": mechanic_assessment_summary,
        "gameplay_summary": gameplay_summary,
        "authority_boundary": {
            "training_performed": False,
            "behavior_cloning_performed": False,
            "production_mechanic_detector_defined": False,
            "source_bound_behavior_cloning_candidate_selection_performed": bool(attempts),
            "mechanic_rewards_assigned": False,
            "all_attempts_preserved": True,
        },
    }
    _write_json(output_dir / "sessions" / f"{session_dir.name}.json", report)
    return report, attempts


def _markdown_report(index: dict[str, Any], groupings: dict[str, Any]) -> str:
    lines = [
        "# Rival 2.0 human-demo review V2",
        "",
        f"Generated: `{index['generated_utc']}`",
        "",
        (
            "This is an inventory, paired-stream integrity review, reset-aware segmentation, "
            "and source-bound mechanic adjudication pass. It identifies high-confidence "
            "demonstrations for later behavior-cloning curation, but does not behavior-clone, "
            "train, define a production mechanic detector, or assign rewards. Successful, "
            "failed, and ambiguous evidence is preserved together."
        ),
        "",
        "## Inventory and validation",
        "",
        (
            "| Session | Class | Declared label | Frames | Attempts | Review-usable | "
            "Strict recorder diagnostic |"
        ),
        "|---|---|---:|---:|---:|---:|---|",
    ]
    for session in index["sessions"]:
        issue = "; ".join(session["validation_errors"] + session["completeness_errors"]) or "none"
        lines.append(
            f"| `{session['session_uuid']}` | {session['classification']} | "
            f"{session['declared_label']} | {session['frame_count']} | "
            f"{session['attempt_count']} | {str(session['review_usable']).lower()} | {issue} |"
        )
    lines.extend(
        [
            "",
            "## Reset-aware segmentation authority",
            "",
            (
                f"Version: `{SEGMENTATION_VERSION}`. Local-human native jump-onset events "
                "are activity anchors. Global physics-frame discontinuities and native reset, "
                "respawn, and local-car rebind events are hard boundaries. A sustained "
                f"{RECOVERY_RUN_TICKS}-tick period with at least "
                f"{MIN_RECOVERY_WHEEL_CONTACTS} world-contact wheels, "
                "grounded state, and inactive jump/dodge/flip components separates attempts. "
                "The segmentation rule never branches on the declared mechanic label. Events "
                "after the last paired action frame are excluded rather than clamped onto it."
            ),
            "",
            "## Mechanic-specific source-bound adjudication",
            "",
            (
                "Each exact source recording is reviewed against a declared-mechanic physical "
                "criterion using native 120 Hz state, effective action, and event evidence. "
                "Only high-confidence successes are listed as later behavior-cloning "
                "candidates. This adjudication is not a reusable production detector."
            ),
            "",
            (
                "| Label | Attempts | BC candidates | Failed | Ambiguous |"
            ),
            "|---|---:|---:|---:|---:|",
        ]
    )
    for row in index["sessions"]:
        if row["classification"] != "freeplay_mechanic_practice":
            continue
        counts = row["mechanic_verdict_counts"]
        lines.append(
            f"| {row['declared_label']} | {row['attempt_count']} | "
            f"{counts.get('success', 0)} | {counts.get('failure', 0)} | "
            f"{counts.get('ambiguous', 0)} |"
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
                "paired-stream verdict, event inventory, segmentation audit, and criteria."
            ),
            (
                "- `attempts/*.jsonl` preserves every attempt with raw timing references, "
                "physical outcome telemetry, and its source-bound adjudication."
            ),
            "- `mechanic_assessments.json` summarizes criteria and all verdict counts.",
            "- `behavior_cloning_candidates.json` contains only high-confidence source spans.",
            "- `groupings.json` is a compact descriptive grouping index.",
            "- `artifact_manifest.json` hashes the generated review package.",
            "- Re-run `python benchmarks/review_rival2_human_demos.py --verify-only` "
            "to verify source and review hashes plus attempt/index invariants.",
            "",
            "## Known evidence boundary",
            "",
            (
                "The strict recorder completeness verdict is retained unchanged as a diagnostic. "
                "The review verdict instead asks whether every retained frame has a valid paired "
                "state/action and whether global frame-ID gaps are explained by native lifecycle "
                "events. No tick is synthesized. The Nexto match is complete at the captured "
                "match-ended state even though a later post-match car-spawn callback caused a "
                "local-human identity shutdown."
            ),
            "",
        ]
    )
    return "\n".join(lines)


def _select_session_dirs(
    source_root: Path, session_ids: Sequence[str] | None = None
) -> list[Path]:
    available = {
        path.name: path
        for path in source_root.iterdir()
        if path.is_dir() and (path / "manifest.json").is_file()
    }
    if session_ids:
        requested = [str(session_id) for session_id in session_ids]
        duplicates = sorted(
            session_id for session_id, count in Counter(requested).items() if count > 1
        )
        if duplicates:
            raise ValueError(f"duplicate --session-id values: {', '.join(duplicates)}")
        missing = sorted(set(requested) - set(available))
        if missing:
            raise ValueError(f"requested session directories not found: {', '.join(missing)}")
        selected = [available[session_id] for session_id in requested]
    else:
        selected = list(available.values())
    return sorted(
        selected,
        key=lambda path: json.loads((path / "manifest.json").read_text(encoding="utf-8")).get(
            "capture_start_utc", ""
        ),
    )


def run(
    source_root: Path,
    output_dir: Path,
    session_ids: Sequence[str] | None = None,
) -> dict[str, Any]:
    session_dirs = _select_session_dirs(source_root, session_ids)
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
    mechanic_rows = []
    behavior_cloning_candidates = []
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
            "review_usable": report["review_usable"],
            "strict_recorder_validation_valid": report["strict_recorder_validation_valid"],
            "container_valid": validation["container_valid"],
            "clean_termination": validation["clean_termination"],
            "capture_complete": validation["capture_complete"],
            "missing_physics_frame_count": validation["missing_physics_frame_count"],
            "validation_errors": validation["errors"],
            "completeness_errors": validation["completeness_errors"],
            "lifecycle_gap_episode_count": report["paired_stream_assessment"][
                "lifecycle_gap_episode_count"
            ],
            "lifecycle_explained_global_frame_id_gap_count": report[
                "paired_stream_assessment"
            ]["lifecycle_explained_global_frame_id_gap_count"],
            "unexplained_global_frame_id_gap_count": report["paired_stream_assessment"][
                "unexplained_global_frame_id_gap_count"
            ],
            "unpaired_event_count": report["paired_stream_assessment"][
                "unpaired_event_count"
            ],
            "unpaired_marker_count": report["paired_stream_assessment"][
                "unpaired_marker_count"
            ],
            "match_ended_in_paired_frames": report["paired_stream_assessment"][
                "match_ended_in_paired_frames"
            ],
            "post_match_identity_shutdown": report["paired_stream_assessment"][
                "post_match_identity_shutdown"
            ],
            "mechanic_verdict_counts": (
                report["mechanic_assessment_summary"]["counts"]
                if report["mechanic_assessment_summary"]
                else {}
            ),
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
        if report["mechanic_assessment_summary"] is not None:
            summary = report["mechanic_assessment_summary"]
            mechanic_rows.append(
                {
                    "session_uuid": report["session_uuid"],
                    "declared_label": report["declared_label"],
                    "attempt_count": report["attempt_count"],
                    "criterion": summary["criterion"],
                    "counts": summary["counts"],
                    "source_binding": summary["source_binding"],
                }
            )
            behavior_cloning_candidates.extend(
                {
                    **candidate,
                    "session_uuid": report["session_uuid"],
                    "declared_label": report["declared_label"],
                    "source_file_set_sha256": report["source_file_set_sha256"],
                }
                for candidate in summary["behavior_cloning_candidates"]
            )
    demo_rows = [row for row in index_rows if row["review_scope"] == "new_demonstration"]
    index = {
        "format": FORMAT,
        "generated_utc": generated_utc,
        "source_root": str(source_root.resolve()),
        "session_selection": {
            "mode": "explicit_session_ids" if session_ids else "all_sessions",
            "session_ids": [path.name for path in session_dirs],
        },
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
        "unusable_new_demonstration_session_count": sum(
            not row["review_usable"] for row in demo_rows
        ),
        "strict_recorder_validation_failure_count": sum(
            not row["strict_recorder_validation_valid"] for row in demo_rows
        ),
        "behavior_cloning_candidate_count": len(behavior_cloning_candidates),
        "mechanic_verdict_counts": dict(
            sorted(
                sum(
                    (Counter(row["mechanic_verdict_counts"]) for row in demo_rows),
                    Counter(),
                ).items()
            )
        ),
        "sessions": index_rows,
        "authority_boundary": {
            "training_performed": False,
            "behavior_cloning_performed": False,
            "production_mechanic_detector_defined": False,
            "source_bound_behavior_cloning_candidate_selection_performed": True,
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
    _write_json(
        output_dir / "mechanic_assessments.json",
        {
            "format": FORMAT,
            "version": MECHANIC_ASSESSMENT_VERSION,
            "adjudication_file": str(ADJUDICATION_PATH.resolve()),
            "adjudication_file_sha256": _sha256(ADJUDICATION_PATH),
            "sessions": mechanic_rows,
            "all_attempts_preserved": True,
            "production_detector": False,
        },
    )
    _write_json(
        output_dir / "behavior_cloning_candidates.json",
        {
            "format": FORMAT,
            "version": MECHANIC_ASSESSMENT_VERSION,
            "candidate_count": len(behavior_cloning_candidates),
            "candidates": behavior_cloning_candidates,
            "training_performed": False,
            "behavior_cloning_performed": False,
            "use_restriction": (
                "Candidate source spans only. A later adapter must still define observation, "
                "action-window, split, and training semantics."
            ),
        },
    )
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
    mechanic = json.loads(
        (output_dir / "mechanic_assessments.json").read_text(encoding="utf-8")
    )
    mechanic_by_uuid = {str(row["session_uuid"]): row for row in mechanic.get("sessions", [])}
    candidates_document = json.loads(
        (output_dir / "behavior_cloning_candidates.json").read_text(encoding="utf-8")
    )
    if mechanic.get("adjudication_file_sha256") != _sha256(ADJUDICATION_PATH):
        errors.append("mechanic adjudication file hash mismatch")
    candidate_ids = {
        str(row["attempt_id"]) for row in candidates_document.get("candidates", [])
    }
    observed_candidate_ids: set[str] = set()
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
            "production_mechanic_detector_defined": False,
            "source_bound_behavior_cloning_candidate_selection_performed": bool(
                session_report["attempt_count"]
            ),
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
            assessment = attempt.get("mechanic_assessment")
            if assessment is None:
                errors.append(f"missing mechanic assessment: {attempt_id}")
            elif assessment.get("verdict") not in {"success", "failure", "ambiguous"}:
                errors.append(f"invalid mechanic verdict: {attempt_id}")
            elif bool(assessment.get("behavior_cloning_eligible")) != (
                assessment.get("verdict") == "success"
            ):
                errors.append(f"behavior-cloning eligibility mismatch: {attempt_id}")
            elif assessment.get("behavior_cloning_eligible"):
                observed_candidate_ids.add(attempt_id)
                quality = attempt["data_quality"]
                if (
                    int(quality["noncontiguous_adjacent_frame_count"]) != 0
                    or int(quality["invalid_rival_action_frame_count"]) != 0
                    or not bool(quality["complete_human_car_coverage"])
                ):
                    errors.append(
                        f"behavior-cloning candidate has invalid paired span: {attempt_id}"
                    )
        attempt_total += len(attempts)
        if expected_attempts:
            grouping_row = grouping_by_uuid.get(session_uuid)
            if grouping_row is None:
                errors.append(f"outcome grouping missing: {session_uuid}")
            elif sum(int(value) for value in grouping_row["counts"].values()) != len(attempts):
                errors.append(f"outcome grouping count mismatch: {session_uuid}")
            mechanic_row = mechanic_by_uuid.get(session_uuid)
            if mechanic_row is None:
                errors.append(f"mechanic assessment summary missing: {session_uuid}")
            elif sum(int(value) for value in mechanic_row["counts"].values()) != len(attempts):
                errors.append(f"mechanic assessment count mismatch: {session_uuid}")

    if attempt_total != int(index["mechanic_attempt_count"]):
        errors.append(
            f"global attempt count mismatch: {attempt_total} != {index['mechanic_attempt_count']}"
        )
    if observed_candidate_ids != candidate_ids:
        errors.append("behavior-cloning candidate index does not match successful attempts")
    if len(candidate_ids) != int(index["behavior_cloning_candidate_count"]):
        errors.append("behavior-cloning candidate count does not match index")
    if len(candidate_ids) != int(candidates_document.get("candidate_count", -1)):
        errors.append("behavior-cloning candidate count does not match candidate document")
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
        default=Path("results/rival2/human_demo_review_v2"),
    )
    parser.add_argument(
        "--session-id",
        action="append",
        default=[],
        help=(
            "Review only this session directory under --source-root. Repeat for an exact "
            "cohort; missing or duplicate values fail loudly."
        ),
    )
    parser.add_argument("--verify-only", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.verify_only:
        verification = verify_review(args.output_dir.resolve())
        print(json.dumps(verification, sort_keys=True))
        return 0 if verification["valid"] else 1
    index = run(
        args.source_root.resolve(),
        args.output_dir.resolve(),
        session_ids=args.session_id or None,
    )
    print(
        json.dumps(
            {
                "output_dir": str(args.output_dir.resolve()),
                "sessions": index["session_count"],
                "new_demonstrations": index["new_demonstration_session_count"],
                "mechanic_attempts": index["mechanic_attempt_count"],
                "review_usable_new_demonstrations": (
                    index["new_demonstration_session_count"]
                    - index["unusable_new_demonstration_session_count"]
                ),
                "behavior_cloning_candidates": index["behavior_cloning_candidate_count"],
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
