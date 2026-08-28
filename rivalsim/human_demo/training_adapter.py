"""Read-only, fail-closed 120 Hz human-demonstration training adapter.

The adapter never mutates a recording and never invents a Rival observation value. It
always exposes the exact eight-channel action target. It exposes a complete 182-element
observation only when every field is source-exact; otherwise ``observation`` is ``None``
and ``partial_observation`` contains NaN for every unresolved field.
"""

from __future__ import annotations

import bisect
import hashlib
import math
from collections import defaultdict
from collections.abc import Iterable, Iterator, Sequence
from dataclasses import dataclass
from itertools import pairwise
from pathlib import Path
from typing import Any

import numpy as np

from rivalsim.human_demo.reader import SessionReader
from rivalsim.rival2_contracts import (
    ACTION_CONTRACT_V2_120HZ_HASH,
    ACTION_NAMES,
    ANGULAR_SPEED_SCALE,
    BALL_LINEAR_SPEED_SCALE,
    CAR_LINEAR_SPEED_SCALE,
    OBS_DIM,
    OBS_FIELD_NAMES,
    OBSERVATION_SCHEMA_V2_120HZ_HASH,
    POSITION_SCALE,
    RIVAL2_ACTION_V2_120HZ_VERSION,
    RIVAL2_OBS_V2_120HZ_VERSION,
)

ADAPTER_FORMAT = "RIVAL2_HUMAN_DEMO_120HZ_ADAPTER_V1"
SPLIT_FORMAT = "RIVAL2_HUMAN_DEMO_FROZEN_SPLIT_V1"
MECHANIC_SPLIT_SEED = "RIVAL2_MECHANIC_BC_SPLIT_08A55E8_V1"
GAMEPLAY_SPLIT_SEED = "RIVAL2_GAMEPLAY_SPLIT_08A55E8_V1"
HARD_BOUNDARY_KINDS = {
    "freeplay_reset",
    "kickoff_or_round_reset",
    "local_car_rebind",
    "respawn",
}

_FIELD_INDEX = {name: index for index, name in enumerate(OBS_FIELD_NAMES)}
_CAR_EXACT_SUFFIXES = (
    "position.x",
    "position.y",
    "position.z",
    "linear_velocity.x",
    "linear_velocity.y",
    "linear_velocity.z",
    "forward.x",
    "forward.y",
    "forward.z",
    "up.x",
    "up.y",
    "up.z",
    "angular_velocity.x",
    "angular_velocity.y",
    "angular_velocity.z",
    "on_ground",
    "has_jumped",
    "has_double_jumped",
    "jump_available",
    "wheel_contact.front_left",
    "wheel_contact.front_right",
    "wheel_contact.back_left",
    "wheel_contact.back_right",
    "is_supersonic",
)
_RIVAL_INTERNAL_CAR_SUFFIXES = (
    "boost",
    "is_jumping",
    "has_flipped",
    "is_flipping",
    "dodge_available",
    "is_demoed",
    "demo_timer_remaining",
    "jump_time",
    "air_time",
    "air_time_since_jump",
    "flip_time",
    "boosting_time",
)
_UNAVAILABLE_CAR_SUFFIXES = (
    "time_since_boosted",
    "supersonic_time",
    "sticky_ticks",
)


def _readonly(value: np.ndarray) -> np.ndarray:
    result = np.ascontiguousarray(value)
    result.flags.writeable = False
    return result


def _orientation_basis(rotation: Sequence[int | float]) -> tuple[np.ndarray, np.ndarray]:
    pitch, yaw, roll = [float(value) * 2.0 * math.pi / 65536.0 for value in rotation]
    sp, cp = math.sin(pitch), math.cos(pitch)
    sy, cy = math.sin(yaw), math.cos(yaw)
    sr, cr = math.sin(roll), math.cos(roll)
    forward = np.asarray((cp * cy, cp * sy, sp), dtype=np.float32)
    up = np.asarray(
        (-(cr * sp * cy + sr * sy), cy * sr - cr * sp * sy, cr * cp),
        dtype=np.float32,
    )
    return forward, up


def _stable_hash(seed: str, identity: str) -> str:
    return hashlib.sha256(f"{seed}|{identity}".encode()).hexdigest().upper()


def action_target(frame: dict[str, Any]) -> np.ndarray:
    """Return the exact one-frame ``RIVAL2_ACTION_V2_120HZ`` target."""

    source = frame.get("rival_action", {})
    values = np.asarray([float(source[name]) for name in ACTION_NAMES], dtype=np.float32)
    if values.shape != (8,) or not bool(np.isfinite(values).all()):
        raise ValueError("native Rival action is not a finite eight-channel target")
    if bool(np.any(values[:5] < -1.0)) or bool(np.any(values[:5] > 1.0)):
        raise ValueError("native Rival analog action is outside [-1, 1]")
    if not bool(np.isin(values[5:], np.asarray((0.0, 1.0), dtype=np.float32)).all()):
        raise ValueError("native Rival button action is not binary")
    return _readonly(values)


def frames_are_contiguous(previous: dict[str, Any], current: dict[str, Any]) -> bool:
    return bool(
        int(current["sequence"]) == int(previous["sequence"]) + 1
        and int(current["physics_frame"]) == int(previous["physics_frame"]) + 1
    )


def _human(frame: dict[str, Any]) -> dict[str, Any] | None:
    cars = [car for car in frame.get("cars", []) if car.get("flags", {}).get("is_local_human")]
    return cars[0] if len(cars) == 1 else None


def _opponent(frame: dict[str, Any], human: dict[str, Any]) -> dict[str, Any] | None:
    cars = [car for car in frame.get("cars", []) if car is not human]
    return cars[0] if len(cars) == 1 else None


def _event_actor(event: dict[str, Any]) -> str:
    return str(event.get("contacting_car_id") or event.get("actor_id") or "")


def _boundary_between(
    previous: dict[str, Any] | None,
    current: dict[str, Any],
    boundary_event_physics: Sequence[int],
) -> bool:
    if previous is None or not frames_are_contiguous(previous, current):
        return True
    left = int(previous["physics_frame"])
    right = int(current["physics_frame"])
    position = bisect.bisect_right(boundary_event_physics, left)
    return bool(
        position < len(boundary_event_physics)
        and int(boundary_event_physics[position]) <= right
    )


@dataclass(frozen=True, slots=True)
class AdaptedSample:
    """One source frame and its exactness result."""

    session_uuid: str
    sequence: int
    physics_frame: int
    observation: np.ndarray | None
    partial_observation: np.ndarray
    exact_field_mask: np.ndarray
    action: np.ndarray
    previous_action_source_sequence: int | None
    blocked_fields: tuple[str, ...]
    blocker_reasons: tuple[str, ...]

    @property
    def usable(self) -> bool:
        return self.observation is not None


def _set_values(
    observation: np.ndarray,
    exact: np.ndarray,
    names: Sequence[str],
    values: Sequence[float | int | bool],
) -> None:
    if len(names) != len(values):
        raise ValueError("observation field/value length mismatch")
    for name, value in zip(names, values, strict=True):
        index = _FIELD_INDEX[name]
        observation[index] = np.float32(value)
        exact[index] = True


def _set_car_exact(
    observation: np.ndarray,
    exact: np.ndarray,
    prefix: str,
    car: dict[str, Any],
    sign: np.ndarray,
) -> bool:
    position = np.asarray(car["position"], dtype=np.float32) * sign
    velocity = np.asarray(car["linear_velocity"], dtype=np.float32) * sign
    angular = np.asarray(car["angular_velocity"], dtype=np.float32) * sign
    forward, up = _orientation_basis(car["rotation"])
    forward *= sign
    up *= sign
    names = [
        *(f"{prefix}.position.{axis}" for axis in "xyz"),
        *(f"{prefix}.linear_velocity.{axis}" for axis in "xyz"),
        *(f"{prefix}.forward.{axis}" for axis in "xyz"),
        *(f"{prefix}.up.{axis}" for axis in "xyz"),
        *(f"{prefix}.angular_velocity.{axis}" for axis in "xyz"),
        f"{prefix}.on_ground",
        f"{prefix}.has_jumped",
        f"{prefix}.has_double_jumped",
        f"{prefix}.jump_available",
        f"{prefix}.is_supersonic",
    ]
    flags = car["flags"]
    values = [
        *(position / np.asarray(POSITION_SCALE, dtype=np.float32)),
        *(velocity / np.float32(CAR_LINEAR_SPEED_SCALE)),
        *forward,
        *up,
        *(angular / np.float32(ANGULAR_SPEED_SCALE)),
        bool(flags.get("on_ground")),
        bool(flags.get("jumped")),
        bool(flags.get("double_jumped")),
        not bool(flags.get("jumped")),
        bool(flags.get("supersonic")),
    ]
    _set_values(observation, exact, names, values)
    wheel_rows = list(car.get("wheels", ()))
    wheel_indices = [int(wheel.get("index", -1)) for wheel in wheel_rows]
    if sorted(wheel_indices) != [0, 1, 2, 3]:
        return False
    wheels = {index: wheel for index, wheel in zip(wheel_indices, wheel_rows, strict=True)}
    _set_values(
        observation,
        exact,
        [
            f"{prefix}.wheel_contact.front_left",
            f"{prefix}.wheel_contact.front_right",
            f"{prefix}.wheel_contact.back_left",
            f"{prefix}.wheel_contact.back_right",
        ],
        [bool(wheels[index].get("has_world_contact")) for index in range(4)],
    )
    return True


def adapt_frame(
    frame: dict[str, Any],
    *,
    session_uuid: str,
    previous_frame: dict[str, Any] | None,
    events_at_physics_frame: Sequence[dict[str, Any]] = (),
    lifecycle_boundary_before: bool = False,
) -> AdaptedSample:
    """Adapt a native frame without synthesizing unresolved observation fields."""

    observation = np.full(OBS_DIM, np.nan, dtype=np.float32)
    exact = np.zeros(OBS_DIM, dtype=bool)
    reasons: set[str] = set()
    human = _human(frame)
    if human is None:
        reasons.add("local_human_not_unique")
    else:
        team = int(human.get("team", -1))
        if team not in (0, 1):
            reasons.add("local_human_team_not_binary")
        else:
            sign = np.asarray((1.0, 1.0, 1.0) if team == 0 else (-1.0, -1.0, 1.0))
            ball = frame["ball"]
            ball_position = np.asarray(ball["position"], dtype=np.float32) * sign
            ball_velocity = np.asarray(ball["linear_velocity"], dtype=np.float32) * sign
            ball_angular = np.asarray(ball["angular_velocity"], dtype=np.float32) * sign
            _set_values(
                observation,
                exact,
                [
                    *(f"ball.position.{axis}" for axis in "xyz"),
                    *(f"ball.linear_velocity.{axis}" for axis in "xyz"),
                    *(f"ball.angular_velocity.{axis}" for axis in "xyz"),
                ],
                [
                    *(ball_position / np.asarray(POSITION_SCALE, dtype=np.float32)),
                    *(ball_velocity / np.float32(BALL_LINEAR_SPEED_SCALE)),
                    *(ball_angular / np.float32(ANGULAR_SPEED_SCALE)),
                ],
            )
            if not _set_car_exact(observation, exact, "self", human, sign):
                reasons.add("self_wheel_identity_not_unique")
            human_position = np.asarray(human["position"], dtype=np.float32)
            human_velocity = np.asarray(human["linear_velocity"], dtype=np.float32)
            _set_values(
                observation,
                exact,
                [f"relative.ball_position.{axis}" for axis in "xyz"],
                [
                    *((np.asarray(ball["position"]) - human_position) * sign / POSITION_SCALE),
                ],
            )

            relative_velocity_names = [
                f"relative.ball_velocity.{axis}" for axis in "xyz"
            ]
            relative_velocity_values = (
                (np.asarray(ball["linear_velocity"], dtype=np.float32) - human_velocity)
                * sign
                / np.float32(BALL_LINEAR_SPEED_SCALE)
            )
            _set_values(
                observation,
                exact,
                relative_velocity_names,
                relative_velocity_values,
            )

            opponent = _opponent(frame, human)
            if opponent is None:
                reasons.add("opponent_absent_from_native_freeplay_state")
            else:
                if not _set_car_exact(observation, exact, "opponent", opponent, sign):
                    reasons.add("opponent_wheel_identity_not_unique")
                opponent_position = np.asarray(opponent["position"], dtype=np.float32)
                opponent_velocity = np.asarray(opponent["linear_velocity"], dtype=np.float32)
                _set_values(
                    observation,
                    exact,
                    [
                        *(f"relative.opponent_position.{axis}" for axis in "xyz"),
                        *(f"relative.opponent_velocity.{axis}" for axis in "xyz"),
                    ],
                    [
                        *((opponent_position - human_position) * sign / POSITION_SCALE),
                        *(
                            (opponent_velocity - human_velocity)
                            * sign
                            / np.float32(CAR_LINEAR_SPEED_SCALE)
                        ),
                    ],
                )

            if (
                previous_frame is not None
                and frames_are_contiguous(previous_frame, frame)
                and not lifecycle_boundary_before
            ):
                previous_action = action_target(previous_frame)
                _set_values(
                    observation,
                    exact,
                    [f"previous_action.{name}" for name in ACTION_NAMES],
                    previous_action,
                )
                previous_sequence = int(previous_frame["sequence"])
            else:
                reasons.add("previous_action_unavailable_at_span_or_lifecycle_boundary")
                previous_sequence = None

            events_exact = previous_frame is not None or bool(events_at_physics_frame)
            if events_exact:
                human_id = str(human.get("stable_id", ""))
                opponent_id = str(opponent.get("stable_id", "")) if opponent else ""
                touch_actors = {
                    _event_actor(event)
                    for event in events_at_physics_frame
                    if event.get("kind") == "ball_touch"
                }
                demo_actors = {
                    _event_actor(event)
                    for event in events_at_physics_frame
                    if event.get("kind") in {"demolition", "demo", "demolished"}
                }
                _set_values(
                    observation,
                    exact,
                    [
                        "lifecycle.kickoff_reset",
                        "lifecycle.self_touch_event",
                        "lifecycle.opponent_touch_event",
                        "lifecycle.self_demoed_event",
                        "lifecycle.opponent_demoed_event",
                    ],
                    [
                        any(
                            event.get("kind") in HARD_BOUNDARY_KINDS
                            for event in events_at_physics_frame
                        ),
                        human_id in touch_actors,
                        bool(opponent_id and opponent_id in touch_actors),
                        human_id in demo_actors,
                        bool(opponent_id and opponent_id in demo_actors),
                    ],
                )
            else:
                reasons.add("pre_capture_lifecycle_event_history_unavailable")

    # These fields cannot be filled exactly from the committed native schema.
    reasons.add("boost_pad_identity_and_cooldown_unavailable")
    reasons.add("rivalsim_internal_timer_or_state_semantics_not_source_exact")
    reasons.add("native_sdk_does_not_expose_required_rivalsim_memory")
    blocked = tuple(
        name
        for name, available in zip(OBS_FIELD_NAMES, exact, strict=True)
        if not available
    )
    partial = _readonly(observation)
    exact_mask = _readonly(exact)
    complete = bool(exact.all())
    full = _readonly(observation.copy()) if complete else None
    return AdaptedSample(
        session_uuid=session_uuid,
        sequence=int(frame["sequence"]),
        physics_frame=int(frame["physics_frame"]),
        observation=full,
        partial_observation=partial,
        exact_field_mask=exact_mask,
        action=action_target(frame),
        previous_action_source_sequence=previous_sequence,
        blocked_fields=blocked,
        blocker_reasons=tuple(sorted(reasons)),
    )


class ReadOnlyTrajectoryAdapter:
    """Stream complete source spans while retaining the source immediately before them."""

    def __init__(self, session_dir: Path):
        self.session_dir = Path(session_dir)
        self.reader = SessionReader(self.session_dir)
        self.session_uuid = str(self.reader.manifest["session_uuid"])
        self.events = tuple(self.reader.iter_events())
        events_by_physics: dict[int, list[dict[str, Any]]] = defaultdict(list)
        for event in self.events:
            events_by_physics[int(event.get("physics_frame", -1))].append(event)
        self.events_by_physics = dict(events_by_physics)
        self.boundary_event_physics = tuple(
            sorted(
                int(event["physics_frame"])
                for event in self.events
                if event.get("kind") in HARD_BOUNDARY_KINDS
                and int(event.get("physics_frame", -1)) >= 0
            )
        )

    def iter_span(self, start_sequence: int, end_sequence: int) -> Iterator[AdaptedSample]:
        if start_sequence < 0 or end_sequence < start_sequence:
            raise ValueError("invalid source sequence span")
        previous: dict[str, Any] | None = None
        found = 0
        for frame in self.reader.iter_frames():
            sequence = int(frame["sequence"])
            if sequence < start_sequence:
                previous = frame
                continue
            if sequence > end_sequence:
                break
            boundary = _boundary_between(previous, frame, self.boundary_event_physics)
            yield adapt_frame(
                frame,
                session_uuid=self.session_uuid,
                previous_frame=previous,
                events_at_physics_frame=self.events_by_physics.get(
                    int(frame["physics_frame"]), ()
                ),
                lifecycle_boundary_before=boundary,
            )
            previous = frame
            found += 1
        expected = end_sequence - start_sequence + 1
        if found != expected:
            raise ValueError(
                f"source span frame count mismatch: expected {expected}, found {found}"
            )

    def trajectory(self, start_sequence: int, end_sequence: int) -> tuple[AdaptedSample, ...]:
        return tuple(self.iter_span(start_sequence, end_sequence))

    def iter_spans(
        self, spans: Iterable[tuple[str, int, int]]
    ) -> Iterator[tuple[str, AdaptedSample]]:
        """Stream non-overlapping spans in one source pass."""

        ordered = sorted(
            ((str(identity), int(start), int(end)) for identity, start, end in spans),
            key=lambda row: (row[1], row[2], row[0]),
        )
        for identity, start, end in ordered:
            if start < 0 or end < start:
                raise ValueError(f"invalid source sequence span: {identity}")
        for previous_row, current_row in pairwise(ordered):
            if current_row[1] <= previous_row[2]:
                raise ValueError(
                    f"overlapping source spans: {previous_row[0]} and {current_row[0]}"
                )
        if not ordered:
            return
        found = {identity: 0 for identity, _start, _end in ordered}
        span_index = 0
        previous: dict[str, Any] | None = None
        for frame in self.reader.iter_frames():
            sequence = int(frame["sequence"])
            while span_index < len(ordered) and sequence > ordered[span_index][2]:
                span_index += 1
            if span_index >= len(ordered):
                break
            identity, start, end = ordered[span_index]
            if start <= sequence <= end:
                boundary = _boundary_between(previous, frame, self.boundary_event_physics)
                sample = adapt_frame(
                    frame,
                    session_uuid=self.session_uuid,
                    previous_frame=previous,
                    events_at_physics_frame=self.events_by_physics.get(
                        int(frame["physics_frame"]), ()
                    ),
                    lifecycle_boundary_before=boundary,
                )
                found[identity] += 1
                yield identity, sample
            previous = frame
        for identity, start, end in ordered:
            expected = end - start + 1
            if found[identity] != expected:
                raise ValueError(
                    f"source span frame count mismatch for {identity}: "
                    f"expected {expected}, found {found[identity]}"
                )

    def iter_usable_samples(
        self, start_sequence: int, end_sequence: int
    ) -> Iterator[AdaptedSample]:
        yield from (
            sample
            for sample in self.iter_span(start_sequence, end_sequence)
            if sample.usable
        )


def split_mechanic_candidates(
    candidates: Iterable[dict[str, Any]],
    *,
    seed: str = MECHANIC_SPLIT_SEED,
) -> list[dict[str, Any]]:
    """Assign whole attempts with deterministic per-label rare-class protection."""

    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for candidate in candidates:
        grouped[str(candidate["declared_label"])].append(dict(candidate))
    result = []
    for label in sorted(grouped):
        ordered = sorted(
            grouped[label],
            key=lambda row: (
                _stable_hash(seed, str(row["attempt_id"])),
                row["attempt_id"],
            ),
        )
        count = len(ordered)
        if count >= 10:
            test_count = max(1, round(count * 0.10))
            validation_count = max(1, round(count * 0.10))
        elif count >= 3:
            test_count = 1
            validation_count = 1
        elif count == 2:
            test_count = 0
            validation_count = 1
        else:
            test_count = 0
            validation_count = 0
        for index, candidate in enumerate(ordered):
            split = (
                "test"
                if index < test_count
                else "validation"
                if index < test_count + validation_count
                else "train"
            )
            candidate["split"] = split
            candidate["split_hash"] = _stable_hash(seed, str(candidate["attempt_id"]))
            result.append(candidate)
    return sorted(result, key=lambda row: str(row["attempt_id"]))


def split_gameplay_regions(
    regions: Iterable[dict[str, Any]],
    *,
    session_uuid: str,
    seed: str = GAMEPLAY_SPLIT_SEED,
) -> list[dict[str, Any]]:
    """Assign whole lifecycle regions so neighboring frames never cross a split."""

    rows = [dict(region) for region in regions]
    if len(rows) < 3:
        raise ValueError("gameplay split requires at least three lifecycle regions")
    for row in rows:
        identity = f"{session_uuid}:{row['start_sequence']}:{row['end_sequence']}"
        row["split_hash"] = _stable_hash(seed, identity)
    ordered = sorted(rows, key=lambda row: (row["split_hash"], row["start_sequence"]))
    test_count = max(1, round(len(rows) * 0.10))
    validation_count = max(1, round(len(rows) * 0.10))
    for index, row in enumerate(ordered):
        row["split"] = (
            "test"
            if index < test_count
            else "validation"
            if index < test_count + validation_count
            else "train"
        )
    return sorted(rows, key=lambda row: int(row["start_sequence"]))


def contract_identity() -> dict[str, Any]:
    return {
        "adapter_format": ADAPTER_FORMAT,
        "split_format": SPLIT_FORMAT,
        "observation_version": RIVAL2_OBS_V2_120HZ_VERSION,
        "observation_schema_sha256": OBSERVATION_SCHEMA_V2_120HZ_HASH,
        "observation_shape": [OBS_DIM],
        "observation_dtype": "float32",
        "action_version": RIVAL2_ACTION_V2_120HZ_VERSION,
        "action_contract_sha256": ACTION_CONTRACT_V2_120HZ_HASH,
        "action_shape": [len(ACTION_NAMES)],
        "action_dtype": "float32",
        "action_channels": list(ACTION_NAMES),
        "temporal_reduction": None,
    }


__all__ = [
    "ADAPTER_FORMAT",
    "GAMEPLAY_SPLIT_SEED",
    "MECHANIC_SPLIT_SEED",
    "SPLIT_FORMAT",
    "AdaptedSample",
    "ReadOnlyTrajectoryAdapter",
    "action_target",
    "adapt_frame",
    "contract_identity",
    "frames_are_contiguous",
    "split_gameplay_regions",
    "split_mechanic_candidates",
]
