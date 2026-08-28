"""Masked 120 Hz human-demo bridge for future behavior cloning.

This module is intentionally separate from :mod:`training_adapter`.  The exact adapter
answers whether all 182 fields can be reproduced exactly and remains fail-closed.  This
bridge emits a finite training-domain vector plus an explicit per-field quality mask; a
neutral value in an unavailable field is never represented as measured or exact.
"""

from __future__ import annotations

import bisect
import hashlib
import json
from collections import defaultdict
from collections.abc import Iterable, Iterator, Sequence
from dataclasses import asdict, dataclass
from enum import IntEnum
from itertools import pairwise
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F

from rivalsim.human_demo.reader import SessionReader
from rivalsim.human_demo.training_adapter import (
    HARD_BOUNDARY_KINDS,
    AdaptedSample,
    action_target,
    adapt_frame,
    frames_are_contiguous,
)
from rivalsim.rival2_contracts import (
    ACTION_CONTRACT_V2_120HZ_HASH,
    ACTION_NAMES,
    AIR_TIME_SCALE,
    BOOST_SCALE,
    BOOSTING_TIME_SCALE,
    DEMO_TIMER_SCALE,
    EPISODE_AGE_SCALE_TICKS,
    FLIP_TIME_SCALE,
    JUMP_TIME_SCALE,
    NO_TOUCH_AGE_SCALE_TICKS,
    OBS_DIM,
    OBS_FIELD_NAMES,
    OBSERVATION_SCHEMA_V2_120HZ_HASH,
    RIVAL2_ACTION_V2_120HZ_VERSION,
    RIVAL2_OBS_V2_120HZ_VERSION,
)
from rivalsim.rival2_policy import Rival2ActorCritic, Rival2PolicyConfig

BC_BRIDGE_VERSION = "RIVAL2_HUMAN_DEMO_BC_OBSERVATION_BRIDGE_V1"
BC_QUALITY_VERSION = "RIVAL2_HUMAN_DEMO_FIELD_QUALITY_V1"
NEUTRAL_UNAVAILABLE_VALUE = np.float32(0.0)


class FieldQuality(IntEnum):
    """Ordered quality codes; lower values may never be promoted silently."""

    UNAVAILABLE = 0
    APPROXIMATE = 1
    EXACT_DERIVED = 2
    EXACT_DIRECT = 3


_QUALITY_LABELS = {
    FieldQuality.UNAVAILABLE: "unavailable",
    FieldQuality.APPROXIMATE: "approximate_semantically_reconstructed",
    FieldQuality.EXACT_DERIVED: "exactly_derivable",
    FieldQuality.EXACT_DIRECT: "exact_direct",
}

_DIRECT_CAR_SUFFIXES = {
    "on_ground",
    "has_jumped",
    "has_double_jumped",
    "wheel_contact.front_left",
    "wheel_contact.front_right",
    "wheel_contact.back_left",
    "wheel_contact.back_right",
    "is_supersonic",
}
_DERIVED_CAR_SUFFIXES = {
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
    "jump_available",
}
_APPROXIMATE_CAR_SUFFIXES = {
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
}
_UNAVAILABLE_CAR_SUFFIXES = {
    "time_since_boosted",
    "supersonic_time",
    "sticky_ticks",
}
_EXACT_LIFECYCLE_FIELDS = {
    "lifecycle.kickoff_reset",
    "lifecycle.self_touch_event",
    "lifecycle.opponent_touch_event",
    "lifecycle.self_demoed_event",
    "lifecycle.opponent_demoed_event",
}
_CORE_USABILITY_FIELDS = (
    *(f"ball.{kind}.{axis}" for kind in ("position", "linear_velocity") for axis in "xyz"),
    *(f"self.{kind}.{axis}" for kind in ("position", "linear_velocity") for axis in "xyz"),
    *(f"self.{kind}.{axis}" for kind in ("forward", "up") for axis in "xyz"),
)
_FIELD_INDEX = {name: index for index, name in enumerate(OBS_FIELD_NAMES)}


@dataclass(frozen=True, slots=True)
class FieldQualitySpec:
    index: int
    field: str
    quality: int
    classification: str
    source: str
    reconstruction: str


def _field_spec(index: int, field: str) -> FieldQualitySpec:
    if field.startswith("ball."):
        quality = FieldQuality.EXACT_DERIVED
        source = "native frame.ball"
        reconstruction = "team canonicalization and frozen Rival normalization"
    elif field.startswith("relative."):
        quality = FieldQuality.EXACT_DERIVED
        source = "native ball and car transforms"
        reconstruction = "source subtraction, team canonicalization, and frozen normalization"
    elif field.startswith("previous_action."):
        quality = FieldQuality.APPROXIMATE
        source = "preceding contiguous 120 Hz native rival_action"
        reconstruction = "exact predecessor when present; unavailable at a true boundary"
    elif field.startswith("boost_pad."):
        quality = FieldQuality.UNAVAILABLE
        source = "none with canonical Rival pad identity"
        reconstruction = (
            "neutral placeholder with unavailable mask; pointer-only pickup events are not "
            "promoted to canonical pad telemetry"
        )
    elif field in _EXACT_LIFECYCLE_FIELDS:
        quality = FieldQuality.EXACT_DERIVED
        source = "native event stream"
        reconstruction = "one-frame event indicator after capture begins"
    elif field in {"lifecycle.episode_age", "lifecycle.no_touch_age"}:
        quality = FieldQuality.APPROXIMATE
        source = "contiguous 120 Hz span and native reset/touch events"
        reconstruction = "lower-bound counter from the first observed lifecycle boundary"
    elif field.startswith(("self.", "opponent.")):
        suffix = field.split(".", 1)[1]
        if suffix in _DIRECT_CAR_SUFFIXES:
            quality = FieldQuality.EXACT_DIRECT
            source = "native frame.cars wrapper/component"
            reconstruction = "direct boolean or uniquely indexed wheel value"
        elif suffix in _DERIVED_CAR_SUFFIXES:
            quality = FieldQuality.EXACT_DERIVED
            source = "native frame.cars transform/flags"
            reconstruction = "basis conversion, team canonicalization, or boolean derivation"
        elif suffix in _APPROXIMATE_CAR_SUFFIXES:
            quality = FieldQuality.APPROXIMATE
            source = "native car flags, components, and component activity timers"
            reconstruction = "deterministic Rocket League proxy normalized to the Rival field"
        elif suffix in _UNAVAILABLE_CAR_SUFFIXES:
            quality = FieldQuality.UNAVAILABLE
            source = "none in the pinned native recorder schema"
            reconstruction = "neutral placeholder with unavailable mask"
        else:
            raise ValueError(f"unclassified Rival car observation field: {field}")
    else:
        raise ValueError(f"unclassified Rival observation field: {field}")
    return FieldQualitySpec(
        index=index,
        field=field,
        quality=int(quality),
        classification=_QUALITY_LABELS[quality],
        source=source,
        reconstruction=reconstruction,
    )


FIELD_QUALITY_SPECS = tuple(
    _field_spec(index, field) for index, field in enumerate(OBS_FIELD_NAMES)
)
GLOBAL_QUALITY_MASK = np.asarray(
    [row.quality for row in FIELD_QUALITY_SPECS], dtype=np.uint8
)
GLOBAL_QUALITY_MASK.flags.writeable = False


def _canonical_hash(value: object) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":")).encode("ascii")
    return hashlib.sha256(payload).hexdigest().upper()


FIELD_QUALITY_CONTRACT_SHA256 = _canonical_hash(
    {
        "version": BC_QUALITY_VERSION,
        "observation_contract": OBSERVATION_SCHEMA_V2_120HZ_HASH,
        "neutral_unavailable_value": float(NEUTRAL_UNAVAILABLE_VALUE),
        "fields": [asdict(row) for row in FIELD_QUALITY_SPECS],
    }
)


def field_quality_contract() -> dict[str, Any]:
    counts = defaultdict(int)
    for row in FIELD_QUALITY_SPECS:
        counts[row.classification] += 1
    return {
        "version": BC_QUALITY_VERSION,
        "bridge_version": BC_BRIDGE_VERSION,
        "sha256": FIELD_QUALITY_CONTRACT_SHA256,
        "observation_version": RIVAL2_OBS_V2_120HZ_VERSION,
        "observation_contract_sha256": OBSERVATION_SCHEMA_V2_120HZ_HASH,
        "field_count": OBS_DIM,
        "neutral_unavailable_value": float(NEUTRAL_UNAVAILABLE_VALUE),
        "quality_codes": {
            str(int(quality)): label for quality, label in _QUALITY_LABELS.items()
        },
        "counts": dict(sorted(counts.items())),
        "fields": [asdict(row) for row in FIELD_QUALITY_SPECS],
    }


def _readonly(value: np.ndarray) -> np.ndarray:
    result = np.ascontiguousarray(value)
    result.flags.writeable = False
    return result


def _timer(value: float, scale: float) -> np.float32:
    return np.float32(np.clip(np.float32(value) / np.float32(scale), 0.0, 1.0))


def _human(frame: dict[str, Any]) -> dict[str, Any] | None:
    rows = [
        car
        for car in frame.get("cars", ())
        if car.get("flags", {}).get("is_local_human")
    ]
    return rows[0] if len(rows) == 1 else None


def _opponent(frame: dict[str, Any], human: dict[str, Any]) -> dict[str, Any] | None:
    rows = [car for car in frame.get("cars", ()) if car is not human]
    return rows[0] if len(rows) == 1 else None


def _set_approximate(
    values: np.ndarray,
    quality: np.ndarray,
    field: str,
    value: float | int | bool,
) -> None:
    numeric = np.float32(value)
    if not np.isfinite(numeric):
        return
    index = _FIELD_INDEX[field]
    if GLOBAL_QUALITY_MASK[index] != int(FieldQuality.APPROXIMATE):
        raise RuntimeError(f"attempted approximate write to non-approximate field: {field}")
    values[index] = numeric
    quality[index] = int(FieldQuality.APPROXIMATE)


def _reconstruct_car_approximate(
    values: np.ndarray,
    quality: np.ndarray,
    prefix: str,
    car: dict[str, Any] | None,
) -> None:
    if car is None:
        return
    flags = car.get("flags", {})
    jumped = bool(flags.get("jumped"))
    double_jumped = bool(flags.get("double_jumped"))
    has_native_flip = bool(flags.get("has_flip"))
    jump = car.get("jump_component", {})
    dodge = car.get("dodge_component", {})
    flip = car.get("flip_component", {})
    boost = car.get("boost_component", {})
    is_flipping = bool(dodge.get("active")) or bool(flip.get("active"))
    has_flipped = jumped and not double_jumped and not has_native_flip
    flip_time = max(
        float(dodge.get("activity_time", 0.0)),
        float(flip.get("activity_time", 0.0)),
        float(flip.get("flip_time", 0.0)),
    )
    reconstructed = {
        "boost": np.float32(car.get("boost", 0.0)) / np.float32(BOOST_SCALE),
        "is_jumping": bool(jump.get("active")),
        "has_flipped": has_flipped,
        "is_flipping": is_flipping,
        "dodge_available": has_native_flip,
        "is_demoed": bool(flags.get("demolished")),
        "demo_timer_remaining": _timer(
            float(car.get("respawn_time_remaining", 0.0)), DEMO_TIMER_SCALE
        ),
        "jump_time": _timer(float(jump.get("activity_time", 0.0)), JUMP_TIME_SCALE),
        "air_time": _timer(float(car.get("time_off_ground", 0.0)), AIR_TIME_SCALE),
        "air_time_since_jump": _timer(
            float(car.get("time_off_ground", 0.0)) if jumped else 0.0,
            AIR_TIME_SCALE,
        ),
        "flip_time": _timer(flip_time, FLIP_TIME_SCALE),
        "boosting_time": _timer(
            float(boost.get("activity_time", 0.0)) if boost.get("active") else 0.0,
            BOOSTING_TIME_SCALE,
        ),
    }
    for suffix, value in reconstructed.items():
        _set_approximate(values, quality, f"{prefix}.{suffix}", value)


@dataclass(slots=True)
class TrajectoryReconstructionState:
    """Span-local lower-bound lifecycle counters for approximate age fields."""

    episode_ticks: int = 0
    no_touch_ticks: int = 0

    def reset(self) -> None:
        self.episode_ticks = 0
        self.no_touch_ticks = 0

    def values(self) -> tuple[np.float32, np.float32]:
        return (
            _timer(float(self.episode_ticks), float(EPISODE_AGE_SCALE_TICKS)),
            _timer(float(self.no_touch_ticks), float(NO_TOUCH_AGE_SCALE_TICKS)),
        )

    def advance(self, events: Sequence[dict[str, Any]]) -> None:
        self.episode_ticks += 1
        if any(event.get("kind") == "ball_touch" for event in events):
            self.no_touch_ticks = 0
        else:
            self.no_touch_ticks += 1


@dataclass(frozen=True, slots=True)
class BCBridgeSample:
    session_uuid: str
    sequence: int
    physics_frame: int
    observation: np.ndarray
    quality: np.ndarray
    availability: np.ndarray
    action: np.ndarray
    action_unchanged_from_exact_adapter: bool
    exact_audit_usable: bool
    bc_usable: bool
    previous_action_source_sequence: int | None


def bridge_human_frame(
    frame: dict[str, Any],
    *,
    exact_sample: AdaptedSample,
    trajectory_state: TrajectoryReconstructionState,
    events_at_physics_frame: Sequence[dict[str, Any]] = (),
    span_start: bool = False,
    lifecycle_boundary_before: bool = False,
) -> BCBridgeSample:
    """Create one finite masked BC-domain observation from one native frame."""

    if span_start or lifecycle_boundary_before:
        trajectory_state.reset()
    values = np.full(OBS_DIM, NEUTRAL_UNAVAILABLE_VALUE, dtype=np.float32)
    quality = np.zeros(OBS_DIM, dtype=np.uint8)

    for spec in FIELD_QUALITY_SPECS:
        if spec.quality not in {
            int(FieldQuality.EXACT_DIRECT),
            int(FieldQuality.EXACT_DERIVED),
        }:
            continue
        if exact_sample.exact_field_mask[spec.index]:
            value = exact_sample.partial_observation[spec.index]
            if np.isfinite(value):
                values[spec.index] = value
                quality[spec.index] = spec.quality

    human = _human(frame)
    opponent = _opponent(frame, human) if human is not None else None
    _reconstruct_car_approximate(values, quality, "self", human)
    _reconstruct_car_approximate(values, quality, "opponent", opponent)

    for channel in ACTION_NAMES:
        field = f"previous_action.{channel}"
        index = _FIELD_INDEX[field]
        if exact_sample.exact_field_mask[index]:
            _set_approximate(
                values,
                quality,
                field,
                exact_sample.partial_observation[index],
            )

    episode_age, no_touch_age = trajectory_state.values()
    _set_approximate(values, quality, "lifecycle.episode_age", episode_age)
    _set_approximate(values, quality, "lifecycle.no_touch_age", no_touch_age)
    trajectory_state.advance(events_at_physics_frame)

    if bool(np.any(quality > GLOBAL_QUALITY_MASK)):
        raise RuntimeError("BC bridge silently promoted field quality")
    availability = quality != int(FieldQuality.UNAVAILABLE)
    core_indices = np.asarray(
        [_FIELD_INDEX[field] for field in _CORE_USABILITY_FIELDS], dtype=np.int64
    )
    action = action_target(frame)
    action_unchanged = bool(np.array_equal(action, exact_sample.action))
    usable = bool(
        availability[core_indices].all()
        and np.isfinite(values).all()
        and np.isfinite(action).all()
    )
    return BCBridgeSample(
        session_uuid=exact_sample.session_uuid,
        sequence=exact_sample.sequence,
        physics_frame=exact_sample.physics_frame,
        observation=_readonly(values),
        quality=_readonly(quality),
        availability=_readonly(availability),
        action=action,
        action_unchanged_from_exact_adapter=action_unchanged,
        exact_audit_usable=exact_sample.usable,
        bc_usable=usable,
        previous_action_source_sequence=exact_sample.previous_action_source_sequence,
    )


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


class BCBridgeTrajectoryAdapter:
    """Stream non-overlapping native spans with masked BC observations."""

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

    def iter_spans(
        self, spans: Iterable[tuple[str, int, int]]
    ) -> Iterator[tuple[str, BCBridgeSample]]:
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
        states = {
            identity: TrajectoryReconstructionState()
            for identity, _start, _end in ordered
        }
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
                boundary = _boundary_between(
                    previous, frame, self.boundary_event_physics
                )
                events = self.events_by_physics.get(int(frame["physics_frame"]), ())
                exact = adapt_frame(
                    frame,
                    session_uuid=self.session_uuid,
                    previous_frame=previous,
                    events_at_physics_frame=events,
                    lifecycle_boundary_before=boundary,
                )
                sample = bridge_human_frame(
                    frame,
                    exact_sample=exact,
                    trajectory_state=states[identity],
                    events_at_physics_frame=events,
                    span_start=sequence == start,
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

    def trajectory(self, start_sequence: int, end_sequence: int) -> tuple[BCBridgeSample, ...]:
        rows = self.iter_spans((("trajectory", start_sequence, end_sequence),))
        return tuple(sample for _identity, sample in rows)


def degrade_simulator_observations(
    true_observations: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Apply the BC availability contract to complete authoritative observations.

    Simulator-native values are the numerical target of each deterministic human
    reconstruction, so direct, derived, and approximate fields retain their true value while
    their quality code remains distinct.  Truly unavailable fields receive the neutral value
    and an unavailable mask.  This isolates information loss caused by missing fields; it does
    not claim that cross-domain approximate semantics have zero error.
    """

    source = np.asarray(true_observations, dtype=np.float32)
    if source.ndim != 2 or source.shape[1] != OBS_DIM:
        raise ValueError("simulator observations must have shape [N, 182]")
    if not bool(np.isfinite(source).all()):
        raise ValueError("simulator observations contain nonfinite values")
    quality = np.broadcast_to(GLOBAL_QUALITY_MASK, source.shape).copy()
    degraded = source.copy()
    degraded[quality == int(FieldQuality.UNAVAILABLE)] = NEUTRAL_UNAVAILABLE_VALUE
    return _readonly(degraded), _readonly(quality)


@dataclass(frozen=True, slots=True)
class DistillationObjectiveResult:
    loss: torch.Tensor
    per_action_channel_kl: torch.Tensor
    per_sample_kl: torch.Tensor
    teacher_actor: torch.Tensor
    student_actor: torch.Tensor
    unavailable_fraction: torch.Tensor


def actor_distribution_distillation_objective(
    teacher_model: Rival2ActorCritic,
    student_model: Rival2ActorCritic,
    true_observation: torch.Tensor,
    degraded_observation: torch.Tensor,
    quality: torch.Tensor,
    *,
    policy_config: Rival2PolicyConfig | None = None,
) -> DistillationObjectiveResult:
    """Return a differentiable student objective without stepping an optimizer.

    The teacher and student must be independent modules so a future optimizer cannot move the
    frozen teacher across objective calls.  The teacher path is detached; the student path is
    differentiable so a separately authorized future task can use the interface.  This function
    creates no optimizer, calls no backward pass, and mutates neither module nor its gradients.
    """

    if teacher_model is student_model:
        raise ValueError("teacher and student models must be independent modules")
    config = policy_config or student_model.config
    if teacher_model.config != student_model.config:
        raise ValueError("teacher/student policy configurations differ")
    if true_observation.shape != degraded_observation.shape:
        raise ValueError("teacher/student observation shapes differ")
    if true_observation.ndim != 2 or true_observation.shape[1] != OBS_DIM:
        raise ValueError("paired distillation observations must have shape [N, 182]")
    if quality.shape != true_observation.shape:
        raise ValueError("paired distillation quality mask shape mismatch")
    with torch.no_grad():
        teacher_actor, _teacher_value = teacher_model(true_observation)
        teacher_actor = teacher_actor.detach()
    student_actor, _student_value = student_model(degraded_observation)

    teacher_mean = teacher_actor[:, :5]
    teacher_log_std = teacher_actor[:, 5:10].clamp(
        config.log_std_min, config.log_std_max
    )
    student_mean = student_actor[:, :5]
    student_log_std = student_actor[:, 5:10].clamp(
        config.log_std_min, config.log_std_max
    )
    teacher_variance = torch.exp(2.0 * teacher_log_std)
    student_variance = torch.exp(2.0 * student_log_std)
    analog_kl = (
        student_log_std
        - teacher_log_std
        + (
            teacher_variance
            + (teacher_mean - student_mean).square()
        )
        / (2.0 * student_variance)
        - 0.5
    )

    teacher_logits = teacher_actor[:, 10:13]
    student_logits = student_actor[:, 10:13]
    teacher_probability = torch.sigmoid(teacher_logits)
    button_kl = teacher_probability * (
        F.logsigmoid(teacher_logits) - F.logsigmoid(student_logits)
    ) + (1.0 - teacher_probability) * (
        F.logsigmoid(-teacher_logits) - F.logsigmoid(-student_logits)
    )
    channel_kl = torch.cat((analog_kl, button_kl), dim=-1).clamp_min(0.0)
    per_sample = channel_kl.sum(dim=-1)
    unavailable = (quality == int(FieldQuality.UNAVAILABLE)).to(torch.float32).mean()
    return DistillationObjectiveResult(
        loss=per_sample.mean(),
        per_action_channel_kl=channel_kl.mean(dim=0),
        per_sample_kl=per_sample,
        teacher_actor=teacher_actor,
        student_actor=student_actor,
        unavailable_fraction=unavailable,
    )


def bridge_contract() -> dict[str, Any]:
    return {
        "version": BC_BRIDGE_VERSION,
        "quality_version": BC_QUALITY_VERSION,
        "quality_contract_sha256": FIELD_QUALITY_CONTRACT_SHA256,
        "observation_version": RIVAL2_OBS_V2_120HZ_VERSION,
        "observation_contract_sha256": OBSERVATION_SCHEMA_V2_120HZ_HASH,
        "observation_shape": [OBS_DIM],
        "observation_dtype": "float32",
        "quality_shape": [OBS_DIM],
        "quality_dtype": "uint8",
        "action_version": RIVAL2_ACTION_V2_120HZ_VERSION,
        "action_contract_sha256": ACTION_CONTRACT_V2_120HZ_HASH,
        "action_shape": [len(ACTION_NAMES)],
        "action_dtype": "float32",
        "architecture_change_required": False,
        "student_model_input": "masked neutral-domain float32[182]",
        "quality_usage": "retained alongside input for auditing and future sampling/loss analysis",
        "neutral_value_is_exact": False,
    }


__all__ = [
    "BC_BRIDGE_VERSION",
    "BC_QUALITY_VERSION",
    "FIELD_QUALITY_CONTRACT_SHA256",
    "FIELD_QUALITY_SPECS",
    "GLOBAL_QUALITY_MASK",
    "BCBridgeSample",
    "BCBridgeTrajectoryAdapter",
    "DistillationObjectiveResult",
    "FieldQuality",
    "TrajectoryReconstructionState",
    "actor_distribution_distillation_objective",
    "bridge_contract",
    "bridge_human_frame",
    "degrade_simulator_observations",
    "field_quality_contract",
]
