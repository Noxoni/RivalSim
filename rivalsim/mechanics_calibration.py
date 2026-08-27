"""Read-only Rival 2.0 mechanics detectors and calibration helpers.

This module deliberately has no reward or training integration.  The detector
state is resident on the GPU and is attached to an existing world as an
additional post-physics observer.  Host copies are limited to the bounded
calibration/evaluation export paths.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import warp as wp

PHYSICS_HZ = 120
POLICY_HZ = 30
EPS_LINEAR_SPEED_UU_S = 1.0
EPS_BALL_DELTA_V_UU_S = 1.0
DASH_AIR_TICKS = 42
DASH_LANDING_TICKS = 24
ZAP_JUMP_TICKS = 12
ZAP_DODGE_TICKS = 30
DOUBLE_DASH_TICKS = 90
SURFACE_FLOOR_CEILING_NZ = 0.85
SURFACE_WALL_NZ = 0.25

FAMILY_NAMES = (
    "speedflip",
    "half_flip",
    "possession",
    "ground_carry",
    "musty",
    "breezi",
    "redirect",
    "pinch",
    "pogo",
)
FAMILY_ID = {name: index for index, name in enumerate(FAMILY_NAMES)}
FAMILY_COUNT = len(FAMILY_NAMES)

STATUS_CALIBRATED = "CALIBRATED"
STATUS_NOT_READY = "NOT_READY_FOR_REWARD"
RESET_SUPPORT_BALL = 1
RESET_SUPPORT_CAR = 2

# Stable threshold slots consumed by the GPU kernel.  The JSON evidence keeps
# the human-readable names; this mapping is the compact runtime ABI.
THRESHOLD_NAMES = (
    "speedflip_pitch_rotation_max",
    "speedflip_alignment_min",
    "speedflip_cancel_ticks_max",
    "half_flip_pitch_rotation_max",
    "half_flip_heading_dot_max",
    "half_flip_new_forward_speed_min",
    "possession_distance_max",
    "possession_relative_speed_max",
    "possession_gap_ticks_max",
    "carry_distance_max",
    "carry_relative_speed_max",
    "carry_support_ticks_min",
    "musty_rotational_normal_speed_min",
    "musty_rotational_fraction_min",
    "musty_ball_delta_v_min",
    "breezi_roll_path_min",
    "breezi_yaw_path_min",
    "breezi_setup_ticks_min",
    "redirect_incoming_speed_min",
    "redirect_outgoing_speed_min",
    "redirect_angle_min_radians",
    "pinch_overlap_ticks_max",
    "pinch_opposition_min",
    "pinch_closing_speed_min",
    "pogo_corner_region_min",
    "pogo_incoming_normal_speed_min",
    "pogo_outgoing_normal_speed_min",
    "pogo_wheel_support_max",
    "pogo_separation_ticks_max",
    "half_flip_cancel_ticks_min",
    "half_flip_cancel_ticks_max",
    "pinch_ball_delta_v_min",
    "breezi_setup_ticks_max",
    "breezi_nose_up_min",
    "breezi_inverted_depth_min",
    "breezi_nose_down_depth_min",
)
THRESHOLD_INDEX = {name: index for index, name in enumerate(THRESHOLD_NAMES)}


@dataclass(frozen=True, slots=True)
class ScalarBoundary:
    """One physically directed scalar identity boundary."""

    feature: str
    direction: str
    threshold: float
    positive_edge: float
    negative_edge: float
    margin: float

    def accepts(self, features: dict[str, float]) -> bool:
        value = float(features[self.feature])
        if self.direction == "min":
            return value >= self.threshold
        if self.direction == "max":
            return value <= self.threshold
        raise ValueError(f"unknown boundary direction: {self.direction}")


def midpoint_boundary(
    feature: str,
    direction: str,
    positives: list[dict[str, float]],
    negatives: list[dict[str, float]],
) -> ScalarBoundary | None:
    """Return the midpoint of the narrowest clean physical separation."""

    positive_values = np.asarray([row[feature] for row in positives], dtype=np.float64)
    negative_values = np.asarray([row[feature] for row in negatives], dtype=np.float64)
    if direction == "min":
        positive_edge = float(np.min(positive_values))
        negative_edge = float(np.max(negative_values))
        margin = positive_edge - negative_edge
    elif direction == "max":
        positive_edge = float(np.max(positive_values))
        negative_edge = float(np.min(negative_values))
        margin = negative_edge - positive_edge
    else:
        raise ValueError("direction must be 'min' or 'max'")
    if not margin > 0.0:
        return None
    return ScalarBoundary(
        feature=feature,
        direction=direction,
        threshold=float((positive_edge + negative_edge) * 0.5),
        positive_edge=positive_edge,
        negative_edge=negative_edge,
        margin=float(margin),
    )


def classify(features: dict[str, float], boundaries: list[ScalarBoundary]) -> bool:
    return bool(boundaries) and all(boundary.accepts(features) for boundary in boundaries)


def source_exact_reset_acquisition(
    *,
    pre_untimed_resource: bool,
    post_untimed_resource: bool,
    supporting_wheels: int,
    support_body: int,
    expected_support_body: int,
    separated: bool,
    airborne: bool,
    new_ground_jump: bool,
) -> bool:
    """Frozen ball/car reset resource transition, with no fitted threshold."""

    return bool(
        not pre_untimed_resource
        and post_untimed_resource
        and supporting_wheels >= 3
        and support_body == expected_support_body
        and separated
        and airborne
        and not new_ground_jump
    )


def source_exact_reset_rearmed(
    *, acquired_token_active: bool, resource_consumed_or_lost: bool, lockout_ended: bool
) -> bool:
    """Chain/pre-flip reset re-arm requires loss/consumption or lockout end."""

    return bool(acquired_token_active and (resource_consumed_or_lost or lockout_ended))


def canonical_family_events(events: list[tuple[str, str]]) -> list[tuple[str, str]]:
    """Deduplicate subtype labels while retaining genuine compound families."""

    result: list[tuple[str, str]] = []
    slot_by_family: dict[str, int] = {}
    subtype_specificity = {"musty": 1, "breezi": 2}
    for family, subtype in events:
        if family not in slot_by_family:
            slot_by_family[family] = len(result)
            result.append((family, subtype))
            continue
        slot = slot_by_family[family]
        retained_subtype = result[slot][1]
        if subtype_specificity.get(subtype, 0) > subtype_specificity.get(
            retained_subtype, 0
        ):
            result[slot] = (family, subtype)
    return result


def load_runtime_thresholds(path: str | Path) -> tuple[np.ndarray, np.ndarray]:
    """Load calibrated thresholds and family readiness into stable arrays."""

    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    values = np.zeros(len(THRESHOLD_NAMES), dtype=np.float32)
    ready = np.zeros(FAMILY_COUNT, dtype=np.int32)
    for family_name, record in payload["detectors"].items():
        family = FAMILY_ID[family_name]
        ready[family] = int(record["status"] == STATUS_CALIBRATED)
        for boundary in record.get("boundaries", []):
            feature = str(boundary["runtime_threshold"])
            if feature in THRESHOLD_INDEX:
                values[THRESHOLD_INDEX[feature]] = np.float32(boundary["threshold"])
    return values, ready


@wp.func
def _safe_unit(value: wp.vec3) -> wp.vec3:
    length = wp.length(value)
    if length > 1.0e-6:
        return value / length
    return wp.vec3(0.0, 0.0, 0.0)


@wp.func
def _emit_mechanic(
    car: int,
    family: int,
    subtype: int,
    tick: int,
    feature0: float,
    feature1: float,
    feature2: float,
    feature3: float,
    evidence_capacity: int,
    family_event_count: wp.array(dtype=wp.int32),
    family_lockout: wp.array(dtype=wp.int32),
    duplicate_suppression: wp.array(dtype=wp.int32),
    evidence_count: wp.array(dtype=wp.int32),
    evidence_family: wp.array(dtype=wp.int32),
    evidence_subtype: wp.array(dtype=wp.int32),
    evidence_tick: wp.array(dtype=wp.int32),
    evidence_features: wp.array(dtype=wp.float32),
):
    family_slot = car * FAMILY_COUNT + family
    if family_lockout[family_slot] != 0:
        duplicate_suppression[family_slot] = duplicate_suppression[family_slot] + 1
        return
    family_event_count[family_slot] = family_event_count[family_slot] + 1
    family_lockout[family_slot] = 1
    slot = evidence_count[car]
    evidence_count[car] = slot + 1
    if slot < evidence_capacity:
        base = car * evidence_capacity + slot
        evidence_family[base] = family
        evidence_subtype[base] = subtype
        evidence_tick[base] = tick
        value_base = base * 4
        evidence_features[value_base] = feature0
        evidence_features[value_base + 1] = feature1
        evidence_features[value_base + 2] = feature2
        evidence_features[value_base + 3] = feature3


@wp.kernel
def initialize_mechanics_shadow(
    car_vel: wp.array(dtype=wp.vec3),
    car_quat: wp.array(dtype=wp.quat),
    car_ang_vel: wp.array(dtype=wp.vec3),
    has_flipped: wp.array(dtype=wp.int32),
    wheel_contact: wp.array(dtype=wp.int32),
    chassis_contact_count: wp.array(dtype=wp.int32),
    ball_vel: wp.array(dtype=wp.vec3),
    previous_velocity: wp.array(dtype=wp.vec3),
    previous_quaternion: wp.array(dtype=wp.quat),
    previous_angular_velocity: wp.array(dtype=wp.vec3),
    previous_has_flipped: wp.array(dtype=wp.int32),
    previous_wheel_mask: wp.array(dtype=wp.int32),
    previous_chassis_contacts: wp.array(dtype=wp.int32),
    previous_ball_velocity: wp.array(dtype=wp.vec3),
):
    car = wp.tid()
    env = car // 2
    previous_velocity[car] = car_vel[car]
    previous_quaternion[car] = car_quat[car]
    previous_angular_velocity[car] = car_ang_vel[car]
    previous_has_flipped[car] = has_flipped[car]
    mask = 0
    for wheel in range(4):
        if wheel_contact[car * 4 + wheel] != 0:
            mask = mask | (1 << wheel)
    previous_wheel_mask[car] = mask
    previous_chassis_contacts[car] = chassis_contact_count[car]
    previous_ball_velocity[env] = ball_vel[env]


@wp.kernel
def collect_mechanics_shadow_tick(
    evidence_capacity: int,
    thresholds: wp.array(dtype=wp.float32),
    family_ready: wp.array(dtype=wp.int32),
    episode_tick: wp.array(dtype=wp.int32),
    done: wp.array(dtype=wp.int32),
    car_pos: wp.array(dtype=wp.vec3),
    car_vel: wp.array(dtype=wp.vec3),
    car_quat: wp.array(dtype=wp.quat),
    car_ang_vel: wp.array(dtype=wp.vec3),
    on_ground: wp.array(dtype=wp.int32),
    has_flipped: wp.array(dtype=wp.int32),
    is_flipping: wp.array(dtype=wp.int32),
    flip_rel_torque: wp.array(dtype=wp.vec3),
    ball_pos: wp.array(dtype=wp.vec3),
    ball_vel: wp.array(dtype=wp.vec3),
    control_pitch: wp.array(dtype=wp.float32),
    control_yaw: wp.array(dtype=wp.float32),
    control_roll: wp.array(dtype=wp.float32),
    car_a_hit: wp.array(dtype=wp.int32),
    car_b_hit: wp.array(dtype=wp.int32),
    car_a_pre_velocity_bt: wp.array(dtype=wp.vec3),
    car_b_pre_velocity_bt: wp.array(dtype=wp.vec3),
    car_a_pre_angular_velocity: wp.array(dtype=wp.vec3),
    car_b_pre_angular_velocity: wp.array(dtype=wp.vec3),
    car_a_contact_point_bt: wp.array(dtype=wp.vec3),
    car_b_contact_point_bt: wp.array(dtype=wp.vec3),
    car_a_contact_normal: wp.array(dtype=wp.vec3),
    car_b_contact_normal: wp.array(dtype=wp.vec3),
    car_a_ball_delta_v: wp.array(dtype=wp.vec3),
    car_b_ball_delta_v: wp.array(dtype=wp.vec3),
    ball_world_contact_count: wp.array(dtype=wp.int32),
    ball_world_normal: wp.array(dtype=wp.vec3),
    chassis_contact_count: wp.array(dtype=wp.int32),
    chassis_local_point: wp.array(dtype=wp.vec3),
    chassis_normal: wp.array(dtype=wp.vec3),
    wheel_contact: wp.array(dtype=wp.int32),
    previous_velocity: wp.array(dtype=wp.vec3),
    previous_quaternion: wp.array(dtype=wp.quat),
    previous_angular_velocity: wp.array(dtype=wp.vec3),
    previous_has_flipped: wp.array(dtype=wp.int32),
    previous_wheel_mask: wp.array(dtype=wp.int32),
    previous_chassis_contacts: wp.array(dtype=wp.int32),
    previous_ball_velocity: wp.array(dtype=wp.vec3),
    flip_kind: wp.array(dtype=wp.int32),
    flip_age: wp.array(dtype=wp.int32),
    flip_cancel_age: wp.array(dtype=wp.int32),
    flip_pitch_path: wp.array(dtype=wp.float32),
    flip_roll_path: wp.array(dtype=wp.float32),
    flip_yaw_path: wp.array(dtype=wp.float32),
    flip_initial_forward: wp.array(dtype=wp.vec3),
    flip_initial_tangent_speed: wp.array(dtype=wp.float32),
    touch_latched: wp.array(dtype=wp.int32),
    possession_owner: wp.array(dtype=wp.int32),
    possession_gap: wp.array(dtype=wp.int32),
    possession_touch_count: wp.array(dtype=wp.int32),
    carry_support_ticks: wp.array(dtype=wp.int32),
    setup_ticks: wp.array(dtype=wp.int32),
    setup_roll_path: wp.array(dtype=wp.float32),
    setup_yaw_path: wp.array(dtype=wp.float32),
    setup_orientation_stage: wp.array(dtype=wp.int32),
    pogo_pending: wp.array(dtype=wp.int32),
    pogo_age: wp.array(dtype=wp.int32),
    pogo_features: wp.array(dtype=wp.float32),
    family_event_count: wp.array(dtype=wp.int32),
    family_lockout: wp.array(dtype=wp.int32),
    family_rearm_count: wp.array(dtype=wp.int32),
    duplicate_suppression: wp.array(dtype=wp.int32),
    impossible_count: wp.array(dtype=wp.int32),
    evidence_count: wp.array(dtype=wp.int32),
    evidence_family: wp.array(dtype=wp.int32),
    evidence_subtype: wp.array(dtype=wp.int32),
    evidence_tick: wp.array(dtype=wp.int32),
    evidence_features: wp.array(dtype=wp.float32),
):
    car = wp.tid()
    env = car // 2
    local_car = car - env * 2
    tick = episode_tick[env]

    velocity = car_vel[car]
    quat = car_quat[car]
    angular = car_ang_vel[car]
    forward = wp.quat_rotate(quat, wp.vec3(1.0, 0.0, 0.0))
    right = wp.quat_rotate(quat, wp.vec3(0.0, 1.0, 0.0))
    up = wp.quat_rotate(quat, wp.vec3(0.0, 0.0, 1.0))
    wheel_mask = 0
    wheel_count = 0
    for wheel in range(4):
        if wheel_contact[car * 4 + wheel] != 0:
            wheel_mask = wheel_mask | (1 << wheel)
            wheel_count = wheel_count + 1

    if done[env] == 0:
        # Keep orientation-path history continuously.  It is consumed only by
        # terminal Breezi/Musty candidates and therefore cannot create events
        # from controls alone.
        setup_ticks[car] = setup_ticks[car] + 1
        setup_roll_path[car] = setup_roll_path[car] + wp.abs(wp.dot(angular, forward)) / 120.0
        setup_yaw_path[car] = setup_yaw_path[car] + wp.abs(wp.dot(angular, up)) / 120.0
        if setup_orientation_stage[car] == 0 and forward[2] >= thresholds[33]:
            setup_orientation_stage[car] = 1
        elif setup_orientation_stage[car] == 1 and -up[2] >= thresholds[34]:
            setup_orientation_stage[car] = 2
        elif setup_orientation_stage[car] == 2 and -forward[2] >= thresholds[35]:
            setup_orientation_stage[car] = 3

        new_flip = has_flipped[car] != 0 and previous_has_flipped[car] == 0
        if new_flip:
            torque = flip_rel_torque[car]
            flip_kind[car] = 1
            if torque[0] > 0.25:
                flip_kind[car] = 2
            flip_age[car] = 0
            flip_cancel_age[car] = -1
            flip_pitch_path[car] = 0.0
            flip_roll_path[car] = 0.0
            flip_yaw_path[car] = 0.0
            flip_initial_forward[car] = forward
            tangent = wp.vec3(velocity[0], velocity[1], 0.0)
            flip_initial_tangent_speed[car] = wp.length(tangent)

        if flip_kind[car] != 0 and flip_kind[car] != 3:
            flip_age[car] = flip_age[car] + 1
            flip_pitch_path[car] = flip_pitch_path[car] + wp.abs(wp.dot(angular, right)) / 120.0
            flip_roll_path[car] = flip_roll_path[car] + wp.abs(wp.dot(angular, forward)) / 120.0
            flip_yaw_path[car] = flip_yaw_path[car] + wp.abs(wp.dot(angular, up)) / 120.0
            if flip_cancel_age[car] < 0 and (
                (flip_kind[car] == 1 and control_pitch[car] > 0.25)
                or (flip_kind[car] == 2 and control_pitch[car] < -0.25)
            ):
                flip_cancel_age[car] = flip_age[car]

            if flip_kind[car] == 1 and flip_age[car] == 36:
                tangent = wp.vec3(velocity[0], velocity[1], 0.0)
                alignment = wp.dot(_safe_unit(forward), _safe_unit(tangent))
                accepted = (
                    family_ready[0] != 0
                    and flip_cancel_age[car] >= 0
                    and float(flip_cancel_age[car]) <= thresholds[2]
                    and flip_pitch_path[car] <= thresholds[0]
                    and alignment >= thresholds[1]
                )
                if accepted:
                    _emit_mechanic(
                        car,
                        0,
                        1,
                        tick,
                        float(flip_cancel_age[car]),
                        flip_pitch_path[car],
                        alignment,
                        wp.length(tangent) - flip_initial_tangent_speed[car],
                        evidence_capacity,
                        family_event_count,
                        family_lockout,
                        duplicate_suppression,
                        evidence_count,
                        evidence_family,
                        evidence_subtype,
                        evidence_tick,
                        evidence_features,
                    )
                flip_kind[car] = 0
            elif flip_kind[car] == 2 and flip_age[car] == 72:
                accepted = (
                    family_ready[1] != 0
                    and flip_cancel_age[car] >= 0
                    and float(flip_cancel_age[car]) >= thresholds[29]
                    and float(flip_cancel_age[car]) <= thresholds[30]
                    and flip_pitch_path[car] <= thresholds[3]
                )
                if accepted:
                    flip_kind[car] = 3
                else:
                    flip_kind[car] = 0

        if flip_kind[car] == 3 and (on_ground[car] != 0 or wheel_count > 0):
            heading_dot = wp.dot(_safe_unit(forward), _safe_unit(flip_initial_forward[car]))
            new_forward_speed = wp.dot(velocity, forward)
            if heading_dot <= thresholds[4] and new_forward_speed >= thresholds[5]:
                _emit_mechanic(
                    car,
                    1,
                    1,
                    tick,
                    float(flip_cancel_age[car]),
                    flip_pitch_path[car],
                    heading_dot,
                    new_forward_speed,
                    evidence_capacity,
                    family_event_count,
                    family_lockout,
                    duplicate_suppression,
                    evidence_count,
                    evidence_family,
                    evidence_subtype,
                    evidence_tick,
                    evidence_features,
                )
            flip_kind[car] = 0

        reports_contact = car_a_hit[env] if local_car == 0 else car_b_hit[env]
        touch_onset = reports_contact != 0 and touch_latched[car] == 0
        touch_latched[car] = int(reports_contact != 0)
        other = env * 2 + (1 - local_car)
        ball_relative = ball_pos[env] - car_pos[car]
        relative_velocity = ball_vel[env] - velocity
        distance = wp.length(ball_relative)
        relative_speed = wp.length(relative_velocity)

        if touch_onset:
            possession_owner[env] = local_car
            possession_gap[env] = 0
            possession_touch_count[car] = possession_touch_count[car] + 1
            possession_touch_count[other] = 0
            if possession_touch_count[car] >= 2:
                accepted = (
                    family_ready[2] != 0
                    and distance <= thresholds[6]
                    and relative_speed <= thresholds[7]
                )
                if accepted:
                    _emit_mechanic(
                        car,
                        2,
                        possession_touch_count[car],
                        tick,
                        distance,
                        relative_speed,
                        float(possession_gap[env]),
                        float(possession_touch_count[car]),
                        evidence_capacity,
                        family_event_count,
                        family_lockout,
                        duplicate_suppression,
                        evidence_count,
                        evidence_family,
                        evidence_subtype,
                        evidence_tick,
                        evidence_features,
                    )

            pre_velocity_bt = (
                car_a_pre_velocity_bt[env] if local_car == 0 else car_b_pre_velocity_bt[env]
            )
            pre_angular = (
                car_a_pre_angular_velocity[env]
                if local_car == 0
                else car_b_pre_angular_velocity[env]
            )
            point_bt = (
                car_a_contact_point_bt[env] if local_car == 0 else car_b_contact_point_bt[env]
            )
            contact_normal = (
                car_a_contact_normal[env] if local_car == 0 else car_b_contact_normal[env]
            )
            delta_v = car_a_ball_delta_v[env] if local_car == 0 else car_b_ball_delta_v[env]
            center_bt = car_pos[car] * 0.02
            r_bt = point_bt - center_bt
            rotational_bt = wp.cross(pre_angular, r_bt)
            rotational_normal = wp.abs(wp.dot(rotational_bt * 50.0, contact_normal))
            translation_normal = wp.abs(wp.dot(pre_velocity_bt * 50.0, contact_normal))
            fraction = rotational_normal / wp.max(rotational_normal + translation_normal, 1.0e-6)
            ball_delta = wp.length(delta_v)
            backward_dodge = is_flipping[car] != 0 and flip_rel_torque[car][0] > 0.25
            musty = (
                family_ready[4] != 0
                and backward_dodge
                and rotational_normal >= thresholds[12]
                and fraction >= thresholds[13]
                and ball_delta >= thresholds[14]
            )
            breezi = (
                family_ready[5] != 0
                and musty
                and setup_roll_path[car] >= thresholds[15]
                and setup_yaw_path[car] >= thresholds[16]
                and float(setup_ticks[car]) >= thresholds[17]
                and float(setup_ticks[car]) <= thresholds[32]
                and setup_orientation_stage[car] >= 3
            )
            if breezi:
                _emit_mechanic(
                    car,
                    5,
                    1,
                    tick,
                    setup_roll_path[car],
                    setup_yaw_path[car],
                    float(setup_ticks[car]),
                    rotational_normal,
                    evidence_capacity,
                    family_event_count,
                    family_lockout,
                    duplicate_suppression,
                    evidence_count,
                    evidence_family,
                    evidence_subtype,
                    evidence_tick,
                    evidence_features,
                )
                family_slot = car * FAMILY_COUNT + 4
                if family_lockout[family_slot] == 0:
                    family_lockout[family_slot] = 1
                    duplicate_suppression[family_slot] = duplicate_suppression[family_slot] + 1
            elif musty:
                _emit_mechanic(
                    car,
                    4,
                    1,
                    tick,
                    rotational_normal,
                    fraction,
                    ball_delta,
                    translation_normal,
                    evidence_capacity,
                    family_event_count,
                    family_lockout,
                    duplicate_suppression,
                    evidence_count,
                    evidence_family,
                    evidence_subtype,
                    evidence_tick,
                    evidence_features,
                )

            incoming = previous_ball_velocity[env]
            outgoing = ball_vel[env]
            incoming_speed = wp.length(incoming)
            outgoing_speed = wp.length(outgoing)
            cosine = wp.clamp(wp.dot(_safe_unit(incoming), _safe_unit(outgoing)), -1.0, 1.0)
            angle = wp.acos(cosine)
            if (
                family_ready[6] != 0
                and incoming_speed >= thresholds[18]
                and outgoing_speed >= thresholds[19]
                and angle >= thresholds[20]
                and ball_relative[2] > 80.0
            ):
                _emit_mechanic(
                    car,
                    6,
                    1,
                    tick,
                    incoming_speed,
                    outgoing_speed,
                    angle,
                    ball_relative[2],
                    evidence_capacity,
                    family_event_count,
                    family_lockout,
                    duplicate_suppression,
                    evidence_count,
                    evidence_family,
                    evidence_subtype,
                    evidence_tick,
                    evidence_features,
                )

            if ball_world_contact_count[env] > 0:
                world_normal = ball_world_normal[env * 16]
                car_to_ball_normal = -contact_normal
                opposition = -wp.dot(_safe_unit(car_to_ball_normal), _safe_unit(world_normal))
                closing = wp.abs(wp.dot((pre_velocity_bt * 50.0) - incoming, world_normal))
                if (
                    family_ready[7] != 0
                    and thresholds[21] >= 0.0
                    and opposition > 0.0
                    and opposition >= thresholds[22]
                    and closing >= thresholds[23]
                    and ball_delta >= thresholds[31]
                ):
                    _emit_mechanic(
                        car,
                        7,
                        1,
                        tick,
                        0.0,
                        opposition,
                        closing,
                        ball_delta,
                        evidence_capacity,
                        family_event_count,
                        family_lockout,
                        duplicate_suppression,
                        evidence_count,
                        evidence_family,
                        evidence_subtype,
                        evidence_tick,
                        evidence_features,
                    )

            setup_ticks[car] = 0
            setup_roll_path[car] = 0.0
            setup_yaw_path[car] = 0.0
            setup_orientation_stage[car] = 0
        elif possession_owner[env] == local_car:
            possession_gap[env] = possession_gap[env] + 1
            if (
                distance > thresholds[6]
                or relative_speed > thresholds[7]
                or float(possession_gap[env]) > thresholds[8]
            ):
                possession_owner[env] = -1
                possession_touch_count[car] = 0
                slot = car * FAMILY_COUNT + 2
                if family_lockout[slot] != 0:
                    family_lockout[slot] = 0
                    family_rearm_count[slot] = family_rearm_count[slot] + 1

        upper_support = reports_contact != 0 and on_ground[car] != 0 and ball_relative[2] > 80.0
        if upper_support and distance <= thresholds[9] and relative_speed <= thresholds[10]:
            carry_support_ticks[car] = carry_support_ticks[car] + 1
            if float(carry_support_ticks[car]) >= thresholds[11] and family_ready[3] != 0:
                _emit_mechanic(
                    car,
                    3,
                    1,
                    tick,
                    distance,
                    relative_speed,
                    float(carry_support_ticks[car]),
                    ball_relative[2],
                    evidence_capacity,
                    family_event_count,
                    family_lockout,
                    duplicate_suppression,
                    evidence_count,
                    evidence_family,
                    evidence_subtype,
                    evidence_tick,
                    evidence_features,
                )
        else:
            carry_support_ticks[car] = 0
            slot = car * FAMILY_COUNT + 3
            if family_lockout[slot] != 0 and reports_contact == 0:
                family_lockout[slot] = 0
                family_rearm_count[slot] = family_rearm_count[slot] + 1

        new_chassis = chassis_contact_count[car] > 0 and previous_chassis_contacts[car] == 0
        if new_chassis and previous_wheel_mask[car] == 0:
            local_point = chassis_local_point[car * 12]
            normal = chassis_normal[car * 12]
            world_r_before = wp.quat_rotate(previous_quaternion[car], local_point) * 50.0
            world_r_after = wp.quat_rotate(quat, local_point) * 50.0
            point_before = previous_velocity[car] + wp.cross(
                previous_angular_velocity[car], world_r_before
            )
            point_after = velocity + wp.cross(angular, world_r_after)
            incoming_normal = -wp.dot(point_before, normal)
            outgoing_normal = wp.dot(point_after, normal)
            corner_x = wp.abs(local_point[0]) / 2.3602
            corner_y = wp.abs(local_point[1]) / 1.6840
            corner_z = wp.abs(local_point[2]) / 0.7232
            # Second-largest normalized coordinate: an edge/corner needs two
            # axes near the Octane hitbox extents; max alone is a face test.
            corner = wp.max(
                wp.min(corner_x, corner_y),
                wp.min(wp.max(corner_x, corner_y), corner_z),
            )
            base = car * 4
            pogo_features[base] = corner
            pogo_features[base + 1] = incoming_normal
            pogo_features[base + 2] = outgoing_normal
            pogo_features[base + 3] = float(wheel_count)
            pogo_pending[car] = 1
            pogo_age[car] = 0
        elif pogo_pending[car] != 0:
            pogo_age[car] = pogo_age[car] + 1
            if wheel_count >= 3 or float(pogo_age[car]) > thresholds[28]:
                pogo_pending[car] = 0
            elif chassis_contact_count[car] == 0 and float(pogo_age[car]) <= thresholds[28]:
                base = car * 4
                if (
                    family_ready[8] != 0
                    and pogo_features[base] >= thresholds[24]
                    and pogo_features[base + 1] >= thresholds[25]
                    and pogo_features[base + 2] >= thresholds[26]
                    and pogo_features[base + 3] <= thresholds[27]
                ):
                    _emit_mechanic(
                        car,
                        8,
                        1,
                        tick,
                        pogo_features[base],
                        pogo_features[base + 1],
                        pogo_features[base + 2],
                        float(pogo_age[car]),
                        evidence_capacity,
                        family_event_count,
                        family_lockout,
                        duplicate_suppression,
                        evidence_count,
                        evidence_family,
                        evidence_subtype,
                        evidence_tick,
                        evidence_features,
                    )
                pogo_pending[car] = 0

        # Rearm bounded one-shot families only after the relevant physical
        # state has ended.  A subtype cannot stack another family event.
        for family in range(FAMILY_COUNT):
            slot = car * FAMILY_COUNT + family
            if family_lockout[slot] != 0:
                release = False
                if family == 0 or family == 1:
                    release = has_flipped[car] == 0
                elif family == 4 or family == 5 or family == 6 or family == 7:
                    release = reports_contact == 0 and possession_gap[env] > 12
                elif family == 8:
                    release = (
                        chassis_contact_count[car] == 0
                        and wheel_count == 0
                        and pogo_pending[car] == 0
                    )
                if release:
                    family_lockout[slot] = 0
                    family_rearm_count[slot] = family_rearm_count[slot] + 1

    previous_velocity[car] = velocity
    previous_quaternion[car] = quat
    previous_angular_velocity[car] = angular
    previous_has_flipped[car] = has_flipped[car]
    previous_wheel_mask[car] = wheel_mask
    previous_chassis_contacts[car] = chassis_contact_count[car]
    if local_car == 0:
        previous_ball_velocity[env] = ball_vel[env]


class MechanicsShadowObserver:
    """GPU-resident, reward-free mechanics observer for a Rival2 world."""

    def __init__(
        self,
        world: Any,
        threshold_path: str | Path,
        *,
        done: wp.array | None = None,
        evidence_capacity: int = 16,
    ):
        self.world = world
        self.num_worlds = int(world.num_envs)
        self.car_count = self.num_worlds * 2
        self.device = world.device
        self.evidence_capacity = int(evidence_capacity)
        threshold_values, family_ready = load_runtime_thresholds(threshold_path)
        self.thresholds = wp.array(threshold_values, dtype=wp.float32, device=self.device)
        self.family_ready = wp.array(family_ready, dtype=wp.int32, device=self.device)
        self.done = (
            done
            if done is not None
            else wp.zeros(self.num_worlds, dtype=wp.int32, device=self.device)
        )

        def ints(count: int) -> wp.array:
            return wp.zeros(count, dtype=wp.int32, device=self.device)

        def floats(count: int) -> wp.array:
            return wp.zeros(count, dtype=wp.float32, device=self.device)

        def vectors(count: int) -> wp.array:
            return wp.zeros(count, dtype=wp.vec3, device=self.device)

        self.previous_velocity = vectors(self.car_count)
        self.previous_quaternion = wp.zeros(self.car_count, dtype=wp.quat, device=self.device)
        self.previous_angular_velocity = vectors(self.car_count)
        self.previous_has_flipped = ints(self.car_count)
        self.previous_wheel_mask = ints(self.car_count)
        self.previous_chassis_contacts = ints(self.car_count)
        self.previous_ball_velocity = vectors(self.num_worlds)
        self.flip_kind = ints(self.car_count)
        self.flip_age = ints(self.car_count)
        self.flip_cancel_age = wp.full(self.car_count, -1, dtype=wp.int32, device=self.device)
        self.flip_pitch_path = floats(self.car_count)
        self.flip_roll_path = floats(self.car_count)
        self.flip_yaw_path = floats(self.car_count)
        self.flip_initial_forward = vectors(self.car_count)
        self.flip_initial_tangent_speed = floats(self.car_count)
        self.touch_latched = ints(self.car_count)
        self.possession_owner = wp.full(self.num_worlds, -1, dtype=wp.int32, device=self.device)
        self.possession_gap = ints(self.num_worlds)
        self.possession_touch_count = ints(self.car_count)
        self.carry_support_ticks = ints(self.car_count)
        self.setup_ticks = ints(self.car_count)
        self.setup_roll_path = floats(self.car_count)
        self.setup_yaw_path = floats(self.car_count)
        self.setup_orientation_stage = ints(self.car_count)
        self.pogo_pending = ints(self.car_count)
        self.pogo_age = ints(self.car_count)
        self.pogo_features = floats(self.car_count * 4)
        self.family_event_count = ints(self.car_count * FAMILY_COUNT)
        self.family_lockout = ints(self.car_count * FAMILY_COUNT)
        self.family_rearm_count = ints(self.car_count * FAMILY_COUNT)
        self.duplicate_suppression = ints(self.car_count * FAMILY_COUNT)
        self.impossible_count = ints(self.car_count)
        self.reward_contribution = floats(self.car_count)
        self.evidence_count = ints(self.car_count)
        evidence_count = self.car_count * self.evidence_capacity
        self.evidence_family = wp.full(evidence_count, -1, dtype=wp.int32, device=self.device)
        self.evidence_subtype = ints(evidence_count)
        self.evidence_tick = wp.full(evidence_count, -1, dtype=wp.int32, device=self.device)
        self.evidence_features = floats(evidence_count * 4)
        wp.launch(
            initialize_mechanics_shadow,
            dim=self.car_count,
            inputs=[
                world.state.car_vel,
                world.state.car_quat,
                world.state.car_ang_vel,
                world.state.has_flipped,
                world.vehicle.wheel_contact,
                world.vehicle.contact_count,
                world.state.ball_vel,
                self.previous_velocity,
                self.previous_quaternion,
                self.previous_angular_velocity,
                self.previous_has_flipped,
                self.previous_wheel_mask,
                self.previous_chassis_contacts,
                self.previous_ball_velocity,
            ],
            device=self.device,
        )
        self._original_launch: Any | None = None

    def attach(self) -> None:
        if self._original_launch is not None:
            raise RuntimeError("mechanics shadow observer is already attached")
        original_launch = self.world._launch_tick
        self._original_launch = original_launch

        def instrumented_launch() -> None:
            original_launch()
            self._launch_tick()

        self.world._launch_tick = instrumented_launch

    def _launch_tick(self) -> None:
        world = self.world
        wp.launch(
            collect_mechanics_shadow_tick,
            dim=self.car_count,
            inputs=[
                self.evidence_capacity,
                self.thresholds,
                self.family_ready,
                world.rival2.episode_ticks,
                self.done,
                world.state.car_pos,
                world.state.car_vel,
                world.state.car_quat,
                world.state.car_ang_vel,
                world.state.on_ground,
                world.state.has_flipped,
                world.state.is_flipping,
                world.state.flip_rel_torque,
                world.state.ball_pos,
                world.state.ball_vel,
                world.controls.pitch,
                world.controls.yaw,
                world.controls.roll,
                world.car_ball.hit_this_tick,
                world.car_ball_b.hit_this_tick,
                world.car_ball.pre_car_velocity_bt,
                world.car_ball_b.pre_car_velocity_bt,
                world.car_ball.pre_car_angular_velocity,
                world.car_ball_b.pre_car_angular_velocity,
                world.car_ball.contact_point_a_bt,
                world.car_ball_b.contact_point_a_bt,
                world.car_ball.contact_normal,
                world.car_ball_b.contact_normal,
                world.car_ball.extra_hit_velocity_uu,
                world.car_ball_b.extra_hit_velocity_uu,
                world.ball_world.contact_count,
                world.ball_world.contact_normal,
                world.vehicle.contact_count,
                world.vehicle.contact_local_a,
                world.vehicle.contact_normal,
                world.vehicle.wheel_contact,
                self.previous_velocity,
                self.previous_quaternion,
                self.previous_angular_velocity,
                self.previous_has_flipped,
                self.previous_wheel_mask,
                self.previous_chassis_contacts,
                self.previous_ball_velocity,
                self.flip_kind,
                self.flip_age,
                self.flip_cancel_age,
                self.flip_pitch_path,
                self.flip_roll_path,
                self.flip_yaw_path,
                self.flip_initial_forward,
                self.flip_initial_tangent_speed,
                self.touch_latched,
                self.possession_owner,
                self.possession_gap,
                self.possession_touch_count,
                self.carry_support_ticks,
                self.setup_ticks,
                self.setup_roll_path,
                self.setup_yaw_path,
                self.setup_orientation_stage,
                self.pogo_pending,
                self.pogo_age,
                self.pogo_features,
                self.family_event_count,
                self.family_lockout,
                self.family_rearm_count,
                self.duplicate_suppression,
                self.impossible_count,
                self.evidence_count,
                self.evidence_family,
                self.evidence_subtype,
                self.evidence_tick,
                self.evidence_features,
            ],
            device=self.device,
        )

    def numpy(self) -> dict[str, np.ndarray]:
        wp.synchronize_device(self.device)
        car_shape = (self.num_worlds, 2)
        family_shape = (self.num_worlds, 2, FAMILY_COUNT)
        evidence_shape = (self.num_worlds, 2, self.evidence_capacity)
        return {
            "family_event_count": np.asarray(self.family_event_count.numpy()).reshape(family_shape),
            "family_lockout": np.asarray(self.family_lockout.numpy()).reshape(family_shape),
            "family_rearm_count": np.asarray(self.family_rearm_count.numpy()).reshape(family_shape),
            "duplicate_suppression": np.asarray(self.duplicate_suppression.numpy()).reshape(
                family_shape
            ),
            "impossible_count": np.asarray(self.impossible_count.numpy()).reshape(car_shape),
            "reward_contribution": np.asarray(self.reward_contribution.numpy()).reshape(car_shape),
            "evidence_count": np.asarray(self.evidence_count.numpy()).reshape(car_shape),
            "evidence_family": np.asarray(self.evidence_family.numpy()).reshape(evidence_shape),
            "evidence_subtype": np.asarray(self.evidence_subtype.numpy()).reshape(evidence_shape),
            "evidence_tick": np.asarray(self.evidence_tick.numpy()).reshape(evidence_shape),
            "evidence_features": np.asarray(self.evidence_features.numpy()).reshape(
                *evidence_shape, 4
            ),
        }


__all__ = [
    "DASH_AIR_TICKS",
    "DASH_LANDING_TICKS",
    "DOUBLE_DASH_TICKS",
    "EPS_BALL_DELTA_V_UU_S",
    "EPS_LINEAR_SPEED_UU_S",
    "FAMILY_COUNT",
    "FAMILY_ID",
    "FAMILY_NAMES",
    "PHYSICS_HZ",
    "POLICY_HZ",
    "RESET_SUPPORT_BALL",
    "RESET_SUPPORT_CAR",
    "STATUS_CALIBRATED",
    "STATUS_NOT_READY",
    "SURFACE_FLOOR_CEILING_NZ",
    "SURFACE_WALL_NZ",
    "THRESHOLD_INDEX",
    "THRESHOLD_NAMES",
    "ZAP_DODGE_TICKS",
    "ZAP_JUMP_TICKS",
    "MechanicsShadowObserver",
    "ScalarBoundary",
    "canonical_family_events",
    "classify",
    "load_runtime_thresholds",
    "midpoint_boundary",
    "source_exact_reset_acquisition",
    "source_exact_reset_rearmed",
]
