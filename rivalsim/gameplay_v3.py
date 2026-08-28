"""Native GPU-resident Gameplay V3 detector, accounting, and calibration helpers.

The read-only mechanics observer remains calibration infrastructure.  This
module owns the production state and launches the validated continuous detector
directly at the post-physics boundary.  Per-event evidence storage is optional
and is disabled by default; it is used only by bounded validation worlds.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import warp as wp

from rivalsim.mechanics_calibration import (
    DASH_AIR_TICKS,
    DASH_LANDING_TICKS,
    DOUBLE_DASH_TICKS,
    FAMILY_COUNT,
    MUSTY_SWEEP_HISTORY,
    SURFACE_FLOOR_CEILING_NZ,
    SURFACE_WALL_NZ,
    ZAP_DODGE_TICKS,
    ZAP_JUMP_TICKS,
    collect_mechanics_shadow_tick,
    initialize_mechanics_shadow,
    load_runtime_thresholds,
)
from rivalsim.rival2_contracts import (
    GAMEPLAY_V3_MAX_PAID_MECHANICS_EVENTS,
    GAMEPLAY_V3_MECHANICS_EVENT_REWARD,
    GAMEPLAY_V3_UNNECESSARY_FLIP_PENALTY,
)

CANONICAL_MECHANIC_NAMES = (
    "speedflip",
    "half_flip",
    "musty",
    "breezi",
    "redirect",
    "pinch",
    "pogo",
    "successful_dash",
    "ball_reset",
    "car_reset",
)
CANONICAL_MECHANIC_COUNT = len(CANONICAL_MECHANIC_NAMES)

OUTCOME_NONE = 0
OUTCOME_EXEMPT_RECOGNIZED_MECHANIC = 1
OUTCOME_EXEMPT_CONTROLLED_FLICK = 2
OUTCOME_EXEMPT_CONTESTED_50 = 3
OUTCOME_EXEMPT_POWER_CONTACT = 4
OUTCOME_UNNECESSARY_FLIP_THROUGH_CONTACT = 5
OUTCOME_NAMES = (
    "NONE",
    "EXEMPT_RECOGNIZED_MECHANIC",
    "EXEMPT_CONTROLLED_FLICK",
    "EXEMPT_CONTESTED_50",
    "EXEMPT_POWER_CONTACT",
    "UNNECESSARY_FLIP_THROUGH_CONTACT",
)


def flip_contact_candidate(
    *,
    touch_onset: bool,
    is_flipping: bool,
    has_flipped: bool,
    directional_torque_norm: float,
) -> bool:
    """Pure test/calibration form of the authoritative candidate conjunction."""

    return bool(
        touch_onset
        and is_flipping
        and has_flipped
        and directional_torque_norm > 0.25
    )


def primary_flip_outcome(
    *,
    recognized_mechanic: bool,
    controlled_flick: bool,
    contested_50: bool,
    power_contact: bool,
) -> int:
    """Return the frozen deterministic primary-reason precedence."""

    if recognized_mechanic:
        return OUTCOME_EXEMPT_RECOGNIZED_MECHANIC
    if controlled_flick:
        return OUTCOME_EXEMPT_CONTROLLED_FLICK
    if contested_50:
        return OUTCOME_EXEMPT_CONTESTED_50
    if power_contact:
        return OUTCOME_EXEMPT_POWER_CONTACT
    return OUTCOME_UNNECESSARY_FLIP_THROUGH_CONTACT

# Prospectively frozen from the deterministic calibration corpus emitted by
# benchmarks/run_rival2_gameplay_v3_validation.py.  These are physical-identity
# separators, not quality scaling.
CONTEST_CONTACT_WINDOW_TICKS = 2
CONTEST_ASSOCIATION_BALL_DISPLACEMENT_MAX = 300.0
CONTEST_OPPONENT_DISTANCE_MAX = 500.0
CONTEST_SELF_CLOSING_SPEED_MIN = 150.0
CONTEST_OPPONENT_CLOSING_SPEED_MIN = 150.0
CONTEST_TIME_TO_BALL_DELTA_MAX = 0.12

POWER_TOTAL_CLOSING_SPEED_MIN = 300.0
POWER_ROTATIONAL_CLOSING_SPEED_MIN = 100.0
POWER_ROTATIONAL_SHARE_MIN = 0.18
POWER_BALL_DELTA_V_MIN = 175.0

CONTROL_DISTANCE_MAX = 220.0
CONTROL_RELATIVE_SPEED_MAX = 260.0
CONTROL_HISTORY_TICKS_MIN = 4
CONTROL_RELEASE_DISTANCE_MIN = 245.0
CONTROL_RELEASE_BALL_DELTA_V_MIN = 120.0


def contest_convergence_exempt(
    *,
    opponent_distance: float,
    self_closing_speed: float,
    opponent_closing_speed: float,
    time_to_ball_delta: float,
) -> bool:
    """Pure calibration/test form of the device convergence conjunction."""

    return bool(
        opponent_distance <= CONTEST_OPPONENT_DISTANCE_MAX
        and self_closing_speed >= CONTEST_SELF_CLOSING_SPEED_MIN
        and opponent_closing_speed >= CONTEST_OPPONENT_CLOSING_SPEED_MIN
        and time_to_ball_delta <= CONTEST_TIME_TO_BALL_DELTA_MAX
    )


def power_contact_exempt(
    *,
    total_closing_speed: float,
    rotational_closing_speed: float,
    rotational_share: float,
    ball_delta_v: float,
) -> bool:
    """Pure calibration/test form of the dodge-powered contact conjunction."""

    return bool(
        total_closing_speed >= POWER_TOTAL_CLOSING_SPEED_MIN
        and rotational_closing_speed >= POWER_ROTATIONAL_CLOSING_SPEED_MIN
        and rotational_share >= POWER_ROTATIONAL_SHARE_MIN
        and ball_delta_v >= POWER_BALL_DELTA_V_MIN
    )


def controlled_flick_exempt(
    *,
    control_ticks: int,
    control_max_distance: float,
    control_max_relative_speed: float,
    release_distance: float,
    ball_delta_v: float,
) -> bool:
    """Pure calibration/test form of the exemption-only control/release rule."""

    return bool(
        control_ticks >= CONTROL_HISTORY_TICKS_MIN
        and control_max_distance <= CONTROL_DISTANCE_MAX
        and control_max_relative_speed <= CONTROL_RELATIVE_SPEED_MAX
        and release_distance >= CONTROL_RELEASE_DISTANCE_MIN
        and ball_delta_v >= CONTROL_RELEASE_BALL_DELTA_V_MIN
    )


@wp.func
def _safe_unit(value: wp.vec3) -> wp.vec3:
    length = wp.length(value)
    if length > 1.0e-6:
        return value / length
    return wp.vec3(0.0, 0.0, 0.0)


@wp.func
def _wheel_mask(car: int, wheel_contact: wp.array(dtype=wp.int32)) -> int:
    mask = wp.int32(0)
    for wheel in range(4):
        if wheel_contact[car * 4 + wheel] != 0:
            mask = mask | wp.int32(1 << wheel)
    return mask


@wp.func
def _count_mask(mask: int) -> int:
    count = wp.int32(0)
    for bit in range(4):
        if (mask & (1 << bit)) != 0:
            count = count + 1
    return count


@wp.func
def _canonical_from_continuous_family(family: int) -> int:
    result = -1
    if family == 0:
        result = 0
    elif family == 1:
        result = 1
    elif family == 4:
        result = 2
    elif family == 5:
        result = 3
    elif family == 6:
        result = 4
    elif family == 7:
        result = 5
    elif family == 8:
        result = 6
    return result


@wp.func
def _emit_canonical(
    car: int,
    canonical: int,
    count: int,
    interval_requested: wp.array(dtype=wp.int32),
    interval_detected: wp.array(dtype=wp.int32),
    total_detected: wp.array(dtype=wp.int32),
):
    if canonical >= 0 and canonical < CANONICAL_MECHANIC_COUNT and count > 0:
        interval_requested[car] = interval_requested[car] + count
        slot = car * CANONICAL_MECHANIC_COUNT + canonical
        interval_detected[slot] = interval_detected[slot] + count
        total_detected[slot] = total_detected[slot] + count


@wp.func
def _resolve_flip_candidate(
    car: int,
    env: int,
    tick: int,
    current_distance: float,
    evidence_capacity: int,
    pending_active: wp.array(dtype=wp.int32),
    pending_recognized: wp.array(dtype=wp.int32),
    pending_controlled: wp.array(dtype=wp.int32),
    pending_contest: wp.array(dtype=wp.int32),
    pending_power: wp.array(dtype=wp.int32),
    pending_features: wp.array(dtype=wp.float32),
    interval_bad_flip: wp.array(dtype=wp.int32),
    interval_exemptions: wp.array(dtype=wp.int32),
    outcome_total: wp.array(dtype=wp.int32),
    exemption_flag_total: wp.array(dtype=wp.int32),
    outcome_evidence_count: wp.array(dtype=wp.int32),
    outcome_evidence_outcome: wp.array(dtype=wp.int32),
    outcome_evidence_tick: wp.array(dtype=wp.int32),
    outcome_evidence_features: wp.array(dtype=wp.float32),
):
    outcome = OUTCOME_UNNECESSARY_FLIP_THROUGH_CONTACT
    feature_base = car * 8
    controlled = (
        pending_controlled[car] != 0
        and pending_features[feature_base + 5] >= float(CONTROL_HISTORY_TICKS_MIN)
        and pending_features[feature_base + 6] <= CONTROL_DISTANCE_MAX
        and pending_features[feature_base + 7] <= CONTROL_RELATIVE_SPEED_MAX
        and current_distance >= CONTROL_RELEASE_DISTANCE_MIN
        and pending_features[feature_base + 4] >= CONTROL_RELEASE_BALL_DELTA_V_MIN
    )
    if pending_recognized[car] != 0:
        outcome = OUTCOME_EXEMPT_RECOGNIZED_MECHANIC
    elif controlled:
        outcome = OUTCOME_EXEMPT_CONTROLLED_FLICK
    elif pending_contest[car] != 0:
        outcome = OUTCOME_EXEMPT_CONTESTED_50
    elif pending_power[car] != 0:
        outcome = OUTCOME_EXEMPT_POWER_CONTACT

    outcome_total[car * 6 + outcome] = outcome_total[car * 6 + outcome] + 1
    if outcome == OUTCOME_UNNECESSARY_FLIP_THROUGH_CONTACT:
        interval_bad_flip[car] = interval_bad_flip[car] + 1
    if pending_recognized[car] != 0:
        interval_exemptions[car * 6 + OUTCOME_EXEMPT_RECOGNIZED_MECHANIC] += 1
        exemption_flag_total[car * 6 + OUTCOME_EXEMPT_RECOGNIZED_MECHANIC] += 1
    if controlled:
        interval_exemptions[car * 6 + OUTCOME_EXEMPT_CONTROLLED_FLICK] += 1
        exemption_flag_total[car * 6 + OUTCOME_EXEMPT_CONTROLLED_FLICK] += 1
    if pending_contest[car] != 0:
        interval_exemptions[car * 6 + OUTCOME_EXEMPT_CONTESTED_50] += 1
        exemption_flag_total[car * 6 + OUTCOME_EXEMPT_CONTESTED_50] += 1
    if pending_power[car] != 0:
        interval_exemptions[car * 6 + OUTCOME_EXEMPT_POWER_CONTACT] += 1
        exemption_flag_total[car * 6 + OUTCOME_EXEMPT_POWER_CONTACT] += 1

    if evidence_capacity > 0:
        slot = outcome_evidence_count[car]
        outcome_evidence_count[car] = slot + 1
        if slot < evidence_capacity:
            record = car * evidence_capacity + slot
            outcome_evidence_outcome[record] = outcome
            outcome_evidence_tick[record] = tick
            record_features = record * 8
            for feature in range(8):
                outcome_evidence_features[record_features + feature] = pending_features[
                    feature_base + feature
                ]

    pending_active[car] = 0
    pending_recognized[car] = 0
    pending_controlled[car] = 0
    pending_contest[car] = 0
    pending_power[car] = 0


@wp.kernel(enable_backward=False)
def gameplay_v3_begin_decision(
    interval_requested: wp.array(dtype=wp.int32),
    interval_detected: wp.array(dtype=wp.int32),
    interval_paid: wp.array(dtype=wp.int32),
    interval_budget_suppressed: wp.array(dtype=wp.int32),
    interval_bad_flip: wp.array(dtype=wp.int32),
    interval_exemptions: wp.array(dtype=wp.int32),
    mechanics_component: wp.array(dtype=wp.float32),
    bad_flip_component: wp.array(dtype=wp.float32),
    total_component: wp.array(dtype=wp.float32),
):
    car = wp.tid()
    env = car // 2
    interval_requested[car] = 0
    interval_paid[car] = 0
    interval_budget_suppressed[car] = 0
    interval_bad_flip[car] = 0
    for outcome in range(6):
        interval_exemptions[car * 6 + outcome] = 0
    for family in range(CANONICAL_MECHANIC_COUNT):
        interval_detected[car * CANONICAL_MECHANIC_COUNT + family] = 0
    if car % 2 == 0:
        mechanics_component[env] = 0.0
        bad_flip_component[env] = 0.0
        total_component[env] = 0.0


@wp.kernel(enable_backward=False)
def gameplay_v3_track_tick(
    evidence_capacity: int,
    episode_ticks: wp.array(dtype=wp.int32),
    car_pos: wp.array(dtype=wp.vec3),
    car_vel: wp.array(dtype=wp.vec3),
    on_ground: wp.array(dtype=wp.int32),
    has_jumped: wp.array(dtype=wp.int32),
    has_double_jumped: wp.array(dtype=wp.int32),
    has_flipped: wp.array(dtype=wp.int32),
    is_flipping: wp.array(dtype=wp.int32),
    air_time: wp.array(dtype=wp.float32),
    flip_rel_torque: wp.array(dtype=wp.vec3),
    ball_pos: wp.array(dtype=wp.vec3),
    ball_vel: wp.array(dtype=wp.vec3),
    wheel_contact: wp.array(dtype=wp.int32),
    wheel_hit_normal: wp.array(dtype=wp.vec3),
    wheel_hit_face: wp.array(dtype=wp.int32),
    car_a_hit: wp.array(dtype=wp.int32),
    car_b_hit: wp.array(dtype=wp.int32),
    car_a_pre_velocity_bt: wp.array(dtype=wp.vec3),
    car_b_pre_velocity_bt: wp.array(dtype=wp.vec3),
    car_a_pre_angular_velocity: wp.array(dtype=wp.vec3),
    car_b_pre_angular_velocity: wp.array(dtype=wp.vec3),
    car_a_contact_point_bt: wp.array(dtype=wp.vec3),
    car_b_contact_point_bt: wp.array(dtype=wp.vec3),
    car_a_ball_delta_v: wp.array(dtype=wp.vec3),
    car_b_ball_delta_v: wp.array(dtype=wp.vec3),
    continuous_event_count: wp.array(dtype=wp.int32),
    continuous_seen_count: wp.array(dtype=wp.int32),
    interval_requested: wp.array(dtype=wp.int32),
    interval_detected: wp.array(dtype=wp.int32),
    total_detected: wp.array(dtype=wp.int32),
    dash_previous_on_ground: wp.array(dtype=wp.int32),
    dash_previous_has_jumped: wp.array(dtype=wp.int32),
    dash_previous_has_flipped: wp.array(dtype=wp.int32),
    dash_previous_air_time: wp.array(dtype=wp.float32),
    dash_previous_wheel_mask: wp.array(dtype=wp.int32),
    dash_previous_velocity: wp.array(dtype=wp.vec3),
    dash_pending_flip_tick: wp.array(dtype=wp.int32),
    dash_pending_pre_velocity: wp.array(dtype=wp.vec3),
    dash_last_success_flip_tick: wp.array(dtype=wp.int32),
    dash_last_success_landing_tick: wp.array(dtype=wp.int32),
    dash_last_pre_tangent_speed: wp.array(dtype=wp.float32),
    dash_success_total: wp.array(dtype=wp.int32),
    dash_surface_total: wp.array(dtype=wp.int32),
    zap_stage: wp.array(dtype=wp.int32),
    zap_stage_tick: wp.array(dtype=wp.int32),
    zap_total: wp.array(dtype=wp.int32),
    double_dash_total: wp.array(dtype=wp.int32),
    reset_previous_untimed: wp.array(dtype=wp.int32),
    reset_previous_has_flipped: wp.array(dtype=wp.int32),
    reset_previous_has_double_jumped: wp.array(dtype=wp.int32),
    reset_pending_body: wp.array(dtype=wp.int32),
    reset_pending_preflip: wp.array(dtype=wp.int32),
    reset_armed: wp.array(dtype=wp.int32),
    reset_completion_total: wp.array(dtype=wp.int32),
    chain_reset_total: wp.array(dtype=wp.int32),
    preflip_reset_total: wp.array(dtype=wp.int32),
    v3_touch_latched: wp.array(dtype=wp.int32),
    legitimate_touch_total: wp.array(dtype=wp.int32),
    flip_touch_total: wp.array(dtype=wp.int32),
    control_ticks: wp.array(dtype=wp.int32),
    control_max_distance: wp.array(dtype=wp.float32),
    control_max_relative_speed: wp.array(dtype=wp.float32),
    pending_active: wp.array(dtype=wp.int32),
    pending_age: wp.array(dtype=wp.int32),
    pending_recognized: wp.array(dtype=wp.int32),
    pending_controlled: wp.array(dtype=wp.int32),
    pending_contest: wp.array(dtype=wp.int32),
    pending_power: wp.array(dtype=wp.int32),
    pending_ball_position: wp.array(dtype=wp.vec3),
    pending_features: wp.array(dtype=wp.float32),
    interval_bad_flip: wp.array(dtype=wp.int32),
    interval_exemptions: wp.array(dtype=wp.int32),
    outcome_total: wp.array(dtype=wp.int32),
    exemption_flag_total: wp.array(dtype=wp.int32),
    impossible_total: wp.array(dtype=wp.int32),
    outcome_evidence_count: wp.array(dtype=wp.int32),
    outcome_evidence_outcome: wp.array(dtype=wp.int32),
    outcome_evidence_tick: wp.array(dtype=wp.int32),
    outcome_evidence_features: wp.array(dtype=wp.float32),
):
    car = wp.tid()
    env = car // 2
    local = car - env * 2
    other = env * 2 + (1 - local)
    tick = episode_ticks[env] + 1

    recognized_this_contact = False
    for family in range(FAMILY_COUNT):
        slot = car * FAMILY_COUNT + family
        current = continuous_event_count[slot]
        delta = current - continuous_seen_count[slot]
        if delta < 0:
            impossible_total[car] = impossible_total[car] + 1
            delta = 0
        if delta > 0:
            canonical = _canonical_from_continuous_family(family)
            _emit_canonical(
                car,
                canonical,
                delta,
                interval_requested,
                interval_detected,
                total_detected,
            )
            if family == 4 or family == 5:
                recognized_this_contact = True
        continuous_seen_count[slot] = current

    current_mask = _wheel_mask(car, wheel_contact)
    wheel_count = _count_mask(current_mask)
    prior_mask = dash_previous_wheel_mask[car]
    new_jump = dash_previous_has_jumped[car] == 0 and has_jumped[car] != 0
    new_flip = dash_previous_has_flipped[car] == 0 and has_flipped[car] != 0

    # Zap sequence: front wheels first, then a non-flat three-wheel jump, then
    # a directional landing dodge whose terminal dash must independently pass.
    if prior_mask == 0 and current_mask != 0:
        front = current_mask & 0b0011
        rear = current_mask & 0b1100
        if front != 0 and rear == 0:
            zap_stage[car] = 1
            zap_stage_tick[car] = tick
    if zap_stage[car] == 1:
        if tick - zap_stage_tick[car] > ZAP_JUMP_TICKS:
            zap_stage[car] = 0
        elif new_jump and wheel_count == 3 and current_mask != 0b1111:
            zap_stage[car] = 2
            zap_stage_tick[car] = tick
    if zap_stage[car] == 2:
        if tick - zap_stage_tick[car] > ZAP_DODGE_TICKS:
            zap_stage[car] = 0
        elif new_flip and wp.length(flip_rel_torque[car]) > 0.25:
            zap_stage[car] = 3
            zap_stage_tick[car] = tick

    pending_tick = dash_pending_flip_tick[car]
    if pending_tick >= 0 and tick - pending_tick > DASH_LANDING_TICKS:
        pending_tick = -1
    if new_flip:
        air_ticks = wp.int32(wp.floor(dash_previous_air_time[car] * 120.0 + 0.5))
        if (
            prior_mask == 0
            and dash_previous_on_ground[car] == 0
            and air_ticks >= 0
            and air_ticks <= DASH_AIR_TICKS
        ):
            pending_tick = tick
            dash_pending_pre_velocity[car] = dash_previous_velocity[car]
        else:
            pending_tick = -1

    landing = prior_mask == 0 and current_mask != 0
    if landing and pending_tick >= 0 and tick - pending_tick <= DASH_LANDING_TICKS:
        normal_sum = wp.vec3(0.0, 0.0, 0.0)
        for wheel in range(4):
            wheel_slot = car * 4 + wheel
            if wheel_contact[wheel_slot] != 0:
                normal_sum = normal_sum + wheel_hit_normal[wheel_slot]
        normal = _safe_unit(normal_sum)
        pre_velocity = dash_pending_pre_velocity[car]
        pre_tangent = pre_velocity - normal * wp.dot(pre_velocity, normal)
        post_tangent = car_vel[car] - normal * wp.dot(car_vel[car], normal)
        pre_speed = wp.length(pre_tangent)
        post_speed = wp.length(post_tangent)
        if post_speed - pre_speed > 1.0:
            _emit_canonical(
                car,
                7,
                1,
                interval_requested,
                interval_detected,
                total_detected,
            )
            dash_success_total[car] = dash_success_total[car] + 1
            abs_nz = wp.abs(normal[2])
            surface = 2
            if abs_nz >= SURFACE_FLOOR_CEILING_NZ:
                surface = 0
                if normal[2] < 0.0:
                    surface = 3
            elif abs_nz <= SURFACE_WALL_NZ:
                surface = 1
            dash_surface_total[car * 4 + surface] = (
                dash_surface_total[car * 4 + surface] + 1
            )
            if zap_stage[car] == 3:
                zap_total[car] = zap_total[car] + 1
                zap_stage[car] = 0
            prior_success = dash_last_success_flip_tick[car]
            if (
                prior_success >= 0
                and pending_tick - prior_success > 0
                and pending_tick - prior_success <= DOUBLE_DASH_TICKS
                and dash_last_success_landing_tick[car] <= pending_tick
                and post_speed - dash_last_pre_tangent_speed[car] > 1.0
            ):
                double_dash_total[car] = double_dash_total[car] + 1
            dash_last_success_flip_tick[car] = pending_tick
            dash_last_success_landing_tick[car] = tick
            dash_last_pre_tangent_speed[car] = pre_speed
        pending_tick = -1
    dash_pending_flip_tick[car] = pending_tick

    # Source-exact dynamic support identity uses the pinned wheel-ray face IDs:
    # -6 ball, -7 the other Octane.  World faces are non-negative.
    ball_support = 0
    car_support = 0
    world_support = 0
    for wheel in range(4):
        wheel_slot = car * 4 + wheel
        if wheel_contact[wheel_slot] != 0:
            face = wheel_hit_face[wheel_slot]
            if face == -6:
                ball_support = ball_support + 1
            elif face == -7:
                car_support = car_support + 1
            elif face >= 0:
                world_support = world_support + 1
    untimed = wp.int32(
        on_ground[car] == 0
        and has_jumped[car] == 0
        and has_flipped[car] == 0
        and has_double_jumped[car] == 0
    )
    support_body = 0
    if ball_support >= 3:
        support_body = 1
    elif car_support >= 3:
        support_body = 2
    if (
        reset_armed[car] != 0
        and reset_pending_body[car] == 0
        and reset_previous_untimed[car] == 0
        and support_body != 0
        and has_jumped[car] == 0
        and has_flipped[car] == 0
        and has_double_jumped[car] == 0
    ):
        reset_pending_body[car] = support_body
        reset_pending_preflip[car] = wp.int32(
            reset_previous_has_flipped[car] != 0
            or reset_previous_has_double_jumped[car] != 0
        )
    reset_completed_preflip = False
    pending_body = reset_pending_body[car]
    pending_still_supported = (
        (pending_body == 1 and ball_support >= 3)
        or (pending_body == 2 and car_support >= 3)
    )
    if pending_body != 0 and not pending_still_supported:
        if untimed != 0 and world_support == 0 and new_jump == 0:
            canonical = 8
            if pending_body == 2:
                canonical = 9
            _emit_canonical(
                car,
                canonical,
                1,
                interval_requested,
                interval_detected,
                total_detected,
            )
            if reset_completion_total[car] > 0:
                chain_reset_total[car] = chain_reset_total[car] + 1
            reset_completion_total[car] = reset_completion_total[car] + 1
            if reset_pending_preflip[car] != 0:
                preflip_reset_total[car] = preflip_reset_total[car] + 1
                reset_completed_preflip = True
            reset_armed[car] = 0
        reset_pending_body[car] = 0
        reset_pending_preflip[car] = 0
    if reset_armed[car] == 0 and untimed == 0 and reset_pending_body[car] == 0:
        reset_armed[car] = 1
    reset_previous_untimed[car] = untimed
    reset_previous_has_flipped[car] = has_flipped[car]
    reset_previous_has_double_jumped[car] = has_double_jumped[car]

    reports = car_a_hit[env] if local == 0 else car_b_hit[env]
    other_reports = car_b_hit[env] if local == 0 else car_a_hit[env]
    touch_onset = reports != 0 and v3_touch_latched[car] == 0
    v3_touch_latched[car] = wp.int32(reports != 0)
    distance = wp.length(ball_pos[env] - car_pos[car])
    relative_speed = wp.length(ball_vel[env] - car_vel[car])

    # Resolve an older pending candidate before a genuine new re-contact is
    # considered.  This preserves exactly-once outcomes while permitting a
    # separated re-contact to create a new candidate.
    if pending_active[car] != 0:
        if (
            other_reports != 0
            and wp.length(ball_pos[env] - pending_ball_position[car])
            <= CONTEST_ASSOCIATION_BALL_DISPLACEMENT_MAX
        ):
            pending_contest[car] = 1
        pending_age[car] = pending_age[car] + 1
        if pending_age[car] >= CONTEST_CONTACT_WINDOW_TICKS or touch_onset:
            _resolve_flip_candidate(
                car,
                env,
                tick,
                distance,
                evidence_capacity,
                pending_active,
                pending_recognized,
                pending_controlled,
                pending_contest,
                pending_power,
                pending_features,
                interval_bad_flip,
                interval_exemptions,
                outcome_total,
                exemption_flag_total,
                outcome_evidence_count,
                outcome_evidence_outcome,
                outcome_evidence_tick,
                outcome_evidence_features,
            )

    if distance <= CONTROL_DISTANCE_MAX and relative_speed <= CONTROL_RELATIVE_SPEED_MAX:
        if control_ticks[car] == 0:
            control_max_distance[car] = distance
            control_max_relative_speed[car] = relative_speed
        control_ticks[car] = control_ticks[car] + 1
        control_max_distance[car] = wp.max(control_max_distance[car], distance)
        control_max_relative_speed[car] = wp.max(
            control_max_relative_speed[car], relative_speed
        )
    elif reports == 0:
        control_ticks[car] = 0
        control_max_distance[car] = 0.0
        control_max_relative_speed[car] = 0.0

    if touch_onset:
        legitimate_touch_total[car] = legitimate_touch_total[car] + 1
        torque = flip_rel_torque[car]
        active_directional_dodge = (
            is_flipping[car] != 0
            and has_flipped[car] != 0
            and wp.length(torque) > 0.25
        )
        if active_directional_dodge:
            flip_touch_total[car] = flip_touch_total[car] + 1
            pre_velocity_bt = (
                car_a_pre_velocity_bt[env] if local == 0 else car_b_pre_velocity_bt[env]
            )
            pre_angular = (
                car_a_pre_angular_velocity[env]
                if local == 0
                else car_b_pre_angular_velocity[env]
            )
            point_bt = (
                car_a_contact_point_bt[env] if local == 0 else car_b_contact_point_bt[env]
            )
            ball_delta_vector = (
                car_a_ball_delta_v[env] if local == 0 else car_b_ball_delta_v[env]
            )
            pre_velocity = pre_velocity_bt * 50.0
            contact_world = point_bt * 50.0
            offset = contact_world - car_pos[car]
            direction = _safe_unit(ball_pos[env] - car_pos[car])
            rotational = wp.max(wp.dot(wp.cross(pre_angular, offset), direction), 0.0)
            translational = wp.max(wp.dot(pre_velocity - ball_vel[env], direction), 0.0)
            total_closing = rotational + translational
            rotational_share = rotational / wp.max(total_closing, 1.0e-6)
            ball_delta = wp.length(ball_delta_vector)

            self_direction = _safe_unit(ball_pos[env] - car_pos[car])
            opponent_direction = _safe_unit(ball_pos[env] - car_pos[other])
            self_closing = wp.dot(car_vel[car] - ball_vel[env], self_direction)
            opponent_closing = wp.dot(car_vel[other] - ball_vel[env], opponent_direction)
            opponent_distance = wp.length(ball_pos[env] - car_pos[other])
            self_time = distance / wp.max(self_closing, 1.0e-6)
            opponent_time = opponent_distance / wp.max(opponent_closing, 1.0e-6)
            time_delta = wp.abs(self_time - opponent_time)
            convergence = (
                opponent_distance <= CONTEST_OPPONENT_DISTANCE_MAX
                and self_closing >= CONTEST_SELF_CLOSING_SPEED_MIN
                and opponent_closing >= CONTEST_OPPONENT_CLOSING_SPEED_MIN
                and time_delta <= CONTEST_TIME_TO_BALL_DELTA_MAX
            )
            power = (
                total_closing >= POWER_TOTAL_CLOSING_SPEED_MIN
                and rotational >= POWER_ROTATIONAL_CLOSING_SPEED_MIN
                and rotational_share >= POWER_ROTATIONAL_SHARE_MIN
                and ball_delta >= POWER_BALL_DELTA_V_MIN
            )
            controlled = (
                control_ticks[car] >= CONTROL_HISTORY_TICKS_MIN
                and control_max_distance[car] <= CONTROL_DISTANCE_MAX
                and control_max_relative_speed[car] <= CONTROL_RELATIVE_SPEED_MAX
            )
            pending_active[car] = 1
            pending_age[car] = 0
            pending_recognized[car] = wp.int32(
                recognized_this_contact or reset_completed_preflip
            )
            pending_controlled[car] = wp.int32(controlled)
            pending_contest[car] = wp.int32(other_reports != 0 or convergence)
            pending_power[car] = wp.int32(power)
            pending_ball_position[car] = ball_pos[env]
            feature_base = car * 8
            pending_features[feature_base] = opponent_distance
            pending_features[feature_base + 1] = self_closing
            pending_features[feature_base + 2] = opponent_closing
            pending_features[feature_base + 3] = time_delta
            pending_features[feature_base + 4] = ball_delta
            pending_features[feature_base + 5] = float(control_ticks[car])
            pending_features[feature_base + 6] = control_max_distance[car]
            pending_features[feature_base + 7] = control_max_relative_speed[car]
        control_ticks[car] = 0
        control_max_distance[car] = 0.0
        control_max_relative_speed[car] = 0.0

    dash_previous_on_ground[car] = on_ground[car]
    dash_previous_has_jumped[car] = has_jumped[car]
    dash_previous_has_flipped[car] = has_flipped[car]
    dash_previous_air_time[car] = air_time[car]
    dash_previous_wheel_mask[car] = current_mask
    dash_previous_velocity[car] = car_vel[car]


@wp.kernel(enable_backward=False)
def gameplay_v3_compose_reward(
    interval_tick: wp.array(dtype=wp.int32),
    reward: wp.array(dtype=wp.float32),
    interval_requested: wp.array(dtype=wp.int32),
    interval_detected: wp.array(dtype=wp.int32),
    interval_paid: wp.array(dtype=wp.int32),
    interval_budget_suppressed: wp.array(dtype=wp.int32),
    interval_bad_flip: wp.array(dtype=wp.int32),
    mechanics_paid_episode: wp.array(dtype=wp.int32),
    budget_exhausted_latched: wp.array(dtype=wp.int32),
    budget_exhausted_total: wp.array(dtype=wp.int32),
    total_paid: wp.array(dtype=wp.int32),
    total_budget_suppressed: wp.array(dtype=wp.int32),
    mechanics_component: wp.array(dtype=wp.float32),
    bad_flip_component: wp.array(dtype=wp.float32),
    total_component: wp.array(dtype=wp.float32),
):
    env = wp.tid()
    if interval_tick[env] != 4:
        return
    car_base = env * 2
    paid_blue = wp.int32(0)
    paid_orange = wp.int32(0)
    for local in range(2):
        car = car_base + local
        remaining = wp.max(
            0, GAMEPLAY_V3_MAX_PAID_MECHANICS_EVENTS - mechanics_paid_episode[car]
        )
        paid = wp.min(interval_requested[car], remaining)
        interval_paid[car] = paid
        suppressed = interval_requested[car] - paid
        interval_budget_suppressed[car] = suppressed
        mechanics_paid_episode[car] = mechanics_paid_episode[car] + paid
        total_budget_suppressed[car] = total_budget_suppressed[car] + suppressed
        if (
            mechanics_paid_episode[car] >= GAMEPLAY_V3_MAX_PAID_MECHANICS_EVENTS
            and budget_exhausted_latched[car] == 0
        ):
            budget_exhausted_total[car] = budget_exhausted_total[car] + 1
            budget_exhausted_latched[car] = 1
        remaining_paid = paid
        for family in range(CANONICAL_MECHANIC_COUNT):
            slot = car * CANONICAL_MECHANIC_COUNT + family
            family_paid = wp.min(interval_detected[slot], remaining_paid)
            total_paid[slot] = total_paid[slot] + family_paid
            remaining_paid = remaining_paid - family_paid
        if local == 0:
            paid_blue = paid
        else:
            paid_orange = paid

    mechanics = GAMEPLAY_V3_MECHANICS_EVENT_REWARD * float(
        paid_blue - paid_orange
    )
    bad_flip = GAMEPLAY_V3_UNNECESSARY_FLIP_PENALTY * float(
        interval_bad_flip[car_base] - interval_bad_flip[car_base + 1]
    )
    blue = reward[car_base] + mechanics + bad_flip
    reward[car_base] = blue
    reward[car_base + 1] = -blue
    mechanics_component[env] = mechanics
    bad_flip_component[env] = bad_flip
    total_component[env] = blue


@wp.kernel(enable_backward=False)
def gameplay_v3_reset_continuous(
    reset_mask: wp.array(dtype=wp.int32),
    car_pos: wp.array(dtype=wp.vec3),
    car_vel: wp.array(dtype=wp.vec3),
    car_quat: wp.array(dtype=wp.quat),
    car_ang_vel: wp.array(dtype=wp.vec3),
    has_flipped: wp.array(dtype=wp.int32),
    wheel_contact: wp.array(dtype=wp.int32),
    chassis_contact_count: wp.array(dtype=wp.int32),
    ball_pos: wp.array(dtype=wp.vec3),
    ball_vel: wp.array(dtype=wp.vec3),
    previous_position: wp.array(dtype=wp.vec3),
    previous_velocity: wp.array(dtype=wp.vec3),
    previous_quaternion: wp.array(dtype=wp.quat),
    previous_angular_velocity: wp.array(dtype=wp.vec3),
    previous_has_flipped: wp.array(dtype=wp.int32),
    previous_wheel_mask: wp.array(dtype=wp.int32),
    previous_chassis_contacts: wp.array(dtype=wp.int32),
    previous_ball_velocity: wp.array(dtype=wp.vec3),
    previous_ball_position: wp.array(dtype=wp.vec3),
    previous_episode_tick: wp.array(dtype=wp.int32),
    history_car_position: wp.array(dtype=wp.vec3),
    history_car_quaternion: wp.array(dtype=wp.quat),
    history_ball_position: wp.array(dtype=wp.vec3),
    history_cursor: wp.array(dtype=wp.int32),
    history_count: wp.array(dtype=wp.int32),
    flip_kind: wp.array(dtype=wp.int32),
    flip_age: wp.array(dtype=wp.int32),
    flip_cancel_age: wp.array(dtype=wp.int32),
    flip_pitch_path: wp.array(dtype=wp.float32),
    flip_roll_path: wp.array(dtype=wp.float32),
    flip_yaw_path: wp.array(dtype=wp.float32),
    touch_latched: wp.array(dtype=wp.int32),
    possession_owner: wp.array(dtype=wp.int32),
    possession_gap: wp.array(dtype=wp.int32),
    possession_touch_count: wp.array(dtype=wp.int32),
    carry_support_ticks: wp.array(dtype=wp.int32),
    setup_ticks: wp.array(dtype=wp.int32),
    setup_roll_path: wp.array(dtype=wp.float32),
    setup_yaw_path: wp.array(dtype=wp.float32),
    setup_orientation_stage: wp.array(dtype=wp.int32),
    setup_combined_ticks: wp.array(dtype=wp.int32),
    setup_control_active: wp.array(dtype=wp.int32),
    setup_terminal_pending: wp.array(dtype=wp.int32),
    setup_control_max_distance: wp.array(dtype=wp.float32),
    setup_control_max_relative_speed: wp.array(dtype=wp.float32),
    pogo_pending: wp.array(dtype=wp.int32),
    pogo_age: wp.array(dtype=wp.int32),
    family_event_count: wp.array(dtype=wp.int32),
    family_lockout: wp.array(dtype=wp.int32),
    continuous_seen_count: wp.array(dtype=wp.int32),
):
    car = wp.tid()
    env = car // 2
    if reset_mask[env] == 0:
        return
    previous_position[car] = car_pos[car]
    previous_velocity[car] = car_vel[car]
    previous_quaternion[car] = car_quat[car]
    previous_angular_velocity[car] = car_ang_vel[car]
    previous_has_flipped[car] = has_flipped[car]
    previous_wheel_mask[car] = _wheel_mask(car, wheel_contact)
    previous_chassis_contacts[car] = chassis_contact_count[car]
    previous_episode_tick[car] = 0
    if car % 2 == 0:
        previous_ball_velocity[env] = ball_vel[env]
        previous_ball_position[env] = ball_pos[env]
        possession_owner[env] = -1
        possession_gap[env] = 0
    history_cursor[car] = 0
    history_count[car] = 1
    for sample in range(MUSTY_SWEEP_HISTORY):
        history_slot = car * MUSTY_SWEEP_HISTORY + sample
        history_car_position[history_slot] = car_pos[car]
        history_car_quaternion[history_slot] = car_quat[car]
        history_ball_position[history_slot] = ball_pos[env]
    flip_kind[car] = 0
    flip_age[car] = 0
    flip_cancel_age[car] = -1
    flip_pitch_path[car] = 0.0
    flip_roll_path[car] = 0.0
    flip_yaw_path[car] = 0.0
    touch_latched[car] = 0
    possession_touch_count[car] = 0
    carry_support_ticks[car] = 0
    setup_ticks[car] = 0
    setup_roll_path[car] = 0.0
    setup_yaw_path[car] = 0.0
    setup_orientation_stage[car] = 0
    setup_combined_ticks[car] = 0
    setup_control_active[car] = 0
    setup_terminal_pending[car] = 0
    setup_control_max_distance[car] = 0.0
    setup_control_max_relative_speed[car] = 0.0
    pogo_pending[car] = 0
    pogo_age[car] = 0
    for family in range(FAMILY_COUNT):
        slot = car * FAMILY_COUNT + family
        family_lockout[slot] = 0
        continuous_seen_count[slot] = family_event_count[slot]


@wp.kernel(enable_backward=False)
def gameplay_v3_reset_source_state(
    reset_mask: wp.array(dtype=wp.int32),
    on_ground: wp.array(dtype=wp.int32),
    has_jumped: wp.array(dtype=wp.int32),
    has_double_jumped: wp.array(dtype=wp.int32),
    has_flipped: wp.array(dtype=wp.int32),
    air_time: wp.array(dtype=wp.float32),
    car_vel: wp.array(dtype=wp.vec3),
    wheel_contact: wp.array(dtype=wp.int32),
    dash_previous_on_ground: wp.array(dtype=wp.int32),
    dash_previous_has_jumped: wp.array(dtype=wp.int32),
    dash_previous_has_flipped: wp.array(dtype=wp.int32),
    dash_previous_air_time: wp.array(dtype=wp.float32),
    dash_previous_wheel_mask: wp.array(dtype=wp.int32),
    dash_previous_velocity: wp.array(dtype=wp.vec3),
    dash_pending_flip_tick: wp.array(dtype=wp.int32),
    dash_last_success_flip_tick: wp.array(dtype=wp.int32),
    dash_last_success_landing_tick: wp.array(dtype=wp.int32),
    zap_stage: wp.array(dtype=wp.int32),
    zap_stage_tick: wp.array(dtype=wp.int32),
    reset_previous_untimed: wp.array(dtype=wp.int32),
    reset_previous_has_flipped: wp.array(dtype=wp.int32),
    reset_previous_has_double_jumped: wp.array(dtype=wp.int32),
    reset_pending_body: wp.array(dtype=wp.int32),
    reset_pending_preflip: wp.array(dtype=wp.int32),
    reset_armed: wp.array(dtype=wp.int32),
    v3_touch_latched: wp.array(dtype=wp.int32),
    control_ticks: wp.array(dtype=wp.int32),
    control_max_distance: wp.array(dtype=wp.float32),
    control_max_relative_speed: wp.array(dtype=wp.float32),
    pending_active: wp.array(dtype=wp.int32),
    pending_age: wp.array(dtype=wp.int32),
    pending_recognized: wp.array(dtype=wp.int32),
    pending_controlled: wp.array(dtype=wp.int32),
    pending_contest: wp.array(dtype=wp.int32),
    pending_power: wp.array(dtype=wp.int32),
    mechanics_paid_episode: wp.array(dtype=wp.int32),
    budget_exhausted_latched: wp.array(dtype=wp.int32),
):
    car = wp.tid()
    env = car // 2
    if reset_mask[env] == 0:
        return
    dash_previous_on_ground[car] = on_ground[car]
    dash_previous_has_jumped[car] = has_jumped[car]
    dash_previous_has_flipped[car] = has_flipped[car]
    dash_previous_air_time[car] = air_time[car]
    dash_previous_wheel_mask[car] = _wheel_mask(car, wheel_contact)
    dash_previous_velocity[car] = car_vel[car]
    dash_pending_flip_tick[car] = -1
    dash_last_success_flip_tick[car] = -1
    dash_last_success_landing_tick[car] = -1
    zap_stage[car] = 0
    zap_stage_tick[car] = -1
    untimed = wp.int32(
        on_ground[car] == 0
        and has_jumped[car] == 0
        and has_flipped[car] == 0
        and has_double_jumped[car] == 0
    )
    reset_previous_untimed[car] = untimed
    reset_previous_has_flipped[car] = has_flipped[car]
    reset_previous_has_double_jumped[car] = has_double_jumped[car]
    reset_pending_body[car] = 0
    reset_pending_preflip[car] = 0
    reset_armed[car] = 1
    v3_touch_latched[car] = 0
    control_ticks[car] = 0
    control_max_distance[car] = 0.0
    control_max_relative_speed[car] = 0.0
    pending_active[car] = 0
    pending_age[car] = 0
    pending_recognized[car] = 0
    pending_controlled[car] = 0
    pending_contest[car] = 0
    pending_power[car] = 0
    mechanics_paid_episode[car] = 0
    budget_exhausted_latched[car] = 0


class Rival2GameplayV3State:
    """Optional native state allocated only by Gameplay V3 worlds."""

    def __init__(
        self,
        world: Any,
        threshold_path: str | Path,
        *,
        evidence_capacity: int = 0,
    ) -> None:
        self.world = world
        self.num_worlds = int(world.num_envs)
        self.car_count = self.num_worlds * 2
        self.device = world.device
        self.evidence_capacity = int(evidence_capacity)
        if self.evidence_capacity < 0:
            raise ValueError("V3 evidence capacity must be non-negative")
        self._inventory: dict[str, dict[str, Any]] = {}

        def register(name: str, array: wp.array, count: int, item_bytes: int) -> wp.array:
            setattr(self, name, array)
            self._inventory[name] = {
                "shape": list(array.shape),
                "dtype": str(array.dtype),
                "logical_bytes": int(count * item_bytes),
                "evaluation_only": name.startswith("evidence_")
                or name.startswith("outcome_evidence_"),
            }
            return array

        def ints(name: str, count: int, *, fill: int = 0) -> wp.array:
            array = (
                wp.zeros(count, dtype=wp.int32, device=self.device)
                if fill == 0
                else wp.full(count, fill, dtype=wp.int32, device=self.device)
            )
            return register(name, array, count, 4)

        def floats(name: str, count: int) -> wp.array:
            return register(
                name,
                wp.zeros(count, dtype=wp.float32, device=self.device),
                count,
                4,
            )

        def vectors(name: str, count: int) -> wp.array:
            return register(
                name,
                wp.zeros(count, dtype=wp.vec3, device=self.device),
                count,
                12,
            )

        def quaternions(name: str, count: int) -> wp.array:
            return register(
                name,
                wp.zeros(count, dtype=wp.quat, device=self.device),
                count,
                16,
            )

        threshold_values, family_ready = load_runtime_thresholds(threshold_path)
        register(
            "thresholds",
            wp.array(threshold_values, dtype=wp.float32, device=self.device),
            len(threshold_values),
            4,
        )
        register(
            "family_ready",
            wp.array(family_ready, dtype=wp.int32, device=self.device),
            len(family_ready),
            4,
        )
        ints("done", self.num_worlds)

        vectors("previous_position", self.car_count)
        vectors("previous_velocity", self.car_count)
        quaternions("previous_quaternion", self.car_count)
        vectors("previous_angular_velocity", self.car_count)
        ints("previous_has_flipped", self.car_count)
        ints("previous_wheel_mask", self.car_count)
        ints("previous_chassis_contacts", self.car_count)
        vectors("previous_ball_velocity", self.num_worlds)
        vectors("previous_ball_position", self.num_worlds)
        ints("previous_episode_tick", self.car_count)
        history = self.car_count * MUSTY_SWEEP_HISTORY
        vectors("history_car_position", history)
        quaternions("history_car_quaternion", history)
        vectors("history_ball_position", history)
        ints("history_cursor", self.car_count)
        ints("history_count", self.car_count)
        ints("flip_kind", self.car_count)
        ints("flip_age", self.car_count)
        ints("flip_cancel_age", self.car_count, fill=-1)
        floats("flip_pitch_path", self.car_count)
        floats("flip_roll_path", self.car_count)
        floats("flip_yaw_path", self.car_count)
        vectors("flip_initial_forward", self.car_count)
        floats("flip_initial_tangent_speed", self.car_count)
        ints("touch_latched", self.car_count)
        ints("possession_owner", self.num_worlds, fill=-1)
        ints("possession_gap", self.num_worlds)
        ints("possession_touch_count", self.car_count)
        ints("carry_support_ticks", self.car_count)
        ints("setup_ticks", self.car_count)
        floats("setup_roll_path", self.car_count)
        floats("setup_yaw_path", self.car_count)
        ints("setup_orientation_stage", self.car_count)
        ints("setup_combined_ticks", self.car_count)
        ints("setup_control_active", self.car_count)
        ints("setup_terminal_pending", self.car_count)
        floats("setup_control_max_distance", self.car_count)
        floats("setup_control_max_relative_speed", self.car_count)
        ints("pogo_pending", self.car_count)
        ints("pogo_age", self.car_count)
        floats("pogo_features", self.car_count * 4)
        ints("family_event_count", self.car_count * FAMILY_COUNT)
        ints("family_lockout", self.car_count * FAMILY_COUNT)
        ints("family_rearm_count", self.car_count * FAMILY_COUNT)
        ints("duplicate_suppression", self.car_count * FAMILY_COUNT)
        ints("impossible_count", self.car_count)
        if self.evidence_capacity > 0:
            ints("evidence_count", self.car_count)
            evidence_records = self.car_count * self.evidence_capacity
            ints("evidence_family", evidence_records, fill=-1)
            ints("evidence_subtype", evidence_records)
            ints("evidence_tick", evidence_records, fill=-1)
            floats("evidence_features", evidence_records * 4)
        else:
            # The calibration kernel ABI accepts these arrays, but capacity=0
            # prevents all access.  Two constant-size sentinels replace the
            # per-car/per-event diagnostic buffers in production.
            self._disabled_evidence_int = wp.zeros(1, dtype=wp.int32, device=self.device)
            self._disabled_evidence_float = wp.zeros(
                1, dtype=wp.float32, device=self.device
            )
            self.evidence_count = self._disabled_evidence_int
            self.evidence_family = self._disabled_evidence_int
            self.evidence_subtype = self._disabled_evidence_int
            self.evidence_tick = self._disabled_evidence_int
            self.evidence_features = self._disabled_evidence_float

        ints("continuous_seen_count", self.car_count * FAMILY_COUNT)
        ints("interval_requested", self.car_count)
        ints("interval_detected", self.car_count * CANONICAL_MECHANIC_COUNT)
        ints("interval_paid", self.car_count)
        ints("interval_budget_suppressed", self.car_count)
        ints("mechanics_paid_episode", self.car_count)
        ints("budget_exhausted_latched", self.car_count)
        ints("budget_exhausted_total", self.car_count)
        ints("total_detected", self.car_count * CANONICAL_MECHANIC_COUNT)
        ints("total_paid", self.car_count * CANONICAL_MECHANIC_COUNT)
        ints("total_budget_suppressed", self.car_count)

        ints("dash_previous_on_ground", self.car_count)
        ints("dash_previous_has_jumped", self.car_count)
        ints("dash_previous_has_flipped", self.car_count)
        floats("dash_previous_air_time", self.car_count)
        ints("dash_previous_wheel_mask", self.car_count)
        vectors("dash_previous_velocity", self.car_count)
        ints("dash_pending_flip_tick", self.car_count, fill=-1)
        vectors("dash_pending_pre_velocity", self.car_count)
        ints("dash_last_success_flip_tick", self.car_count, fill=-1)
        ints("dash_last_success_landing_tick", self.car_count, fill=-1)
        floats("dash_last_pre_tangent_speed", self.car_count)
        ints("dash_success_total", self.car_count)
        ints("dash_surface_total", self.car_count * 4)
        ints("zap_stage", self.car_count)
        ints("zap_stage_tick", self.car_count, fill=-1)
        ints("zap_total", self.car_count)
        ints("double_dash_total", self.car_count)

        ints("reset_previous_untimed", self.car_count)
        ints("reset_previous_has_flipped", self.car_count)
        ints("reset_previous_has_double_jumped", self.car_count)
        ints("reset_pending_body", self.car_count)
        ints("reset_pending_preflip", self.car_count)
        ints("reset_armed", self.car_count, fill=1)
        ints("reset_completion_total", self.car_count)
        ints("chain_reset_total", self.car_count)
        ints("preflip_reset_total", self.car_count)

        ints("v3_touch_latched", self.car_count)
        ints("legitimate_touch_total", self.car_count)
        ints("flip_touch_total", self.car_count)
        ints("control_ticks", self.car_count)
        floats("control_max_distance", self.car_count)
        floats("control_max_relative_speed", self.car_count)
        ints("pending_active", self.car_count)
        ints("pending_age", self.car_count)
        ints("pending_recognized", self.car_count)
        ints("pending_controlled", self.car_count)
        ints("pending_contest", self.car_count)
        ints("pending_power", self.car_count)
        vectors("pending_ball_position", self.car_count)
        floats("pending_features", self.car_count * 8)
        ints("interval_bad_flip", self.car_count)
        ints("interval_exemptions", self.car_count * 6)
        ints("outcome_total", self.car_count * 6)
        ints("exemption_flag_total", self.car_count * 6)
        ints("impossible_total", self.car_count)
        if self.evidence_capacity > 0:
            ints("outcome_evidence_count", self.car_count)
            outcome_records = self.car_count * self.evidence_capacity
            ints("outcome_evidence_outcome", outcome_records)
            ints("outcome_evidence_tick", outcome_records, fill=-1)
            floats("outcome_evidence_features", outcome_records * 8)
        else:
            self.outcome_evidence_count = self._disabled_evidence_int
            self.outcome_evidence_outcome = self._disabled_evidence_int
            self.outcome_evidence_tick = self._disabled_evidence_int
            self.outcome_evidence_features = self._disabled_evidence_float

        floats("mechanics_component", self.num_worlds)
        floats("bad_flip_component", self.num_worlds)
        floats("total_component", self.num_worlds)

        wp.launch(
            initialize_mechanics_shadow,
            dim=self.car_count,
            inputs=[
                world.state.car_pos,
                world.state.car_vel,
                world.state.car_quat,
                world.state.car_ang_vel,
                world.state.has_flipped,
                world.vehicle.wheel_contact,
                world.vehicle.contact_count,
                world.state.ball_pos,
                world.state.ball_vel,
                self.previous_position,
                self.previous_velocity,
                self.previous_quaternion,
                self.previous_angular_velocity,
                self.previous_has_flipped,
                self.previous_wheel_mask,
                self.previous_chassis_contacts,
                self.previous_ball_velocity,
                self.previous_ball_position,
                self.history_car_position,
                self.history_car_quaternion,
                self.history_ball_position,
                self.history_cursor,
                self.history_count,
            ],
            device=self.device,
        )
        self.reset(wp.ones(self.num_worlds, dtype=wp.int32, device=self.device))

    @property
    def logical_bytes(self) -> int:
        return sum(int(item["logical_bytes"]) for item in self._inventory.values())

    def memory_inventory(self) -> dict[str, Any]:
        return {
            "logical_bytes": self.logical_bytes,
            "evidence_capacity_per_car": self.evidence_capacity,
            "calibration_evidence_buffers_allocated": self.evidence_capacity > 0,
            "disabled_kernel_abi_sentinel_bytes": 8 if self.evidence_capacity == 0 else 0,
            "arrays": {name: dict(item) for name, item in self._inventory.items()},
        }

    def begin_decision(self) -> None:
        wp.launch(
            gameplay_v3_begin_decision,
            dim=self.car_count,
            inputs=[
                self.interval_requested,
                self.interval_detected,
                self.interval_paid,
                self.interval_budget_suppressed,
                self.interval_bad_flip,
                self.interval_exemptions,
                self.mechanics_component,
                self.bad_flip_component,
                self.total_component,
            ],
            device=self.device,
        )

    def launch_tick(self) -> None:
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
                self.previous_position,
                self.previous_velocity,
                self.previous_quaternion,
                self.previous_angular_velocity,
                self.previous_has_flipped,
                self.previous_wheel_mask,
                self.previous_chassis_contacts,
                self.previous_ball_velocity,
                self.previous_ball_position,
                self.previous_episode_tick,
                self.history_car_position,
                self.history_car_quaternion,
                self.history_ball_position,
                self.history_cursor,
                self.history_count,
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
                self.setup_combined_ticks,
                self.setup_control_active,
                self.setup_terminal_pending,
                self.setup_control_max_distance,
                self.setup_control_max_relative_speed,
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
        wp.launch(
            gameplay_v3_track_tick,
            dim=self.car_count,
            inputs=[
                self.evidence_capacity,
                world.rival2.episode_ticks,
                world.state.car_pos,
                world.state.car_vel,
                world.state.on_ground,
                world.state.has_jumped,
                world.state.has_double_jumped,
                world.state.has_flipped,
                world.state.is_flipping,
                world.state.air_time,
                world.state.flip_rel_torque,
                world.state.ball_pos,
                world.state.ball_vel,
                world.vehicle.wheel_contact,
                world.vehicle.wheel_hit_normal,
                world.vehicle.wheel_hit_face,
                world.car_ball.hit_this_tick,
                world.car_ball_b.hit_this_tick,
                world.car_ball.pre_car_velocity_bt,
                world.car_ball_b.pre_car_velocity_bt,
                world.car_ball.pre_car_angular_velocity,
                world.car_ball_b.pre_car_angular_velocity,
                world.car_ball.contact_point_a_bt,
                world.car_ball_b.contact_point_a_bt,
                world.car_ball.extra_hit_velocity_uu,
                world.car_ball_b.extra_hit_velocity_uu,
                self.family_event_count,
                self.continuous_seen_count,
                self.interval_requested,
                self.interval_detected,
                self.total_detected,
                self.dash_previous_on_ground,
                self.dash_previous_has_jumped,
                self.dash_previous_has_flipped,
                self.dash_previous_air_time,
                self.dash_previous_wheel_mask,
                self.dash_previous_velocity,
                self.dash_pending_flip_tick,
                self.dash_pending_pre_velocity,
                self.dash_last_success_flip_tick,
                self.dash_last_success_landing_tick,
                self.dash_last_pre_tangent_speed,
                self.dash_success_total,
                self.dash_surface_total,
                self.zap_stage,
                self.zap_stage_tick,
                self.zap_total,
                self.double_dash_total,
                self.reset_previous_untimed,
                self.reset_previous_has_flipped,
                self.reset_previous_has_double_jumped,
                self.reset_pending_body,
                self.reset_pending_preflip,
                self.reset_armed,
                self.reset_completion_total,
                self.chain_reset_total,
                self.preflip_reset_total,
                self.v3_touch_latched,
                self.legitimate_touch_total,
                self.flip_touch_total,
                self.control_ticks,
                self.control_max_distance,
                self.control_max_relative_speed,
                self.pending_active,
                self.pending_age,
                self.pending_recognized,
                self.pending_controlled,
                self.pending_contest,
                self.pending_power,
                self.pending_ball_position,
                self.pending_features,
                self.interval_bad_flip,
                self.interval_exemptions,
                self.outcome_total,
                self.exemption_flag_total,
                self.impossible_total,
                self.outcome_evidence_count,
                self.outcome_evidence_outcome,
                self.outcome_evidence_tick,
                self.outcome_evidence_features,
            ],
            device=self.device,
        )

    def compose_reward(self) -> None:
        world = self.world
        wp.launch(
            gameplay_v3_compose_reward,
            dim=self.num_worlds,
            inputs=[
                world.rival2.interval_tick,
                world.rival2.reward,
                self.interval_requested,
                self.interval_detected,
                self.interval_paid,
                self.interval_budget_suppressed,
                self.interval_bad_flip,
                self.mechanics_paid_episode,
                self.budget_exhausted_latched,
                self.budget_exhausted_total,
                self.total_paid,
                self.total_budget_suppressed,
                self.mechanics_component,
                self.bad_flip_component,
                self.total_component,
            ],
            device=self.device,
        )

    def reset(self, reset_mask: wp.array) -> None:
        world = self.world
        wp.launch(
            gameplay_v3_reset_continuous,
            dim=self.car_count,
            inputs=[
                reset_mask,
                world.state.car_pos,
                world.state.car_vel,
                world.state.car_quat,
                world.state.car_ang_vel,
                world.state.has_flipped,
                world.vehicle.wheel_contact,
                world.vehicle.contact_count,
                world.state.ball_pos,
                world.state.ball_vel,
                self.previous_position,
                self.previous_velocity,
                self.previous_quaternion,
                self.previous_angular_velocity,
                self.previous_has_flipped,
                self.previous_wheel_mask,
                self.previous_chassis_contacts,
                self.previous_ball_velocity,
                self.previous_ball_position,
                self.previous_episode_tick,
                self.history_car_position,
                self.history_car_quaternion,
                self.history_ball_position,
                self.history_cursor,
                self.history_count,
                self.flip_kind,
                self.flip_age,
                self.flip_cancel_age,
                self.flip_pitch_path,
                self.flip_roll_path,
                self.flip_yaw_path,
                self.touch_latched,
                self.possession_owner,
                self.possession_gap,
                self.possession_touch_count,
                self.carry_support_ticks,
                self.setup_ticks,
                self.setup_roll_path,
                self.setup_yaw_path,
                self.setup_orientation_stage,
                self.setup_combined_ticks,
                self.setup_control_active,
                self.setup_terminal_pending,
                self.setup_control_max_distance,
                self.setup_control_max_relative_speed,
                self.pogo_pending,
                self.pogo_age,
                self.family_event_count,
                self.family_lockout,
                self.continuous_seen_count,
            ],
            device=self.device,
        )
        wp.launch(
            gameplay_v3_reset_source_state,
            dim=self.car_count,
            inputs=[
                reset_mask,
                world.state.on_ground,
                world.state.has_jumped,
                world.state.has_double_jumped,
                world.state.has_flipped,
                world.state.air_time,
                world.state.car_vel,
                world.vehicle.wheel_contact,
                self.dash_previous_on_ground,
                self.dash_previous_has_jumped,
                self.dash_previous_has_flipped,
                self.dash_previous_air_time,
                self.dash_previous_wheel_mask,
                self.dash_previous_velocity,
                self.dash_pending_flip_tick,
                self.dash_last_success_flip_tick,
                self.dash_last_success_landing_tick,
                self.zap_stage,
                self.zap_stage_tick,
                self.reset_previous_untimed,
                self.reset_previous_has_flipped,
                self.reset_previous_has_double_jumped,
                self.reset_pending_body,
                self.reset_pending_preflip,
                self.reset_armed,
                self.v3_touch_latched,
                self.control_ticks,
                self.control_max_distance,
                self.control_max_relative_speed,
                self.pending_active,
                self.pending_age,
                self.pending_recognized,
                self.pending_controlled,
                self.pending_contest,
                self.pending_power,
                self.mechanics_paid_episode,
                self.budget_exhausted_latched,
            ],
            device=self.device,
        )


def default_threshold_path() -> Path:
    return (
        Path(__file__).resolve().parents[1]
        / "results"
        / "rival2"
        / "mechanics_calibration_v1"
        / "thresholds.json"
    )


__all__ = [
    "CANONICAL_MECHANIC_COUNT",
    "CANONICAL_MECHANIC_NAMES",
    "CONTEST_CONTACT_WINDOW_TICKS",
    "OUTCOME_NAMES",
    "Rival2GameplayV3State",
    "contest_convergence_exempt",
    "controlled_flick_exempt",
    "default_threshold_path",
    "flip_contact_candidate",
    "gameplay_v3_begin_decision",
    "gameplay_v3_compose_reward",
    "gameplay_v3_reset_continuous",
    "gameplay_v3_reset_source_state",
    "gameplay_v3_track_tick",
    "power_contact_exempt",
    "primary_flip_outcome",
]
