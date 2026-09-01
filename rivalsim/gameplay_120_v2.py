"""Gameplay 120 V2 competitive control and physical bad-flip guard.

This production path is deliberately independent from Gameplay V3.  It uses
only authoritative kinematics/contact state, contains no named-mechanic
classifier, and never allocates Gameplay V3 detector state.
"""

from __future__ import annotations

from typing import Any

import warp as wp

from rivalsim.gameplay_120 import (
    CONTEST_ASSOCIATION_BALL_DISPLACEMENT_MAX,
    CONTEST_CONTACT_WINDOW_TICKS,
    CONTEST_OPPONENT_CLOSING_SPEED_MIN,
    CONTEST_OPPONENT_DISTANCE_MAX,
    CONTEST_SELF_CLOSING_SPEED_MIN,
    CONTEST_TIME_TO_BALL_DELTA_MAX,
    POWER_BALL_DELTA_V_MIN,
    POWER_ROTATIONAL_CLOSING_SPEED_MIN,
    POWER_ROTATIONAL_SHARE_MIN,
    POWER_TOTAL_CLOSING_SPEED_MIN,
)
from rivalsim.rival2_contracts import (
    GAMEPLAY_120_V2_CONTROL_DISTANCE_MAX,
    GAMEPLAY_120_V2_CONTROL_DISTANCE_RANGE,
    GAMEPLAY_120_V2_CONTROL_RELATIVE_SPEED_SCALE,
    GAMEPLAY_120_V2_CONTROL_REWARD,
    GAMEPLAY_120_V2_RETAINED_CONTROL_THRESHOLD,
    GAMEPLAY_120_V2_RETAINED_CONTROL_WINDOW_TICKS,
    GAMEPLAY_120_V2_UNNECESSARY_FLIP_PENALTY,
)

NO_TOUCH_TIMEOUT_TICKS = 15 * 120
EPISODE_LIMIT_TICKS = 45 * 120

OUTCOME_NONE = 0
OUTCOME_EXEMPT_CONTESTED_50 = 1
OUTCOME_EXEMPT_POWER_CONTACT = 2
OUTCOME_EXEMPT_RETAINED_CONTROL = 3
OUTCOME_UNNECESSARY_FLIP_THROUGH_CONTACT = 4
OUTCOME_COUNT = 5


def physical_control_score(
    *, distance: float, relative_speed: float
) -> float:
    """Return the frozen V2 score for focused deterministic tests."""

    proximity = max(
        0.0,
        min(
            1.0,
            (GAMEPLAY_120_V2_CONTROL_DISTANCE_MAX - distance)
            / GAMEPLAY_120_V2_CONTROL_DISTANCE_RANGE,
        ),
    )
    velocity_match = max(
        0.0,
        min(
            1.0,
            1.0 - relative_speed / GAMEPLAY_120_V2_CONTROL_RELATIVE_SPEED_SCALE,
        ),
    )
    return proximity * velocity_match


def physical_flip_outcome(
    *, contested_50: bool, power_contact: bool, retained_control: bool
) -> int:
    """Pure deterministic V2 exemption precedence."""

    if contested_50:
        return OUTCOME_EXEMPT_CONTESTED_50
    if power_contact:
        return OUTCOME_EXEMPT_POWER_CONTACT
    if retained_control:
        return OUTCOME_EXEMPT_RETAINED_CONTROL
    return OUTCOME_UNNECESSARY_FLIP_THROUGH_CONTACT


@wp.func
def _safe_unit(value: wp.vec3) -> wp.vec3:
    length = wp.length(value)
    if length <= 1.0e-6:
        return wp.vec3(0.0)
    return value / length


@wp.func
def _control_score(
    car_position: wp.vec3,
    car_velocity: wp.vec3,
    ball_position: wp.vec3,
    ball_velocity: wp.vec3,
) -> float:
    distance = wp.length(ball_position - car_position)
    relative_speed = wp.length(ball_velocity - car_velocity)
    proximity = wp.clamp(
        (GAMEPLAY_120_V2_CONTROL_DISTANCE_MAX - distance)
        / GAMEPLAY_120_V2_CONTROL_DISTANCE_RANGE,
        0.0,
        1.0,
    )
    velocity_match = wp.clamp(
        1.0 - relative_speed / GAMEPLAY_120_V2_CONTROL_RELATIVE_SPEED_SCALE,
        0.0,
        1.0,
    )
    return proximity * velocity_match


@wp.func
def _resolve_pending(
    car: int,
    pending_active: wp.array(dtype=wp.int32),
    pending_contest: wp.array(dtype=wp.int32),
    pending_power: wp.array(dtype=wp.int32),
    pending_retained_control: wp.array(dtype=wp.int32),
    pending_age: wp.array(dtype=wp.int32),
    pending_self_contact_tick: wp.array(dtype=wp.int32),
    interval_bad_flip: wp.array(dtype=wp.int32),
    interval_contest_exempt: wp.array(dtype=wp.int32),
    interval_power_exempt: wp.array(dtype=wp.int32),
    interval_retained_control_exempt: wp.array(dtype=wp.int32),
    bad_flip_total: wp.array(dtype=wp.int32),
    contest_exempt_total: wp.array(dtype=wp.int32),
    power_exempt_total: wp.array(dtype=wp.int32),
    retained_control_exempt_total: wp.array(dtype=wp.int32),
):
    if pending_contest[car] != 0:
        interval_contest_exempt[car] = interval_contest_exempt[car] + 1
        contest_exempt_total[car] = contest_exempt_total[car] + 1
    elif pending_power[car] != 0:
        interval_power_exempt[car] = interval_power_exempt[car] + 1
        power_exempt_total[car] = power_exempt_total[car] + 1
    elif pending_retained_control[car] != 0:
        interval_retained_control_exempt[car] = (
            interval_retained_control_exempt[car] + 1
        )
        retained_control_exempt_total[car] = retained_control_exempt_total[car] + 1
    else:
        interval_bad_flip[car] = interval_bad_flip[car] + 1
        bad_flip_total[car] = bad_flip_total[car] + 1
    pending_active[car] = 0
    pending_contest[car] = 0
    pending_power[car] = 0
    pending_retained_control[car] = 0
    pending_age[car] = 0
    pending_self_contact_tick[car] = -1


@wp.kernel(enable_backward=False)
def gameplay_120_v2_begin_decision(
    interval_bad_flip: wp.array(dtype=wp.int32),
    interval_contest_exempt: wp.array(dtype=wp.int32),
    interval_power_exempt: wp.array(dtype=wp.int32),
    interval_retained_control_exempt: wp.array(dtype=wp.int32),
    bad_flip_component: wp.array(dtype=wp.float32),
    control_component: wp.array(dtype=wp.float32),
):
    car = wp.tid()
    env = car // 2
    interval_bad_flip[car] = 0
    interval_contest_exempt[car] = 0
    interval_power_exempt[car] = 0
    interval_retained_control_exempt[car] = 0
    if car % 2 == 0:
        bad_flip_component[env] = 0.0
        control_component[env] = 0.0


@wp.kernel(enable_backward=False)
def gameplay_120_v2_track_tick(
    episode_ticks: wp.array(dtype=wp.int32),
    no_touch_ticks: wp.array(dtype=wp.int32),
    goal_scored: wp.array(dtype=wp.int32),
    car_pos: wp.array(dtype=wp.vec3),
    car_vel: wp.array(dtype=wp.vec3),
    has_flipped: wp.array(dtype=wp.int32),
    is_flipping: wp.array(dtype=wp.int32),
    flip_rel_torque: wp.array(dtype=wp.vec3),
    ball_pos: wp.array(dtype=wp.vec3),
    ball_vel: wp.array(dtype=wp.vec3),
    car_a_hit: wp.array(dtype=wp.int32),
    car_b_hit: wp.array(dtype=wp.int32),
    car_a_pre_velocity_bt: wp.array(dtype=wp.vec3),
    car_b_pre_velocity_bt: wp.array(dtype=wp.vec3),
    car_a_pre_ball_velocity_bt: wp.array(dtype=wp.vec3),
    car_b_pre_ball_velocity_bt: wp.array(dtype=wp.vec3),
    car_a_pre_angular_velocity: wp.array(dtype=wp.vec3),
    car_b_pre_angular_velocity: wp.array(dtype=wp.vec3),
    car_a_contact_point_bt: wp.array(dtype=wp.vec3),
    car_b_contact_point_bt: wp.array(dtype=wp.vec3),
    car_a_ball_delta_v: wp.array(dtype=wp.vec3),
    car_b_ball_delta_v: wp.array(dtype=wp.vec3),
    touch_latched: wp.array(dtype=wp.int32),
    opponent_touch_latched: wp.array(dtype=wp.int32),
    recent_opponent_contact_tick: wp.array(dtype=wp.int32),
    recent_opponent_contact_ball_position: wp.array(dtype=wp.vec3),
    pending_active: wp.array(dtype=wp.int32),
    pending_age: wp.array(dtype=wp.int32),
    pending_contest: wp.array(dtype=wp.int32),
    pending_power: wp.array(dtype=wp.int32),
    pending_retained_control: wp.array(dtype=wp.int32),
    pending_self_contact_tick: wp.array(dtype=wp.int32),
    pending_ball_position: wp.array(dtype=wp.vec3),
    interval_bad_flip: wp.array(dtype=wp.int32),
    interval_contest_exempt: wp.array(dtype=wp.int32),
    interval_power_exempt: wp.array(dtype=wp.int32),
    interval_retained_control_exempt: wp.array(dtype=wp.int32),
    legitimate_touch_total: wp.array(dtype=wp.int32),
    flip_touch_total: wp.array(dtype=wp.int32),
    bad_flip_total: wp.array(dtype=wp.int32),
    contest_exempt_total: wp.array(dtype=wp.int32),
    power_exempt_total: wp.array(dtype=wp.int32),
    retained_control_exempt_total: wp.array(dtype=wp.int32),
):
    car = wp.tid()
    env = car // 2
    local = car % 2
    other = env * 2 + (1 - local)
    tick = episode_ticks[env] + 1
    reports = car_a_hit[env] if local == 0 else car_b_hit[env]
    other_reports = car_b_hit[env] if local == 0 else car_a_hit[env]
    touch_onset = reports != 0 and touch_latched[car] == 0
    opponent_touch_onset = other_reports != 0 and opponent_touch_latched[car] == 0
    touch_latched[car] = wp.int32(reports != 0)
    opponent_touch_latched[car] = wp.int32(other_reports != 0)

    next_no_touch = no_touch_ticks[env] + 1
    if car_a_hit[env] != 0 or car_b_hit[env] != 0:
        next_no_touch = 0
    reset_transition = (
        goal_scored[env] != 0
        or next_no_touch >= NO_TOUCH_TIMEOUT_TICKS
        or tick >= EPISODE_LIMIT_TICKS
    )
    if reset_transition:
        # The reset kernel clears pending candidates. Neither control reward nor
        # a retained-control decision is allowed on the transition tick.
        return

    if opponent_touch_onset:
        recent_opponent_contact_tick[car] = tick
        recent_opponent_contact_ball_position[car] = ball_pos[env]

    if pending_active[car] != 0:
        candidate_age = tick - pending_self_contact_tick[car]
        if (
            opponent_touch_onset
            and candidate_age <= CONTEST_CONTACT_WINDOW_TICKS
            and wp.length(ball_pos[env] - pending_ball_position[car])
            <= CONTEST_ASSOCIATION_BALL_DISPLACEMENT_MAX
        ):
            pending_contest[car] = 1
        if (
            candidate_age >= 1
            and candidate_age <= GAMEPLAY_120_V2_RETAINED_CONTROL_WINDOW_TICKS
            and _control_score(
                car_pos[car], car_vel[car], ball_pos[env], ball_vel[env]
            )
            >= GAMEPLAY_120_V2_RETAINED_CONTROL_THRESHOLD
        ):
            pending_retained_control[car] = 1
        pending_age[car] = candidate_age
        if (
            candidate_age >= GAMEPLAY_120_V2_RETAINED_CONTROL_WINDOW_TICKS
            or touch_onset
        ):
            _resolve_pending(
                car,
                pending_active,
                pending_contest,
                pending_power,
                pending_retained_control,
                pending_age,
                pending_self_contact_tick,
                interval_bad_flip,
                interval_contest_exempt,
                interval_power_exempt,
                interval_retained_control_exempt,
                bad_flip_total,
                contest_exempt_total,
                power_exempt_total,
                retained_control_exempt_total,
            )

    if touch_onset:
        legitimate_touch_total[car] = legitimate_touch_total[car] + 1
        directional_dodge = (
            is_flipping[car] != 0
            and has_flipped[car] != 0
            and wp.length(flip_rel_torque[car]) > 0.25
        )
        if directional_dodge:
            flip_touch_total[car] = flip_touch_total[car] + 1
            pre_velocity_bt = (
                car_a_pre_velocity_bt[env]
                if local == 0
                else car_b_pre_velocity_bt[env]
            )
            opponent_pre_velocity_bt = (
                car_b_pre_velocity_bt[env]
                if local == 0
                else car_a_pre_velocity_bt[env]
            )
            pre_ball_velocity_bt = (
                car_a_pre_ball_velocity_bt[env]
                if local == 0
                else car_b_pre_ball_velocity_bt[env]
            )
            pre_angular = (
                car_a_pre_angular_velocity[env]
                if local == 0
                else car_b_pre_angular_velocity[env]
            )
            point_bt = (
                car_a_contact_point_bt[env]
                if local == 0
                else car_b_contact_point_bt[env]
            )
            ball_delta_vector = (
                car_a_ball_delta_v[env]
                if local == 0
                else car_b_ball_delta_v[env]
            )
            pre_velocity = pre_velocity_bt * 50.0
            opponent_pre_velocity = opponent_pre_velocity_bt * 50.0
            pre_ball_velocity = pre_ball_velocity_bt * 50.0
            contact_world = point_bt * 50.0
            offset = contact_world - car_pos[car]
            direction = _safe_unit(ball_pos[env] - car_pos[car])
            rotational = wp.max(
                wp.dot(wp.cross(pre_angular, offset), direction), 0.0
            )
            translational = wp.max(
                wp.dot(pre_velocity - pre_ball_velocity, direction), 0.0
            )
            total_closing = rotational + translational
            rotational_share = rotational / wp.max(total_closing, 1.0e-6)
            ball_delta = wp.length(ball_delta_vector)

            opponent_direction = _safe_unit(ball_pos[env] - car_pos[other])
            self_closing = wp.dot(pre_velocity - pre_ball_velocity, direction)
            opponent_closing = wp.dot(
                opponent_pre_velocity - pre_ball_velocity, opponent_direction
            )
            self_distance = wp.length(ball_pos[env] - car_pos[car])
            opponent_distance = wp.length(ball_pos[env] - car_pos[other])
            self_time = self_distance / wp.max(self_closing, 1.0e-6)
            opponent_time = opponent_distance / wp.max(opponent_closing, 1.0e-6)
            convergence = (
                opponent_distance <= CONTEST_OPPONENT_DISTANCE_MAX
                and self_closing >= CONTEST_SELF_CLOSING_SPEED_MIN
                and opponent_closing >= CONTEST_OPPONENT_CLOSING_SPEED_MIN
                and wp.abs(self_time - opponent_time)
                <= CONTEST_TIME_TO_BALL_DELTA_MAX
            )
            power = (
                total_closing >= POWER_TOTAL_CLOSING_SPEED_MIN
                and rotational >= POWER_ROTATIONAL_CLOSING_SPEED_MIN
                and rotational_share >= POWER_ROTATIONAL_SHARE_MIN
                and ball_delta >= POWER_BALL_DELTA_V_MIN
            )
            recent_tick = recent_opponent_contact_tick[car]
            recent_age = tick - recent_tick
            adjacent_recent = (
                recent_tick >= 0
                and recent_age >= 0
                and recent_age <= CONTEST_CONTACT_WINDOW_TICKS
                and wp.length(
                    ball_pos[env] - recent_opponent_contact_ball_position[car]
                )
                <= CONTEST_ASSOCIATION_BALL_DISPLACEMENT_MAX
            )
            pending_active[car] = 1
            pending_age[car] = 0
            pending_contest[car] = wp.int32(adjacent_recent or convergence)
            pending_power[car] = wp.int32(power)
            pending_retained_control[car] = 0
            pending_self_contact_tick[car] = tick
            pending_ball_position[car] = ball_pos[env]


@wp.kernel(enable_backward=False)
def gameplay_120_v2_compose_reward(
    interval_tick: wp.array(dtype=wp.int32),
    reset_mask: wp.array(dtype=wp.int32),
    car_pos: wp.array(dtype=wp.vec3),
    car_vel: wp.array(dtype=wp.vec3),
    ball_pos: wp.array(dtype=wp.vec3),
    ball_vel: wp.array(dtype=wp.vec3),
    reward: wp.array(dtype=wp.float32),
    interval_bad_flip: wp.array(dtype=wp.int32),
    bad_flip_component: wp.array(dtype=wp.float32),
    control_component: wp.array(dtype=wp.float32),
    control_score_current: wp.array(dtype=wp.float32),
    control_score_sum_total: wp.array(dtype=wp.float32),
    control_score_tick_total: wp.array(dtype=wp.int32),
    control_score_positive_total: wp.array(dtype=wp.int32),
    control_score_ge_025_total: wp.array(dtype=wp.int32),
    control_score_ge_05_total: wp.array(dtype=wp.int32),
    control_reward_sum_total: wp.array(dtype=wp.float32),
):
    env = wp.tid()
    if interval_tick[env] != 1:
        return
    car_base = env * 2
    if reset_mask[env] != 0:
        control_score_current[car_base] = 0.0
        control_score_current[car_base + 1] = 0.0
        control_component[env] = 0.0
        bad_flip_component[env] = 0.0
        return

    blue_score = _control_score(
        car_pos[car_base], car_vel[car_base], ball_pos[env], ball_vel[env]
    )
    orange_score = _control_score(
        car_pos[car_base + 1], car_vel[car_base + 1], ball_pos[env], ball_vel[env]
    )
    control_score_current[car_base] = blue_score
    control_score_current[car_base + 1] = orange_score
    for local in range(2):
        car = car_base + local
        score = blue_score if local == 0 else orange_score
        control_score_sum_total[car] = control_score_sum_total[car] + score
        control_score_tick_total[car] = control_score_tick_total[car] + 1
        if score > 0.0:
            control_score_positive_total[car] = control_score_positive_total[car] + 1
        if score >= 0.25:
            control_score_ge_025_total[car] = control_score_ge_025_total[car] + 1
        if score >= 0.5:
            control_score_ge_05_total[car] = control_score_ge_05_total[car] + 1

    competitive_control = GAMEPLAY_120_V2_CONTROL_REWARD * (
        blue_score - orange_score
    )
    bad_flip = GAMEPLAY_120_V2_UNNECESSARY_FLIP_PENALTY * float(
        interval_bad_flip[car_base] - interval_bad_flip[car_base + 1]
    )
    component = competitive_control + bad_flip
    blue = reward[car_base] + component
    reward[car_base] = blue
    reward[car_base + 1] = -blue
    control_component[env] = competitive_control
    bad_flip_component[env] = bad_flip
    control_reward_sum_total[env] = (
        control_reward_sum_total[env] + competitive_control
    )


@wp.kernel(enable_backward=False)
def gameplay_120_v2_reset(
    reset_mask: wp.array(dtype=wp.int32),
    touch_latched: wp.array(dtype=wp.int32),
    opponent_touch_latched: wp.array(dtype=wp.int32),
    recent_opponent_contact_tick: wp.array(dtype=wp.int32),
    recent_opponent_contact_ball_position: wp.array(dtype=wp.vec3),
    pending_active: wp.array(dtype=wp.int32),
    pending_age: wp.array(dtype=wp.int32),
    pending_contest: wp.array(dtype=wp.int32),
    pending_power: wp.array(dtype=wp.int32),
    pending_retained_control: wp.array(dtype=wp.int32),
    pending_self_contact_tick: wp.array(dtype=wp.int32),
    pending_ball_position: wp.array(dtype=wp.vec3),
):
    car = wp.tid()
    env = car // 2
    if reset_mask[env] == 0:
        return
    touch_latched[car] = 0
    opponent_touch_latched[car] = 0
    recent_opponent_contact_tick[car] = -1
    recent_opponent_contact_ball_position[car] = wp.vec3(0.0)
    pending_active[car] = 0
    pending_age[car] = 0
    pending_contest[car] = 0
    pending_power[car] = 0
    pending_retained_control[car] = 0
    pending_self_contact_tick[car] = -1
    pending_ball_position[car] = wp.vec3(0.0)


class Rival2Gameplay120V2State:
    """Bounded device state for V2 control telemetry and physical guard."""

    def __init__(self, world: Any):
        self.world = world
        self.device = world.device
        self.num_worlds = world.num_envs
        self.car_count = self.num_worlds * 2
        self._inventory: dict[str, int] = {}

        def ints(name: str, count: int, *, fill: int = 0) -> None:
            value = (
                wp.zeros(count, dtype=wp.int32, device=self.device)
                if fill == 0
                else wp.full(count, fill, dtype=wp.int32, device=self.device)
            )
            setattr(self, name, value)
            self._inventory[name] = count * 4

        def floats(name: str, count: int) -> None:
            setattr(self, name, wp.zeros(count, dtype=wp.float32, device=self.device))
            self._inventory[name] = count * 4

        def vectors(name: str, count: int) -> None:
            setattr(self, name, wp.zeros(count, dtype=wp.vec3, device=self.device))
            self._inventory[name] = count * 12

        ints("touch_latched", self.car_count)
        ints("opponent_touch_latched", self.car_count)
        ints("recent_opponent_contact_tick", self.car_count, fill=-1)
        vectors("recent_opponent_contact_ball_position", self.car_count)
        ints("pending_active", self.car_count)
        ints("pending_age", self.car_count)
        ints("pending_contest", self.car_count)
        ints("pending_power", self.car_count)
        ints("pending_retained_control", self.car_count)
        ints("pending_self_contact_tick", self.car_count, fill=-1)
        vectors("pending_ball_position", self.car_count)
        ints("interval_bad_flip", self.car_count)
        ints("interval_contest_exempt", self.car_count)
        ints("interval_power_exempt", self.car_count)
        ints("interval_retained_control_exempt", self.car_count)
        ints("legitimate_touch_total", self.car_count)
        ints("flip_touch_total", self.car_count)
        ints("bad_flip_total", self.car_count)
        ints("contest_exempt_total", self.car_count)
        ints("power_exempt_total", self.car_count)
        ints("retained_control_exempt_total", self.car_count)
        floats("bad_flip_component", self.num_worlds)
        floats("control_component", self.num_worlds)
        floats("control_score_current", self.car_count)
        floats("control_score_sum_total", self.car_count)
        ints("control_score_tick_total", self.car_count)
        ints("control_score_positive_total", self.car_count)
        ints("control_score_ge_025_total", self.car_count)
        ints("control_score_ge_05_total", self.car_count)
        floats("control_reward_sum_total", self.num_worlds)
        self.reset(wp.ones(self.num_worlds, dtype=wp.int32, device=self.device))

    @property
    def logical_bytes(self) -> int:
        return sum(self._inventory.values())

    def memory_inventory(self) -> dict[str, Any]:
        return {
            "logical_bytes": self.logical_bytes,
            "named_mechanics_arrays": 0,
            "controlled_flick_arrays": 0,
            "arrays": dict(self._inventory),
        }

    def begin_decision(self) -> None:
        wp.launch(
            gameplay_120_v2_begin_decision,
            dim=self.car_count,
            inputs=[
                self.interval_bad_flip,
                self.interval_contest_exempt,
                self.interval_power_exempt,
                self.interval_retained_control_exempt,
                self.bad_flip_component,
                self.control_component,
            ],
            device=self.device,
        )

    def launch_tick(self) -> None:
        world = self.world
        wp.launch(
            gameplay_120_v2_track_tick,
            dim=self.car_count,
            inputs=[
                world.rival2.episode_ticks,
                world.rival2.no_touch_ticks,
                world.lifecycle.goal_scored,
                world.state.car_pos,
                world.state.car_vel,
                world.state.has_flipped,
                world.state.is_flipping,
                world.state.flip_rel_torque,
                world.state.ball_pos,
                world.state.ball_vel,
                world.car_ball.hit_this_tick,
                world.car_ball_b.hit_this_tick,
                world.car_ball.pre_car_velocity_bt,
                world.car_ball_b.pre_car_velocity_bt,
                world.car_ball.pre_ball_velocity_bt,
                world.car_ball_b.pre_ball_velocity_bt,
                world.car_ball.pre_car_angular_velocity,
                world.car_ball_b.pre_car_angular_velocity,
                world.car_ball.contact_point_a_bt,
                world.car_ball_b.contact_point_a_bt,
                world.car_ball.extra_hit_velocity_uu,
                world.car_ball_b.extra_hit_velocity_uu,
                self.touch_latched,
                self.opponent_touch_latched,
                self.recent_opponent_contact_tick,
                self.recent_opponent_contact_ball_position,
                self.pending_active,
                self.pending_age,
                self.pending_contest,
                self.pending_power,
                self.pending_retained_control,
                self.pending_self_contact_tick,
                self.pending_ball_position,
                self.interval_bad_flip,
                self.interval_contest_exempt,
                self.interval_power_exempt,
                self.interval_retained_control_exempt,
                self.legitimate_touch_total,
                self.flip_touch_total,
                self.bad_flip_total,
                self.contest_exempt_total,
                self.power_exempt_total,
                self.retained_control_exempt_total,
            ],
            device=self.device,
        )

    def compose_reward(self) -> None:
        world = self.world
        wp.launch(
            gameplay_120_v2_compose_reward,
            dim=self.num_worlds,
            inputs=[
                world.rival2.interval_tick,
                world.rival2.reset_mask,
                world.state.car_pos,
                world.state.car_vel,
                world.state.ball_pos,
                world.state.ball_vel,
                world.rival2.reward,
                self.interval_bad_flip,
                self.bad_flip_component,
                self.control_component,
                self.control_score_current,
                self.control_score_sum_total,
                self.control_score_tick_total,
                self.control_score_positive_total,
                self.control_score_ge_025_total,
                self.control_score_ge_05_total,
                self.control_reward_sum_total,
            ],
            device=self.device,
        )

    def reset(self, mask: wp.array) -> None:
        wp.launch(
            gameplay_120_v2_reset,
            dim=self.car_count,
            inputs=[
                mask,
                self.touch_latched,
                self.opponent_touch_latched,
                self.recent_opponent_contact_tick,
                self.recent_opponent_contact_ball_position,
                self.pending_active,
                self.pending_age,
                self.pending_contest,
                self.pending_power,
                self.pending_retained_control,
                self.pending_self_contact_tick,
                self.pending_ball_position,
            ],
            device=self.device,
        )


__all__ = [
    "OUTCOME_EXEMPT_CONTESTED_50",
    "OUTCOME_EXEMPT_POWER_CONTACT",
    "OUTCOME_EXEMPT_RETAINED_CONTROL",
    "OUTCOME_UNNECESSARY_FLIP_THROUGH_CONTACT",
    "Rival2Gameplay120V2State",
    "physical_control_score",
    "physical_flip_outcome",
]
